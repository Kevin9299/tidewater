import { FloatType } from 'three/webgpu';
import {
	Fn, float, vec2, vec4, uv, ivec2, mix, max, min, exp, floor, luminance, property, outputStruct, texture, context, uniform, bool,
} from 'three/tsl';
import TAAUNode from 'three/addons/tsl/display/TAAUNode.js';
import { clipAABB, flickerReduction, sampleCurrentDepth, samplePreviousDepth } from 'three/addons/tsl/utils/TAAUtils.js';

// Temporal anti-aliasing / upscaling, based on three's TAAUNode (same history, depth dilation,
// variance clipping and anti-flicker weighting) with two changes that keep the image sharp:
//  - the history is resampled with a 5-tap Catmull-Rom filter instead of bilinear. Bilinear
//    resampling blurs the accumulated image a little more on every frame the camera moves.
//  - the current-frame weight is a uniform (higher at 1:1 than when upscaling from lower res).
// The camera jitter is driven by the app (scene rendered before the post pipeline runs): the app
// calls setViewOffset() before the scene and clearViewOffset() after the post chain.
export class TemporalUpscale extends TAAUNode {

	constructor( beautyNode, depthNode, velocityNode, camera, waterMaskTexture = null ) {

		super( beautyNode, depthNode, velocityNode, camera );
		this.frameWeight = uniform( 0.06 ).setName( 'taaWeight' );
		// water surface pixels (SceneRenderer waterMask, g = 1): see resolve()
		this.waterMaskNode = waterMaskTexture ? texture( waterMaskTexture ) : null;

	}

	setup( builder ) {

		const state = builder.context.renderPipelineState;
		if ( state ) state.viewOffsetOwner = this;

		if ( builder.renderer.reversedDepthBuffer === true ) this._previousDepthRenderTarget.depthTexture.type = FloatType;

		const historyNode = texture( this._historyRenderTarget.textures[ 0 ] );
		const lockNode = texture( this._historyRenderTarget.textures[ 1 ] );

		const colorOutput = property( 'vec4' );
		const lockOutput = property( 'vec4' );
		const outputNode = outputStruct( colorOutput, lockOutput );

		// Catmull-Rom history lookup with 5 bilinear taps (the 4 corner taps are dropped)
		const sampleHistory = ( uvIn ) => {

			const size = vec2( historyNode.size() );
			const samplePos = uvIn.mul( size );
			const texPos1 = floor( samplePos.sub( 0.5 ) ).add( 0.5 );
			const f = samplePos.sub( texPos1 );
			const w0 = f.mul( f.mul( f.mul( - 0.5 ).add( 1.0 ) ).sub( 0.5 ) );
			const w1 = f.mul( f ).mul( f.mul( 1.5 ).sub( 2.5 ) ).add( 1.0 );
			const w2 = f.mul( f.mul( f.mul( - 1.5 ).add( 2.0 ) ).add( 0.5 ) );
			const w3 = f.mul( f ).mul( f.mul( 0.5 ).sub( 0.5 ) );
			const w12 = w1.add( w2 );
			const tp0 = texPos1.sub( 1 ).div( size );
			const tp3 = texPos1.add( 2 ).div( size );
			const tp12 = texPos1.add( w2.div( w12 ) ).div( size );
			const s = ( x, y ) => historyNode.sample( vec2( x, y ) );
			const wa = w12.x.mul( w0.y ), wb = w0.x.mul( w12.y ), wc = w12.x.mul( w12.y ), wd = w3.x.mul( w12.y ), we = w12.x.mul( w3.y );
			const sum = s( tp12.x, tp0.y ).mul( wa ).add( s( tp0.x, tp12.y ).mul( wb ) ).add( s( tp12.x, tp12.y ).mul( wc ) )
				.add( s( tp3.x, tp12.y ).mul( wd ) ).add( s( tp12.x, tp3.y ).mul( we ) );
			return max( sum.div( wa.add( wb ).add( wc ).add( wd ).add( we ) ), vec4( 0 ) );

		};

		const resolve = Fn( () => {

			const uvNode = uv();
			const inputSizeF = vec2( this.beautyNode.size() );

			// output pixel center in input-pixel coordinates; closest jittered input tap
			const pIn = uvNode.mul( inputSizeF );
			const closestTapF = pIn.sub( vec2( 0.5 ).add( this._jitterOffset ) ).round();
			const closestTap = ivec2( closestTapF );

			// depth dilation around the closest input tap
			const currentDepth = sampleCurrentDepth( this.depthNode, closestTapF, this._cameraNearFar );
			const closestDepth = currentDepth.get( 'closestDepth' );
			const closestPositionTexel = currentDepth.get( 'closestPositionTexel' );
			const farthestDepth = currentDepth.get( 'farthestDepth' );

			// reproject using the velocity at the dilated depth tap
			const offsetUV = this.velocityNode.load( closestPositionTexel ).xy.mul( vec2( 0.5, - 0.5 ) );
			const historyUV = uvNode.sub( offsetUV );
			const previousDepth = samplePreviousDepth( this._previousDepthNode, historyUV, this._previousCameraProjectionMatrixInverse, this._previousCameraWorldMatrix, this._cameraWorldMatrixInverse, this._cameraNearFar, this.camera );

			// history validity
			const isValidUV = historyUV.greaterThanEqual( 0 ).all().and( historyUV.lessThanEqual( 1 ).all() );
			const isEdge = farthestDepth.sub( closestDepth ).greaterThan( this.edgeDepthDiff );
			const isDisocclusion = closestDepth.sub( previousDepth ).greaterThan( this.depthThreshold );
			// The water surface moves on its own (waves, the wake and bow wave travelling with a boat) while
			// its motion vectors only carry the camera's motion: near the camera its depth changes between
			// frames by more than the disocclusion threshold. Rejecting its history there shows the raw
			// jittered frame (flicker), so water always keeps its history and relies on the variance clip.
			const isWater = this.waterMaskNode ? this.waterMaskNode.load( closestTap ).g.greaterThan( 0.5 ) : bool( false );
			const hasValidHistory = isValidUV.and( isEdge.or( isDisocclusion.not() ).or( isWater ) );

			// 9-tap Blackman-Harris (Gaussian approximation) reconstruction of the current frame and
			// the moments for the variance clip
			const sumColor = vec4( 0 ).toVar();
			const sumWeight = float( 0 ).toVar();
			const moment1 = vec4( 0 ).toVar();
			const moment2 = vec4( 0 ).toVar();
			for ( const [ x, y ] of [ [ - 1, - 1 ], [ 0, - 1 ], [ 1, - 1 ], [ - 1, 0 ], [ 0, 0 ], [ 1, 0 ], [ - 1, 1 ], [ 0, 1 ], [ 1, 1 ] ] ) {

				const tap = closestTap.add( ivec2( x, y ) );
				const delta = pIn.sub( vec2( tap ).add( vec2( 0.5 ).add( this._jitterOffset ) ) );
				const w = exp( delta.dot( delta ).mul( - 2.29 ) );
				const c = this.beautyNode.load( tap ).max( 0 );
				sumColor.addAssign( c.mul( w ) );
				sumWeight.addAssign( w );
				moment1.addAssign( c );
				moment2.addAssign( c.pow2() );

			}

			const currentColor = sumColor.div( sumWeight.max( 1e-5 ) );

			const mean = moment1.div( 9 );
			const motionFactor = uvNode.sub( historyUV ).mul( inputSizeF ).length().div( this.maxVelocityLength ).saturate();
			const varianceGamma = mix( 0.5, 1, motionFactor.oneMinus().pow2() );
			const variance = moment2.div( 9 ).sub( mean.pow2() ).max( 0 ).sqrt().mul( varianceGamma );
			const minColor = mean.sub( variance );
			const maxColor = mean.add( variance );

			const historyColor = sampleHistory( historyUV );
			const clippedHistoryColor = clipAABB( mean.clamp( minColor, maxColor ), historyColor, minColor, maxColor );

			// thin features lock the history a little (less flicker on wires, masts, leaves)
			const meanLuma = luminance( mean.rgb ).toConst();
			const thinFeature = luminance( currentColor.rgb ).sub( meanLuma ).abs().div( meanLuma ).smoothstep( 0, 0.2 );
			const isDepthChanged = closestDepth.sub( previousDepth ).abs().greaterThan( this.depthThreshold );
			const canLock = isValidUV.and( isDepthChanged.not() );
			const gatedThinFeature = canLock.select( thinFeature, float( 0 ) );
			const decay = isDisocclusion.select( 0, 0.5 );
			const lock = max( gatedThinFeature, lockNode.sample( historyUV ).r.mul( decay ) ).saturate();
			const lockedHistoryColor = mix( clippedHistoryColor, historyColor, lock );

			// fast camera motion trusts the current frame more; capped on water, whose fine detail shimmers under the jitter
			const motionW = isWater.select( min( motionFactor, 0.15 ), motionFactor );
			const currentWeight = hasValidHistory.select( this.frameWeight.add( motionW ).saturate(), float( 1 ) );
			colorOutput.assign( flickerReduction( currentColor, lockedHistoryColor, currentWeight ) );
			lockOutput.assign( lock );

			return vec4( 0 );

		} );

		const sharedContext = context( builder.getSharedContext() );
		this._resolveMaterial.contextNode = sharedContext;
		this._resolveMaterial.colorNode = resolve();
		this._resolveMaterial.outputNode = outputNode;

		this._seedMaterial.colorNode = Fn( () => {

			colorOutput.assign( this.beautyNode.sample( uv() ) );
			lockOutput.assign( 0 );
			return vec4( 0 );

		} )();
		this._seedMaterial.contextNode = sharedContext;
		this._seedMaterial.outputNode = outputNode;

		return this._textureNode;

	}

}
