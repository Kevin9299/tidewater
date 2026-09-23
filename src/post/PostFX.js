import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, vec2, vec3, vec4, texture, screenUV, rtt, mix, smoothstep, length, dot, max,
	luminance, pow, select, fract, sin, instancedArray, workgroupArray, workgroupBarrier, localId, textureLoad,
	textureSize, ivec2, uvec2, int, uint, Loop, If, exp, exp2, log2, clamp, floor, min, abs,
} from 'three/tsl';
import { G } from '../core/Globals.js';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import { TemporalUpscale } from './TemporalUpscale.js';
import { LensDroplets } from './LensDroplets.js';
import { LensFlare } from './LensFlare.js';
import { MotionBlur } from './MotionBlur.js';

// Post chain (internal resolution = drawing buffer * scale up to the TAAU resolve):
//   scene (HDR + velocity) -> GTAO (half internal res, temporally rotated)
//   -> AO composite + underwater / waterline (one pass)
//   -> TAAU (anti-aliasing + upscale to the output resolution)
//   -> bloom (13-tap downsample / tent upsample chain from half res)
//   -> grading (saturation, contrast, warmth) + vignette + grain -> ACES (renderOutput)
export class PostFX {

	constructor( renderer, { sceneRenderer, camera, underwater, clouds = null, sunDir = null, haze = null } ) {

		this.renderer = renderer;
		this.camera = camera;
		this.sceneRenderer = sceneRenderer;
		this.scale = 1;

		this.params = {
			aoStrength: uniform( 1.0 ).setName( 'ppAO' ),
			bloom: uniform( 0.05 ).setName( 'ppBloom' ),
			vignette: uniform( 0.28 ).setName( 'ppVignette' ),
			saturation: uniform( 1.06 ).setName( 'ppSat' ),
			contrast: uniform( 1.04 ).setName( 'ppContrast' ),
			warmth: uniform( 0.02 ).setName( 'ppWarm' ),
			grain: uniform( 0.012 ).setName( 'ppGrain' ),
			sharpen: uniform( 0.45 ).setName( 'ppSharpen' ), // RCAS strength (0 = off, 1 = strong)
		};

		const P = this.params;
		const sceneRT = sceneRenderer.sceneRT;
		const opaque = sceneRenderer.opaqueCopy;
		const sceneColor = texture( sceneRT.texture );
		const opaqueDepth = texture( opaque.depthTexture );
		const finalDepth = texture( sceneRT.depthTexture );

		// ---- ambient occlusion on the opaque depth (normals reconstructed from depth)
		this.aoPass = ao( opaqueDepth, null, camera );
		this.aoPass.resolutionScale = 0.5;
		this.aoPass.radius.value = 2.2;
		this.aoPass.thickness.value = 2.0;
		this.aoPass.distanceExponent.value = 1.4;
		this.aoPass.scale.value = 1.6;
		this.aoPass.samples.value = 12;
		this.aoPass.useTemporalFiltering = true;
		// spatial denoise at the AO resolution (separable 5 + 5 taps, depth-aware): GTAO rotates its
		// directions over a 5x5 pattern and jitters its steps per pixel and per frame, which only the
		// temporal resolve averaged out; where it can't accumulate (the rocking helm, fast turns) that
		// noise flickered
		const aoBlur = ( src, dx, dy ) => rtt( Fn( () => {

			const size = vec2( src.size() );
			const dC = opaqueDepth.sample( screenUV ).x;
			const sum = float( 0 ).toVar(), wSum = float( 0 ).toVar();
			for ( let k = - 2; k <= 2; k ++ ) {

				const uvK = screenUV.add( vec2( dx * k, dy * k ).div( size ) );
				const rel = abs( opaqueDepth.sample( uvK ).x.sub( dC ) ).div( max( dC, 1e-7 ) );
				const w = float( 1 ).div( rel.mul( 40 ).add( 1 ).pow2() );
				sum.addAssign( src.sample( uvK ).r.mul( w ) );
				wSum.addAssign( w );

			}

			return vec4( sum.div( wSum ), 0, 0, 1 );

		} )(), null, null, { type: THREE.HalfFloatType, resolutionScale: 0.5 } );
		this.aoBlurX = aoBlur( this.aoPass.getTextureNode(), 1, 0 );
		this.aoBlurY = aoBlur( this.aoBlurX, 0, 1 );
		const aoTex = this.aoBlurY;

		// scene color with AO applied to opaque pixels that aren't behind water
		const colorAO = ( uv ) => {

			const c = sceneColor.sample( uv ).rgb;
			const dO = opaqueDepth.sample( uv ).x;
			const dF = finalDepth.sample( uv ).x;
			// reversed depth: sky = 0; water in front of the opaque surface has a larger depth value
			const isSky = dO.lessThan( 1e-7 );
			const covered = dF.greaterThan( dO.add( 1e-7 ) );
			// depth-aware upsample of the half-res AO: the 4 nearest AO texels, weighted by how close
			// their depth is to this pixel's (no dark halos bleeding across depth edges)
			const aoSize = vec2( aoTex.size() );
			const pa = uv.mul( aoSize ).sub( 0.5 );
			const i0 = floor( pa );
			const fr = pa.sub( i0 );
			const aSum = float( 0 ).toVar(), wSum = float( 1e-4 ).toVar();
			for ( const [ ox, oy ] of [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ], [ 1, 1 ] ] ) {

				const uvT = i0.add( vec2( ox, oy ) ).add( 0.5 ).div( aoSize );
				const dT = opaqueDepth.sample( uvT ).x;
				const wBil = ( ox ? fr.x : fr.x.oneMinus() ).mul( oy ? fr.y : fr.y.oneMinus() );
				// reversed-Z depth ~ near / z, so the relative depth difference ~ |dT - dO| / dO
				const rel = abs( dT.sub( dO ) ).div( max( dO, 1e-7 ) );
				const wDepth = float( 1 ).div( rel.mul( 40 ).add( 1 ).pow2() );
				const wt = wBil.mul( wDepth ).add( 1e-5 );
				aSum.addAssign( aoTex.sample( uvT ).r.mul( wt ) );
				wSum.addAssign( wt );

			}

			const a = aSum.div( wSum );
			// multi-bounce approximation (Jimenez 2016) for a typical outdoor albedo of ~0.35
			const aMB = max( a, a.mul( 0.382 ).sub( 1.036 ).mul( a ).add( 1.654 ).mul( a ) );
			// AO only removes ambient light: full effect where the pixel is lit by the sky alone,
			// a third of it on sunlit surfaces (still grounds objects without dirty halos)
			const ambientOnly = float( 1 ).sub( smoothstep( 0.12, 0.9, luminance( c ) ) );
			const k = select( isSky.or( covered ), float( 0 ), P.aoStrength.mul( mix( 0.35, 1.0, ambientOnly ) ) );
			return vec4( c.mul( mix( float( 1 ), aMB, k ) ), 1 );

		};

		// medium at the near clip plane per pixel (the waterline on the lens), then the composite
		this.medium = rtt( underwater.mediumNode(), null, null, { type: THREE.UnsignedByteType, format: THREE.RedFormat } );
		underwater.mediumTexture = this.medium;
		// air: aerial perspective, marine haze and volumetric sun shafts (AirHaze) on the lit scene
		this.haze = haze;
		if ( haze ) haze.mediumTexture = this.medium;
		const sceneSample = haze ? ( uv ) => haze.apply( uv, colorAO( uv ) ) : colorAO;
		this.beauty = rtt( underwater.node( { sample: sceneSample } ), null, null, { type: THREE.HalfFloatType } );

		// ---- temporal anti-aliasing + upscale
		this.taau = new TemporalUpscale( this.beauty, finalDepth, texture( sceneRenderer.velocityTexture ), camera, sceneRenderer.waterMaskTexture );
		const resolved = this.taau.getTextureNode();

		// ---- camera + object motion blur on the resolved image (gathered in the final pass, before
		// bloom and the screen-fixed lens effects)
		this.motionBlur = new MotionBlur( { velocityTexture: sceneRenderer.velocityTexture, depthTexture: sceneRenderer.sceneRT.depthTexture, color: resolved } );

		// RCAS (AMD FSR1 robust contrast-adaptive sharpening) on the resolved image: TAA converges to a
		// slightly soft image, RCAS restores the detail without halos (the lobe is limited by local
		// contrast). Done on a tonemapped proxy of the HDR values and inverted afterwards.
		const rcas = ( uvIn ) => {

			const size = ivec2( resolved.size() );
			const pc = ivec2( uvIn.mul( vec2( size ) ) ).clamp( ivec2( 1 ), size.sub( 2 ) ).toVar();
			const tm = ( c ) => c.div( max( c.r, max( c.g, c.b ) ).add( 1 ) );
			const e = tm( resolved.load( pc ).rgb ).toVar();
			const b = tm( resolved.load( pc.add( ivec2( 0, - 1 ) ) ).rgb ), d = tm( resolved.load( pc.add( ivec2( - 1, 0 ) ) ).rgb );
			const f = tm( resolved.load( pc.add( ivec2( 1, 0 ) ) ).rgb ), h = tm( resolved.load( pc.add( ivec2( 0, 1 ) ) ).rgb );
			const mn4 = min( min( b, d ), min( f, h ) );
			const mx4 = max( max( b, d ), max( f, h ) );
			const hitMin = min( mn4, e ).div( mx4.mul( 4.0 ).add( 1e-5 ) );
			const hitMax = vec3( 1.0 ).sub( max( mx4, e ) ).div( mn4.mul( 4.0 ).sub( 4.0 ) );
			const lobeRGB = max( hitMin.negate(), hitMax );
			const lobe = max( float( - ( 0.25 - 1.0 / 16.0 ) ), min( max( lobeRGB.r, max( lobeRGB.g, lobeRGB.b ) ), float( 0 ) ) ).mul( P.sharpen );
			const r = b.add( d ).add( f ).add( h ).mul( lobe ).add( e ).div( lobe.mul( 4.0 ).add( 1.0 ) ).max( 0 );
			// back to HDR (inverse of the max-channel Reinhard)
			return r.div( max( float( 1 ).sub( max( r.r, max( r.g, r.b ) ) ), 1e-3 ) );

		};

		// ---- bloom
		const { bloom: bloomTex, half, meter } = this._buildBloom( resolved );

		// ---- auto exposure (eye adaptation), metered on the GPU from the 1/16 bloom level
		this.autoExposure = {
			enabled: uniform( 1 ).setName( 'aeOn' ),
			ref: uniform( 0.25 ).setName( 'aeRef' ), // scene average luminance that needs no correction
			min: uniform( 0.6 ).setName( 'aeMin' ),
			max: uniform( 6.0 ).setName( 'aeMax' ),
			up: uniform( 1.6 ).setName( 'aeUp' ), // adaptation rates (1/s): brightening, darkening
			down: uniform( 1.1 ).setName( 'aeDown' ),
		};
		this.exposure = instancedArray( new Float32Array( [ 1 ] ), 'float' ).setName( 'autoExposure' );
		this._buildMeter( meter );
		const exposureMul = this.exposure.element( uint( 0 ) );

		// ---- sun flare in the lens (screen-fixed, after the temporal resolve; its visibility is measured
		// from the scene depth by a compute pass the app runs after the scene)
		this.flare = sunDir ? new LensFlare( { depthTexture: sceneRenderer.sceneRT.depthTexture, clouds, sunDir } ) : null;
		const flareAt = ( uv ) => this.flare ? this.flare.node( uv ) : vec3( 0 );

		// ---- water on the lens after surfacing (screen-fixed, so after the temporal resolve)
		this.lens = new LensDroplets();
		const lensFn = this.lens.build();

		const graded = Fn( () => {

			const sharp = ( uv ) => this.motionBlur.apply( rcas( uv ), uv ).add( bloomTex.sample( uv ).rgb.mul( P.bloom ) ).add( flareAt( uv ) );
			const blurred = ( uv ) => half.sample( uv ).rgb.add( bloomTex.sample( uv ).rgb.mul( P.bloom ) );
			let c = lensFn( sharp, blurred, screenUV ).mul( exposureMul );
			// white balance nudge + saturation + contrast around mid grey (in linear HDR)
			c = c.mul( vec3( float( 1 ).add( P.warmth ), 1, float( 1 ).sub( P.warmth ) ) );
			const l = luminance( c );
			c = mix( vec3( l ), c, P.saturation );
			c = pow( max( c, vec3( 0 ) ).div( 0.18 ), vec3( P.contrast ) ).mul( 0.18 );
			// vignette (elliptical, soft)
			const d = screenUV.sub( 0.5 ).mul( vec2( 1.0, 0.8 ) );
			const v = float( 1 ).sub( smoothstep( 0.25, 0.75, length( d ) ).mul( P.vignette ) );
			c = c.mul( v );
			// fine film grain (luminance-weighted, hides banding in dark gradients)
			const n = fract( sin( dot( screenUV.mul( vec2( 1920, 1080 ) ), vec2( 12.9898, 78.233 ) ) ).mul( 43758.5453 ) ).sub( 0.5 );
			c = c.add( c.mul( n.mul( P.grain ) ) );
			return vec4( c, 1 );

		} )();

		this.pipeline = new THREE.RenderPipeline( renderer );
		this.pipeline.outputNode = graded;
		this._size = new THREE.Vector2();

	}

	// Physically-motivated bloom (Jimenez 2014): 13-tap downsamples (Karis average on the first
	// one so single bright glints can't flicker), then 3x3 tent upsamples accumulating each level.
	_buildBloom( src ) {

		const opts = ( s ) => ( { type: THREE.HalfFloatType, resolutionScale: s } );
		const tap = ( t, uv, texel, x, y ) => t.sample( uv.add( texel.mul( vec2( x, y ) ) ) ).rgb;
		const karis = ( c ) => c.div( luminance( c ).add( 1 ) );

		const down = ( t, scale, first ) => rtt( Fn( () => {

			const uv = screenUV;
			const texel = vec2( 1 ).div( vec2( t.size() ) );
			const a = tap( t, uv, texel, - 2, - 2 ), b = tap( t, uv, texel, 0, - 2 ), c = tap( t, uv, texel, 2, - 2 );
			const d = tap( t, uv, texel, - 2, 0 ), e = tap( t, uv, texel, 0, 0 ), f = tap( t, uv, texel, 2, 0 );
			const g = tap( t, uv, texel, - 2, 2 ), h = tap( t, uv, texel, 0, 2 ), i = tap( t, uv, texel, 2, 2 );
			const j = tap( t, uv, texel, - 1, - 1 ), k = tap( t, uv, texel, 1, - 1 ), l = tap( t, uv, texel, - 1, 1 ), m = tap( t, uv, texel, 1, 1 );
			if ( first ) {

				// weighted groups of four (Karis average) suppress fireflies
				const g0 = karis( j.add( k ).add( l ).add( m ).mul( 0.25 ) ).mul( 0.5 );
				const g1 = karis( a.add( b ).add( d ).add( e ).mul( 0.25 ) ).mul( 0.125 );
				const g2 = karis( b.add( c ).add( e ).add( f ).mul( 0.25 ) ).mul( 0.125 );
				const g3 = karis( d.add( e ).add( g ).add( h ).mul( 0.25 ) ).mul( 0.125 );
				const g4 = karis( e.add( f ).add( h ).add( i ).mul( 0.25 ) ).mul( 0.125 );
				const s = g0.add( g1 ).add( g2 ).add( g3 ).add( g4 );
				// undo the Karis tonemap on the result
				return vec4( s.div( max( float( 1 ).sub( luminance( s ) ), 0.02 ) ), 1 );

			}

			const s = e.mul( 0.125 ).add( a.add( c ).add( g ).add( i ).mul( 0.03125 ) )
				.add( b.add( d ).add( f ).add( h ).mul( 0.0625 ) ).add( j.add( k ).add( l ).add( m ).mul( 0.125 ) );
			return vec4( s, 1 );

		} )(), null, null, opts( scale ) );

		const up = ( small, base, scale ) => rtt( Fn( () => {

			const uv = screenUV;
			const texel = vec2( 1 ).div( vec2( small.size() ) );
			const s = tap( small, uv, texel, 0, 0 ).mul( 4 )
				.add( tap( small, uv, texel, - 1, 0 ).add( tap( small, uv, texel, 1, 0 ) ).add( tap( small, uv, texel, 0, - 1 ) ).add( tap( small, uv, texel, 0, 1 ) ).mul( 2 ) )
				.add( tap( small, uv, texel, - 1, - 1 ).add( tap( small, uv, texel, 1, - 1 ) ).add( tap( small, uv, texel, - 1, 1 ) ).add( tap( small, uv, texel, 1, 1 ) ) )
				.div( 16 );
			return vec4( base.sample( uv ).rgb.add( s ), 1 );

		} )(), null, null, opts( scale ) );

		const d1 = down( src, 0.5, true );
		const d2 = down( d1, 0.25, false );
		const d3 = down( d2, 0.125, false );
		const d4 = down( d3, 0.0625, false );
		const d5 = down( d4, 0.03125, false );
		const u4 = up( d5, d4, 0.0625 );
		const u3 = up( u4, d3, 0.125 );
		const u2 = up( u3, d2, 0.25 );
		const u1 = up( u2, d1, 0.5 );
		return { bloom: u1, half: d1, meter: d4 };

	}

	// Average log luminance (centre weighted) of a small copy of the frame, one workgroup; the
	// adapted exposure multiplier lives in a storage buffer read by the grading pass next frame.
	_buildMeter( meterRTT ) {

		const W = 256;
		const A = this.autoExposure;
		const exposure = this.exposure;
		const tex = texture( meterRTT.value );

		this.meterKernel = Fn( () => {

			const t = localId.x;
			const sumL = workgroupArray( 'float', W );
			const sumW = workgroupArray( 'float', W );
			const size = uvec2( textureSize( tex, 0 ) ).toVar();
			const n = size.x.mul( size.y ).toVar();
			const accL = float( 0 ).toVar(), accW = float( 0 ).toVar();
			Loop( 160, ( { i } ) => {

				const idx = t.add( uint( i ).mul( uint( W ) ) );
				If( idx.lessThan( n ), () => {

					const x = idx.mod( size.x ), y = idx.div( size.x );
					const c = textureLoad( tex, ivec2( x, y ) ).rgb;
					const l = log2( max( luminance( c ), 1e-4 ) );
					const uvc = vec2( float( x ).add( 0.5 ).div( float( size.x ) ), float( y ).add( 0.5 ).div( float( size.y ) ) ).sub( 0.5 );
					const w = max( float( 1 ).sub( length( uvc.mul( vec2( 1, 1.4 ) ) ).mul( 1.2 ) ), 0.15 );
					accL.addAssign( l.mul( w ) );
					accW.addAssign( w );

				} );

			} );

			sumL.element( t ).assign( accL );
			sumW.element( t ).assign( accW );
			workgroupBarrier();
			for ( let s = W / 2; s > 0; s >>= 1 ) {

				If( t.lessThan( uint( s ) ), () => {

					sumL.element( t ).addAssign( sumL.element( t.add( uint( s ) ) ) );
					sumW.element( t ).addAssign( sumW.element( t.add( uint( s ) ) ) );

				} );
				workgroupBarrier();

			}

			If( t.equal( uint( 0 ) ), () => {

				const avg = exp2( sumL.element( uint( 0 ) ).div( max( sumW.element( uint( 0 ) ), 1e-4 ) ) );
				// the eye only partly compensates: dark scenes stay darker (dusk and night must not
				// look like day), and at night at most one extra stop
				const ratio = A.ref.div( avg );
				const partial = select( ratio.greaterThan( 1 ), pow( ratio, 0.8 ), ratio );
				const target = clamp( partial, A.min, mix( A.max, float( 2 ), G.night ) );
				const cur = exposure.element( uint( 0 ) );
				const rate = select( target.greaterThan( cur ), A.up, A.down );
				const k = float( 1 ).sub( exp( G.dt.mul( rate ).negate() ) );
				const next = exp2( mix( log2( max( cur, 1e-3 ) ), log2( target ), k ) );
				cur.assign( select( A.enabled.greaterThan( 0.5 ), next, float( 1 ) ) );

			} );

		} )().computeKernel( [ W, 1, 1 ] ).setName( 'Auto Exposure' );

	}

	// internal render resolution (0.5 .. 1) for dynamic resolution
	setScale( s ) {

		if ( s === this.scale ) return;
		this.scale = s;
		this.sceneRenderer.scale = s;
		this.beauty.setResolutionScale( s );
		this.medium.setResolutionScale( s );
		if ( this.haze ) this.haze.setScale( s );
		// upscaling needs more frames to fill the output grid; at 1:1 respond faster (less smear)
		this.taau.frameWeight.value = THREE.MathUtils.lerp( 0.035, 0.06, THREE.MathUtils.clamp( ( s - 0.6 ) / 0.4, 0, 1 ) );
		this.aoPass.resolutionScale = 0.5 * s;
		this.aoBlurX.setResolutionScale( 0.5 * s );
		this.aoBlurY.setResolutionScale( 0.5 * s );

	}

	setBloom( v ) {

		this.params.bloom.value = v;

	}

	// jitter the camera for this frame (call before rendering the scene)
	beginFrame() {

		this.motionBlur.updateCamera( this.camera );
		this.sceneRenderer.internalSize( this._size );
		this.taau.setViewOffset( this._size.x, this._size.y );

	}

	render() {

		this.motionBlur.compute( this.renderer );
		if ( this.haze ) this.haze.update();
		this.pipeline.render();
		// meter this frame's image; the result is used from the next frame on
		this.renderer.compute( this.meterKernel, [ 1, 1, 1 ] );

	}

	endFrame() {

		this.taau.clearViewOffset();

	}

}

