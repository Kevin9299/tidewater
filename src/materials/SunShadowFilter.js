import {
	Fn, float, vec2, vec4, ivec2, If, select, abs, min, max, textureLoad, reference, uniform, renderGroup,
	screenCoordinate, interleavedGradientNoise, vogelDiskSample, viewZToOrthographicDepth, positionView,
	frameId, uint,
} from 'three/tsl';
import { CSMShadowNode } from 'three/addons/csm/CSMShadowNode.js';

// Contact-hardening sun shadows (PCSS) for the cascaded shadow maps. The penumbra of a real sun
// shadow grows with the distance from the occluder to the receiver (the sun is a 0.53° disc): sharp
// where an object touches the ground, soft under a palm crown 10 m up. Per pixel:
//  1. blocker search: average depth of the occluders around the pixel (raw depth loads, no sampler)
//  2. penumbra width = occluder-receiver distance * sun diameter, converted to this cascade's texels
//  3. percentage-closer filtering over that width
// Both sample sets are Vogel disks rotated per pixel and per frame (interleaved gradient noise): the
// TAA resolves the noise into a smooth gradient. All reads are texel loads with manual depth comparisons: no
// sampler at all (three can't mix loads and comparison samples of one depth texture), which also
// frees a sampler per cascade in every lit material.

const SUN_DIAMETER = 0.00925; // rad (tan of the 0.53° angular diameter)
const MAX_OCCLUDER_HEIGHT = 30; // m: search radius covers penumbrae of occluders up to this far above
const SEARCH_TAPS = 8;
const FILTER_TAPS = 12;

export const SunShadowFilter = Fn( ( { depthTexture, shadowCoord, shadow, depthLayer }, builder ) => {

	const reversed = builder.renderer.reversedDepthBuffer === true;
	const cam = shadow.camera;
	const mapSize = reference( 'mapSize', 'vec2', shadow ).setGroup( renderGroup );
	const width = reference( 'right', 'float', cam ).setGroup( renderGroup ).sub( reference( 'left', 'float', cam ).setGroup( renderGroup ) );
	const range = reference( 'far', 'float', cam ).setGroup( renderGroup ).sub( reference( 'near', 'float', cam ).setGroup( renderGroup ) );
	const texel = float( 1 ).div( mapSize.x );
	const zr = shadowCoord.z;

	const rawDepth = ( uv ) => {

		let t = textureLoad( depthTexture, ivec2( uv.clamp( 0, 0.99999 ).mul( mapSize ) ) );
		if ( depthTexture.isArrayTexture ) t = t.depth( depthLayer );
		return t.x;

	};

	// 1 where the receiver is lit by this texel (reversed depth: closer to the light = larger)
	const litBy = ( d ) => reversed ? select( zr.greaterThanEqual( d ), float( 1 ), float( 0 ) ) : select( zr.lessThanEqual( d ), float( 1 ), float( 0 ) );

	// moved every frame (Jimenez 2014): a noise pattern fixed on screen would never average out
	const phi = interleavedGradientNoise( screenCoordinate.xy.add( float( frameId.mod( uint( 64 ) ) ).mul( 5.588238 ) ) ).mul( 6.28318530718 );

	// 1. blockers within the widest penumbra this cascade can show. The texel straight along the light
	// ray comes first: a thin occluder (a log, a rope, a rail) can fall between the disk taps, which
	// left lit dots inside its umbra.
	const searchUV = min( float( MAX_OCCLUDER_HEIGHT * SUN_DIAMETER ).div( width ), texel.mul( 24 ) ).max( texel.mul( 1.5 ) );
	const d0 = rawDepth( shadowCoord.xy );
	const occ0 = reversed ? d0.greaterThan( zr ) : d0.lessThan( zr );
	const blockSum = select( occ0, d0, float( 0 ) ).toVar();
	const blockCount = select( occ0, float( 1 ), float( 0 ) ).toVar();
	for ( let i = 0; i < SEARCH_TAPS; i ++ ) {

		const d = rawDepth( shadowCoord.xy.add( vogelDiskSample( i, SEARCH_TAPS, phi ).mul( searchUV ) ) );
		// reversed depth: an occluder is closer to the light = larger depth
		const occludes = reversed ? d.greaterThan( zr ) : d.lessThan( zr );
		blockSum.addAssign( select( occludes, d, 0 ) );
		blockCount.addAssign( select( occludes, 1, 0 ) );

	}

	const lit = float( 1 ).toVar();
	If( blockCount.greaterThan( 0 ), () => {

		// 2. occluder-receiver distance (orthographic: depth is linear over the camera range)
		const dz = abs( blockSum.div( blockCount ).sub( zr ) ).mul( range );
		const penumbraUV = dz.mul( SUN_DIAMETER ).div( width ).clamp( texel.mul( 1.2 ), texel.mul( 32 ) );

		// 3. PCF over the penumbra
		const sum = float( 0 ).toVar();
		for ( let i = 0; i < FILTER_TAPS; i ++ ) {

			sum.addAssign( litBy( rawDepth( shadowCoord.xy.add( vogelDiskSample( i, FILTER_TAPS, phi.add( 1.7 ) ).mul( penumbraUV ) ) ) ) );

		}

		lit.assign( sum.div( FILTER_TAPS ) );

	} );

	return lit;

} );

// Cascade layout for CSMShadowNode (mode: 'custom'): a tight first cascade for fine detail up close,
// a long last one for rough shadows far away (the island's own long shadows come from the terrain
// horizon map beyond it). Distances in metres.
export const SHADOW_SPLITS = [ 10, 60, 400 ];

export function shadowSplits( cascades, near, far, breaks ) {

	for ( let i = 0; i < cascades; i ++ ) breaks.push( Math.min( SHADOW_SPLITS[ Math.min( i, SHADOW_SPLITS.length - 1 ) ] / far, 1 ) );
	breaks[ breaks.length - 1 ] = 1;

}

// Per-cascade settings once CSMShadowNode has created its cascade lights: the contact-hardening
// filter on the near cascade only (farther out the sun's penumbra is under a texel and the default
// 5-tap PCF looks the same for a quarter of the cost), and a normal bias that scales with the texel
// size (a fixed one detaches fine shadows near the camera).
export function configureCascades( csm, { normalBias = [ 0.015, 0.06, 0.3 ], contactCascades = 1 } = {} ) {

	const init = csm._init.bind( csm );
	csm._init = ( builder ) => {

		init( builder );
		csm.lights.forEach( ( l, i ) => {

			if ( i < contactCascades ) l.shadow.filterNode = SunShadowFilter;
			l.shadow.normalBias = normalBias[ Math.min( i, normalBias.length - 1 ) ];

		} );

	};

}

// Cascade seams blended over a band that grows with their distance (a quarter of it: 2.5 m at the
// 10 m seam, 15 m at 60 m, and the last cascade fades out over its final 100 m). three's default
// band shrinks with the square of the distance (6 cm at a 10 m seam), which reads as a hard line where
// the shadow resolution and filter change. Each cascade's map is widened to cover its part of the
// overlap.
const BLEND = 0.25;
const band = ( edge ) => Math.max( 0.25 * edge * edge, BLEND * edge ); // normalized to shadowFar

export class SoftCSMShadowNode extends CSMShadowNode {

	_updateShadowBounds() {

		const fade = this.fade;
		this.fade = false;
		super._updateShadowBounds();
		this.fade = fade;
		if ( ! fade ) return;
		const far = Math.min( this.camera.far, this.maxFar );
		for ( let i = 0; i < this.lights.length; i ++ ) {

			const cam = this.lights[ i ].shadow.camera;
			// the band beyond this cascade's far seam, plus the frustum's growth over that depth
			const grow = band( this.breaks[ i ] ) * far * 1.5;
			cam.left -= grow * 0.5;
			cam.right += grow * 0.5;
			cam.bottom -= grow * 0.5;
			cam.top += grow * 0.5;
			cam.updateProjectionMatrix();

		}

	}

	_setupFade() {

		const cameraNear = reference( 'camera.near', 'float', this ).setGroup( renderGroup );
		const cascades = reference( '_cascades', 'vec2', this ).setGroup( renderGroup ).setName( 'cascades' );
		const shadowFar = uniform( 'float' ).setGroup( renderGroup ).setName( 'shadowFar' )
			.onRenderUpdate( () => Math.min( this.maxFar, this.camera.far ) );
		const linearDepth = viewZToOrthographicDepth( positionView.z, cameraNear, shadowFar ).toVar( 'linearDepth' );
		const lastCascade = this.cascades - 1;

		return Fn( ( builder ) => {

			this.setupShadowPosition( builder );
			const ret = vec4( 1, 1, 1, 1 ).toVar( 'shadowValue' );
			const cascade = vec2().toVar( 'cascade' );
			const cascadeCenter = float().toVar( 'cascadeCenter' );
			const margin = float().toVar( 'margin' );
			const csmX = float().toVar( 'csmX' );
			const csmY = float().toVar( 'csmY' );
			for ( let i = 0; i < this.cascades; i ++ ) {

				cascade.assign( cascades.element( i ) );
				cascadeCenter.assign( cascade.x.add( cascade.y ).div( 2.0 ) );
				const edge = linearDepth.lessThan( cascadeCenter ).select( cascade.x, cascade.y );
				margin.assign( max( edge.mul( edge ).mul( 0.25 ), edge.mul( BLEND ) ).max( 1e-5 ) );
				csmX.assign( cascade.x.sub( margin.div( 2.0 ) ) );
				csmY.assign( i === lastCascade ? cascade.y : cascade.y.add( margin.div( 2.0 ) ) );
				If( linearDepth.greaterThanEqual( csmX ).and( linearDepth.lessThanEqual( csmY ) ), () => {

					const dist = min( linearDepth.sub( csmX ), csmY.sub( linearDepth ) );
					let ratio = dist.div( margin ).clamp( 0.0, 1.0 );
					// no fade at the near edge of the first cascade
					if ( i === 0 ) ratio = linearDepth.greaterThan( cascadeCenter ).select( ratio, 1 );
					ret.subAssign( this._shadowNodes[ i ].oneMinus().mul( ratio ) );

				} );

			}

			return ret;

		} )();

	}

}
