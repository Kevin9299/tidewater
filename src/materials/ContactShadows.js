import * as THREE from 'three/webgpu';
import {
	If, float, vec2, vec4, ivec2, max, min, abs, fract, select, smoothstep, textureLoad, uniform, dot, dFdx, dFdy,
	positionWorld, positionView, normalWorld, screenCoordinate, interleavedGradientNoise, frameId, cameraNear,
	renderGroup,
} from 'three/tsl';
import { G } from '../core/Globals.js';
import { prevViewProj } from '../post/CameraVelocity.js';
import { SceneLighting } from './SceneLighting.js';

// Screen-space contact shadows for the sun: the fine shadows the cascaded maps miss (pebbles, shells,
// debris, grass, feet, rope, deck fittings). The scene is forward rendered, so the march reads the
// previous frame's opaque depth (SceneRenderer.opaqueCopy, still last frame's during the opaque
// pass) reprojected with last frame's view-projection. Per sunlit fragment within MAX_DIST: 8 steps
// toward the sun (4 beyond 12 m) over 0.3 m (near) .. 1 m (far), denser near the start, jittered per pixel and per
// frame (the TAAU resolves the noise), in two groups of independent loads with an early exit between. A sample occludes when it lies
// behind the depth buffer by more than a slope-scaled bias (no acne on flat ground) and less than a
// thickness that grows along the ray (no long false shadows behind thin or distant objects).
// Applied to the key light only, in the opaque pass only.
const GROUP = 4;
const GROUPS = 2;
const STEPS = GROUP * GROUPS;
const MAX_DIST = 32;
const NEAR_DIST = 12; // the second group of steps only runs nearer than this

export const ContactShadows = {
	strength: uniform( 1 ).setName( 'csStrength' ),
	depthTexture: null,
	skipRoots: new Set(), // receivers under these objects are skipped (foliage: overdraw, still casts)
};

const texSize = uniform( new THREE.Vector2( 1, 1 ) ).setName( 'csSize' ).setGroup( renderGroup ).onRenderUpdate( () => {

	const img = ContactShadows.depthTexture && ContactShadows.depthTexture.image;
	if ( img && img.width ) texSize.value.set( img.width, img.height );

} );

export function installContactShadows( { depthTexture, skip = [] } ) {

	ContactShadows.depthTexture = depthTexture;
	for ( const o of skip ) if ( o ) ContactShadows.skipRoots.add( o );
	SceneLighting.contactShadow = contactShadow;

}

// 0 (occluded) .. 1 visibility node for the key light, or null where it does not apply
export function contactShadow( builder, lightColor ) {

	const mat = builder.material, obj = builder.object;
	const depth = ContactShadows.depthTexture;
	if ( ! depth || ! mat || mat.isWaterMaterial || mat.contactShadows === false ) return null;
	// late (transparent) pass: last frame's depth is this frame's there
	if ( obj && obj.layers && ( obj.layers.mask & 1 ) === 0 ) return null;
	for ( let o = obj; o; o = o.parent ) if ( ContactShadows.skipRoots.has( o ) ) return null;

	const P = positionWorld;
	const L = G.sunDir;
	const w0 = positionView.z.negate();
	// depth change per pixel on this surface (uniform control flow: before any branch)
	const slope = max( abs( dFdx( w0 ) ), abs( dFdy( w0 ) ) ).toVar();
	const vis = float( 1 ).toVar();
	const lit = dot( lightColor, lightColor ).greaterThan( 1e-8 ).and( dot( normalWorld, L ).greaterThan( 0.02 ) );
	If( ContactShadows.strength.greaterThan( 0 ).and( w0.lessThan( MAX_DIST ) ).and( lit ).and( P.y.greaterThan( G.seaLevel.sub( 0.3 ) ) ), () => {

		const len = smoothstep( 2, 30, w0 ).mul( 0.7 ).add( 0.3 );
		const q0 = prevViewProj.mul( vec4( P, 1 ) ).toVar();
		const qd = prevViewProj.mul( vec4( L.mul( len ), 0 ) ).toVar();
		const bias = slope.mul( 2.0 ).add( w0.mul( 0.002 ) ).add( 0.01 ).toVar();
		const jit = fract( interleavedGradientNoise( screenCoordinate.xy ).add( float( frameId ).mul( 0.618034 ) ) ).toVar();
		const hit = float( 2 ).toVar(); // ray parameter of the first occluded sample (> 1: none)
		// groups of independent loads (latency hiding), early exit between groups; far away the ray
		// covers few pixels and the first group is enough. Depths compare in reversed-Z device depth
		// (d ~ near / w): the bias and thickness in metres scale by near / w^2.
		for ( let g = 0; g < GROUPS; g ++ ) {

			const go = g === 0 ? hit.greaterThan( 1 ) : hit.greaterThan( 1 ).and( w0.lessThan( NEAR_DIST ) );
			If( go, () => {

				for ( let k = 0; k < GROUP; k ++ ) {

					const s = jit.add( g * GROUP + k ).div( STEPS );
					const u = s.mul( s ).mul( 0.85 ).add( s.mul( 0.15 ) ); // denser near the contact
					const q = q0.add( qd.mul( u ) );
					const iw = float( 1 ).div( q.w );
					const uv = q.xy.mul( iw ).mul( vec2( 0.5, - 0.5 ) ).add( 0.5 );
					const d = textureLoad( depth, ivec2( uv.clamp( 0, 1 ).mul( texSize ).min( texSize.sub( 1 ) ) ) ).x;
					const k2 = cameraNear.mul( iw ).mul( iw );
					const diff = d.sub( q.z.mul( iw ) ); // > 0: the depth buffer is in front of the ray
					const thick = bias.add( 0.06 ).add( u.mul( len ).mul( 0.3 ) );
					const occ = diff.greaterThan( bias.mul( k2 ) ).and( diff.lessThan( thick.mul( k2 ) ) ).and( uv.x.greaterThan( 0 ) ).and( uv.x.lessThan( 1 ) ).and( uv.y.greaterThan( 0 ) ).and( uv.y.lessThan( 1 ) );
					hit.assign( min( hit, select( occ, u, float( 2 ) ) ) );

				}

			} );

		}

		// the sun's penumbra is millimetres at this range: a hard shadow, faded out toward the end
		// of the ray (no cut-off line at its length)
		vis.assign( select( hit.lessThanEqual( 1 ), smoothstep( 0.55, 1.0, hit ), float( 1 ) ) );
		vis.assign( vis.oneMinus().mul( ContactShadows.strength ).mul( smoothstep( MAX_DIST, MAX_DIST - 8, w0 ) ).oneMinus() );

	} );
	return vis;

}
