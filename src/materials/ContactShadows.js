import { ShaderModule, UniformBlock } from '../engine/gpu/Shader.js';
import { commonModule } from '../engine/render/wgsl/common.js';
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
//
// Installed as the `contactShadow` lighting hook (fn hookContactShadow( P, N ) -> f32). Exclusions
// are per material define (the hook only sees defines): IS_WATER, NO_CONTACT_SHADOWS (set for
// `material.contactShadows = false` via ContactShadows.exclude( material ) and for every material
// under ContactShadows.skipRoots), and the late (transparent) pass (PASS_LATE: last frame's depth is
// this frame's there).
const GROUP = 4;
const GROUPS = 2;
const STEPS = GROUP * GROUPS;
const MAX_DIST = 32;
const NEAR_DIST = 12; // the second group of steps only runs nearer than this

const params = new UniformBlock( 'ContactShadowParams', { strength: [ 'f32', 1 ] }, { label: 'contactShadows' } );

// receivers under these objects are skipped (foliage: overdraw, still casts); adding a root marks
// the materials below it (call ContactShadows.refresh() after adding children to a root later)
class SkipSet extends Set {

	add( o ) {

		super.add( o );
		markTree( o );
		return this;

	}

}

function exclude( m ) {

	if ( m && m.setDefine ) m.setDefine( 'NO_CONTACT_SHADOWS', 1 );

}

function markTree( root ) {

	if ( root && root.traverse ) root.traverse( ( o ) => {

		if ( o.material ) for ( const m of Array.isArray( o.material ) ? o.material : [ o.material ] ) exclude( m );

	} );

}

export const ContactShadows = {
	strength: params.fields.strength,
	depthTexture: null,
	skipRoots: new SkipSet(),
	exclude,
	refresh() {

		for ( const r of this.skipRoots ) markTree( r );

	},
	module: null,
};

export function installContactShadows( { depthTexture, skip = [] } ) {

	ContactShadows.depthTexture = depthTexture;
	for ( const o of skip ) if ( o ) ContactShadows.skipRoots.add( o );
	const loop = [];
	for ( let g = 0; g < GROUPS; g ++ ) {

		let body = '';
		for ( let k = 0; k < GROUP; k ++ ) body += /* wgsl */`
			{
				let s = ( jit + ${ g * GROUP + k }.0 ) / ${ STEPS }.0;
				let u = s * s * 0.85 + s * 0.15; // denser near the contact
				let q = q0 + qd * u;
				let iw = 1.0 / q.w;
				let uv = q.xy * iw * vec2f( 0.5, -0.5 ) + 0.5;
				let d = textureLoad( contactDepth, vec2i( min( clamp( uv, vec2f( 0.0 ), vec2f( 1.0 ) ) * texSize, texSize - 1.0 ) ), 0 );
				let k2 = frame.near * iw * iw;
				let diff = d - q.z * iw; // > 0: the depth buffer is in front of the ray
				let thick = bias + 0.06 + u * len * 0.3;
				let occ = diff > bias * k2 && diff < thick * k2 && uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0;
				hit = min( hit, select( 2.0, u, occ ) );
			}`;
		const go = g === 0 ? 'hit > 1.0' : `hit > 1.0 && w0 < ${ NEAR_DIST }.0`;
		loop.push( `\t\tif ( ${ go } ) {${ body }\n\t\t}` );

	}

	ContactShadows.module = new ShaderModule( {
		name: 'hook-contactShadow',
		deps: [ commonModule ],
		uniforms: params,
		uniformName: 'contactShadowParams',
		bindings: { contactDepth: { texture: () => ContactShadows.depthTexture } },
		code: /* wgsl */`
// 0 (occluded) .. 1 visibility of the key light
fn hookContactShadow( P: vec3f, N: vec3f ) -> f32 {
#if IS_WATER || NO_CONTACT_SHADOWS || PASS_LATE || PASS_DEPTH || PASS_COLOR
	return 1.0;
#else
	let L = frame.sunDir;
	let fwd = -vec3f( frame.view[ 0 ][ 2 ], frame.view[ 1 ][ 2 ], frame.view[ 2 ][ 2 ] );
	let w0 = dot( P - frame.cameraPos, fwd );
	// depth change per pixel on this surface (before any branch)
	let slope = max( abs( dpdx( w0 ) ), abs( dpdy( w0 ) ) );
	var vis = 1.0;
	let lit = dot( frame.sunColor, frame.sunColor ) > 1e-8 && dot( N, L ) > 0.02;
	if ( contactShadowParams.strength > 0.0 && w0 < ${ MAX_DIST }.0 && lit && P.y > frame.seaLevel - 0.3 ) {

		let texSize = vec2f( textureDimensions( contactDepth ) );
		let len = smoothstep( 2.0, 30.0, w0 ) * 0.7 + 0.3;
		let q0 = frame.prevViewProjNoJitter * vec4f( P, 1.0 );
		let qd = frame.prevViewProjNoJitter * vec4f( L * len, 0.0 );
		let bias = slope * 2.0 + w0 * 0.002 + 0.01;
		// screen coordinate of P (the hook has no fragment coordinate)
		let cc = frame.viewProj * vec4f( P, 1.0 );
		let pix = floor( ( cc.xy / cc.w * vec2f( 0.5, -0.5 ) + 0.5 ) * frame.resolution );
		let jit = fract( interleavedGradientNoise( pix ) + f32( frame.frameIndex ) * 0.618034 );
		var hit = 2.0; // ray parameter of the first occluded sample (> 1: none)
		// groups of independent loads (latency hiding), early exit between groups; far away the ray
		// covers few pixels and the first group is enough. Depths compare in reversed-Z device depth
		// (d ~ near / w): the bias and thickness in metres scale by near / w^2.
${ loop.join( '\n' ) }

		// the sun's penumbra is millimetres at this range: a hard shadow, faded out toward the end
		// of the ray (no cut-off line at its length)
		vis = select( 1.0, smoothstep( 0.55, 1.0, hit ), hit <= 1.0 );
		vis = 1.0 - ( 1.0 - vis ) * contactShadowParams.strength * smoothstep( ${ MAX_DIST }.0, ${ MAX_DIST - 8 }.0, w0 );

	}
	return vis;
#endif
}
`,
	} );
	SceneLighting.set( 'contactShadow', ContactShadows.module );

}
