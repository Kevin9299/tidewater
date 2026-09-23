import * as THREE from 'three/webgpu';
import {
	Fn, If, float, vec2, vec3, vec4, uint, attribute, positionLocal, normalLocal, positionView, mrt,
	cameraProjectionMatrix, cameraViewMatrix, min,
	normalView, normalWorldGeometry, faceDirection, varyingProperty, floor, step, texture, sin, cos, mix, smoothstep, saturate, dot, cross,
	normalize, length, abs, fract, max, exp, pow, fwidth, sqrt, select,
	mx_noise_float, mx_worley_noise_vec2, mx_cell_noise_float, atan, interleavedGradientNoise, screenCoordinate, texture3D, Discard,
} from 'three/tsl';
import { bayer4 } from '../../materials/LODFade.js';
import { G } from '../../core/Globals.js';
import { WORLD } from '../WorldLayout.js';
import { physical } from '../../materials/Materials.js';
import { staticVelocity } from '../../post/CameraVelocity.js';
import { NOISE_SCALE } from './ReefNoise.js';

// TSL materials of the reef. All surface detail is procedural; the only texture is a small
// tileable 3D noise volume (ReefNoise.js) that replaces per-pixel noise evaluation: one fetch
// gives four noise values, which made the reef several times cheaper to shade.
//
// Both reef batches share the vertex stage: the instance record (see Reef.js) places,
// rotates and scales the model, massive forms get a per-colony warp, flexible organisms
// sway with the surge of the swell, and everything the fragment stage needs is passed as
// a few varyings.
//
// Instance record (4 x vec4):
//   r0 = ( position.xyz, scale )    r1 = orientation quaternion
//   r2 = ( colour 1, colour 2 (packed sRGB 8:8:8), draw distance (whole m) + seed, surface type )
//   r3 = ( stretch.xyz, flexibility (> 0) or shape warp (< 0) )
//
// The hard batch has no discard, which keeps the GPU's hidden surface removal effective
// (overdraw is cheap). Surfaces are close to water in refractive index, so underwater they
// show almost no specular reflection: specular intensity is kept low except on glossy spines.

export const aData = attribute( 'aData', 'vec4' );

export const SURFACE = {
	rock: 0, star: 1, brain: 2, brainWide: 3, porites: 4, acropora: 5, finger: 6, fire: 7, plate: 8, pillar: 9,
	gorgonian: 10, sponge: 11, urchin: 12, anemone: 13, rubble: 14, grass: 15, fan: 16, plume: 17,
	lettuce: 18, algae: 19, sargassum: 20,
};

const SWELL = new THREE.Vector2( WORLD.swellDir.x, WORLD.swellDir.y ).normalize();
const swellDir = vec2( SWELL.x, SWELL.y );
const swellSide = vec2( - SWELL.y, SWELL.x );
const TAU = Math.PI * 2;

export const srgb = ( hex ) => {

	const c = new THREE.Color( hex );
	return vec3( c.r, c.g, c.b );

};

// packs a colour as 8 bit sRGB into a float (exact: 24 bits)
export const packColor = ( c ) => c.getHex();

const unpackColor = ( f ) => {

	const u = uint( f.add( 0.5 ) ); // the value is exact, but interpolation may leave it a hair below
	const c = vec3( float( u.shiftRight( 16 ).bitAnd( 255 ) ), float( u.shiftRight( 8 ).bitAnd( 255 ) ), float( u.bitAnd( 255 ) ) ).div( 255 );
	return pow( c, vec3( 2.2 ) );

};

export const rotateQ = ( q, v ) => v.add( cross( q.xyz, cross( q.xyz, v ).add( v.mul( q.w ) ) ).mul( 2 ) );

// Horizontal water displacement (m) at xz and the given depth, which sways the reef back and
// forth: the long-period surge of the ground swell (a coherent analytic wave train travelling
// with the swell: in 2 - 10 m of water its orbits reach the seabed almost undiminished) plus the
// orbital motion of the simulated waves overhead (long FFT cascades, when the ocean is known).
// Returns { now, prev }: prev is the analytic part one frame earlier (for motion vectors; the
// FFT part changes little in a frame).
export function makeSurge( getOcean ) {

	const swell = ( xz, t ) => {

		const along = dot( xz, swellDir ), across = dot( xz, swellSide );
		const a = sin( t.mul( TAU / 8.2 ).sub( along.mul( TAU / 70 ) ).add( sin( across.mul( 0.021 ) ).mul( 1.5 ) ) ).mul( 0.3 );
		const b = sin( t.mul( TAU / 5.3 ).sub( along.mul( TAU / 34 ) ).add( across.mul( 0.09 ) ) ).mul( 0.1 );
		const c = sin( t.mul( TAU / 11.7 ).add( across.mul( TAU / 45 ) ) ).mul( 0.07 );
		return swellDir.mul( a.add( b ) ).add( swellSide.mul( c ) );

	};

	return ( xz, depth ) => {

		let d = vec2( 0 );
		const fft = getOcean ? getOcean() : null;
		if ( fft ) {

			for ( let k = 0; k < Math.min( 2, fft.cascades ); k ++ ) {

				const L = fft.sizes[ k ];
				const smp = texture( fft.displacementTexture, xz.div( L ) ).depth( k ).level( 3 );
				d = d.add( vec2( smp.x, smp.z ).mul( exp( depth.mul( - TAU / L ) ) ) );

			}

		}

		return { now: swell( xz, G.time ).add( d ), prev: swell( xz, G.time.sub( G.dt ) ).add( d ) };

	};

}

// varyings shared by the reef materials (kept small: on tile-based GPUs every vertex output
// is written to memory)
const vColors = varyingProperty( 'vec2', 'vReefColors' ); // two packed sRGB colours
const vVelocity = varyingProperty( 'vec2', 'vReefVelocity' ); // own motion in NDC (swaying parts)
export const V = {
	local: varyingProperty( 'vec3', 'vReefLocal' ), // object-space position (m)
	data: varyingProperty( 'vec4', 'vReefData' ), // aData
	info: varyingProperty( 'vec3', 'vReefInfo' ), // seed, surface type, scale
};

// Fragment stage: the two instance colours (linear).
export const reefColors = () => [ unpackColor( vColors.x ), unpackColor( vColors.y ) ];

// Motion vectors: camera motion plus the sway of flexible organisms.
export const reefVelocityMRT = mrt( { velocity: staticVelocity.add( vVelocity ) } );

const ndc = ( p ) => {

	const c = cameraProjectionMatrix.mul( cameraViewMatrix ).mul( vec4( p, 1 ) );
	return c.xy.div( c.w );

};

// Vertex stage: returns the world position (the batch meshes have identity transforms)
// and writes the normal and the varyings. view = { position, range } uniforms: the main
// camera (not the shadow camera) and the draw distance; instances shrink away over the last
// fifth of their draw distance so that nothing pops when the culling adds or drops them.
// level-of-detail cross-fade (ReefBatch fade channel): share of this level, outgoing (1) or not
const vFade = varyingProperty( 'vec2', 'vReefFade' );
const fadeDiscard = () => {

	const t = bayer4();
	Discard( select( vFade.y.greaterThan( 0.5 ), t.lessThan( vFade.x ), t.greaterThanEqual( vFade.x ) ) );

};

export function reefVertex( batch, surge, view, fade = false ) {

	return Fn( () => {

		let index;
		if ( fade ) {

			const e = batch.fadeEntry();
			vFade.assign( vec2( e.fade, e.outgoing ) );
			index = e.index;

		} else index = batch.recordIndex();
		const rec = batch.record( index );
		const r0 = rec[ 0 ], q = rec[ 1 ], r2 = rec[ 2 ], r3 = rec[ 3 ];
		const seed = fract( r2.z ); // integer part: the instance's draw distance (m)
		const fadeEnd = min( floor( r2.z ), view.range );
		const grow = float( 1 ).sub( smoothstep( fadeEnd.mul( 0.75 ), fadeEnd.mul( 0.95 ), length( r0.xyz.sub( view.position ) ) ) );
		const scale = r3.xyz.mul( r0.w );
		const p = positionLocal.mul( scale ).toVar();
		V.local.assign( p );
		V.data.assign( aData );
		vColors.assign( r2.xy );
		V.info.assign( vec3( seed, r2.w, r0.w ) );
		vVelocity.assign( vec2( 0 ) );
		normalLocal.assign( normalize( rotateQ( q, normalLocal.div( scale ) ) ) );
		const flex = r3.w;
		If( flex.lessThan( 0 ), () => {

			// massive forms: a smooth per-colony warp so no two look cloned (the base stays put)
			const w = positionLocal.mul( 3.1 ).add( seed.mul( 17 ) );
			const off = vec3( sin( w.y.mul( 1.7 ).add( w.z ) ), sin( w.z.mul( 1.3 ).add( w.x.mul( 1.1 ) ) ).mul( 0.6 ), sin( w.x.mul( 1.9 ).add( w.y.mul( 0.7 ) ) ) );
			p.addAssign( off.mul( flex.negate().mul( r0.w ).mul( aData.x.mul( 0.8 ).add( 0.2 ) ) ) );

		} );
		const world = rotateQ( q, p.mul( grow ) ).add( r0.xyz ).toVar();
		If( flex.greaterThan( 0 ), () => {

			// bend grows with the square of the position along the organism; tips lag behind
			const t = aData.x;
			const h = max( world.y.sub( r0.y ), 0 );
			const depth = max( G.seaLevel.sub( world.y ), 0 );
			const s = surge( r0.xz.sub( swellDir.mul( h.mul( 4 ) ) ), depth );
			const ph = G.time.mul( 1.7 ).add( seed.mul( 40 ) ).add( aData.w.mul( 12 ) );
			const phPrev = ph.sub( G.dt.mul( 1.7 ) );
			const bend = flex.mul( t ).mul( t ).mul( grow );
			const offset = ( sv, phase ) => {

				const d = sv.add( vec2( sin( phase ), cos( phase.mul( 0.77 ) ) ).mul( 0.06 ) ).mul( bend );
				return vec3( d.x, dot( d, d ).div( max( h, 0.1 ).mul( 2 ) ).negate(), d.y );

			};

			const now = offset( s.now, ph ).toVar();
			world.addAssign( now );
			vVelocity.assign( ndc( world ).sub( ndc( world.sub( now ).add( offset( s.prev, phPrev ) ) ) ) );

		} );
		return world;

	} )();

}

// ---------------------------------------------------------------------------
// fragment helpers

// pixel footprint in meters, and a 1 -> 0 fade for patterns of the given spacing
const pixel = () => length( fwidth( positionView ) ).mul( 0.7 );
const fade = ( spacing, px ) => float( 1 ).sub( smoothstep( 0.25, 0.6, px.div( spacing ) ) );

// view-space bump mapping from a scalar height field (Mikkelsen's surface gradient)
export const bumpNormal = ( height ) => {

	const dpdx = positionView.dFdx(), dpdy = positionView.dFdy();
	const dhdx = height.dFdx(), dhdy = height.dFdy();
	const n = normalView.mul( faceDirection );
	const r1 = cross( dpdy, n ), r2 = cross( n, dpdx );
	const det = dot( dpdx, r1 );
	const grad = det.sign().mul( dhdx.mul( r1 ).add( dhdy.mul( r2 ) ) );
	return normalize( det.abs().mul( n ).sub( grad ) );

};

const hue = ( c, n, amount ) => c.mul( vec3( 1 ).add( vec3( n, n.mul( 0.5 ), n.mul( - 0.5 ) ).mul( amount ) ) );

// cheap packed polyp / corallite bumps (0..1) with about `spacing` meters between them
const polyps = ( p, spacing ) => {

	const q = p.mul( Math.PI / spacing );
	const w = q.add( sin( q.yzx.mul( 0.47 ) ).mul( 1.4 ) ).add( sin( q.zxy.mul( 0.29 ).add( 1.7 ) ).mul( 1.1 ) );
	return abs( sin( w.x ).mul( sin( w.y ) ).mul( sin( w.z ) ) );

};

// Round features on a jittered 3D grid of `size` meters (one hash per pixel): distance from
// the cell's feature point (0 .. ~0.9) and a random value per cell. The surface slices the
// cells at random heights, so the cups / pits / pores vary in size.
const cells = ( p, size ) => {

	const q = p.div( size );
	const h = mx_cell_noise_float( floor( q ) );
	const c = vec3( h, fract( h.mul( 7.13 ) ), fract( h.mul( 3.71 ) ) ).mul( 0.3 ).add( 0.35 );
	return { d: length( fract( q ).sub( c ) ), h };

};

// ---------------------------------------------------------------------------

// fade: the materials of the level-of-detail cross-fade channels (dithered, see LODFade.js)
export function createReefMaterials( { hard, soft, getOcean, noise, view, fade: lodFade = false } ) {

	const surge = makeSurge( getOcean );
	// four noise values per fetch; the tile holds 8 noise cells, so sampling at p * k gives
	// features of 1 / ( 8 k ) m
	const volNode = texture3D( noise );
	const vol = ( p ) => volNode.sample( p ).sub( 0.5 ).mul( NOISE_SCALE );

	// ---- hard batch: everything opaque (no discard keeps hidden surface removal intact)
	const mat = physical( { roughness: 0.8, metalness: 0 } );
	mat.name = lodFade ? 'ReefFade' : 'Reef';
	mat.positionNode = reefVertex( hard, surge, view, lodFade );
	mat.mrtNode = reefVelocityMRT;

	const albedo = vec3( 0 ).toVar( 'reefAlbedo' );
	const height = float( 0 ).toVar( 'reefHeight' );
	const rough = float( 0.8 ).toVar( 'reefRough' );
	const occl = float( 1 ).toVar( 'reefOcc' );
	const spec = float( 0.3 ).toVar( 'reefSpec' );
	const thin = float( 0 ).toVar( 'reefThin' ); // light passing through thin tissue (blades, fronds, rods)

	mat.colorNode = Fn( () => {

		if ( lodFade ) fadeDiscard();
		// everything shared by the branches is computed up front (a node first built inside a
		// branch would only be assigned there)
		const [ uc1, uc2 ] = reefColors();
		const L = V.local.toVar(), D = V.data.toVar(), c1 = uc1.toVar(), c2 = uc2.toVar();
		const seed = V.info.x.toVar(), type = V.info.y.add( 0.5 ).floor().toVar();
		const t = D.x.toVar(), ao = D.y.toVar(), part = D.z.toVar(), rnd = D.w.toVar();
		const px = pixel().toVar();
		const P = L.add( seed.mul( 17.0 ) ).toVar();
		const A = vol( P.mul( 0.4 ) ).toVar(); // ~30 cm features
		const B = vol( P.mul( 1.25 ).add( 0.37 ) ).toVar(); // ~10 cm features
		const n1 = A.x, n2 = B.x; // colony-scale mottling, blotches
		const base = hue( c1, n1, 0.18 ).mul( float( 1 ).add( n2.mul( 0.12 ) ) ).toVar();
		albedo.assign( base );
		occl.assign( ao );
		height.assign( 0 );
		rough.assign( 0.8 );
		spec.assign( 0.3 );
		thin.assign( select( type.equal( SURFACE.grass ).or( type.equal( SURFACE.sargassum ) ), 0.55, select( type.equal( SURFACE.gorgonian ).or( type.equal( SURFACE.lettuce ) ).or( type.equal( SURFACE.algae ) ), 0.3, 0 ) ) );

		// turf-covered dead base where a colony meets the seabed
		const turf = mix( srgb( 0x5e5234 ), srgb( 0x725c46 ), smoothstep( - 0.3, 0.4, n2 ) ).toVar();
		const up = saturate( normalWorldGeometry.y ).toVar();

		If( type.equal( SURFACE.rock ), () => {

			// dead coral framework: algal turf with a sediment veneer on the tops, crustose coralline
			// algae (pink / lavender) spreading over sides and edges, small encrusting colonies and
			// sponges, bio-eroded pits and cracks
			const n3 = A.y, n4 = B.y;
			// fine speckle and grain (turf filaments, coralline bumps, pores), fetched only where
			// they are resolved (most reef pixels are distant)
			const speck = float( 0 ).toVar(), grainF = float( 0 ).toVar();
			const fs = fade( 0.04, px );
			If( fs.greaterThan( 0.001 ), () => {

				speck.assign( vol( P.mul( 3.2 ).add( 0.71 ) ).x.mul( fs ) );
				const fg = fade( 0.012, px );
				If( fg.greaterThan( 0.001 ), () => {

					const gv = vol( P.mul( 12 ).add( 1.7 ) );
					grainF.assign( gv.x.add( gv.y.mul( 0.5 ) ).mul( fg ) );

				} );

			} );
			const turfC = mix( srgb( 0x6a5c44 ), srgb( 0x86744e ), smoothstep( - 0.4, 0.4, n2.add( speck.mul( 0.6 ) ) ) );
			let c = mix( base, turfC, smoothstep( - 0.3, 0.3, n1.add( up.mul( 0.3 ) ) ).mul( 0.7 ) );
			// sediment veneer on flat tops
			c = mix( c, srgb( 0xa89e88 ), smoothstep( 0.82, 0.97, up ).mul( smoothstep( - 0.1, 0.3, A.z ) ).mul( 0.55 ) );
			// coralline crusts
			const cca = smoothstep( 0.05, 0.3, n3.mul( 0.7 ).add( n4.mul( 0.5 ) ) ).mul( float( 1 ).sub( up.mul( 0.5 ) ) );
			c = mix( c, c2.mul( float( 0.88 ).add( speck.mul( 0.2 ) ).add( grainF.mul( 0.25 ) ) ), cca.mul( 0.8 ) );
			// small encrusting colonies (mustard / brown / olive) with corallites
			const col = smoothstep( 0.18, 0.26, B.z.add( A.w.mul( 0.25 ) ) );
			const colC = mix( mix( srgb( 0xa89048 ), srgb( 0x8a6a48 ), smoothstep( - 0.3, 0.3, A.w ) ), srgb( 0x7a7c4a ), smoothstep( 0.2, 0.5, A.x ) );
			const colPol = polyps( P, 0.006 ).mul( fade( 0.006, px ) ).mul( col );
			c = mix( c, colC.mul( float( 0.8 ).add( colPol.mul( 0.35 ) ).add( grainF.mul( 0.2 ) ) ), col.mul( 0.75 ) );
			// encrusting sponges on the sides
			const sp = smoothstep( 0.3, 0.38, B.w.add( A.z.mul( 0.3 ) ) ).mul( float( 1 ).sub( up.mul( 0.7 ) ) );
			const spC = mix( mix( srgb( 0xc0662c ), srgb( 0xa83a30 ), smoothstep( - 0.2, 0.2, A.y ) ), mix( srgb( 0x7a3a82 ), srgb( 0xc8a02c ), smoothstep( 0.1, 0.3, B.y ) ), smoothstep( 0.2, 0.4, A.w ) );
			c = mix( c, spC, sp.mul( 0.9 ) );
			// bio-eroded pits of varied size, coral recruits here and there
			const pc = cells( P, 0.03 );
			const pitR = pc.h.mul( 0.16 ).add( 0.06 );
			const pit = float( 1 ).sub( smoothstep( pitR, pitR.add( 0.08 ), pc.d ) ).mul( step( 0.55, fract( pc.h.mul( 5.3 ) ) ) ).mul( smoothstep( 0.0, 0.3, A.w.negate() ) ).mul( fade( 0.045, px ) );
			const rc = cells( P, 0.08 );
			const recruit = float( 1 ).sub( smoothstep( 0.2, 0.3, rc.d ) ).mul( step( 0.93, rc.h ) );
			c = mix( c, mix( srgb( 0xa89648 ), srgb( 0xb07a50 ), fract( rc.h.mul( 11.3 ) ) ), recruit.mul( 0.9 ) );
			// rugged relief: ridged creases at two scales, rough grain; crevices stay dark
			const r3 = float( 1 ).sub( abs( n3 ) ).mul( 2 ).sub( 1 );
			const r4 = float( 1 ).sub( abs( n4 ) ).mul( 2 ).sub( 1 );
			const grain = speck.mul( 1.4 );
			const cav = smoothstep( - 0.9, 0.2, r4.add( grain.mul( 0.4 ) ).add( grainF.mul( 0.3 ) ) );
			c = c.mul( float( 0.88 ).add( speck.mul( 0.25 ) ).add( grainF.mul( 0.3 ) ) ).mul( float( 1 ).sub( pit.mul( mix( 0.35, 0.7, pc.h ) ) ) ).mul( mix( 0.72, 1.0, cav ) );
			albedo.assign( c );
			height.assign( n1.mul( 0.04 ).add( r3.mul( 0.016 ) ).add( r4.mul( 0.008 ) ).add( grain.mul( 0.003 ) ).add( grainF.mul( 0.0018 ) ).add( col.mul( 0.003 ) )
				.add( colPol.mul( 0.0008 ) ).add( sp.mul( 0.002 ) ).add( recruit.mul( 0.004 ) ).sub( pit.mul( 0.008 ) ) );
			occl.assign( occl.mul( float( 1 ).sub( pit.mul( 0.5 ) ) ).mul( mix( 0.65, 1.0, cav ) ) );
			rough.assign( 0.95 );

		} ).ElseIf( type.equal( SURFACE.star ), () => {

			// star coral: packed corallite cups with raised walls, mottled colony tissue
			const f = fade( 0.006, px );
			const cup = cells( P, 0.0055 );
			const rim = smoothstep( 0.3, 0.55, cup.d ).mul( f );
			const lump = B.z.mul( fade( 0.05, px ) );
			const tone = mix( base, c2, smoothstep( 0.0, 0.3, n1.add( n2.mul( 0.4 ) ) ).mul( 0.7 ) ).mul( float( 0.9 ).add( lump.mul( 0.15 ) ) );
			albedo.assign( mix( tone.mul( 0.7 ), tone.mul( 1.12 ), rim.add( float( 1 ).sub( f ).mul( 0.5 ) ) ) );
			height.assign( rim.mul( 0.0018 ).add( n2.mul( 0.004 ) ).add( lump.mul( 0.004 ) ).add( n1.mul( 0.01 ) ) );
			rough.assign( 0.78 );

		} ).ElseIf( type.equal( SURFACE.brain ).or( type.equal( SURFACE.brainWide ) ), () => {

			// brain corals: meandering valleys (the zero set of warped noise); wide-valley species
			// have a pale groove along the ridge
			const wide = type.equal( SURFACE.brainWide );
			const freq = select( wide, 28, 52 ).mul( mix( 0.85, 1.15, fract( seed.mul( 7.3 ) ) ) );
			const q = P.mul( freq );
			const wq = q.add( vol( q.mul( 0.31 / 8 ).add( seed ) ).xyz.mul( 0.9 ) );
			const mn = mx_noise_float( wq );
			const aa = px.mul( freq ).mul( 2.2 ).max( 1e-4 ); // ~ fwidth( mn ): no derivatives in divergent branches
			const width = select( wide, 0.12, 0.075 );
			const ridge = mix( 0.55, smoothstep( width.sub( aa ), width.add( aa ), abs( mn ) ), fade( float( 1 ).div( freq ), px ) );
			const valley = mix( c2, base.mul( 0.55 ), 0.35 );
			const top = base.mul( 1.15 );
			const groove = smoothstep( 0.3, 0.34, abs( mn ) ).mul( float( 1 ).sub( smoothstep( 0.36, 0.4, abs( mn ) ) ) ).mul( fade( float( 0.5 ).div( freq ), px ) ).mul( select( wide, 1, 0.7 ) );
			albedo.assign( mix( valley, top, ridge ).mul( float( 1 ).sub( groove.mul( 0.25 ) ) ) );
			height.assign( ridge.mul( select( wide, 0.006, 0.0035 ) ).sub( groove.mul( 0.001 ) ).add( n1.mul( 0.006 ) ) );
			rough.assign( 0.7 );

		} ).ElseIf( type.equal( SURFACE.porites ), () => {

			// mustard hill / starlet corals: fine pitted surface, knobby, mottled
			const pit = polyps( P, 0.004 ).mul( fade( 0.004, px ) );
			const knob = polyps( P.add( 3.3 ), 0.025 ).mul( fade( 0.025, px ) );
			const tone = mix( base, c2, smoothstep( 0.1, 0.4, n1 ).mul( 0.5 ) );
			albedo.assign( tone.mul( float( 0.74 ).add( pit.mul( 0.22 ) ).add( knob.mul( 0.28 ) ) ) );
			height.assign( pit.mul( 0.0008 ).add( knob.mul( 0.006 ) ).add( n2.mul( 0.008 ) ).add( B.z.mul( 0.004 ) ) );
			rough.assign( 0.75 );

		} ).ElseIf( type.equal( SURFACE.acropora ), () => {

			// elkhorn / staghorn: protruding corallites (a rough, bumpy surface), mottled
			// golden-brown tissue, a thin pale growing margin at the very tips
			const f = fade( 0.005, px );
			const bump = polyps( P, 0.005 ).mul( f );
			const rough2 = vol( P.mul( 4.75 ) ).x.mul( fade( 0.04, px ) );
			const tip = smoothstep( 0.6, 1.0, part ).mul( smoothstep( 0.9, 1.0, t ) );
			const mott = mix( base, base.mul( vec3( 1.1, 0.95, 0.8 ) ), smoothstep( - 0.3, 0.4, n2 ) );
			albedo.assign( mix( mott, c2, tip.mul( 0.8 ) ).mul( float( 0.84 ).add( bump.mul( 0.2 ) ).add( rough2.mul( 0.14 ) ) ) );
			height.assign( bump.mul( 0.0016 ).add( rough2.mul( 0.0025 ) ).add( n2.mul( 0.002 ) ) );
			rough.assign( 0.8 );

		} ).ElseIf( type.equal( SURFACE.finger ).or( type.equal( SURFACE.pillar ) ), () => {

			// finger / pillar corals: fuzzy with extended polyps, blunt pale tips
			const fzv = vol( P.mul( 18 ) );
			const fz = fzv.x.mul( fade( 0.008, px ) ).add( fzv.y.mul( fade( 0.03, px ) ).mul( 0.6 ) );
			const tip = smoothstep( 0.5, 1.0, part );
			albedo.assign( mix( base, c2, tip.mul( 0.6 ) ).mul( float( 0.88 ).add( fz.mul( 0.25 ) ) ) );
			height.assign( fz.mul( 0.002 ) );
			rough.assign( 0.95 );

		} ).ElseIf( type.equal( SURFACE.fire ), () => {

			// fire coral: smooth, finely porous, mottled mustard with white growing edges
			const edge = smoothstep( 0.86, 1.0, t ).mul( part.mul( 0.5 ).add( 0.5 ) );
			const mott = mix( base, base.mul( vec3( 1.1, 1.0, 0.75 ) ), smoothstep( - 0.3, 0.3, n2 ) );
			const pore = polyps( P, 0.003 ).mul( fade( 0.003, px ) );
			albedo.assign( mix( mott, c2, edge.mul( 0.85 ) ).mul( float( 0.92 ).add( pore.mul( 0.12 ) ) ) );
			height.assign( n2.mul( 0.004 ).add( B.z.mul( 0.002 ) ).add( pore.mul( 0.0004 ) ) );
			rough.assign( 0.6 );

		} ).ElseIf( type.equal( SURFACE.plate ), () => {

			// plate corals: concentric ridges on top, pale growing margin, darker underside
			const r = length( L.xz );
			const ridges = sin( r.mul( 260 ).add( n1.mul( 3 ) ) ).mul( fade( 0.025, px ) ).mul( part );
			const margin = smoothstep( 0.85, 1.0, t );
			albedo.assign( mix( base.mul( mix( 0.55, 1.0, part ) ), c2, margin.mul( 0.7 ) ).mul( float( 1 ).add( ridges.mul( 0.08 ) ) ) );
			height.assign( ridges.mul( 0.0012 ) );
			rough.assign( 0.7 );

		} ).ElseIf( type.equal( SURFACE.gorgonian ), () => {

			// sea rods / whips: fuzzy with polyps; tips a little paler
			const fz = vol( P.mul( 27 ) ).x.mul( fade( 0.006, px ) );
			albedo.assign( mix( base, c2, smoothstep( 0.6, 1.0, t ).mul( 0.5 ) ).mul( float( 0.85 ).add( fz.mul( 0.25 ) ) ) );
			height.assign( fz.mul( 0.001 ) );
			rough.assign( 0.95 );

		} ).ElseIf( type.equal( SURFACE.sponge ), () => {

			// sponges: pores and oscula, a dark inner cavity
			const f = fade( 0.008, px );
			const pc = cells( P, 0.009 );
			const pore = float( 1 ).sub( smoothstep( 0.1, 0.25, pc.d ) ).mul( f );
			const inner = part;
			albedo.assign( base.mul( float( 1 ).sub( pore.mul( 0.45 ) ) ).mul( mix( 1.0, 0.3, inner ) ) );
			height.assign( pore.mul( - 0.0015 ).add( n2.mul( 0.004 ) ) );
			occl.assign( ao.mul( mix( 1.0, 0.45, inner ) ) );
			rough.assign( 0.92 );

		} ).ElseIf( type.equal( SURFACE.urchin ), () => {

			// long-spined urchin: glossy black spines with faint bands, dark purple test
			const band = sin( t.mul( 40 ) ).mul( 0.5 ).add( 0.5 ).mul( part ).mul( 0.15 );
			albedo.assign( mix( c2, c1, part ).add( band.mul( 0.02 ) ) );
			rough.assign( 0.35 );
			spec.assign( 1 );

		} ).ElseIf( type.equal( SURFACE.anemone ), () => {

			// anemone: pale translucent tentacles with coloured tips
			albedo.assign( mix( c1, c2, smoothstep( 0.6, 1.0, t ).mul( part ) ) );
			rough.assign( 0.6 );
			spec.assign( 0.6 );

		} ).ElseIf( type.equal( SURFACE.rubble ), () => {

			// rubble: bleached fragments and turf-covered pieces
			const bleached = mix( srgb( 0xd4ccbc ), srgb( 0xb9ad98 ), n2.mul( 0.5 ).add( 0.5 ) );
			const turfed = mix( turf, c2, smoothstep( 0.2, 0.6, n1 ).mul( 0.6 ) );
			albedo.assign( mix( turfed, bleached, part ) );
			height.assign( polyps( P, 0.004 ).mul( 0.0008 ).mul( fade( 0.004, px ) ) );
			rough.assign( 0.9 );

		} ).ElseIf( type.equal( SURFACE.lettuce ), () => {

			// lettuce coral: thin brown blades with fine ridges running up to a pale, growing edge
			const r = fade( 0.004, px );
			const ridges = sin( L.x.add( L.z ).mul( 700 ).add( n1.mul( 4 ) ) ).mul( r );
			const edgeK = smoothstep( 0.82, 1.0, t );
			albedo.assign( mix( base.mul( float( 0.9 ).add( ridges.mul( 0.08 ) ) ), c2, edgeK.mul( 0.75 ) ) );
			height.assign( ridges.mul( 0.0007 ).add( n2.mul( 0.002 ) ) );
			rough.assign( 0.7 );

		} ).ElseIf( type.equal( SURFACE.algae ), () => {

			// calcareous green algae: segments / tufts dusted with lime, paler at the edges
			const lime = smoothstep( 0.0, 0.6, n2.add( t.mul( 0.4 ) ) ).mul( 0.35 );
			const tuft = part;
			albedo.assign( mix( base, c2, lime.add( tuft.mul( 0.2 ) ) ).mul( mix( 0.85, 1.05, rnd ) ) );
			rough.assign( 0.85 );

		} ).ElseIf( type.equal( SURFACE.sargassum ), () => {

			// Sargassum: olive-golden fronds, darker wiry axes, amber gas bladders
			const leaf = select( part.greaterThan( 0.5 ).and( part.lessThan( 1.5 ) ), 1, 0 );
			const bladder = select( part.greaterThan( 1.5 ), 1, 0 );
			const tone = base.mul( mix( 0.8, 1.15, rnd ) ).mul( mix( 0.75, 1.05, t ) );
			albedo.assign( mix( mix( tone.mul( 0.6 ), tone, leaf ), srgb( 0xa87a30 ), bladder.mul( 0.7 ) ) );
			rough.assign( 0.55 );
			spec.assign( 0.45 );

		} ).Else( () => {

			// seagrass: dark bases, epiphyte-covered older tips
			const aged = mix( base, srgb( 0x8a7c48 ), 0.6 );
			const tip = smoothstep( 0.5, 1.0, t ).mul( fract( rnd.mul( 13.7 ) ).mul( 0.7 ).add( 0.3 ) );
			albedo.assign( mix( base.mul( mix( 0.55, 1.0, smoothstep( 0.0, 0.4, t ) ) ), aged, tip ) );
			rough.assign( 0.6 );

		} );

		// the attached base of colonies is dead, turf-covered skeleton, with a paler growing
		// margin of living tissue just above it
		const isColony = type.greaterThan( 0.5 ).and( type.lessThan( 9.5 ) );
		const colonyK = select( isColony, 1, 0 );
		const edge = t.add( n2.mul( 0.03 ) );
		const deadBase = float( 1 ).sub( smoothstep( 0.0, 0.06, edge ) ).mul( colonyK );
		const margin = smoothstep( 0.04, 0.07, edge ).mul( float( 1 ).sub( smoothstep( 0.07, 0.12, edge ) ) ).mul( colonyK );
		albedo.assign( mix( albedo, turf, deadBase.mul( 0.9 ) ).mul( float( 1 ).add( margin.mul( 0.25 ) ) ) );
		// crevices also receive less direct light
		return albedo.mul( mix( 0.6, 1.0, occl ) );

	} )();
	mat.normalNode = Fn( () => bumpNormal( height ) )();
	mat.roughnessNode = rough;
	mat.aoNode = occl;
	mat.specularIntensityNode = spec;
	mat.translucencyNode = ( lightColor ) => {

		const back = saturate( dot( normalWorldGeometry.negate(), G.sunDir ) ).mul( 0.8 ).add( 0.2 );
		return lightColor.mul( albedo ).mul( thin.mul( back ) );

	};

	// ---- soft batch: alpha-tested sea fans and plumes (double-sided, swaying)
	const fanMat = physical( { roughness: 0.85, metalness: 0, side: THREE.DoubleSide } );
	fanMat.name = lodFade ? 'ReefSoftFade' : 'ReefSoft';
	fanMat.positionNode = reefVertex( soft, surge, view, lodFade );
	fanMat.mrtNode = reefVelocityMRT;
	const mask = float( 1 ).toVar( 'fanMask' );
	const fanAlbedo = vec3( 0 ).toVar( 'fanAlbedo' );
	fanMat.maskNode = Fn( () => {

		if ( lodFade ) fadeDiscard();
		const [ uc1, uc2 ] = reefColors();
		const L = V.local.toVar(), D = V.data.toVar(), c1 = uc1.toVar(), c2 = uc2.toVar();
		const seed = V.info.x.toVar(), type = V.info.y.add( 0.5 ).floor().toVar(), s = V.info.z.toVar();
		const t = D.x.toVar(), part = D.z.toVar(), rnd = D.w.toVar();
		const px = pixel().toVar();
		const dither = interleavedGradientNoise( screenCoordinate.xy.add( fract( G.time.mul( 7.3 ) ).mul( 97 ) ) ).toVar();
		If( type.equal( SURFACE.fan ), () => {

			// fine net of anastomosing branchlets (~7 mm mesh, cells stretched along the radial
			// veins) and main veins forking outward from the stalk. Where the mesh gets too fine
			// to resolve the holes fill in (a solid silhouette): holes cost overdraw.
			const q = L.xy.add( seed.mul( 3.1 ) );
			const cell = 0.007;
			const resolve = smoothstep( 0.3, 1.0, px.div( cell ) );
			const rel = L.xy.sub( vec2( 0, s.mul( 0.06 ) ) );
			const r = length( rel ).div( s );
			const th = atan( rel.x, rel.y );
			const forks = select( r.lessThan( 0.22 ), 5, select( r.lessThan( 0.48 ), 10, 20 ) );
			const wv = vol( vec3( r.mul( 0.75 ), seed.mul( 1.1 ), q.x.mul( 0.5 ) ) );
			const vv = abs( fract( th.mul( forks ).div( Math.PI ).add( wv.x.mul( 0.35 ) ) ).sub( 0.5 ) ).mul( 2 );
			const veinW = mix( 0.12, 0.035, r );
			const vein = float( 1 ).sub( smoothstep( veinW, veinW.add( 0.03 ), vv ) );
			const strand = select( dither.lessThan( 0.72 ), 1, 0 ).toVar();
			If( resolve.lessThan( 0.98 ), () => {

				const net = mx_worley_noise_vec2( vec2( q.x, q.y.mul( 0.55 ) ).div( cell ), 0.9 );
				const edge = sqrt( net.y ).sub( sqrt( net.x ) );
				strand.assign( select( edge.lessThan( 0.16 ).or( dither.lessThan( resolve.mul( 0.72 ) ) ), 1, 0 ) );

			} );
			mask.assign( select( strand.greaterThan( 0.5 ).or( vein.greaterThan( 0.5 ) ).or( part.greaterThan( 0.5 ) ).or( t.lessThan( 0.05 ) ), 1, 0 ) );
			const tone = wv.y.mul( 0.5 ).add( 0.5 );
			// a filled-in (distant) fan is darker: holes let the background through
			const col = mix( c1.mul( mix( 0.8, 1.1, tone ) ), c2, vein.mul( 0.35 ) );
			fanAlbedo.assign( mix( col, vec3( dot( col, vec3( 0.3, 0.5, 0.2 ) ) ), resolve.mul( 0.45 ) ).mul( mix( 1.0, 0.7, resolve ) ) );

		} ).Else( () => {

			// plume: pinnules angled toward the branch tip on both sides of the rachis, of
			// uneven length; filled in where they can't be resolved
			const a = abs( part );
			const stem = part.greaterThan( 3 );
			const k = t.mul( 150 ).sub( a.mul( 2.4 ) ).add( rnd.mul( 5 ) );
			const lines = fract( k );
			const reach = float( 0.55 ).add( fract( floor( k ).mul( 0.618 ).add( rnd ) ).mul( 0.45 ) );
			const pin = lines.lessThan( 0.32 ).and( a.lessThan( reach ) );
			const resolve = smoothstep( 0.3, 1.0, px.div( s.mul( 0.0045 ) ) );
			const edge = a.lessThan( mix( 1.0, 0.75, resolve ) );
			const solid = stem.or( a.lessThan( 0.07 ) ).or( pin ).or( dither.lessThan( resolve ).and( edge ) );
			mask.assign( select( solid, 1, 0 ) );
			fanAlbedo.assign( mix( c1, c2, a.mul( 0.5 ) ).mul( mix( 0.8, 1.1, rnd ) ).mul( mix( 1.0, 0.8, resolve ) ) );

		} );
		return mask.greaterThan( 0.5 );

	} )();
	fanMat.colorNode = fanAlbedo;
	// light shining through the thin tissue
	fanMat.translucencyNode = ( lightColor ) => {

		const back = saturate( dot( normalWorldGeometry.mul( faceDirection ).negate(), G.sunDir ) );
		return lightColor.mul( fanAlbedo ).mul( back.mul( 0.4 ).add( 0.05 ) );

	};
	fanMat.specularIntensityNode = float( 0.2 );

	return { hard: mat, soft: fanMat };

}
