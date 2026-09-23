import * as THREE from 'three/webgpu';
import {
	Fn, If, float, vec2, vec3, vec4, attribute, uv, positionLocal, normalLocal, normalView, normalViewGeometry,
	normalWorldGeometry, positionWorld, positionViewDirection, cameraPosition, cameraViewMatrix, faceDirection,
	sin, cos, fract, floor, abs, dot, mix, smoothstep, max, min, pow, sqrt, length, normalize,
	saturate, select, fwidth, step, texture, uniform, property, luminance, normalFlat,
} from 'three/tsl';
import { physical } from '../../materials/Materials.js';
import { G } from '../../core/Globals.js';
import { hash12, vnoise, windStrength, windDir3, windPerp3, gustAt, plantDeform, vTrunkY, vTrunkT, UP, lobeScale, uCamPos, lodDither, LOD_BAND } from './VegNodes.js';
import { bayer4 } from '../../materials/LODFade.js';

// Vegetation materials (all built with the shared scene material classes).
//
// TSL note: values shared by several select()/If branches are materialised with toVar()
// before branching. TSL turns statement-producing branches into if/else blocks and would
// otherwise cache a sub-expression inside one branch and read it unassigned in another.
// Also: normalView / normalWorld must not be referenced from emissiveNode (it re-roots the
// lighting normal inside that expression); the interpolated geometry normal is used there.

// Linear-space colour helper (hex in sRGB).
const C = ( hex ) => {

	const c = new THREE.Color( hex );
	return vec3( c.r, c.g, c.b );

};

// Leaf translucency: sunlight transmitted through the leaf when it is lit from behind (relative
// to the viewer). Evaluated in the lighting model with the shadowed light colour, so leaves in
// shadow (behind a hill at sunset, inside the canopy) don't glow.
export const translucency = ( albedo, N, strength, lightColor ) => {

	const V = normalize( cameraPosition.sub( positionWorld ) );
	const L = G.sunDir;
	const back = saturate( dot( N.negate(), L ) );
	const forward = pow( saturate( dot( V.negate(), L ) ), 3 ).mul( 0.7 ).add( 0.3 );
	const tint = albedo.mul( vec3( 1.25, 1.45, 0.55 ) ).add( vec3( 0.012, 0.018, 0 ) );
	return tint.mul( lightColor ).mul( back.mul( forward ).mul( strength ) ).mul( float( 1 ).sub( G.night ) );

};

// Coconut palm bark: irregular leaf-scar rings (uneven spacing, closer below the crown, wavy,
// partial), fine vertical fissures, grey-brown to silver weathering, lichen, dark stains and the
// root mass at the base, the fibrous old frond bases under the crown. Returns the albedo and a
// relief height (m) for the bump. y: height along the stem (m), a: 0..1 around, H: stem height.
const palmBark = ( y, a, H, seed, iv ) => {

	const A = a.mul( 6.2832 );
	const ca = cos( A ), sa = sin( A );
	const yn = y.div( H );
	// rings: phase grows faster toward the crown, jittered per ring band and around the trunk
	const wob = vnoise( vec2( ca.mul( 1.3 ).add( y.mul( 0.35 ) ), sa.mul( 1.3 ).add( seed.mul( 9 ) ) ) ).sub( 0.5 ).mul( 0.7 )
		.add( sin( A.mul( 2 ).add( y.mul( 1.1 ) ).add( seed.mul( 5 ) ) ).mul( 0.1 ) );
	const phase = y.mul( mix( 6.2, 8.2, iv ) ).add( pow( yn, 2.2 ).mul( H ).mul( 5.5 ) ).add( vnoise( vec2( y.mul( 0.8 ), seed.mul( 7.3 ) ) ).mul( 1.8 ) ).add( wob );
	const k = floor( phase );
	const f = phase.sub( k );
	const groove = smoothstep( 0.1, 0.0, f ).add( smoothstep( 0.93, 1.0, f ) );
	const ridge = smoothstep( 0.07, 0.15, f ).mul( smoothstep( 0.45, 0.16, f ) );
	// partial rings: each ring fades out around part of the circumference
	const amp = smoothstep( 0.28, 0.6, vnoise( vec2( ca.mul( 1.8 ).add( k.mul( 3.17 ) ), sa.mul( 1.8 ).add( k.mul( 1.71 ) ).add( seed.mul( 11 ) ) ) ) ).mul( 0.75 ).add( 0.25 );
	// fine vertical fissures (level set of a vertically stretched noise) and bark plates
	const nf = vnoise( vec2( a.mul( 54 ), y.mul( 1.6 ).add( seed.mul( 41 ) ) ) );
	const crack = smoothstep( 0.07, 0.0, abs( nf.sub( 0.5 ) ) ).mul( smoothstep( 0.3, 0.7, vnoise( vec2( a.mul( 13 ), y.mul( 0.5 ).add( seed.mul( 3 ) ) ) ) ).mul( 0.6 ).add( 0.4 ) );
	const plate = vnoise( vec2( a.mul( 24 ), y.mul( 3.5 ).add( seed.mul( 17 ) ) ) );
	const blotch = vnoise( vec2( a.mul( 6 ), y.mul( 0.4 ).add( seed.mul( 13 ) ) ) );
	// grey-brown bark weathering to silver-grey, per palm
	let bark = mix( C( 0x6a6356 ), C( 0xa39e90 ), saturate( blotch.mul( 0.55 ).add( plate.mul( 0.25 ) ).add( iv.sub( 0.5 ).mul( 0.5 ) ).add( yn.mul( 0.15 ) ) ) );
	bark = bark.mul( mix( 1.0, 0.62, groove.mul( amp ) ) ).mul( ridge.mul( amp ).mul( 0.1 ).add( 1 ) ).mul( mix( 1.0, 0.55, crack ) );
	// lichen: pale grey-green crusts and white spots, a little orange; darker rain streaks
	const lic = smoothstep( 0.6, 0.72, vnoise( vec2( a.mul( 9 ), y.mul( 2.1 ).add( seed.mul( 23 ) ) ) ).add( vnoise( vec2( a.mul( 31 ), y.mul( 7 ) ) ).sub( 0.5 ).mul( 0.3 ) ) ).mul( smoothstep( 0.9, 0.2, yn ) );
	const licC = mix( C( 0x8e917f ), C( 0xa9a799 ), plate );
	bark = mix( bark, select( fract( k.mul( 0.37 ).add( seed.mul( 3.1 ) ) ).lessThan( 0.08 ), C( 0x9a7438 ), licC ), lic.mul( 0.45 ) );
	const streak = smoothstep( 0.62, 0.8, vnoise( vec2( a.mul( 16 ), y.mul( 0.12 ).add( seed.mul( 5 ) ) ) ) ).mul( smoothstep( 1.0, 0.6, yn ) );
	bark = bark.mul( float( 1 ).sub( streak.mul( 0.28 ) ) );
	// damp, dark base (splash of sand and soil) and the mass of exposed roots at the ground
	const baseK = smoothstep( 1.4, 0.2, y.add( blotch.sub( 0.5 ).mul( 0.8 ) ) );
	bark = mix( bark, bark.mul( vec3( 0.62, 0.56, 0.48 ) ), baseK );
	const rootN = vnoise( vec2( a.mul( 26 ), y.mul( 2.2 ).add( seed.mul( 19 ) ) ) );
	const roots = smoothstep( 0.42, 0.05, y ).mul( smoothstep( 0.35, 0.6, rootN ) );
	bark = mix( bark, mix( C( 0x3a2e22 ), C( 0x5e4a36 ), rootN ), smoothstep( 0.5, 0.0, y ).mul( 0.85 ) );
	// fibrous old frond bases (boot) under the crown: criss-cross fibre mat, brown
	const boot = smoothstep( H.sub( 1.0 ), H.sub( 0.35 ), y.add( blotch.sub( 0.5 ).mul( 0.3 ) ) );
	const fib = sin( A.mul( 34 ).add( y.mul( 30 ) ) ).mul( sin( A.mul( 34 ).sub( y.mul( 30 ) ) ) ).mul( 0.5 ).add( 0.5 );
	bark = mix( bark, mix( C( 0x4d3722 ), C( 0x8a744c ), fib.mul( 0.6 ).add( plate.mul( 0.4 ) ) ), boot );
	const hd = ridge.mul( amp ).mul( 0.004 ).sub( groove.mul( amp ).mul( 0.007 ) ).sub( crack.mul( 0.0035 ) ).add( plate.mul( 0.0015 ) )
		.add( roots.mul( 0.01 ) ).mul( float( 1 ).sub( boot ) ).add( boot.mul( fib ).mul( 0.004 ) );
	return { bark, hd };

};


// Palms (trunk, coconuts, fronds), young palms, banana plants, ferns -------------------------
//
// aMat = (part, age / stem colour, leaflet length (m), frond seed); uv = (s along, t across)
// parts: 0 stem, 1 coconut frond, 2 fern frond, 3 banana leaf, 5 coconut

export function createPlantLeafMaterial() {

	const mat = physical( { side: THREE.DoubleSide, specularIntensity: 0.7 } );
	mat.name = 'veg-plant-leaf';
	mat.positionNode = plantDeform();

	const aMat = attribute( 'aMat', 'vec4' );
	const part = aMat.x;
	const age = aMat.y;
	const Ll = aMat.z;
	const fseed = aMat.w;
	const seed = attribute( 'iDat', 'vec4' ).w;
	const H = attribute( 'iDat', 'vec4' ).z;
	const isStem = part.lessThan( 0.5 );
	const isNut = part.greaterThan( 4.5 );
	const isLeaf = isStem.not().and( isNut.not() );

	mat.maskNode = Fn( () => {

		const st = uv().toVar();
		const s = st.x;
		const t = st.y;
		const fwS = fwidth( s ).toVar(); // evaluated in uniform control flow, before the branches
		const m = float( 1 ).toVar();

		If( part.greaterThan( 0.5 ).and( part.lessThan( 1.5 ) ), () => {

			// coconut frond: ~58 leaflets per side
			const N = 58;
			const x = s.mul( N );
			const k = floor( x );
			const r1 = hash12( vec2( k, fseed.mul( 91.7 ) ) );
			const r2 = hash12( vec2( k.mul( 1.37 ).add( 3.1 ), fseed.mul( 17.3 ).add( seed.mul( 5 ) ) ) );
			const fx = fract( x ).sub( 0.5 ).sub( r1.sub( 0.5 ).mul( 0.3 ) );
			const tEnd = mix( 0.7, 1.0, r2 ).mul( select( r1.lessThan( 0.05 ), 0.4, 1 ) );
			const tt = t.div( tEnd );
			const hw = pow( max( tt.oneMinus(), 0 ), 0.55 ).mul( 0.23 ).mul( smoothstep( 0, 0.12, tt ).mul( 0.45 ).add( 0.55 ) );
			// sub-pixel leaflets widen instead of aliasing (fronds turn solid in the distance)
			const hwE = max( hw, min( fwS.mul( N * 0.6 ), 0.5 ).mul( select( tt.lessThan( 1 ), 1, 0 ) ) );
			const leaf = abs( fx ).lessThan( hwE ).and( tt.lessThan( 1 ) ).and( s.greaterThan( 0.06 ) );
			const rachis = t.mul( Ll ).lessThan( 0.028 );
			m.assign( select( leaf.or( rachis ), 1, 0 ) );

		} ).ElseIf( part.greaterThan( 1.5 ).and( part.lessThan( 2.5 ) ), () => {

			// fern: rounded pinnae
			const N = 26;
			const x = s.mul( N );
			const k = floor( x );
			const r2 = hash12( vec2( k, fseed.mul( 31.1 ) ) );
			const fx = fract( x ).sub( 0.5 );
			const tt = t.div( mix( 0.8, 1.0, r2 ) );
			const hw = sqrt( max( tt.mul( tt ).oneMinus(), 0 ) ).mul( 0.34 );
			const hwE = max( hw, min( fwS.mul( N * 0.6 ), 0.5 ).mul( select( tt.lessThan( 1 ), 1, 0 ) ) );
			const leaf = abs( fx ).lessThan( hwE ).and( tt.lessThan( 1 ) ).and( s.greaterThan( 0.04 ) );
			const rachis = t.mul( Ll ).lessThan( 0.006 );
			m.assign( select( leaf.or( rachis ), 1, 0 ) );

		} ).ElseIf( part.greaterThan( 2.5 ).and( part.lessThan( 3.5 ) ), () => {

			// banana: full blade, torn along lateral veins, ragged edge
			const x = s.mul( 13 ).add( sin( s.mul( 31 ).add( fseed.mul( 10 ) ) ).mul( 0.35 ) );
			const k = floor( x );
			const r1 = hash12( vec2( k, fseed.mul( 7.7 ) ) );
			const r2 = hash12( vec2( k.add( 0.5 ), fseed.mul( 3.3 ).add( seed ) ) );
			const fx = abs( fract( x ).sub( 0.5 ) );
			// dead leaves are shredded
			const tear = r1.lessThan( mix( 0.7, 0.95, step( 0.8, age ) ) ).and( t.greaterThan( mix( 0.1, 0.7, r2 ).mul( mix( 1, 0.4, step( 0.8, age ) ) ) ) ).and( fx.greaterThan( mix( 0.46, 0.4, r2 ).sub( step( 0.8, age ).mul( 0.12 ) ) ) );
			const edge = t.lessThan( float( 0.985 ).sub( vnoise( vec2( s.mul( 60 ), fseed.mul( 9 ) ) ).mul( 0.07 ) ) );
			m.assign( select( tear.not().and( edge ), 1, 0 ) );

		} );

		return m.greaterThan( 0.5 ).and( lodDither( attribute( 'iPos', 'vec4' ).xyz ) );

	} )();

	const albedo = Fn( () => {

		const st = uv().toVar();
		const s = st.x;
		const t = st.y;
		const col = vec3( 0 ).toVar();
		const iv = hash12( vec2( seed.mul( 37.1 ), 1.7 ) ).toVar();

		If( part.lessThan( 0.5 ), () => {

			// stems: age 0 -> weathered, ringed palm trunk; 1 -> green banana pseudostem
			const y = vTrunkY;
			const a = st.x;
			const PB = palmBark( y, a, H, seed, iv );
			const bark = PB.bark;
			const fiss = vnoise( vec2( a.mul( 46 ), y.mul( 1.1 ).add( seed.mul( 50 ) ) ) );
			const blotch = vnoise( vec2( a.mul( 7 ), y.mul( 0.45 ).add( seed.mul( 13 ) ) ) );
			// banana pseudostem: overlapping sheaths (vertical streaks), dark blotches, dry brown
			// sheath strips peeling off low down (no leaf-scar rings)
			const streakS = vnoise( vec2( a.mul( 24 ), y.mul( 0.35 ).add( seed.mul( 7 ) ) ) );
			let green = mix( C( 0x4e6a2a ), C( 0x6b8438 ), streakS.mul( 0.7 ).add( fiss.mul( 0.3 ) ) );
			green = mix( green, C( 0x3b3322 ), smoothstep( 0.62, 0.82, blotch ).mul( 0.55 ) );
			green = mix( green, mix( C( 0x6e5534 ), C( 0x8f7a52 ), fiss ), smoothstep( 0.55, 0.75, streakS ).mul( smoothstep( 1.1, 0.3, y ) ) );
			col.assign( mix( bark, green, age ) );

		} ).ElseIf( part.lessThan( 1.5 ), () => {

			const g0 = mix( C( 0x4e7220 ), C( 0x6c8c2a ), iv );
			const leafTip = mix( g0, C( 0x37561a ), smoothstep( 0.35, 1.0, t ).mul( 0.6 ) );
			const leafBase = mix( leafTip, C( 0x93a844 ), smoothstep( 0.18, 0.0, t ).mul( 0.5 ) );
			const young = mix( leafBase, C( 0x9db24a ), smoothstep( 0.2, 0.0, age ).mul( 0.35 ) );
			const perLeaf = hash12( vec2( floor( s.mul( 58 ) ), fseed.mul( 13.1 ) ) );
			let c = young.mul( perLeaf.mul( 0.25 ).add( 0.88 ) );
			const dry = smoothstep( 0.6, 0.95, age );
			c = mix( c, mix( C( 0x8a7148 ), C( 0x6b5434 ), perLeaf ), dry );
			const rachis = t.mul( Ll ).lessThan( 0.028 );
			col.assign( select( rachis, mix( C( 0xa29652 ), C( 0x7d6a40 ), dry ), c ) );

		} ).ElseIf( part.lessThan( 2.5 ), () => {

			const c = mix( C( 0x345c20 ), C( 0x55802c ), smoothstep( 0.1, 1.0, s ).mul( 0.6 ).add( iv.mul( 0.4 ) ) );
			col.assign( c.mul( mix( 0.85, 1.1, hash12( vec2( floor( s.mul( 26 ) ), fseed ) ) ) ) );

		} ).ElseIf( part.lessThan( 3.5 ), () => {

			const base = mix( C( 0x4d7c2a ), C( 0x639034 ), iv );
			const vein = sin( s.mul( 260 ) ).mul( 0.5 ).add( 0.5 ).mul( 0.08 );
			let c = base.mul( vein.add( 0.94 ) );
			c = mix( c, C( 0xc9cf86 ), smoothstep( 0.05, 0.0, t ).mul( 0.8 ) ); // midrib
			const dryEdge = smoothstep( 0.8, 1.0, t ).mul( vnoise( vec2( s.mul( 25 ), fseed.mul( 4 ) ) ) );
			// old leaves hang dead: brown, darker where they fold
			const deadC = mix( C( 0x5e4a2c ), C( 0x7d6a42 ), vnoise( vec2( s.mul( 18 ), t.mul( 3 ).add( fseed.mul( 5 ) ) ) ) );
			c = mix( c, deadC, max( dryEdge.mul( 0.8 ), smoothstep( 0.6, 0.95, age ) ) );
			col.assign( c );

		} ).Else( () => {

			// coconuts: green -> yellow -> brown
			col.assign( mix( mix( C( 0x68762a ), C( 0x9c8a34 ), smoothstep( 0.3, 0.7, age ) ), C( 0x5c4122 ), smoothstep( 0.82, 0.95, age ) ) );

		} );

		return col;

	} )();

	mat.colorNode = albedo;

	// stems: leaf-scar ring bump along the trunk axis; leaves: normals bent slightly towards
	// the viewer to soften grazing-angle Fresnel
	mat.normalNode = Fn( () => {

		// palm trunks: bark relief (rings, fissures, roots, fibres) along the trunk axis (finite
		// difference of the bark height, faded with distance); leaves: normals bent towards the viewer
		const y = vTrunkY;
		const a = uv().x;
		const ivN = hash12( vec2( seed.mul( 37.1 ), 1.7 ) );
		const e = 0.004;
		const slope = palmBark( y.add( e ), a, H, seed, ivN ).hd.sub( palmBark( y, a, H, seed, ivN ).hd ).div( e );
		const fadeB = float( 1 ).sub( smoothstep( 7, 22, length( cameraPosition.sub( positionWorld ) ) ) );
		const d = slope.mul( fadeB ).mul( select( isStem, float( 1 ).sub( age ), float( 0 ) ) );
		const Tv = cameraViewMatrix.mul( vec4( vTrunkT, 0 ) ).xyz;
		return normalize( normalView.sub( Tv.mul( d ) ).add( positionViewDirection.mul( select( isLeaf, 0.4, 0 ) ) ) );

	} )();

	// waxy but not glossy: a sharper, stronger sheen mirrored the bright sky near the sun and read as a
	// white film over backlit foliage
	mat.roughnessNode = select( isStem, float( 0.92 ), select( part.lessThan( 1.5 ).or( isNut ), float( 0.62 ), float( 0.7 ) ) );
	mat.metalnessNode = float( 0 );
	mat.specularIntensityNode = select( isStem, float( 0.3 ), float( 0.4 ) );
	mat.translucencyNode = ( lightColor ) => select( isLeaf, translucency( albedo, normalWorldGeometry.mul( faceDirection ), 0.4, lightColor ), vec3( 0 ) );

	return mat;

}

// Tree / shrub foliage (leaf-cluster cards) and tree bark ------------------------------------

// Trees / shrubs / impostors. aVeg = (height fraction, branch flex, flutter weight, phase);
// iDat = (yaw, vertical scale (negative: shrub), plant height (m), seed). With `lobes`, leaf cards
// move towards / onto their lobe centre (aLobe) by the per-instance lobe scale: dropped lobes
// vanish, the others vary in size, so every instance gets its own irregular crown.
const canopyDeform = ( lobes ) => Fn( () => {

	const iPos = attribute( 'iPos', 'vec4' );
	const iDat = attribute( 'iDat', 'vec4' );
	const veg = attribute( 'aVeg', 'vec4' );
	const base = iPos.xyz.toVar();
	const sc = iPos.w.toVar();
	const Hh = iDat.z.toVar();
	const seed = iDat.w.toVar();
	const hf = veg.x;
	const flex = veg.y;
	const flut = veg.z;
	const ph = veg.w;
	const P = positionLocal.toVar();
	const isShrubI = iDat.y.lessThan( 0 ).toVar();
	let keep = float( 1 );
	if ( lobes ) {

		// merged tree + shrub geometry: keep the parts of this instance's plant type
		const isShrubPart = attribute( 'aMat', 'vec4' ).x.greaterThan( 3.5 );
		keep = select( isShrubPart.equal( isShrubI ), float( 1 ), float( 0 ) );
		// leaf cards move towards their lobe centre by the variant's lobe scale (0: dropped lobe)
		const lobe = attribute( 'aLobe', 'vec4' );
		const k = select( lobe.w.greaterThanEqual( 0 ), float( 1 ).sub( lobeScale( seed, lobe.w, isShrubI ) ), float( 0 ) );
		const d = lobe.xyz.mul( k );
		const yaw = iDat.x, cy = cos( yaw ), sy = sin( yaw );
		const sv = abs( iDat.y );
		P.addAssign( vec3( d.x.mul( cy ).add( d.z.mul( sy ) ), d.y.mul( sv ), d.z.mul( cy ).sub( d.x.mul( sy ) ) ).mul( sc ) );

	}

	const N = normalLocal.toVar();
	const w = windStrength.toVar();
	const g = gustAt( base.xz ).toVar();
	const t = G.time;
	const ph0 = seed.mul( 6.2832 ).toVar();
	const h2 = hf.mul( hf );
	const sway = w.mul( w ).mul( 0.009 ).mul( g.mul( 0.8 ).add( 0.3 ) )
		.add( sin( t.mul( 0.9 ).add( ph0 ) ).mul( w ).mul( 0.0045 ).mul( g.add( 0.4 ) ) ).mul( Hh ).mul( h2 );
	const swayP = sin( t.mul( 0.67 ).add( ph0.mul( 1.3 ) ) ).mul( w ).mul( 0.002 ).mul( Hh ).mul( h2 );
	const branch = sin( t.mul( ph.add( 1.7 ) ).add( ph.mul( 20 ) ).add( ph0 ) ).mul( flex ).mul( w ).mul( sc ).mul( 0.07 ).mul( g.add( 0.5 ) )
		.sub( flex.mul( w ).mul( w ).mul( sc ).mul( 0.04 ).mul( g.add( 0.3 ) ) ); // branches sag / stream in strong gusts
	const flutter = sin( t.mul( 9.5 ).add( ph.mul( 50 ) ).add( P.x.mul( 1.9 ) ).add( P.z.mul( 2.3 ) ) )
		.mul( flut ).mul( sc ).mul( w.mul( 0.035 ).add( 0.005 ) );
	const pos = P.add( windDir3.mul( sway ) ).add( windPerp3.mul( swayP ) ).add( UP.mul( branch ) ).add( N.mul( flutter ) );
	// near LOD: trees and shrubs hand over to the impostors at their own distance
	const dCam = length( uCamPos.sub( base ) );
	const nearK = select( dCam.lessThan( select( isShrubI, uCanopyNear.y, uCanopyNear.x ).mul( 1 + LOD_BAND / 2 ) ), float( 1 ), float( 0 ) );
	return base.add( pos.sub( base ).mul( keep.mul( nearK ) ) );

} )();

// near -> impostor switch distance (m) of trees (x) and shrubs (y)
export const uCanopyNear = uniform( new THREE.Vector2( 75, 45 ) ).setName( 'vegCanopyNear' );

// aMat = (part, canopy exposure (ao), colour rand, card rand); parts: 0 bark, 1 tree card, 4 shrub card
// Species per instance (seed): trees 0 dark glossy (bronze new flush), 1 mid green, 2 yellow-green,
// 3 blue-green; shrubs 0 sea grape (round leaves, red veins), 1 croton (variegated), 2 hibiscus
// (flowering). The far impostors use the same palette (canopyLeafColor) and brightness structure.
const pick4 = ( s4, a, b, c, d ) => select( s4.lessThan( 0.5 ), a, select( s4.lessThan( 1.5 ), b, select( s4.lessThan( 2.5 ), c, d ) ) );
const pick3 = ( s3, a, b, c ) => select( s3.lessThan( 0.5 ), a, select( s3.lessThan( 1.5 ), b, c ) );
const treeSpecies = ( seed ) => floor( fract( seed.mul( 5.31 ) ).mul( 4 ) );
// shrubs: 0 sea grape 45 %, 1 croton 15 %, 2 hibiscus 40 %
const shrubSpecies = ( seed ) => {

	const h = fract( seed.mul( 3.17 ) );
	return select( h.lessThan( 0.45 ), float( 0 ), select( h.lessThan( 0.6 ), float( 1 ), float( 2 ) ) );

};

// base leaf colour of an instance (species, per-card random cr, per-instance tint)
export const canopyLeafColor = ( seed, cr, isShrub ) => {

	const spT = treeSpecies( seed ), spS = shrubSpecies( seed );
	// tree species: dark glossy (bronze flush), fresh mid green, yellow-green, blue-green
	// (Caribbean hillside forest: deep, olive and yellow-greens, muted, with a few dry / bronze
	// and flowering crowns)
	const t0 = mix( mix( C( 0x283a1b ), C( 0x364a23 ), cr ), C( 0x5e4a2e ), smoothstep( 0.96, 0.995, cr ).mul( 0.5 ) );
	const t1 = mix( C( 0x34491f ), C( 0x485c27 ), cr );
	const t2 = mix( C( 0x4f5a27 ), C( 0x646a31 ), cr );
	const t3 = mix( C( 0x2a3b2a ), C( 0x3a4a36 ), cr );
	const s0 = mix( C( 0x3f5522 ), C( 0x52662a ), cr );
	const s1 = mix( C( 0x2c421e ), C( 0x44561f ), cr );
	const s2 = mix( C( 0x34521c ), C( 0x466624 ), cr );
	const iv = hash12( vec2( seed.mul( 17.3 ), 4.1 ) );
	const iv2 = hash12( vec2( seed.mul( 5.9 ), 8.3 ) );
	let c = select( isShrub, pick3( spS, s0, s1, s2 ), pick4( spT, t0, t1, t2, t3 ) ).mul( iv.mul( 0.36 ).add( 0.74 ) );
	// a few trees dry / dropping leaves (brown-olive), or flowering (flamboyant, orange-red)
	const dry = step( iv2, 0.035 ).mul( select( isShrub, float( 0 ), float( 1 ) ) );
	c = mix( c, mix( C( 0x5c5234 ), C( 0x6e5a3a ), cr ), dry.mul( 0.5 ) );
	const flower = step( 0.988, iv2 ).mul( select( isShrub, float( 0 ), float( 1 ) ) ).mul( step( 0.5, cr ) );
	c = mix( c, C( 0x8a4a2c ), flower.mul( 0.5 ) );
	return mix( vec3( luminance( c ) ), c, 0.85 );

};

const barkColor = ( n, n2 ) => mix( mix( C( 0x302a22 ), C( 0x5c5549 ), n.mul( 0.6 ).add( n2.mul( 0.4 ) ) ), C( 0x7b7b6a ), smoothstep( 0.64, 0.8, n2 ).mul( 0.3 ) );

// leaf-cluster tile of a card: trees broad / narrow leaves by species, shrubs round / narrow
const leafTile = ( seed, isShrub ) => {

	const spT = treeSpecies( seed ), spS = shrubSpecies( seed );
	return select( isShrub, select( spS.equal( 1 ), float( 3 ), float( 2 ) ), select( fract( spT.mul( 0.5 ) ).greaterThan( 0.25 ), float( 1 ), float( 0 ) ) );

};

// alpha-test threshold compensating the coverage loss of the minified (mipmapped) leaf texture
const coverageThreshold = ( st ) => {

	const lod = max( fwidth( st.x ), fwidth( st.y ) ).mul( 256 ).max( 1e-4 ).log2();
	return mix( float( 0.5 ), float( 0.3 ), saturate( lod.div( 4 ) ) );

};

export function createCanopyMaterial( leafAtlas ) {

	const mat = physical( { side: THREE.DoubleSide, specularIntensity: 0.15 } );
	mat.name = 'veg-canopy';
	mat.positionNode = canopyDeform( true );

	const aMat = attribute( 'aMat', 'vec4' );
	const part = aMat.x;
	const ao = aMat.y;
	const cr = aMat.z;
	const seed = attribute( 'iDat', 'vec4' ).w;
	const isBark = part.lessThan( 0.5 ).or( part.greaterThan( 4.5 ) );
	const isShrub = part.greaterThan( 2.5 );
	const spT = treeSpecies( seed ), spS = shrubSpecies( seed );

	// leaf cluster sampled once in the mask (first in the fragment shader), shared through vars
	// shared between mask and colour: plain properties (a toVar() initialiser would be re-emitted
	// in every branch that reads them)
	const vBright = property( 'float', 'canBright' ), vCell = property( 'float', 'canCell' );
	mat.maskNode = Fn( () => {

		const st = uv().toVar();
		const L = leafAtlas.sample( st, leafTile( seed, isShrub ) );
		vBright.assign( L.y.mul( 1.4 ) );
		vCell.assign( L.z );
		// cross-fade into the impostors (outgoing level of the band around uCanopyNear)
		const iP = attribute( 'iPos', 'vec4' );
		const nearD = select( attribute( 'iDat', 'vec4' ).y.lessThan( 0 ), uCanopyNear.y, uCanopyNear.x );
		const fade = smoothstep( nearD.mul( 1 - LOD_BAND / 2 ), nearD.mul( 1 + LOD_BAND / 2 ), length( uCamPos.sub( iP.xyz ) ) );
		// cards seen edge-on thin out (no sliver lines through the crown)
		const facing = abs( dot( normalFlat, positionViewDirection ) );
		const thr = coverageThreshold( st ).add( float( 1 ).sub( smoothstep( 0.08, 0.35, facing ) ).mul( 0.45 ) );
		return isBark.or( L.x.greaterThan( thr ) ).and( bayer4().greaterThanEqual( fade ) );

	} )();

	const albedo = Fn( () => {

		const cell = vCell;
		let c = canopyLeafColor( seed, cr, isShrub ).mul( vBright );
		// species details: red-veined old leaves (sea grape), variegation (croton), flowers (hibiscus)
		const shrub0 = isShrub.and( spS.lessThan( 0.5 ) ), shrub1 = isShrub.and( spS.equal( 1 ) ), shrub2 = isShrub.and( spS.greaterThan( 1.5 ) );
		c = mix( c, C( 0x7a3a22 ), smoothstep( 0.86, 0.98, cell ).mul( 0.55 ).mul( select( shrub0, float( 1 ), float( 0 ) ) ) );
		const vari = select( fract( cell.mul( 7.3 ) ).greaterThan( 0.5 ), C( 0x9a8a30 ), C( 0x7a3a22 ) );
		c = mix( c, vari, smoothstep( 0.72, 0.9, cell ).mul( 0.55 ).mul( select( shrub1, float( 1 ), float( 0 ) ) ) );
		c = select( shrub2.and( cell.greaterThan( 0.92 ) ), C( 0xb3261e ), c );
		// trees: a few old leaves turning red / yellow before they drop (sea almond)
		const treeK = select( isShrub, float( 0 ), float( 1 ) );
		c = mix( c, mix( C( 0x8a7a3a ), C( 0x7e3e22 ), step( 0.992, fract( cell.mul( 3.7 ) ) ) ), step( 0.984, fract( cell.mul( 3.7 ) ) ).mul( treeK ).mul( 0.6 ) );
		// sunlit outer / upper leaves brighter and a little yellow-green; shaded interior kept for contrast
		const outer = smoothstep( 0.62, 1.0, ao );
		c = mix( c, c.mul( vec3( 1.16, 1.22, 0.92 ) ), outer.mul( 0.7 ) );
		const leaf = c.mul( mix( 0.55, 1.0, ao ) ).toVar();
		// bark: grey-brown with vertical streaks, lichen patches
		const st = uv();
		const n2 = vnoise( vec2( st.x.mul( 6 ), st.y.mul( 0.7 ) ) );
		let bark = barkColor( vnoise( vec2( st.x.mul( 30 ), st.y.mul( 2.5 ).add( seed.mul( 40 ) ) ) ), n2 );
		// moss and epiphytes on the humid lower trunk and the upper sides of the limbs
		const hfB = attribute( 'aVeg', 'vec4' ).x;
		bark = mix( bark, mix( C( 0x2c3a18 ), C( 0x44552a ), n2 ), smoothstep( 0.45, 0.7, vnoise( vec2( st.x.mul( 9 ), st.y.mul( 1.3 ).add( seed.mul( 11 ) ) ) ).add( float( 0.35 ).sub( hfB ).mul( 0.6 ) ) ).mul( 0.7 ) );
		return select( isBark, bark, leaf );

	} )();

	mat.colorNode = albedo;
	// bark: the trunk under the crown sees little of the sky
	mat.aoNode = select( isBark, smoothstep( 0.0, 0.75, attribute( 'aVeg', 'vec4' ).x ).mul( 0.45 ).add( 0.4 ), mix( 0.35, 1.0, ao ) );
	mat.roughnessNode = select( isBark, float( 0.92 ), select( isShrub.and( spS.lessThan( 0.5 ) ).or( isShrub.not().and( spT.lessThan( 0.5 ) ) ), float( 0.65 ), float( 0.82 ) ) );
	mat.metalnessNode = float( 0 );
	// leaves: canopy (spherical) normals, not flipped on back faces, bent towards the viewer
	// so they are never shaded at grazing angles (avoids a white Fresnel sheen when backlit)
	mat.normalNode = Fn( () => {

		const V = positionViewDirection.toVar(); // hoisted: the lighting reads it for every part
		const leafN = normalize( normalViewGeometry.add( V.mul( 0.7 ) ) ).toVar();
		const barkN = normalView.toVar();
		return select( isBark, barkN, leafN );

	} )();
	mat.translucencyNode = ( lightColor ) => select( isBark, vec3( 0 ), translucency( albedo, normalWorldGeometry, 0.5, lightColor ).mul( ao.mul( 0.6 ).add( 0.4 ) ) );

	return mat;

}

// Unlit bake materials for the impostor atlases:
//   albedo: (leaf brightness / 1.4 | bark brightness, leaf flag, card colour random, 1)
//   normal: (plant-local normal * 0.5 + 0.5, exposure)
export function createCanopyBakeMaterials( leafAtlas ) {

	const make = ( which ) => {

		const mat = new THREE.MeshBasicNodeMaterial( { side: THREE.DoubleSide } );
		mat.name = 'veg-impostor-bake-' + which;
		const aMat = attribute( 'aMat', 'vec4' );
		const part = aMat.x;
		const isBark = part.lessThan( 0.5 ).or( part.greaterThan( 4.5 ) );
		const isShrub = part.greaterThan( 2.5 );
		const vBright = property( 'float', 'bkBright' );
		mat.maskNode = Fn( () => {

			const st = uv();
			const L = leafAtlas.sample( st, select( isShrub, float( 2 ), float( 0 ) ) );
			vBright.assign( L.y.mul( 1.4 ) );
			return isBark.or( L.x.greaterThan( 0.5 ) );

		} )();
		if ( which === 'albedo' ) {

			mat.colorNode = Fn( () => {

				const st = uv();
				const bark = vnoise( vec2( st.x.mul( 30 ), st.y.mul( 2.5 ) ) ).mul( 0.6 ).add( vnoise( vec2( st.x.mul( 6 ), st.y.mul( 0.7 ) ) ).mul( 0.4 ) );
				const bright = select( isBark, bark.mul( 0.5 ).add( 0.4 ), vBright.div( 1.4 ) );
				return vec4( bright, select( isBark, float( 0 ), float( 1 ) ), aMat.z, 1 );

			} )();

		} else {

			mat.colorNode = vec4( normalize( attribute( 'normal', 'vec3' ) ).mul( 0.5 ).add( 0.5 ), select( isBark, float( 0.6 ), aMat.y ) );

		}

		return mat;

	};

	return { albedo: make( 'albedo' ), normal: make( 'normal' ) };

}

// Runtime colour of an impostor fragment (same palette as the near canopy)
export const impostorColor = ( { seed, cr, leaf, bright, isGroup1 } ) => {

	const leafC = canopyLeafColor( seed, cr, isGroup1 ).mul( bright.mul( 1.4 ) );
	// limbs and twigs seen through the crown gaps are in the crown's shade: dark and a little
	// green-brown, and the leaves dominate the blend (grey limbs made far crowns read grey-beige)
	const barkC = mix( C( 0x1c1a13 ), C( 0x2e2b20 ), bright.sub( 0.4 ).div( 0.5 ) );
	const c = mix( barkC, leafC, smoothstep( 0.05, 0.55, leaf ) );
	// far crowns keep their green through the haze (a little more saturated than the near canopy)
	return mix( vec3( luminance( c ) ), c, 1.25 ).max( 0 );

};
