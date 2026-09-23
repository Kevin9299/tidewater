import {
	Fn, If, float, vec2, vec3, vec4, attribute, uv, mix, smoothstep, clamp, max, min, abs, sin, fract, floor, step, dot,
	length, select, positionWorld, normalWorldGeometry, cameraPosition, texture, transformNormalToView, saturate, Discard,
} from 'three/tsl';
import { standard } from '../../materials/Materials.js';
import { TerrainLightingModel } from '../Terrain.js';
import { srgb, perturbNormal, triWeights } from '../terrain/TerrainShading.js';
import { staticVelocityMRT } from '../../post/CameraVelocity.js';
import { bayer4 } from '../../materials/LODFade.js';

// Material for the natural debris (one draw call): stones, coconuts and husks, seaweed and
// seagrass wrack, shells, coral rubble, dry palm fronds. Base colours are procedural per kind;
// surface detail comes from the terrain's tileable detail texture (3 samples, UVs chosen per kind:
// triplanar for stones / coral, fibre-aligned mesh UVs for husks and fronds, world xz for draped
// weed). Normals use the same surface-gradient bump as the terrain; contact with the ground adds
// occlusion and a dusting of sand; the key light gets the terrain's heightfield sun shadow.
//
// vdata: x seed, y kind, z kind parameter, w object size (m)
//   0 STONE    z = style + palette (style 0 beach stone, 1 field / wall stone with lichen,
//              2 whitewashed; palette fract: basalt .. grey .. coral limestone .. ochre)
//   1 COCONUT  z < 1.5: whole coconut, age 0 green .. 0.5 brown .. 1 grey; z >= 1.5: husk fibre
//   2 WEED     z = dryness (0 fresh golden sargassum .. 1 black, brittle)
//   3 SHELL    z < 1: clam, hue; z >= 1: turban / cone shell, hue (lathe uvs: u along the profile)
//   4 CORAL    z < 2: branch rubble, bleach 0..1; z >= 2: coral head, bleach = z - 2
//   5 FROND    z = dryness
//   6 DRIFT    driftwood: z = bark remnants (0 fully bleached .. 1 patchy bark); uv: u along the
//              grain (m), v around (m)
// tint multiplies the albedo (baked contact occlusion, colour variation).
//
// Small items dissolve with distance (Bayer screen-door over ~ 450-650 x their size): no popping and
// no shimmering specks far away.

export const aTint = attribute( 'tint', 'vec3' );
export const aData = attribute( 'vdata', 'vec4' );

const hashS = ( s, k ) => fract( sin( s.mul( 91.345 ).add( k ) ).mul( 47453.5453 ) );

// Stone surface shared with the pebble field. T: triplanar detail sample (r facets, g soil /
// stones, b grains / pits, a fbm); pal 0..1; style 0 / 1 / 2; returns { albedo, rough, hd }
export function stoneSurface( { T, N, pal, style, seed, tile } ) {

	const basalt = srgb( 0.15, 0.145, 0.14 ), grey = srgb( 0.4, 0.38, 0.35 ), lime = srgb( 0.76, 0.72, 0.63 ), ochre = srgb( 0.55, 0.4, 0.28 );
	let col = mix( basalt, grey, smoothstep( 0.12, 0.42, pal ) );
	col = mix( col, lime, smoothstep( 0.5, 0.72, pal ) );
	col = mix( col, ochre, smoothstep( 0.88, 0.97, pal ) );
	// per stone tone, speckles and pits, soft blotches
	col = col.mul( hashS( seed, 1.7 ).mul( 0.3 ).add( 0.85 ) );
	col = col.mul( T.b.sub( 0.5 ).mul( 0.5 ).add( 1 ) ).mul( T.a.sub( 0.5 ).mul( 0.6 ).add( 1 ) );
	const pits = smoothstep( 0.62, 0.85, T.b ).mul( smoothstep( 0.45, 0.8, pal ) );
	col = mix( col, col.mul( 0.55 ), pits.mul( 0.6 ) );
	// quartz veins / bands in a few of the dark stones
	const vein = smoothstep( 0.03, 0.0, abs( fract( T.a.mul( 7 ).add( seed.mul( 3 ) ) ).sub( 0.5 ) ) ).mul( step( hashS( seed, 5.1 ), 0.25 ) ).mul( float( 1 ).sub( smoothstep( 0.3, 0.5, pal ) ) );
	col = mix( col, srgb( 0.7, 0.68, 0.62 ), vein.mul( 0.7 ) );
	let rough = float( 0.74 ).add( T.b.mul( 0.12 ) ).sub( smoothstep( 0.0, 0.3, pal ).mul( 0.0 ) );
	// field / wall stones: lichen crusts on the upper faces, soil at the bottom
	const s1 = step( 0.5, style ).mul( step( style, 1.5 ) );
	const lichen = smoothstep( 0.55, 0.68, T.a.add( N.y.mul( 0.18 ) ).add( hashS( seed, 2.3 ).mul( 0.15 ) ) ).mul( s1 );
	const lichenCol = mix( srgb( 0.62, 0.64, 0.55 ), srgb( 0.72, 0.52, 0.2 ), step( 0.72, T.g ) );
	col = mix( col, lichenCol, lichen.mul( 0.75 ) );
	col = mix( col, col.mul( 0.62 ), smoothstep( 0.2, - 0.6, N.y ).mul( s1 ) );
	// whitewashed stones: lime wash, worn through on edges and blotches
	const s2 = step( 1.5, style );
	const wash = smoothstep( 0.5, 0.58, T.a.add( T.r.sub( 0.45 ).mul( 0.4 ) ).add( hashS( seed, 6.6 ).mul( 0.12 ) ) ).mul( s2 );
	col = mix( col, srgb( 0.8, 0.79, 0.75 ).mul( T.b.mul( 0.14 ).add( 0.86 ) ).mul( T.a.sub( 0.5 ).mul( 0.3 ).add( 1 ) ), wash );
	rough = mix( rough, float( 0.93 ), wash );
	const hd = T.r.mul( 0.6 ).add( T.b.mul( 0.3 ) ).mul( tile ).mul( mix( 0.06, 0.02, wash ) );
	return { albedo: col, rough, hd };

}

export function createNatureMaterial( gpu, { sunShadow = true } = {} ) {

	const det = gpu.detailTexture;
	const mat = standard( { roughness: 0.8, metalness: 0 } );
	mat.name = 'DebrisNature';
	mat.mrtNode = staticVelocityMRT;
	mat.underwaterLighting = 'lite';
	if ( sunShadow ) mat.setupLightingModel = () => new TerrainLightingModel( mat, gpu.sunShadowAt( positionWorld ) );

	const seed = aData.x, kind = aData.y, prm = aData.z, size = aData.w;

	// ---- distance fade: small items dissolve (Bayer screen-door, see LODFade) far away
	const fadeEnd = clamp( size.mul( 650 ), 22, 450 );

	const outRough = float( 0.8 ).toVar( 'natRough' );
	const outN = vec3( 0, 1, 0 ).toVar( 'natN' );
	const outAO = float( 1 ).toVar( 'natAO' );

	const albedo = Fn( () => {

		const p = positionWorld;
		Discard( bayer4().greaterThanEqual( float( 1 ).sub( smoothstep( fadeEnd.mul( 0.7 ), fadeEnd, length( p.sub( cameraPosition ) ) ) ) ) );
		const N = normalWorldGeometry.toVar();
		const uv0 = uv();
		const isStone = kind.lessThan( 0.5 );
		const isCoral = kind.greaterThan( 3.5 ).and( kind.lessThan( 4.5 ) );
		const isWeed = kind.greaterThan( 1.5 ).and( kind.lessThan( 2.5 ) );
		const isShell = kind.greaterThan( 2.5 ).and( kind.lessThan( 3.5 ) );
		const isFrond = kind.greaterThan( 4.5 ).and( kind.lessThan( 5.5 ) );
		const isDrift = kind.greaterThan( 5.5 );
		const triK = isStone.or( isCoral );

		// ---- texture coordinates per kind (all three samples in uniform control flow)
		const tile = select( isCoral, float( 0.07 ), clamp( size.mul( 1.1 ), 0.07, 0.9 ) );
		const so = vec2( hashS( seed, 3.1 ), hashS( seed, 7.9 ) );
		// fibre-aligned (coconut / frond), world xz (weed), shell uvs
		const fibA = select( isDrift, vec2( uv0.x.mul( 0.3 ), uv0.y.mul( 5.5 ) ), select( isFrond, vec2( uv0.x.mul( 1.1 ), uv0.y.mul( 45 ) ), vec2( uv0.x.mul( 3 ), uv0.y.mul( 26 ) ) ) );
		const uvA0 = select( isWeed, p.xz.div( 0.42 ), select( isShell, uv0.mul( vec2( 0.5, 0.16 ) ), fibA ) );
		const uvB0 = select( isWeed, p.xz.div( 0.11 ).add( 0.3 ), select( isDrift, vec2( uv0.x.mul( 1.4 ), uv0.y.mul( 21 ) ), uv0.mul( 1.3 ) ) );
		const uvC0 = select( isWeed, p.xz.div( 1.4 ), select( isFrond, vec2( uv0.x.mul( 5 ), uv0.y.mul( 80 ) ), select( isDrift, vec2( uv0.x.mul( 0.09 ), uv0.y.mul( 0.9 ) ), uv0.mul( 8 ) ) ) );
		const uvA = select( triK, p.zy.div( tile ), uvA0.add( so ) );
		const uvB = select( triK, p.xz.div( tile ).add( 0.37 ), uvB0.add( so.yx ) );
		const uvC = select( triK, p.xy.div( tile ).add( 0.71 ), uvC0.add( so.mul( 2 ) ) );
		const A = texture( det, uvA ), B = texture( det, uvB ), Cc = texture( det, uvC );
		const w = triWeights( N );
		const T3 = A.mul( w.x ).add( B.mul( w.y ) ).add( Cc.mul( w.z ) );

		const col = vec3( 0.5 ).toVar();
		const rough = float( 0.8 ).toVar();
		const hd = float( 0 ).toVar();

		If( isStone, () => {

			const S = stoneSurface( { T: T3, N, pal: fract( prm ), style: floor( prm ), seed, tile } );
			col.assign( S.albedo );
			rough.assign( S.rough );
			hd.assign( S.hd );

		} ).ElseIf( kind.lessThan( 1.5 ), () => {

			// coconut: fibrous husk; green -> brown -> grey with age; split husk pieces show the
			// paler, coarser fibre
			const age = min( prm, 1 );
			const inner = step( 1.5, prm );
			let c = mix( srgb( 0.4, 0.42, 0.12 ), srgb( 0.37, 0.23, 0.11 ), smoothstep( 0.18, 0.55, age ) );
			c = mix( c, srgb( 0.36, 0.32, 0.27 ), smoothstep( 0.7, 0.95, age ) );
			c = mix( c, srgb( 0.52, 0.37, 0.21 ), inner );
			const fib = A.a.mul( 0.7 ).add( Cc.b.mul( 0.3 ) );
			c = c.mul( fib.sub( 0.5 ).mul( mix( 0.9, 1.4, inner ) ).add( 1 ) );
			c = mix( c, c.mul( 0.62 ), smoothstep( 0.55, 0.78, B.g ).mul( 0.7 ) );
			// the three germination pores / calyx at the stem end: dark
			col.assign( c );
			rough.assign( mix( float( 0.62 ), float( 0.86 ), smoothstep( 0.2, 0.6, age ).max( inner ) ).add( fib.mul( 0.08 ) ) );
			hd.assign( fib.mul( 0.004 ).add( Cc.b.mul( 0.0015 ) ).mul( inner.add( 1 ) ) );

		} ).ElseIf( isWeed, () => {

			// sargassum / seagrass wrack: golden and olive when fresh (wet sheen, air bladders),
			// drying to brittle black with bleached tips
			const dry = prm;
			let c = mix( srgb( 0.46, 0.3, 0.08 ), srgb( 0.26, 0.22, 0.08 ), smoothstep( 0.35, 0.7, Cc.a ) );
			c = mix( c, srgb( 0.12, 0.08, 0.04 ), smoothstep( 0.5, 0.7, A.a.add( Cc.a.sub( 0.5 ).mul( 0.6 ) ) ).mul( 0.6 ) );
			c = mix( c, srgb( 0.075, 0.06, 0.045 ), smoothstep( 0.25, 0.75, dry ) );
			c = mix( c, srgb( 0.42, 0.38, 0.29 ), smoothstep( 0.62, 0.78, Cc.a.add( A.g.mul( 0.2 ) ) ).mul( smoothstep( 0.5, 1, dry ) ).mul( 0.55 ) );
			const leaf = A.g;
			c = c.mul( leaf.sub( 0.4 ).mul( 0.8 ).add( 1 ) );
			const bladder = smoothstep( 0.72, 0.82, B.b ).mul( float( 1 ).sub( dry ) );
			c = mix( c, srgb( 0.62, 0.5, 0.16 ), bladder.mul( 0.6 ) );
			col.assign( c );
			rough.assign( mix( float( 0.55 ), float( 0.88 ), smoothstep( 0.2, 0.7, dry ) ).sub( bladder.mul( 0.12 ) ) );
			hd.assign( leaf.mul( 0.012 ).add( B.b.mul( 0.004 ) ).add( bladder.mul( 0.004 ) ).add( Cc.a.mul( 0.01 ) ) );

		} ).ElseIf( isShell, () => {

			// shells: ribbed cups with growth bands; turban shells with spiral bands
			const turban = step( 1, prm );
			const hue = fract( prm );
			let base = mix( srgb( 0.92, 0.9, 0.84 ), srgb( 0.9, 0.8, 0.62 ), smoothstep( 0.3, 0.5, hue ) );
			base = mix( base, srgb( 0.9, 0.64, 0.6 ), smoothstep( 0.62, 0.72, hue ) );
			base = mix( base, srgb( 0.8, 0.52, 0.3 ), smoothstep( 0.85, 0.95, hue ) );
			const dark = mix( srgb( 0.5, 0.32, 0.2 ), srgb( 0.35, 0.3, 0.32 ), step( 0.5, hashS( seed, 4.4 ) ) );
			const ribs = sin( uv0.y.mul( 19 ) ).mul( 0.5 ).add( 0.5 );
			const growth = sin( uv0.x.mul( 38 ).add( A.a.mul( 6 ) ) ).mul( 0.5 ).add( 0.5 );
			const spiral = sin( uv0.x.mul( 22 ).add( uv0.y.mul( 1.0 ) ).add( A.a.mul( 4 ) ) ).mul( 0.5 ).add( 0.5 );
			const bandK = mix( smoothstep( 0.6, 0.9, growth ).mul( 0.35 ).mul( step( 0.4, hashS( seed, 8.8 ) ) ), smoothstep( 0.45, 0.75, spiral ).mul( 0.75 ), turban );
			let c = mix( base, dark, bandK );
			c = c.mul( ribs.mul( 0.12 ).add( 0.92 ) ).mul( B.a.sub( 0.5 ).mul( 0.2 ).add( 1 ) );
			col.assign( c );
			rough.assign( float( 0.42 ).add( A.b.mul( 0.2 ) ) );
			hd.assign( mix( ribs.mul( 0.0012 ), spiral.mul( 0.0015 ), turban ) );

		} ).ElseIf( isCoral, () => {

			// bleached coral rubble: porous (polyp cups), stained grey / algae in the older pieces;
			// coral heads: meandering grooves
			const head = step( 2, prm );
			const bleach = fract( prm ).max( step( 0.999, prm ) );
			const pit = smoothstep( 0.55, 0.82, T3.b );
			let c = mix( srgb( 0.5, 0.48, 0.42 ), srgb( 0.88, 0.86, 0.8 ), bleach );
			c = mix( c, srgb( 0.4, 0.42, 0.3 ), smoothstep( 0.6, 0.75, T3.a ).mul( float( 1 ).sub( bleach ) ).mul( 0.6 ) );
			const groove = float( 1 ).sub( smoothstep( 0.2, 0.34, T3.r ) ).mul( head );
			c = c.mul( float( 1 ).sub( pit.mul( 0.35 ) ) ).mul( float( 1 ).sub( groove.mul( 0.4 ) ) );
			col.assign( c );
			rough.assign( 0.88 );
			hd.assign( pit.mul( - 0.0012 ).sub( groove.mul( 0.004 ) ).add( T3.a.mul( 0.002 ) ) );

		} ).ElseIf( isDrift, () => {

			// driftwood: sun-bleached silver-grey with warmer, less weathered patches; open grain
			// (dark wavy lines along the wood), deep checks, flaking bark remnants in the grooves
			const bark = prm;
			const lines = smoothstep( 0.06, 0.0, abs( fract( A.a.mul( 3.0 ).add( Cc.a ) ).sub( 0.5 ) ) );
			const crack = smoothstep( 0.03, 0.0, abs( B.a.sub( 0.5 ) ) ).mul( smoothstep( 0.35, 0.65, A.r ) );
			let c = mix( srgb( 0.63, 0.61, 0.57 ), srgb( 0.83, 0.81, 0.77 ), smoothstep( 0.3, 0.7, Cc.a.add( A.a.sub( 0.5 ).mul( 0.5 ) ) ) );
			c = mix( c, srgb( 0.66, 0.57, 0.46 ), smoothstep( 0.6, 0.8, Cc.g ).mul( 0.45 ) );
			c = c.mul( B.a.sub( 0.5 ).mul( 0.5 ).add( 1 ) ).mul( float( 1 ).sub( lines.mul( 0.45 ) ) ).mul( float( 1 ).sub( crack.mul( 0.75 ) ) );
			const barkM = smoothstep( 0.68, 0.72, Cc.g.add( B.g.sub( 0.5 ).mul( 0.25 ) ) ).mul( step( 0.5, bark ) );
			c = mix( c, mix( srgb( 0.14, 0.1, 0.07 ), srgb( 0.27, 0.19, 0.12 ), A.a ), barkM );
			col.assign( c );
			rough.assign( float( 0.8 ).add( crack.mul( 0.12 ) ).sub( barkM.mul( 0.08 ) ) );
			hd.assign( A.a.mul( 0.004 ).add( B.a.mul( 0.0015 ) ).sub( lines.mul( 0.0015 ) ).sub( crack.mul( 0.004 ) ).add( barkM.mul( 0.0025 ) ) );

		} ).Else( () => {

			// dry palm frond: parallel veins, olive-yellow drying to straw, brown and grey
			const dry = prm;
			let c = mix( srgb( 0.44, 0.42, 0.17 ), srgb( 0.6, 0.5, 0.3 ), smoothstep( 0.1, 0.45, dry ) );
			c = mix( c, srgb( 0.33, 0.24, 0.15 ), smoothstep( 0.45, 0.8, dry ).mul( B.g.mul( 0.6 ).add( 0.4 ) ) );
			c = mix( c, srgb( 0.46, 0.42, 0.36 ), smoothstep( 0.8, 1.0, dry ).mul( Cc.a ) );
			const veins = A.a.mul( 0.6 ).add( Cc.a.mul( 0.4 ) );
			c = c.mul( veins.sub( 0.5 ).mul( 0.7 ).add( 1 ) );
			col.assign( c );
			rough.assign( float( 0.66 ).add( dry.mul( 0.2 ) ) );
			hd.assign( veins.mul( 0.0015 ) );

		} );

		// ---- ground contact: occlusion and a dusting of sand on the lower parts; wet near the sea
		const ground = gpu.heightAt( p.xz );
		const above = p.y.sub( ground );
		const contact = float( 1 ).sub( smoothstep( 0.0, max( size.mul( 0.32 ), 0.012 ), above ) ).toVar();
		const sandy = smoothstep( 0.6, 1.6, ground ).mul( float( 1 ).sub( isWeed.select( 1, 0 ) ) );
		const sandDust = contact.mul( sandy ).mul( smoothstep( 0.4, 0.7, T3.a.add( contact.mul( 0.3 ) ) ) );
		col.assign( mix( col.mul( aTint ), srgb( 0.78, 0.7, 0.56 ), sandDust.mul( 0.55 ) ) );
		const wet = float( 1 ).sub( smoothstep( 0.35, 0.9, p.y ) );
		col.assign( col.mul( float( 1 ).sub( wet.mul( 0.4 ) ) ) );
		outRough.assign( clamp( mix( rough, float( 0.28 ), wet.mul( 0.8 ) ).add( sandDust.mul( 0.1 ) ), 0.05, 1 ) );
		outAO.assign( float( 1 ).sub( contact.mul( 0.45 ) ) );
		hd.assign( hd.add( sandDust.mul( 0.0015 ) ) );
		outN.assign( perturbNormal( N, hd, 1.0 ) );
		return vec4( col, 1 );

	} );

	mat.colorNode = albedo();
	mat.roughnessNode = outRough;
	mat.normalNode = transformNormalToView( outN );
	mat.aoNode = saturate( outAO );
	return mat;

}
