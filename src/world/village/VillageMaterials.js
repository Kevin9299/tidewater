import * as THREE from 'three/webgpu';
import {
	attribute, uv, vec2, vec3, float, mix, smoothstep, step, fract, floor, abs, sin, cos, clamp, max, min,
	normalize, length, sqrt, dot, atan, log2, texture, positionWorld, positionLocal,
	normalMap, fwidth, select, transformNormalToView, faceDirection, screenCoordinate,
} from 'three/tsl';
import { standard } from '../../materials/Materials.js';
import { G } from '../../core/Globals.js';
import { VillageTextures } from './TextureBaker.js';

// Shared PBR materials for the village, pier and props.
//
// Surface detail comes from tileable texture sets baked on the GPU at startup (see
// TextureBaker.js): albedo / mask maps plus RG normal, B roughness, A ambient occlusion.
// UVs are in metres (GeoBuilder), so texel density is constant (256-1024 px/m).
// Per-vertex attributes keep every piece unique:
//   tint  (vec3) base color (paint color, wood tone, rope color ...)
//   vdata (vec4) material specific parameters (documented per material below)
// Normal layers are blended in slope (derivative) space; a second, higher frequency detail
// normal fades in near the camera and a low frequency macro layer hides tiling.
// Each material samples at most 4 textures (plus three.js' own DFG LUT for PBR lighting).
// Materials: wood (+ glass, pattern 9), roofMetal, thatch, hard (+ rope), stone, fabric.

const aTint = attribute( 'tint', 'vec3' );
const aData = attribute( 'vdata', 'vec4' );

const TAU = Math.PI * 2;

const hash21 = ( x, y ) => fract( sin( float( x ).mul( 12.9898 ).add( float( y ).mul( 78.233 ) ) ).mul( 43758.5453 ) );
const n01 = ( n ) => n.mul( 0.5 ).add( 0.5 );
const packN = ( n ) => normalize( n ).mul( 0.5 ).add( 0.5 );
// baked normal (RG = xy of a unit normal) -> surface slope ( -dh/du, -dh/dv )
const slopeOf = ( t ) => {

	const xy = t.xy.mul( 2.0 ).sub( 1.0 );
	return xy.div( sqrt( max( float( 1 ).sub( dot( xy, xy ) ), 0.04 ) ) );

};

const normalFromSlope = ( s ) => normalMap( packN( vec3( s, 1.0 ) ) );
const band01 = ( p, k ) => step( k - 0.5, p ).mul( step( p, k + 0.5 ) );

// Tidal zone: algae / barnacles / wet darkening driven by world height above sea level.
function tidal( grimeTex, col, rough ) {

	const pw = positionWorld;
	const q = vec2( pw.x.add( pw.z.mul( 0.7 ) ), pw.y );
	const Gt = texture( grimeTex, q.mul( vec2( 0.6, 1.2 ) ) );
	const y = pw.y.add( Gt.a.sub( 0.5 ).mul( 0.3 ) );
	const wet = float( 1 ).sub( smoothstep( 0.18, 0.5, y ) );
	const damp = float( 1 ).sub( smoothstep( 0.45, 1.05, y ) ).mul( 0.35 );
	const algae = mix( vec3( 0.028, 0.042, 0.02 ), vec3( 0.11, 0.105, 0.05 ), Gt.a.mul( 0.6 ).add( Gt.r.mul( 0.4 ) ) );
	const Gb = texture( grimeTex, q.mul( 3.1 ) );
	const barn = Gb.b.mul( smoothstep( - 1.6, - 0.35, y ) ).mul( float( 1 ).sub( smoothstep( 0.15, 0.4, y ) ) );
	let c = col.mul( float( 1 ).sub( damp ) );
	c = mix( c, algae, wet );
	c = mix( c, vec3( 0.52, 0.5, 0.44 ), barn.mul( 0.9 ) );
	let r = mix( rough, 0.3, wet.mul( 0.9 ) );
	r = mix( r, 0.55, damp );
	r = mix( r, 0.9, barn );
	return { col: c, rough: r, wet };

}

// ---------------------------------------------------------------------------
// WOOD: raw and painted timber.
// vdata: x seed, y paint (0 raw, 0.3 heavily worn .. 1 fresh), z pattern, w weathering (0 fresh .. 1 silver)
// patterns: 0 plain, 1 lap siding, 2 board & batten, 3 vertical tongue & groove,
//           4 louvers, 5 planks along u (staves, clinker), 6 horizontal planks (flush),
//           7 nailed deck plank (nail heads + rust stains every 0.8 m),
//           9 GLASS (window panes / lantern glass): vdata = seed, kind (0 window, 1 lantern), 9, lit
// tint: paint color when painted, otherwise a wood tone multiplier.
// uv: metres, u along the grain; end-grain faces are flagged with u + 1000, post tops with u + 2000.

export function createWoodMaterial( T, trigger ) {

	const m = standard( { roughness: 0.85, metalness: 0 } );
	m.name = 'VillageWood';

	const seed = aData.x;
	const paint = aData.y;
	const pattern = aData.z;
	const weather = aData.w;
	const uv0 = uv();
	const isCap = step( 1500.0, uv0.x );
	const isEnd = step( 500.0, uv0.x ).mul( float( 1 ).sub( isCap ) );
	const endAny = max( isCap, isEnd );
	const uvS = uv0.sub( vec2( isCap.mul( 2000.0 ).add( isEnd.mul( 1000.0 ) ), 0.0 ) );
	const fw = fwidth( uvS );
	const px = max( fw.x, fw.y );
	const near = float( 1 ).sub( smoothstep( 0.0012, 0.005, px ) );

	const mLap = band01( pattern, 1 );
	const mBB = band01( pattern, 2 );
	const mTG = band01( pattern, 3 );
	const mLv = band01( pattern, 4 );
	const mPk = band01( pattern, 5 );
	const mHz = band01( pattern, 6 );
	const mNail = step( 6.5, pattern );
	const isVert = mBB.add( mTG );
	const g = mix( uvS, uvS.yx, isVert ); // g.x along the grain

	// every board of siding / planking gets its own piece of the wood texture
	const boardIdx = floor( uvS.y.div( 0.2 ) ).mul( mLap ).add( floor( uvS.y.div( 0.16 ) ).mul( mHz ) )
		.add( floor( uvS.x.div( 0.42 ) ).mul( mBB ) ).add( floor( uvS.x.div( 0.14 ) ).mul( mTG ) ).add( floor( uvS.y.div( 0.105 ) ).mul( mPk ) );
	const pieceSeed = seed.mul( 131.0 ).add( boardIdx.mul( 7.13 ) );
	const off = vec2( hash21( pieceSeed, 1.7 ), hash21( pieceSeed, 9.2 ) );
	const tuv = g.mul( vec2( 0.5, 1.0 ) ).add( off );
	const A = texture( T.woodA, tuv );
	const N = texture( T.woodN, tuv );
	const D = texture( T.woodN, tuv.mul( vec2( 3.0, 4.0 ) ).add( vec2( 0.37, 0.61 ) ) );

	// grime / salt / macro variation in mesh space
	const Gm = texture( T.grime, uvS.mul( 0.5 ).add( off.yx.mul( 0.5 ) ) );
	const macro = texture( T.grime, uvS.mul( 0.07 ).add( vec2( seed.mul( 0.37 ), seed.mul( 0.71 ) ) ) ).a;
	const hasPaint = step( 0.001, paint );

	// end grain: polar remap of the side-grain texture -> growth rings become arcs (plank ends) or circles (post tops)
	const pith = mix(
		vec2( hash21( seed, 5.3 ).mul( 0.4 ).sub( 0.6 ), hash21( seed, 6.1 ).mul( 0.3 ).add( 0.25 ) ),
		vec2( hash21( seed, 3.1 ).sub( 0.5 ).mul( 0.02 ), hash21( seed, 4.7 ).sub( 0.5 ).mul( 0.02 ) ),
		isCap
	);
	const ev = uvS.sub( pith );
	const er = length( ev );
	const eUV = vec2( atan( ev.y, ev.x ).div( TAU ).mul( 3.0 ).add( off.x ), er.add( off.y ) );
	const eLod = log2( max( fwidth( er ).mul( 1024.0 ), 1.0 ) );
	const E = texture( T.woodA, eUV ).level( eLod );

	// raw timber colour: the baked map is fully weathered silver; less weathered pieces shift to warm brown
	const wW = clamp( weather.add( macro.sub( 0.5 ).mul( 0.35 ) ), 0.0, 1.0 );
	// wood exposed under chipped paint was protected until recently: lighter and warmer
	const wWx = wW.mul( mix( 1.0, 0.6, hasPaint ) );
	const sideCol = mix( A.rgb.mul( vec3( 1.42, 1.0, 0.6 ) ).mul( 1.08 ), A.rgb, smoothstep( 0.12, 0.6, wWx ) );
	const endCol = mix( E.rgb.mul( vec3( 1.3, 1.0, 0.7 ) ), E.rgb, wW ).mul( 0.66 );
	const woodTone = mix( aTint, vec3( 1.2 ), hasPaint );
	const wood = mix( sideCol, endCol, endAny ).mul( woodTone ).mul( macro.mul( 0.2 ).add( 0.9 ) );

	// paint film: chips follow the baked chip field; the film has thickness at the chip edges
	// chip field quantiles: 5% 0.26, 10% 0.31, 20% 0.35, 30% 0.39 -> paint 0.7 leaves ~8% bare wood,
	// 0.6 ~14%, 0.5 ~22%; lap drip edges and the splash zone near wall bottoms wear faster
	const chip = A.a;
	const lapEdge = float( 1 ).sub( smoothstep( 0.0, 0.18, fract( uvS.y.div( 0.2 ) ) ) ).mul( mLap );
	const lowWall = float( 1 ).sub( smoothstep( 0.0, 0.7, uvS.y ) ).mul( mLap.add( mBB ).add( mTG ) );
	const thr = clamp( float( 0.395 ).sub( paint.sub( 0.4 ).mul( 0.32 ) ).add( lapEdge.mul( 0.05 ) ).add( lowWall.mul( 0.05 ) ), 0.15, 0.55 );
	const painted = smoothstep( thr.sub( 0.012 ), thr.add( 0.012 ), chip ).mul( hasPaint ).mul( float( 1 ).sub( endAny ) );
	const e = 1.0 / 1024.0;
	const cX = texture( T.woodA, tuv.add( vec2( e, 0.0 ) ) ).a.sub( texture( T.woodA, tuv.sub( vec2( e, 0.0 ) ) ).a );
	const cY = texture( T.woodA, tuv.add( vec2( 0.0, e ) ) ).a.sub( texture( T.woodA, tuv.sub( vec2( 0.0, e ) ) ).a );
	const edge = float( 1 ).sub( smoothstep( 0.0, 0.025, abs( chip.sub( thr ) ) ) ).mul( hasPaint ).mul( near );
	const edgeSlope = vec2( cX, cY ).mul( edge ).mul( - 9.0 );
	const P = texture( T.paintN, g.mul( vec2( 1.0, 2.0 ) ).add( off.mul( 3.7 ) ) );

	const fade = clamp( Gm.a.mul( 0.55 ).add( wW.mul( 0.25 ) ).add( macro.mul( 0.2 ) ), 0.0, 1.0 );
	const chalk = aTint.mul( 0.64 ).add( vec3( 0.27, 0.265, 0.25 ) );
	const paintCol = mix( aTint, chalk, fade.mul( 0.5 ) ).mul( P.b.sub( 0.5 ).mul( 0.16 ).add( 1.0 ) );

	// siding / plank patterns (shading and slope in mesh space)
	const fvLap = fract( uvS.y.div( 0.2 ) );
	const lapFade = float( 1 ).sub( smoothstep( 0.02, 0.07, fw.y ) );
	const lapShade = mix( 0.92, mix( 0.45, 1.03, smoothstep( 0.0, 0.09, fvLap ) ).sub( fvLap.mul( 0.07 ) ), lapFade );
	const lapSy = mix( 0.0, select( fvLap.lessThan( 0.07 ), float( - 2.4 ), float( - 0.14 ) ), lapFade );

	const fuBB = fract( uvS.x.div( 0.42 ) );
	const bbFade = float( 1 ).sub( smoothstep( 0.03, 0.09, fw.x ) );
	const inBatten = step( fuBB, 0.12 );
	const bbShade = mix( 0.95, mix( 1.0, 0.7, float( 1 ).sub( smoothstep( 0.12, 0.2, fuBB ) ).mul( float( 1 ).sub( inBatten ) ) ).mul( inBatten.mul( 0.06 ).add( 1.0 ) ), bbFade );
	const bbSx = select( fuBB.lessThan( 0.015 ), float( - 2.0 ), select( abs( fuBB.sub( 0.112 ) ).lessThan( 0.012 ), float( 2.0 ), float( 0.0 ) ) ).mul( bbFade );

	const fuTG = fract( uvS.x.div( 0.14 ) );
	const tgFade = float( 1 ).sub( smoothstep( 0.015, 0.05, fw.x ) );
	const tgShade = mix( 0.96, mix( 0.6, 1.0, smoothstep( 0.0, 0.06, fuTG ) ), tgFade );
	const tgSx = select( fuTG.lessThan( 0.03 ), float( - 1.0 ), select( fuTG.lessThan( 0.06 ), float( 1.0 ), float( 0.0 ) ) ).mul( tgFade );

	const fvLv = fract( uvS.y.div( 0.07 ) );
	const lvFade = float( 1 ).sub( smoothstep( 0.008, 0.03, fw.y ) );
	const lvShade = mix( 0.8, mix( 0.42, 1.05, smoothstep( 0.0, 0.8, fvLv ) ), lvFade );
	const lvSy = mix( 0.0, select( fvLv.greaterThan( 0.85 ), float( 1.2 ), float( - 1.0 ) ), lvFade );

	const fvPk = fract( uvS.y.div( 0.105 ) );
	const pkFade = float( 1 ).sub( smoothstep( 0.012, 0.04, fw.y ) );
	const pkShade = mix( 0.95, mix( 0.58, 1.0, smoothstep( 0.0, 0.07, fvPk ) ), pkFade );
	const pkSy = select( fvPk.lessThan( 0.035 ), float( - 1.0 ), select( fvPk.lessThan( 0.07 ), float( 1.0 ), float( 0.0 ) ) ).mul( pkFade );

	const fvHz = fract( uvS.y.div( 0.16 ) );
	const hzFade = float( 1 ).sub( smoothstep( 0.015, 0.05, fw.y ) );
	const hzShade = mix( 0.95, mix( 0.5, 1.0, smoothstep( 0.0, 0.05, fvHz ) ), hzFade );

	const shade = float( 1 )
		.add( mLap.mul( lapShade.sub( 1 ) ) ).add( mBB.mul( bbShade.sub( 1 ) ) ).add( mTG.mul( tgShade.sub( 1 ) ) )
		.add( mLv.mul( lvShade.sub( 1 ) ) ).add( mPk.mul( pkShade.sub( 1 ) ) ).add( mHz.mul( hzShade.sub( 1 ) ) );

	// nail heads with rust bleed on deck planks
	const nu = fract( uvS.x.sub( 0.1 ).div( 0.8 ).add( 0.5 ) ).sub( 0.5 ).mul( 0.8 );
	const nv = min( abs( uvS.y.sub( 0.045 ) ), abs( uvS.y.sub( 0.155 ) ) );
	const nearN = float( 1 ).sub( smoothstep( 0.0015, 0.004, px ) );
	const nail = float( 1 ).sub( smoothstep( 0.0045, 0.0075, length( vec2( nu, nv ) ) ) ).mul( mNail ).mul( nearN );
	const stain = float( 1 ).sub( smoothstep( 0.0, 0.05, length( vec2( nu.mul( 0.3 ), nv.mul( 1.4 ) ) ) ) ).mul( mNail );

	// ---- colour
	let col = mix( wood.mul( stain.mul( - 0.4 ).add( 1.0 ) ), paintCol, painted );
	col = mix( col, vec3( 0.04, 0.032, 0.028 ), nail );
	const dirt = mix( 0.22, 0.5, wW );
	col = col.mul( float( 1 ).sub( Gm.r.mul( dirt ).mul( 0.5 ) ) );
	col = mix( col, vec3( 0.62, 0.61, 0.57 ), Gm.g.mul( 0.28 ).mul( painted.mul( 0.5 ).add( 0.5 ) ) );
	col = col.mul( float( 1 ).sub( Gm.b.mul( 0.3 ) ) );
	const isWall = mLap.add( mBB ).add( mTG );
	col = col.mul( float( 1 ).sub( float( 1 ).sub( smoothstep( 0.0, 0.5, uvS.y ) ).mul( 0.3 ).mul( isWall ) ) );
	const ao = mix( N.a, mix( 1.0, N.a, 0.35 ).mul( P.a ), painted );
	col = col.mul( shade ).mul( mix( 1.0, ao, 0.5 ) );

	// ---- normal (grain space -> mesh space) + analytic pattern relief
	const sWood = slopeOf( N ).add( slopeOf( D ).mul( near ).mul( 0.5 ) );
	const sPaint = slopeOf( N ).mul( 0.3 ).add( slopeOf( P ) );
	const sG = mix( sWood, sPaint, painted ).add( edgeSlope ).mul( float( 1 ).sub( endAny.mul( 0.6 ) ) );
	const sM = mix( sG, sG.yx, isVert );
	const sPat = vec2(
		mBB.mul( bbSx ).add( mTG.mul( tgSx ) ),
		mLap.mul( lapSy ).add( mLv.mul( lvSy ) ).add( mPk.mul( pkSy ) )
	);
	// ---- roughness: satin-ish paint vs dry raw timber; grime and salt dull it
	const roughRaw = N.b.add( wW.mul( 0.03 ) );
	const roughPaint = mix( 0.4, 0.7, P.b ).add( fade.mul( 0.12 ) );
	const rough = mix( roughRaw, roughPaint, painted ).add( Gm.r.mul( 0.06 ) ).add( Gm.g.mul( 0.08 ) ).sub( nail.mul( 0.3 ) );
	const res = tidal( T.grime, col, rough );

	// ---- glass mode (windows and lanterns share this material to save a draw call)
	const isGlass = step( 8.5, pattern );
	const glass = glassNodes( T, seed, paint, weather );
	m.colorNode = mix( res.col, glass.color, isGlass ).add( trigger );
	m.roughnessNode = clamp( mix( res.rough, glass.rough, isGlass ), 0.03, 1.0 );
	m.aoNode = mix( ao, 1.0, isGlass );
	m.emissiveNode = glass.emissive.mul( isGlass );
	m.normalNode = normalFromSlope( sM.add( sPat ).mul( float( 1 ).sub( isGlass ) ) );
	return m;

}

// ---------------------------------------------------------------------------
// CORRUGATED METAL ROOFING
// uv: u = distance up-slope from the eave (m), v = along the eave (m)
// vdata: x seed, y rust amount, z galvanized (1) / painted (0)

export function createRoofMetalMaterial( T, trigger ) {

	const m = standard( { roughness: 0.5, metalness: 0.2 } );
	m.name = 'VillageRoofMetal';

	const seed = aData.x;
	const rustAmt = aData.y;
	const galv = aData.z;
	const uvm = uv();
	const fw = fwidth( uvm );
	const near = float( 1 ).sub( smoothstep( 0.0015, 0.006, max( fw.x, fw.y ) ) );
	const sheetF = uvm.y.div( 0.84 ).add( floor( seed.mul( 13.0 ) ) );
	const sheetId = floor( sheetF );
	const sheetR = hash21( sheetId, seed.mul( 91.0 ) );
	const tuv = vec2( sheetF, uvm.x.div( 1.68 ).add( sheetR.mul( 5.37 ) ) );
	const M = texture( T.roofA, tuv );
	const N = texture( T.roofN, tuv );
	const macro = texture( T.roofA, tuv.mul( vec2( 0.13, 0.09 ) ).add( vec2( seed, seed.mul( 1.7 ) ) ) );

	const eave = float( 1 ).sub( smoothstep( 0.0, 0.9, uvm.x ) );
	const replaced = step( 0.86, sheetR );
	const rustIn = clamp( rustAmt.add( replaced.mul( 0.2 ) ).add( macro.r.sub( 0.4 ).mul( 0.5 ) ), 0.0, 1.0 );
	// rust channel quantiles: 50% 0.25, 70% 0.31, 90% 0.42 -> rust 0.4 ~10%, 0.6 ~25%, 0.8 ~45%
	const thr = mix( 0.62, 0.23, rustIn ).sub( eave.mul( 0.2 ) );
	const rust = smoothstep( thr.sub( 0.04 ), thr.add( 0.04 ), M.r );
	const fade = clamp( M.b.mul( 0.7 ).add( macro.b.mul( 0.3 ) ), 0.0, 1.0 );

	const paintCol = mix( aTint, aTint.mul( 0.7 ).add( vec3( 0.12, 0.1, 0.085 ) ), fade.mul( 0.55 ) ).mul( sheetR.mul( 0.16 ).add( 0.9 ) );
	const patchCol = paintCol.mul( vec3( 0.72, 0.68, 0.66 ) ).add( vec3( 0.03, 0.025, 0.02 ) );
	const galvCol = vec3( 0.34, 0.35, 0.35 ).mul( fade.mul( 0.35 ).add( 0.78 ) );
	const base = mix( mix( paintCol, patchCol, replaced ), galvCol, galv );
	const rustCol = mix( vec3( 0.12, 0.045, 0.02 ), vec3( 0.4, 0.17, 0.065 ), M.g );
	let col = mix( base, rustCol, rust );
	col = col.mul( float( 1 ).sub( M.a.mul( 0.32 ) ) );
	col = col.mul( mix( 1.0, N.a, 0.55 ) );

	// texture X runs along the eave (mesh v), texture Y up the slope (mesh u)
	const sT = slopeOf( N );
	const Hd = texture( T.hardN, uvm.mul( 1.7 ) );
	const s = vec2( sT.y, sT.x ).add( slopeOf( Hd ).mul( rust.mul( 0.8 ).add( 0.2 ) ).mul( near ) );
	m.normalNode = normalFromSlope( s );

	m.colorNode = col.add( trigger );
	m.metalnessNode = mix( mix( 0.04, 0.5, galv ), 0.0, rust );
	m.roughnessNode = clamp( mix( mix( 0.55, 0.42, galv ).add( N.b.sub( 0.5 ).mul( 0.4 ) ).add( fade.mul( 0.15 ) ), 0.9, rust ).add( M.a.mul( 0.08 ) ), 0.05, 1.0 );
	m.aoNode = N.a;
	return m;

}

// ---------------------------------------------------------------------------
// PALM THATCH
// uv: u = distance up-slope from the eave (m), v = along the eave (m)
// vdata: x seed, y age (0 golden .. 1 grey)

export function createThatchMaterial( T, trigger ) {

	const m = standard( { roughness: 0.95, metalness: 0 } );
	m.name = 'VillageThatch';

	const seed = aData.x;
	const age = aData.y;
	const uvm = uv();
	const tuv = vec2( uvm.y.add( hash21( seed, 1.3 ).mul( 7.0 ) ), uvm.x );
	const A = texture( T.thatchA, tuv );
	const N = texture( T.thatchN, tuv );
	// macro variation: patches of newer / older thatch and rain streaks down the slope
	const Gm = texture( T.grime, vec2( tuv.x.mul( 0.23 ), tuv.y.mul( 0.19 ) ).add( seed ) );
	const Gs = texture( T.grime, vec2( tuv.x.mul( 0.5 ), tuv.y.mul( 0.25 ) ).add( seed.mul( 1.7 ) ) );

	const lum = dot( A.rgb, vec3( 0.3, 0.59, 0.11 ) );
	const greyed = vec3( lum ).mul( vec3( 0.95, 0.9, 0.8 ) );
	const ageL = clamp( age.add( Gm.a.sub( 0.5 ).mul( 0.9 ) ), 0.0, 1.0 );
	let col = mix( A.rgb, greyed, ageL.mul( 0.8 ) ).mul( Gm.a.mul( 0.35 ).add( 0.82 ) ).mul( float( 1 ).sub( ageL.mul( 0.2 ) ) );
	col = col.mul( float( 1 ).sub( Gs.r.mul( 0.25 ) ) ).mul( float( 1 ).sub( Gs.b.mul( 0.35 ) ) );
	col = col.mul( mix( 1.0, N.a, 0.55 ) );
	const sT = slopeOf( N );
	m.normalNode = normalFromSlope( vec2( sT.y, sT.x ) );
	m.colorNode = col.add( trigger );
	m.roughnessNode = N.b;
	m.aoNode = N.a;
	return m;

}

// ---------------------------------------------------------------------------
// HARD SURFACES: iron, steel, galvanized, rubber, plastic, painted metal - and rope.
// vdata: x seed, y rust (0..1), z metalness, w roughness
// rope: vdata.w = 2 + rope radius (uv: u along the rope, v around, metres)

export function createHardMaterial( T, trigger ) {

	const m = standard( { roughness: 0.5, metalness: 0.5 } );
	m.name = 'VillageHard';

	const seed = aData.x;
	const rustAmt = aData.y;
	const isRope = step( 1.5, aData.w );
	const uv0 = uv();
	const off = vec2( hash21( seed, 2.3 ), hash21( seed, 8.9 ) );
	const tuv = uv0.add( off );
	const HA = texture( T.hardA, tuv );
	const HN = texture( T.hardN, tuv );
	const hasRust = step( 0.001, rustAmt );
	const thr = mix( 0.95, 0.3, rustAmt );
	const rust = smoothstep( thr.sub( 0.06 ), thr.add( 0.06 ), HA.r ).mul( hasRust );
	let base = aTint.mul( float( 1 ).sub( HA.b.mul( 0.2 ) ) ).mul( HA.r.sub( 0.5 ).mul( 0.1 ).add( 1.0 ) );
	base = mix( base, vec3( 0.3, 0.3, 0.29 ), HA.g.mul( 0.55 ).mul( hasRust ).mul( float( 1 ).sub( rust ) ) );
	const rustCol = mix( vec3( 0.12, 0.045, 0.02 ), vec3( 0.4, 0.17, 0.06 ), smoothstep( 0.3, 0.9, HA.r ) );
	const hardCol = mix( base, rustCol, rust ).mul( mix( 1.0, HN.a, 0.5 ) );
	const hardRough = mix( clamp( aData.w.add( HN.b.sub( 0.5 ).mul( 0.35 ) ).sub( HA.g.mul( 0.15 ).mul( hasRust ) ), 0.04, 1.0 ), 0.88, rust );
	const hardS = slopeOf( HN ).mul( rust.mul( 0.8 ).add( 0.25 ) );

	// rope: uvs normalised by the circumference so the three strands wrap seamlessly
	const ropeR = max( aData.w.sub( 2.0 ), 0.004 );
	const R = texture( T.rope, uv0.div( ropeR.mul( TAU ) ).add( vec2( aData.x.mul( 3.1 ), 0.0 ) ) );
	const ropeCol = aTint.mul( mix( 0.45, 1.25, R.r ) ).mul( mix( 1.0, R.g, 0.6 ) );
	const rn = R.ba.mul( 2.0 ).sub( 1.0 );
	const ropeS = rn.div( sqrt( max( float( 1 ).sub( dot( rn, rn ) ), 0.04 ) ) );

	const res = tidal( T.grime, mix( hardCol, ropeCol, isRope ), mix( hardRough, 0.93, isRope ) );
	m.normalNode = normalFromSlope( mix( hardS, ropeS, isRope ) );
	m.colorNode = res.col.add( trigger );
	m.roughnessNode = res.rough;
	m.metalnessNode = aData.z.mul( float( 1 ).sub( max( rust, res.wet ) ) ).mul( float( 1 ).sub( isRope ) );
	m.aoNode = mix( HN.a, R.g, isRope );
	return m;

}

// ---------------------------------------------------------------------------
// GLASS (evaluated inside the wood material): window panes and lantern glass, emissive at night.
// uv: normalized 0..1 across a pane. kind 0 window / 1 lantern, lit 0/1, tint: curtain / glass color

function glassNodes( T, seed, kind, lit ) {

	const uvm = uv();
	// grime map channels: R streaks, G salt, B spots, A macro
	const Gl = texture( T.grime, uvm.mul( 0.45 ).add( vec2( seed.mul( 3.7 ), seed.mul( 1.3 ) ) ) );
	const edgeD = min( min( uvm.x, float( 1 ).sub( uvm.x ) ), min( uvm.y, float( 1 ).sub( uvm.y ) ) );
	const frameDirt = float( 1 ).sub( smoothstep( 0.0, 0.07, edgeD ) );
	const curtain = float( 1 ).sub( smoothstep( 0.18, 0.3, uvm.x ) ).add( smoothstep( 0.7, 0.82, uvm.x ) ).mul( step( 0.35, seed ) );
	const folds = n01( sin( uvm.x.mul( 70.0 ).add( seed.mul( 20.0 ) ) ) );
	const interior = vec3( 0.012, 0.014, 0.017 ).mul( Gl.a.mul( 0.8 ).add( 0.6 ) );
	let winCol = mix( interior, aTint.mul( 0.16 ).mul( folds.mul( 0.4 ).add( 0.6 ) ), curtain );
	winCol = winCol.add( vec3( 0.05, 0.05, 0.045 ).mul( Gl.g ) ).add( vec3( 0.03, 0.026, 0.02 ).mul( Gl.b.add( frameDirt ) ) );
	const lanternCol = aTint.mul( 0.55 ).mul( float( 1 ).sub( Gl.b.mul( 0.25 ) ) );
	const isLantern = step( 0.5, kind );
	const color = mix( winCol, lanternCol, isLantern );
	const rough = mix( float( 0.035 ).add( Gl.r.mul( 0.22 ) ).add( Gl.g.mul( 0.18 ) ).add( Gl.b.mul( 0.1 ) ).add( frameDirt.mul( 0.25 ) ), float( 0.32 ).add( Gl.r.mul( 0.15 ) ), isLantern );

	const nightOn = smoothstep( 0.15, 0.75, G.night );
	const warm = vec3( 1.0, 0.56, 0.24 );
	const center = float( 1 ).sub( smoothstep( 0.1, 0.75, abs( uvm.x.sub( 0.5 ) ).mul( 1.4 ).add( abs( uvm.y.sub( 0.62 ) ) ) ) );
	const winGlow = mix( warm.mul( center.mul( 0.6 ).add( 0.5 ) ), aTint.mul( 0.5 ).add( warm.mul( 0.5 ) ).mul( folds.mul( 0.3 ).add( 0.45 ) ), curtain )
		.mul( hash21( seed, 3.1 ).mul( 0.5 ).add( 0.75 ) ).mul( 3.2 ).mul( float( 1 ).sub( Gl.r.mul( 0.25 ) ) );
	const flicker = sin( G.time.mul( 9.0 ).add( seed.mul( 40.0 ) ) ).mul( sin( G.time.mul( 5.3 ).add( seed.mul( 13.0 ) ) ) ).mul( 0.12 ).add( 0.9 );
	const lanternGlow = vec3( 1.0, 0.64, 0.3 ).mul( 6.0 ).mul( flicker );
	const emissive = mix( winGlow, lanternGlow, isLantern ).mul( nightOn ).mul( max( lit, isLantern ) );
	return { color, rough, emissive };

}

// ---------------------------------------------------------------------------
// STONE: rubble masonry, lime plaster over stone, sand dusted near the ground
// uv in metres (u horizontal, v vertical from the bottom of the piece).
// vdata: x seed, y style (0 coursed rubble, 1 plaster over stone, 2 small rubble)
// tint: plaster colour

export function createStoneMaterial( T, trigger ) {

	const m = standard( { roughness: 0.92, metalness: 0 } );
	m.name = 'VillageStone';

	const seed = aData.x;
	const style = aData.y;
	const uv0 = uv();
	const isCap = step( 1500.0, uv0.x );
	const isEnd = step( 500.0, uv0.x ).mul( float( 1 ).sub( isCap ) );
	const uvS = uv0.sub( vec2( isCap.mul( 2000.0 ).add( isEnd.mul( 1000.0 ) ), 0.0 ) );
	const isRubble = step( 1.5, style );
	const isPlaster = band01( style, 1 );
	const off = vec2( hash21( seed, 4.1 ), hash21( seed, 7.7 ) );
	const tuv = uvS.mul( mix( 0.5, 0.8, isRubble ) ).add( off );
	const A = texture( T.stoneA, tuv );
	const N = texture( T.stoneN, tuv );
	// lime plaster: trowel marks from the paint film set, stains / dust from the grime map
	const puv = uvS.add( off.mul( 3.0 ) );
	const PN = texture( T.paintN, puv.mul( vec2( 0.7, 1.4 ) ) );
	const PA = texture( T.grime, puv.mul( 0.5 ) );

	// plaster survives where the baked plaster field is high; the plaster edge has thickness
	const pf = A.a.add( PA.a.sub( 0.5 ).mul( 0.12 ) );
	const pm = smoothstep( 0.3, 0.33, pf ).mul( isPlaster );
	const e = 1.0 / 1024.0;
	const gX = texture( T.stoneA, tuv.add( vec2( e, 0.0 ) ) ).a.sub( texture( T.stoneA, tuv.sub( vec2( e, 0.0 ) ) ).a );
	const gY = texture( T.stoneA, tuv.add( vec2( 0.0, e ) ) ).a.sub( texture( T.stoneA, tuv.sub( vec2( 0.0, e ) ) ).a );
	const pEdge = float( 1 ).sub( smoothstep( 0.0, 0.03, abs( pf.sub( 0.315 ) ) ) ).mul( isPlaster );

	const stoneCol = A.rgb.mul( mix( vec3( 1.0 ), aTint.mul( 1.1 ), 0.15 ) );
	let plasterCol = aTint.mul( float( 1 ).sub( PA.r.mul( 0.3 ) ) ).mul( float( 1 ).sub( PA.b.mul( 0.25 ) ) ).mul( PA.a.sub( 0.5 ).mul( 0.16 ).add( 1.0 ) );
	plasterCol = plasterCol.mul( mix( 1.0, PN.a, 0.6 ) ).mul( PN.b.sub( 0.5 ).mul( 0.12 ).add( 1.0 ) );
	let col = mix( stoneCol.mul( mix( 1.0, N.a, 0.5 ) ), plasterCol, pm );
	// sand dust and splash dirt near the ground
	const low = float( 1 ).sub( smoothstep( 0.05, 0.75, uvS.y ) );
	const dust = low.mul( smoothstep( 0.25, 0.7, PA.a ) ).mul( 0.8 );
	col = mix( col, vec3( 0.46, 0.4, 0.3 ), dust );
	col = col.mul( float( 1 ).sub( low.mul( 0.18 ) ) );

	const s = mix( slopeOf( N ), slopeOf( PN ).mul( 2.5 ), pm ).add( vec2( gX, gY ).mul( pEdge ).mul( - 6.0 ) );
	m.normalNode = normalFromSlope( s );
	m.colorNode = col.add( trigger );
	m.roughnessNode = clamp( mix( N.b, PN.b.mul( 0.3 ).add( 0.66 ), pm ).add( dust.mul( 0.05 ) ), 0.05, 1.0 );
	m.aoNode = mix( N.a, PN.a, pm );
	return m;

}

// ---------------------------------------------------------------------------
// FABRIC: laundry / tarps, knotted fishing nets and pennant flags in one double sided,
// alpha tested material (one draw call).
// vdata: x seed, y sway weight (flag: distance from the pole), z mode:
//   z = 0          cloth
//   0 < z < 5000   net, z = mesh size (m)
//   z > 5000       flag, z = pole x + 10000, w = pole z (the cloth swings downwind on the GPU)

export function createFabricMaterial( T, trigger ) {

	const m = standard( { roughness: 0.88, metalness: 0, side: THREE.DoubleSide } );
	m.name = 'VillageFabric';
	m.alphaTest = 0.5;

	const isFlag = step( 5000.0, aData.z );
	const isNet = step( 0.001, aData.z ).mul( float( 1 ).sub( isFlag ) );
	const uvm = uv();

	// net
	const tile = max( aData.z, 0.02 ).mul( 4.0 );
	const nuv = uvm.div( tile );
	const Nt = texture( T.net, nuv );
	const fwn = fwidth( nuv );
	const far = smoothstep( 0.05, 0.1, max( fwn.x, fwn.y ) );
	const sc = screenCoordinate.xy;
	const ign = fract( fract( sc.x.mul( 0.06711056 ).add( sc.y.mul( 0.00583715 ) ) ).mul( 52.9829189 ) );
	const netAlpha = mix( Nt.r, step( ign, 0.55 ), far );
	const netCol = aTint.mul( mix( 0.5, 1.05, Nt.g ) ).mul( mix( 1.0, 0.8, far ) );
	const nn = Nt.ba.mul( 2.0 ).sub( 1.0 );
	const netS = nn.div( sqrt( max( float( 1 ).sub( dot( nn, nn ) ), 0.04 ) ) ).mul( float( 1 ).sub( far ) );

	// cloth: woven canvas, sun faded
	const weave = n01( sin( uvm.x.mul( 900.0 ) ) ).mul( n01( sin( uvm.y.mul( 900.0 ) ) ) );
	const wf = float( 1 ).sub( smoothstep( 0.0005, 0.002, fwidth( uvm.x ) ) );
	const fadeN = texture( T.grime, uvm.mul( 0.8 ).add( aData.x ) ).a;
	const clothCol = mix( aTint, aTint.mul( 0.6 ).add( vec3( 0.3 ) ), fadeN.mul( 0.5 ) ).mul( weave.mul( 0.12 ).mul( wf ).add( 0.94 ) );

	// sway for cloth / nets
	const w = aData.y;
	const ph = G.time.mul( 1.7 ).add( positionLocal.x.mul( 0.6 ) ).add( positionLocal.z.mul( 0.45 ) ).add( aData.x.mul( 20.0 ) );
	const gust = sin( ph ).mul( 0.6 ).add( sin( ph.mul( 2.3 ).add( 1.7 ) ).mul( 0.3 ) ).add( 0.55 );
	const flutter = sin( G.time.mul( 7.0 ).add( positionLocal.y.mul( 9.0 ) ).add( aData.x.mul( 50.0 ) ) ).mul( 0.25 );
	const swayPos = positionLocal.add( vec3( G.windDir.x, flutter.mul( 0.3 ), G.windDir.y ).mul( gust.add( flutter ).mul( G.windSpeed.mul( 0.011 ).mul( w ) ) ) );

	// flag: oriented downwind around its pole
	const a = aData.y;
	const dir = normalize( vec3( G.windDir.x, 0.0, G.windDir.y ) );
	const perp = vec3( dir.z.negate(), 0.0, dir.x );
	const strength = smoothstep( 0.5, 9.0, G.windSpeed );
	const phase = G.time.mul( 7.5 ).sub( a.mul( 5.0 ) ).add( aData.x.mul( 30.0 ) );
	const wave = sin( phase ).mul( a.mul( 0.09 ) ).mul( strength.mul( 0.7 ).add( 0.3 ) );
	const droop = float( 1 ).sub( strength ).mul( a ).mul( 0.55 );
	const flagPos = vec3( aData.z.sub( 10000.0 ), positionLocal.y.sub( droop ), aData.w ).add( dir.mul( a.mul( float( 1 ).sub( droop.mul( 0.4 ) ) ) ) ).add( perp.mul( wave ) );
	const slopeW = cos( phase ).mul( 0.45 ).mul( a.add( 0.2 ) ).mul( strength.mul( 0.7 ).add( 0.3 ) );
	const flagNV = transformNormalToView( normalize( perp.sub( dir.mul( slopeW ) ) ) ).mul( faceDirection );

	m.positionNode = mix( swayPos, flagPos, isFlag );
	m.normalNode = normalize( mix( normalFromSlope( netS.mul( isNet ) ), flagNV, isFlag ) );
	m.opacityNode = mix( 1.0, netAlpha, isNet );
	m.colorNode = mix( mix( clothCol, netCol, isNet ), aTint.mul( fadeN.mul( 0.15 ).add( 0.88 ) ), isFlag ).add( trigger );
	return m;

}

// ---------------------------------------------------------------------------
// NETS: blended, no depth write. Knotted mesh up close (soft-edged strands from the net texture);
// further away the strands average into a see-through veil of the right coverage. Alpha testing
// (or dithering) the sub-pixel strands wrote a noisy foreground depth over whatever is behind the net,
// and the TAA reprojected the background with it: grain and ghosted houses behind every net.
// vdata as for FABRIC nets: x seed, y sway weight, z mesh size (m).

export function createNetMaterial( T, trigger ) {

	const m = standard( { roughness: 0.9, metalness: 0, side: THREE.DoubleSide } );
	m.name = 'VillageNet';
	m.transparent = true;
	m.depthWrite = false;

	const uvm = uv();
	const tile = max( aData.z, 0.02 ).mul( 4.0 );
	const nuv = uvm.div( tile );
	const Nt = texture( T.net, nuv );
	const fwn = fwidth( nuv );
	const far = smoothstep( 0.05, 0.1, max( fwn.x, fwn.y ) );
	const nn = Nt.ba.mul( 2.0 ).sub( 1.0 );
	const netS = nn.div( sqrt( max( float( 1 ).sub( dot( nn, nn ) ), 0.04 ) ) ).mul( float( 1 ).sub( far ) );

	// sway (as the fabric)
	const w = aData.y;
	const ph = G.time.mul( 1.7 ).add( positionLocal.x.mul( 0.6 ) ).add( positionLocal.z.mul( 0.45 ) ).add( aData.x.mul( 20.0 ) );
	const gust = sin( ph ).mul( 0.6 ).add( sin( ph.mul( 2.3 ).add( 1.7 ) ).mul( 0.3 ) ).add( 0.55 );
	const flutter = sin( G.time.mul( 7.0 ).add( positionLocal.y.mul( 9.0 ) ).add( aData.x.mul( 50.0 ) ) ).mul( 0.25 );
	m.positionNode = positionLocal.add( vec3( G.windDir.x, flutter.mul( 0.3 ), G.windDir.y ).mul( gust.add( flutter ).mul( G.windSpeed.mul( 0.011 ).mul( w ) ) ) );

	m.normalNode = normalize( normalFromSlope( netS ) );
	// strands up close; average coverage of the knotted mesh (~0.4) once they are sub-pixel
	m.opacityNode = mix( Nt.r, float( 0.4 ), far ).mul( 0.95 );
	m.colorNode = aTint.mul( mix( 0.5, 1.05, Nt.g ) ).mul( mix( 1.0, 0.85, far ) ).add( trigger );
	return m;

}

export function createVillageMaterials( textures = new VillageTextures() ) {

	const T = textures.textures;
	const trig = textures.trigger;
	return {
		wood: createWoodMaterial( T, trig ),
		roofMetal: createRoofMetalMaterial( T, trig ),
		thatch: createThatchMaterial( T, trig ),
		hard: createHardMaterial( T, trig ),
		stone: createStoneMaterial( T, trig ),
		fabric: createFabricMaterial( T, trig ),
		net: createNetMaterial( T, trig ),
	};

}
