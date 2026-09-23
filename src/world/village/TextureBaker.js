import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, vec2, vec3, vec4, float, int, uint, uv, texture, uniform, mrt, nodeObject, hash, property,
	floor, fract, mix, smoothstep, step, sin, sqrt, abs, min, max, clamp, length, normalize, exp,
} from 'three/tsl';

// GPU-baked, tileable PBR texture sets for the village materials.
//
// Every set is generated once, on the first frame that draws a village material
// (VillageBakeNode.updateBefore), with two shared pipelines:
//   1. an uber "fields" shader (generator picked by a uniform) writes height / roughness /
//      occlusion into a transient RGBA16F target and (MRT) the final albedo or mask map
//      into an RGBA8 target,
//   2. a derivation shader turns the height field into a tangent-space normal (Sobel
//      filtered) plus horizon-based ambient occlusion, packed as RG = normal xy,
//      B = roughness, A = AO.
// All final maps are RGBA8, fully mipmapped, repeat-wrapped, trilinear + 8x anisotropic:
// 43 MB in total. Each transient field map is freed right after its derivation pass.
// Cost: ~50 ms with a warm pipeline cache, ~450 ms on a cold (first ever) shader compile.
//
// All noise is periodic over the texture tile so the maps tile seamlessly.
//
// Maps (tile size in metres):
//   woodA / woodN      weathered timber: grain, growth rings, knots, checks; A = paint chip field
//   paintN             paint film: brush strokes, crazing
//   roofA / roofN      corrugated sheet: ribs, screws, dents; A = rust, rust hue, fade, grime
//   thatchA / thatchN  layered palm fronds with frayed tips; A = tip mask
//   stoneA / stoneN    rubble masonry, pits, lichen; A = plaster survival field
//   hardA / hardN      worn metal: rust blooms, pits, scratches
//   grime              R streaks, G salt, B spots / barnacles, A macro variation
//   rope               three strand twist: R shade, G AO, BA normal
//   net                knotted diamond mesh: R alpha, G shade, BA normal

const { resetRendererState, restoreRendererState } = THREE.RendererUtils;
const TAU = Math.PI * 2;

// ---------------------------------------------------------------------------
// Periodic noise library (compact: loops are real shader loops so pipelines compile fast)

const hashU = ( x, y, s ) => uint( x ).mul( uint( 1597334677 ) )
	.bitXor( uint( y ).mul( uint( 3812015801 ) ) )
	.bitXor( uint( s ).mul( uint( 2654435761 ) ).add( uint( 1013904223 ) ) );

// three independent 10-bit randoms from one hash
const hash3 = Fn( ( [ x, y, s ] ) => {

	const h = uint( hash( hashU( x, y, s ) ).mul( 1073741823.0 ) );
	return vec3(
		float( h.bitAnd( uint( 1023 ) ) ),
		float( h.shiftRight( uint( 10 ) ).bitAnd( uint( 1023 ) ) ),
		float( h.shiftRight( uint( 20 ) ).bitAnd( uint( 1023 ) ) )
	).div( 1023.0 );

} ).setLayout( { name: 'vlgHash3', type: 'vec3', inputs: [ { name: 'x', type: 'float' }, { name: 'y', type: 'float' }, { name: 's', type: 'float' } ] } );

const wrapP = ( i, per ) => i.sub( per.mul( floor( i.div( per ) ) ) );

// Loop bounds / octave counts are routed through this uniform (always 1) so the shader
// compiler cannot specialise and unroll every call site: keeps cold compiles short.
const uOne = uniform( 1.0 );

// periodic 2D gradient noise, roughly in [-1, 1]
const pnoise = Fn( ( [ p, per, s ] ) => {

	const i = floor( p );
	const f = fract( p );
	const u = f.mul( f ).mul( f ).mul( f.mul( f.mul( 6.0 ).sub( 15.0 ) ).add( 10.0 ) );
	const i0 = wrapP( i, per );
	const i1 = wrapP( i.add( 1.0 ), per );
	const g = ( gx, gy, ox, oy ) => {

		const h = hash3( gx, gy, s ).xy.sub( 0.5 );
		return h.x.mul( f.x.sub( ox ) ).add( h.y.mul( f.y.sub( oy ) ) );

	};

	const n00 = g( i0.x, i0.y, 0.0, 0.0 );
	const n10 = g( i1.x, i0.y, 1.0, 0.0 );
	const n01 = g( i0.x, i1.y, 0.0, 1.0 );
	const n11 = g( i1.x, i1.y, 1.0, 1.0 );
	return mix( mix( n00, n10, u.x ), mix( n01, n11, u.x ), u.y ).mul( 3.4 );

} ).setLayout( { name: 'vlgPNoise', type: 'float', inputs: [ { name: 'p', type: 'vec2' }, { name: 'per', type: 'vec2' }, { name: 's', type: 'float' } ] } );

// periodic fBm (octave loop inside the shader)
const pfbm = Fn( ( [ p, per, oct, s, gain ] ) => {

	const sum = float( 0 ).toVar();
	const amp = float( 1 ).toVar();
	const norm = float( 0 ).toVar();
	const q = vec2( p ).toVar();
	const pp = vec2( per ).toVar();
	Loop( { start: int( 0 ), end: int( float( oct ).mul( uOne ) ), type: 'int', condition: '<', name: 'o' }, ( { o } ) => {

		sum.addAssign( pnoise( q, pp, s.add( float( o ).mul( 17.0 ) ) ).mul( amp ) );
		norm.addAssign( amp );
		amp.mulAssign( gain );
		q.mulAssign( 2.0 );
		pp.mulAssign( 2.0 );

	} );
	return sum.div( norm );

} ).setLayout( { name: 'vlgPFbm', type: 'float', inputs: [ { name: 'p', type: 'vec2' }, { name: 'per', type: 'vec2' }, { name: 'oct', type: 'int' }, { name: 's', type: 'float' }, { name: 'gain', type: 'float' } ] } );

// periodic cellular noise: returns vec4( F1, F2, cellId, 0 )
const pworley = Fn( ( [ p, per, s, jit ] ) => {

	const i = floor( p );
	const f = fract( p );
	const d1 = float( 9.0 ).toVar();
	const d2 = float( 9.0 ).toVar();
	const id = float( 0.0 ).toVar();
	const lo = int( uOne.negate() ), hi = int( uOne.add( 1.0 ) );
	Loop( { start: lo, end: hi, type: 'int', condition: '<', name: 'cy' }, { start: lo, end: hi, type: 'int', condition: '<', name: 'cx' }, ( { cx, cy } ) => {

		const o = vec2( float( cx ), float( cy ) );
		const c = wrapP( i.add( o ), per );
		const h = hash3( c.x, c.y, s );
		const d = length( o.add( h.xy.sub( 0.5 ).mul( jit ).add( 0.5 ) ).sub( f ) );
		If( d.lessThan( d1 ), () => {

			d2.assign( d1 );
			d1.assign( d );
			id.assign( h.z );

		} ).ElseIf( d.lessThan( d2 ), () => {

			d2.assign( d );

		} );

	} );
	return vec4( d1, d2, id, 0.0 );

} ).setLayout( { name: 'vlgPWorley', type: 'vec4', inputs: [ { name: 'p', type: 'vec2' }, { name: 'per', type: 'vec2' }, { name: 's', type: 'float' }, { name: 'jit', type: 'float' } ] } );

// periodic cellular noise returning the offset to the nearest feature: vec4( rel.x, rel.y, F1, cellId )
const pworleyRel = Fn( ( [ p, per, s, jit ] ) => {

	const i = floor( p );
	const f = fract( p );
	const d1 = float( 9.0 ).toVar();
	const rel = vec2( 0.0 ).toVar();
	const id = float( 0.0 ).toVar();
	const lo = int( uOne.negate() ), hi = int( uOne.add( 1.0 ) );
	Loop( { start: lo, end: hi, type: 'int', condition: '<', name: 'cy' }, { start: lo, end: hi, type: 'int', condition: '<', name: 'cx' }, ( { cx, cy } ) => {

		const o = vec2( float( cx ), float( cy ) );
		const c = wrapP( i.add( o ), per );
		const h = hash3( c.x, c.y, s );
		const dv = f.sub( o.add( h.xy.sub( 0.5 ).mul( jit ).add( 0.5 ) ) );
		const d = length( dv );
		If( d.lessThan( d1 ), () => {

			d1.assign( d );
			rel.assign( dv );
			id.assign( h.z );

		} );

	} );
	return vec4( rel, d1, id );

} ).setLayout( { name: 'vlgPWorleyRel', type: 'vec4', inputs: [ { name: 'p', type: 'vec2' }, { name: 'per', type: 'vec2' }, { name: 's', type: 'float' }, { name: 'jit', type: 'float' } ] } );

// convenience wrappers (fx, fy: integer frequencies over the tile)
const pn = ( X, Y, fx, fy, s ) => pnoise( vec2( X.mul( fx ), Y.mul( fy ) ), vec2( fx, fy ), float( s ) );
const pf = ( X, Y, fx, fy, oct, s, gain = 0.5 ) => pfbm( vec2( X.mul( fx ), Y.mul( fy ) ), vec2( fx, fy ), int( oct ), float( s ), float( gain ) );
const pw = ( X, Y, fx, fy, s, jit = 1 ) => pworley( vec2( X.mul( fx ), Y.mul( fy ) ), vec2( fx, fy ), float( s ), float( jit ) );
const pwr = ( X, Y, fx, fy, s, jit = 1 ) => pworleyRel( vec2( X.mul( fx ), Y.mul( fy ) ), vec2( fx, fy ), float( s ), float( jit ) );
const n01 = ( n ) => n.mul( 0.5 ).add( 0.5 );
const hf = ( a, b ) => fract( sin( float( a ).mul( 12.9898 ).add( float( b ).mul( 78.233 ) ) ).mul( 43758.5453 ) );
const band = ( x, a, b, c, d ) => smoothstep( a, b, x ).mul( float( 1 ).sub( smoothstep( c, d, x ) ) );

// ---------------------------------------------------------------------------
// Texture set generators. Each returns { fields: vec4( height, roughness, occlusion, 0 ), albedo?: vec4 }
// X, Y are tile coordinates in [0, 1).

// WEATHERED WOOD - tile = 2 m along the grain (X) x 1 m across (Y)
function woodGen( X, Y ) {

	// knots with grain flowing around them
	const kw = pwr( X, Y, 6, 10, 11, 0.7 );
	const kHas = step( 0.9, kw.w );
	const kRad = float( 0.01 ).add( hf( kw.w, 3.3 ).mul( 0.013 ) );
	const kdm = length( vec2( kw.x.mul( 2 / 6 ), kw.y.mul( 1 / 10 ) ) );
	const kInfl = exp( kdm.div( kRad.mul( 2.4 ) ).negate() ).mul( kHas );

	// growth rings with uneven spacing and per-ring width / contrast
	const warpA = pf( X, Y, 2, 4, 3, 21 );
	const warpB = pf( X, Y, 1, 2, 2, 23 );
	const r = Y.mul( 110 ).add( warpA.mul( 5.0 ) ).add( warpB.mul( 14.0 ) ).add( kInfl.mul( 14.0 ) );
	const ringId = floor( r );
	const ringIdW = ringId.sub( floor( ringId.div( 110 ) ).mul( 110 ) );
	const w = fract( r );
	const lw = mix( 0.09, 0.32, hf( ringIdW, 5.1 ) );
	const late = smoothstep( float( 0.93 ).sub( lw ), float( 1.0 ).sub( lw ), w ).mul( float( 1 ).sub( smoothstep( 0.9, 0.995, w ) ) );
	const ringDark = mix( 0.65, 1.15, hf( ringIdW, 9.7 ) );

	// dense fibre streaks along the grain (the signature of sun-weathered timber)
	const fA = n01( pn( X, Y, 12, 300, 53 ) );
	const fB = n01( pn( X, Y, 5, 140, 55 ) );
	const fC = n01( pn( X, Y, 16, 400, 57 ) );
	const streak = fA.mul( 0.5 ).add( fB.mul( 0.3 ) ).add( fC.mul( 0.2 ) );
	const erosion = float( 1 ).sub( late ).mul( fB.mul( 0.6 ).add( 0.4 ) );

	// checks / cracks along the grain, tapering at the ends
	const crN = abs( pn( X, Y, 6, 60, 71 ) );
	const crMask = smoothstep( 0.1, 0.45, pf( X, Y, 3, 4, 2, 73 ) );
	const crack = float( 1 ).sub( smoothstep( 0.012, 0.055, crN.div( crMask.add( 0.05 ) ) ) ).mul( step( 0.05, crMask ) );

	const kCore = float( 1 ).sub( smoothstep( kRad.mul( 0.72 ), kRad, kdm ) ).mul( kHas );
	const kRing = band( kdm, kRad.mul( 0.92 ), kRad.mul( 1.02 ), kRad.mul( 1.05 ), kRad.mul( 1.3 ) ).mul( kHas );

	const height = float( 0.46 ).add( late.mul( 0.22 ).mul( ringDark ) ).sub( erosion.mul( 0.09 ) ).add( streak.sub( 0.5 ).mul( 0.14 ) )
		.sub( crack.mul( 0.36 ) ).add( kCore.mul( 0.08 ) ).sub( kRing.mul( 0.22 ) );

	// silver grey surface, brown-grey eroded grooves, fibre streaks, stains and bleaching
	const blotch = n01( pf( X, Y, 2, 3, 3, 81 ) );
	const bleach = smoothstep( 0.5, 0.82, n01( pf( X, Y, 3, 2, 3, 83 ) ) );
	const stain = smoothstep( 0.58, 0.86, n01( pf( X, Y, 4, 5, 3, 85 ) ) );
	// sun-bleached driftwood grey (linear albedo ~0.2 in the grooves, ~0.45 on the latewood)
	const grooveC = vec3( 0.2, 0.186, 0.168 );
	const silverC = vec3( 0.45, 0.445, 0.428 );
	let col = mix( grooveC, silverC, clamp( late.mul( ringDark ).mul( 0.55 ).add( streak.mul( 0.75 ) ).sub( 0.12 ), 0.0, 1.0 ) );
	col = col.mul( fA.mul( 0.34 ).add( 0.83 ) );
	col = col.mul( float( 1 ).sub( smoothstep( 0.7, 0.95, fC ).mul( 0.3 ) ) );
	col = col.mul( blotch.mul( 0.24 ).add( 0.88 ) );
	col = mix( col, vec3( 0.55, 0.54, 0.51 ), bleach.mul( 0.3 ) );
	col = mix( col, col.mul( vec3( 0.8, 0.68, 0.54 ) ), stain.mul( 0.6 ) );
	col = mix( col, vec3( 0.035, 0.03, 0.026 ), crack.mul( 0.92 ) );
	col = mix( col, mix( vec3( 0.13, 0.085, 0.05 ), vec3( 0.07, 0.045, 0.03 ), fract( kdm.mul( 900 ) ) ), kCore );
	col = mix( col, vec3( 0.05, 0.04, 0.03 ), kRing.mul( 0.8 ) );

	// paint chip field (A): paint lets go in grain-aligned flakes, first at cracks, knots and eroded grain
	const flake = pw( X, Y, 30, 80, 91, 0.9 );
	const chipLarge = n01( pf( X, Y, 4, 8, 4, 97 ) );
	const chip = chipLarge.mul( 0.62 ).add( flake.z.mul( 0.3 ) ).add( late.mul( 0.06 ) ).sub( erosion.mul( 0.04 ) )
		.sub( crack.mul( 0.4 ) ).sub( kCore.mul( 0.1 ) );

	const rough = clamp( float( 0.82 ).add( erosion.mul( 0.1 ) ).add( fA.mul( 0.04 ) ).add( crack.mul( 0.08 ) ).sub( kCore.mul( 0.15 ) ), 0.0, 1.0 );
	const occ = float( 1 ).sub( crack.mul( 0.65 ) ).sub( kRing.mul( 0.4 ) ).sub( erosion.mul( 0.08 ) );
	return { fields: vec4( height, rough, occ, 0 ), albedo: vec4( col, clamp( chip, 0.0, 1.0 ) ) };

}

// PAINT FILM - tile = 1 m along the grain (X) x 0.5 m across (Y): brush strokes and crazing
function paintGen( X, Y ) {

	const strokes = pn( X, Y, 4, 60, 701 ).mul( 0.25 ).add( pn( X, Y, 16, 160, 703 ).mul( 0.12 ) ).add( pn( X, Y, 2, 12, 704 ).mul( 0.2 ) );
	const cz = pw( X, Y, 34, 40, 705, 1 );
	const crazeMask = smoothstep( 0.62, 0.85, n01( pf( X, Y, 3, 2, 3, 707 ) ) );
	const craze = float( 1 ).sub( smoothstep( 0.015, 0.04, cz.y.sub( cz.x ) ) ).mul( crazeMask );
	const height = float( 0.5 ).add( strokes ).sub( craze.mul( 0.12 ) );
	const rough = clamp( float( 0.5 ).add( strokes.mul( 0.35 ) ).add( craze.mul( 0.15 ) ), 0.0, 1.0 );
	const occ = float( 1 ).sub( craze.mul( 0.18 ) );
	return { fields: vec4( height, rough, occ, 0 ) };

}

// GRIME / SALT - tile = 2 m x 2 m in mesh space (X along the wall, Y up)
function grimeGen( X, Y ) {

	const streak = smoothstep( 0.55, 0.9, n01( pn( X, Y, 26, 2, 601 ) ) ).mul( smoothstep( 0.35, 0.7, n01( pf( X, Y, 4, 3, 2, 603 ) ) ) );
	const drip = smoothstep( 0.6, 0.95, n01( pn( X, Y, 70, 4, 604 ) ) ).mul( 0.5 );
	const salt = smoothstep( 0.52, 0.8, n01( pf( X, Y, 4, 4, 5, 605 ) ) ).mul( n01( pn( X, Y, 120, 120, 607 ) ).mul( 0.5 ).add( 0.5 ) );
	const mw = pw( X, Y, 60, 60, 609, 1 );
	const mildew = float( 1 ).sub( smoothstep( 0.05, 0.22, mw.x ) ).mul( step( 0.6, n01( pf( X, Y, 3, 3, 3, 611 ) ) ) );
	const macro = n01( pf( X, Y, 2, 2, 3, 613 ) );
	return vec4( clamp( streak.add( drip ), 0.0, 1.0 ), salt, mildew, macro );

}

// CORRUGATED ROOF SHEET - tile = 0.84 m across the ribs (X, 8 ribs) x 1.68 m down the slope (Y, 3 purlins)
function roofGen( X, Y ) {

	const corr = sin( X.mul( 8 ).sub( 0.25 ).mul( TAU ) ).mul( 0.5 ).add( 0.5 );
	const valley = float( 1 ).sub( corr );
	// screws with washers on every second crest at each purlin
	const sx = fract( X.mul( 4 ).sub( 0.25 ).add( 0.5 ) ).sub( 0.5 ).mul( 0.84 / 4 );
	const sy = fract( Y.mul( 3 ).sub( 0.5 ).add( 0.5 ) ).sub( 0.5 ).mul( 1.68 / 3 );
	const ds = length( vec2( sx, sy ) );
	const washer = float( 1 ).sub( smoothstep( 0.0095, 0.0115, ds ) );
	const head = sqrt( max( float( 1 ).sub( ds.div( 0.0065 ).pow( 2 ) ), 0.0 ) );
	// dents, rust pitting
	const dw = pw( X, Y, 5, 10, 13, 0.9 );
	const dent = step( 0.7, dw.z ).mul( float( 1 ).sub( smoothstep( 0.0, 0.5, dw.x ) ) );
	const blot = n01( pf( X, Y, 3, 5, 5, 31 ) );
	const vstreak = n01( pn( X, Y, 48, 3, 33 ) );
	const screwStreak = exp( sx.div( 0.007 ).pow( 2 ).negate() ).mul( smoothstep( - 0.35, - 0.01, sy ) ).mul( step( sy, 0.004 ) )
		.mul( n01( pn( X, Y, 16, 6, 35 ) ).mul( 0.8 ).add( 0.2 ) );
	const edge = float( 1 ).sub( smoothstep( 0.0, 0.05, min( X, float( 1 ).sub( X ) ).mul( 0.84 ) ) );
	const streakLong = smoothstep( 0.55, 0.85, n01( pn( X, Y, 64, 2, 39 ) ) ).mul( n01( pf( X, Y, 4, 6, 2, 45 ) ) );
	const rust = clamp( blot.mul( 0.42 ).add( vstreak.mul( valley ).mul( 0.25 ) ).add( streakLong.mul( 0.35 ) ).add( screwStreak.mul( 0.7 ) ).add( washer.mul( 0.45 ) ).add( edge.mul( 0.3 ) ).add( dent.mul( 0.15 ) ).sub( 0.05 ), 0.0, 1.0 );
	const pit = pn( X, Y, 150, 300, 37 ).mul( smoothstep( 0.45, 0.8, rust ) );
	const lapLine = float( 1 ).sub( smoothstep( 0.0012, 0.0028, abs( X.mul( 0.84 ).sub( 0.028 ) ) ) );
	const lapTop = float( 1 ).sub( smoothstep( 0.026, 0.03, X.mul( 0.84 ) ) );
	const height = corr.mul( 0.9 ).add( washer.mul( 0.035 ) ).add( head.mul( 0.05 ) ).sub( dent.mul( 0.1 ) ).add( pit.mul( 0.012 ) ).add( lapTop.mul( 0.012 ) );

	const rustVar = n01( pf( X, Y, 10, 20, 3, 41 ) );
	const fade = clamp( n01( pf( X, Y, 2, 4, 4, 43 ) ).mul( 0.8 ).add( corr.mul( 0.25 ) ), 0.0, 1.0 );
	const grime = clamp( valley.mul( n01( pn( X, Y, 32, 2, 47 ) ) ).mul( 0.8 ).add( blot.mul( 0.2 ) ), 0.0, 1.0 );
	const rough = clamp( float( 0.5 ).add( n01( pn( X, Y, 20, 40, 49 ) ).sub( 0.5 ).mul( 0.25 ) ).add( fade.mul( 0.1 ) ), 0.0, 1.0 );
	const occ = float( 1 ).sub( lapLine.mul( 0.7 ) ).sub( band( ds, 0.0095, 0.011, 0.012, 0.016 ).mul( 0.35 ) );
	return { fields: vec4( height, rough, occ, 0 ), albedo: vec4( rust, rustVar, fade, grime ) };

}

// PALM THATCH - tile = 1 m along the eave (X) x 1 m up the slope (Y), 5 layered rows
function thatchGen( X, Y ) {

	const R = 5;
	// rows wave up and down along the eave and are unevenly spaced
	const rowWarp = pn( X, Y, 3, 1, 143 ).mul( 0.22 ).add( pn( X, Y, 7, 2, 145 ).mul( 0.1 ) );
	const rowF = Y.mul( R ).add( rowWarp );
	const k0 = floor( rowF );
	const row = ( k ) => {

		const kk = k.sub( floor( k.div( R ) ).mul( R ) );
		const warp = pnoise( vec2( X.mul( 7 ), kk.mul( 3.1 ) ), vec2( 7, 1000 ), float( 131 ) );
		// strands lean a little, in coherent groups, so they are not perfectly parallel
		const lean = pnoise( vec2( X.mul( 9 ), kk.mul( 5.3 ) ), vec2( 9, 1000 ), float( 133 ) );
		const s = rowF.sub( k ).max( - 0.5 );
		const sc = X.mul( 110 ).add( warp.mul( 1.3 ) ).add( lean.mul( s ).mul( 2.2 ) );
		const sid = floor( sc );
		const sidW = sid.sub( floor( sid.div( 110 ) ).mul( 110 ) );
		const sf = fract( sc );
		const rnd = hf( sidW.add( kk.mul( 211.0 ) ), kk.mul( 7.3 ).add( 1.0 ) );
		const lump = n01( pnoise( vec2( X.mul( 5 ), kk.mul( 2.3 ) ), vec2( 5, 1000 ), float( 137 ) ) );
		const lump2 = n01( pnoise( vec2( X.mul( 2 ), kk.mul( 1.7 ) ), vec2( 2, 1000 ), float( 139 ) ) );
		const hang = float( 0.06 ).add( lump.mul( 0.14 ) ).add( lump2.mul( 0.16 ) ).add( rnd.mul( rnd ).mul( 0.22 ) );
		return { tip: k.sub( hang ), sf, rnd };

	};

	const r0 = row( k0 ), r1 = row( k0.add( 1 ) ), r2 = row( k0.add( 2 ) );
	const in1 = step( r1.tip, rowF );
	const tip = mix( r0.tip, r1.tip, in1 );
	const nextTip = mix( r1.tip, r2.tip, in1 );
	const sf = mix( r0.sf, r1.sf, in1 );
	const rnd = mix( r0.rnd, r1.rnd, in1 );
	const s = rowF.sub( tip );
	const dNext = nextTip.sub( rowF );

	const bulge = sin( sf.mul( Math.PI ) );
	const tipZone = float( 1 ).sub( smoothstep( 0.0, 0.16, s ) );
	// frayed tips and gaps between strands reveal the darker layer underneath
	const gapW = mix( 0.43, 0.3, tipZone ).sub( hf( rnd, 2.2 ).mul( 0.06 ) );
	const gap = smoothstep( gapW, gapW.add( 0.05 ), abs( sf.sub( 0.5 ) ) );
	const fibre = pn( X, Y, 330, 6, 139 );
	const fibre2 = pn( X, Y, 160, 3, 141 );
	const height = float( 0.92 ).sub( min( s, 1.3 ).div( 1.3 ).mul( 0.55 ) ).add( bulge.mul( 0.08 ) ).add( fibre.mul( 0.03 ) ).sub( gap.mul( 0.32 ) );

	const golden = vec3( 0.42, 0.3, 0.13 ), pale = vec3( 0.5, 0.43, 0.27 ), brown = vec3( 0.24, 0.165, 0.09 ), grey = vec3( 0.34, 0.31, 0.26 );
	let col = mix( golden, pale, smoothstep( 0.25, 0.75, rnd ) );
	col = mix( col, brown, step( 0.85, rnd ).mul( 0.75 ) );
	col = mix( col, grey, smoothstep( 0.1, 0.5, hf( rnd, 9.1 ) ).mul( 0.4 ) );
	col = col.mul( n01( fibre ).mul( 0.38 ).add( 0.8 ) ).mul( n01( fibre2 ).mul( 0.2 ).add( 0.9 ) );
	col = mix( col, grey.mul( 1.18 ), tipZone.mul( 0.4 ) );
	col = col.mul( float( 1 ).sub( smoothstep( 0.55, 1.25, s ).mul( 0.3 ) ) );
	col = mix( col, vec3( 0.07, 0.055, 0.035 ), gap.mul( 0.85 ) );
	const occ = smoothstep( 0.0, 0.32, dNext ).mul( 0.7 ).add( 0.3 ).mul( bulge.mul( 0.25 ).add( 0.75 ) ).mul( float( 1 ).sub( gap.mul( 0.6 ) ) );
	const rough = float( 0.88 ).add( fibre.mul( 0.05 ) );
	return { fields: vec4( height, rough, occ, 0 ), albedo: vec4( col, tipZone ) };

}

// RUBBLE STONE WITH MORTAR - tile = 2 m x 2 m: warped, rounded stones with mottling and lichen
function stoneGen( X, Y ) {

	const wx = pf( X, Y, 4, 4, 2, 213 ).mul( 0.45 );
	const wy = pf( X, Y, 4, 4, 2, 215 ).mul( 0.45 );
	const W = pworley( vec2( X.mul( 7 ).add( wx ), Y.mul( 12 ).add( wy ) ), vec2( 7, 12 ), float( 201 ), float( 0.85 ) );
	const edge = W.y.sub( W.x );
	const mortar = float( 1 ).sub( smoothstep( 0.035, 0.085, edge ) );
	const dome = smoothstep( 0.02, 0.5, edge ).pow( 0.7 );
	const surf = pf( X, Y, 24, 24, 4, 203 );
	const pits = pw( X, Y, 70, 70, 207, 1 );
	const porous = step( 0.55, hf( W.z, 4.2 ) );
	const pitSize = mix( 0.1, 0.19, hf( pits.z, 1.7 ) );
	const pit = float( 1 ).sub( smoothstep( pitSize.mul( 0.6 ), pitSize, pits.x ) ).mul( step( 0.4, pits.z ) ).mul( porous.mul( 0.75 ).add( 0.25 ) );
	const sandy = pn( X, Y, 140, 140, 209 );
	const height = float( 0.2 ).add( dome.mul( 0.46 ) ).add( surf.mul( 0.11 ) ).add( pn( X, Y, 200, 200, 229 ).mul( 0.025 ) ).sub( pit.mul( 0.16 ) ).sub( mortar.mul( 0.12 ) ).add( sandy.mul( 0.02 ) );

	const id = W.z;
	// coral stone / limestone: warm beiges with the odd darker basalt and pale block
	const c0 = vec3( 0.55, 0.5, 0.41 ), c1 = vec3( 0.47, 0.45, 0.41 ), c2 = vec3( 0.42, 0.37, 0.3 ), c3 = vec3( 0.3, 0.29, 0.27 ), c4 = vec3( 0.64, 0.6, 0.52 );
	let col = mix( c0, c1, smoothstep( 0.2, 0.4, id ) );
	col = mix( col, c2, smoothstep( 0.5, 0.62, id ) );
	col = mix( col, c3, smoothstep( 0.8, 0.86, id ).mul( 0.8 ) );
	col = mix( col, c4, smoothstep( 0.9, 0.97, id ) );
	// mottling, veins and weathering inside each stone
	const mott = n01( pf( X, Y, 10, 10, 4, 217 ) );
	const vein = float( 1 ).sub( smoothstep( 0.0, 0.035, abs( pf( X, Y, 6, 6, 3, 219 ) ) ) ).mul( step( 0.6, hf( id, 6.6 ) ) );
	const speck = n01( pn( X, Y, 300, 300, 225 ) );
	const grain = n01( pn( X, Y, 120, 120, 227 ) );
	col = col.mul( mott.mul( 0.6 ).add( 0.68 ) ).mul( n01( surf ).mul( 0.3 ).add( 0.85 ) ).mul( speck.mul( 0.34 ).add( 0.83 ) ).mul( grain.mul( 0.22 ).add( 0.89 ) );
	col = col.mul( mix( 0.66, 1.0, smoothstep( 0.04, 0.28, edge ) ) );
	col = mix( col, col.mul( 1.25 ), vein.mul( 0.5 ) );
	// lichen: pale grey-green and dark specks
	const lich = smoothstep( 0.62, 0.8, n01( pf( X, Y, 8, 8, 3, 221 ) ) ).mul( smoothstep( 0.4, 0.6, n01( pn( X, Y, 60, 60, 223 ) ) ) );
	col = mix( col, vec3( 0.42, 0.44, 0.36 ), lich.mul( 0.45 ) );
	col = col.mul( float( 1 ).sub( pit.mul( 0.5 ) ) );
	col = mix( col, vec3( 0.5, 0.47, 0.41 ).mul( n01( sandy ).mul( 0.3 ).add( 0.85 ) ), mortar );
	col = col.mul( float( 1 ).sub( band( edge, 0.03, 0.06, 0.08, 0.16 ).mul( 0.3 ) ) );
	const plaster = clamp( n01( pf( X, Y, 3, 3, 5, 211 ) ).sub( mortar.mul( 0.08 ) ), 0.0, 1.0 );
	const rough = float( 0.84 ).add( surf.mul( 0.06 ) ).add( mortar.mul( 0.1 ) ).sub( lich.mul( 0.05 ) );
	const occ = float( 1 ).sub( pit.mul( 0.55 ) ).sub( mortar.mul( 0.3 ) );
	return { fields: vec4( height, rough, occ, 0 ), albedo: vec4( col, plaster ) };

}

// WORN METAL / PLASTIC - tile = 1 m x 1 m: rust blooms, pits, scratches, grime
function hardGen( X, Y ) {

	const base = n01( pf( X, Y, 4, 4, 5, 401 ) );
	const sp = pw( X, Y, 16, 16, 403, 1 );
	const spots = float( 1 ).sub( smoothstep( 0.1, 0.38, sp.x ) ).mul( step( 0.5, sp.z ) );
	const rust = clamp( base.mul( 0.8 ).add( spots.mul( 0.35 ) ).sub( 0.08 ), 0.0, 1.0 );
	const sMask = smoothstep( 0.2, 0.6, n01( pf( X, Y, 3, 3, 2, 413 ) ) );
	const s1 = float( 1 ).sub( smoothstep( 0.008, 0.03, abs( pn( X, Y, 3, 90, 405 ) ) ) );
	const s2 = float( 1 ).sub( smoothstep( 0.008, 0.03, abs( pn( X, Y, 90, 3, 407 ) ) ) );
	const scratch = max( s1, s2 ).mul( sMask );
	const grime = n01( pf( X, Y, 3, 3, 3, 409 ) );
	const pw2 = pw( X, Y, 90, 90, 411, 1 );
	const pit = float( 1 ).sub( smoothstep( 0.1, 0.28, pw2.x ) ).mul( smoothstep( 0.4, 0.8, rust ) );
	const bump = pn( X, Y, 40, 40, 415 ).mul( 0.04 ).mul( rust );
	const height = float( 0.5 ).sub( pit.mul( 0.3 ) ).sub( scratch.mul( 0.12 ) ).add( bump );
	const rough = clamp( float( 0.5 ).add( rust.mul( 0.3 ) ).sub( scratch.mul( 0.25 ) ).add( grime.mul( 0.08 ) ), 0.0, 1.0 );
	const occ = float( 1 ).sub( pit.mul( 0.5 ) );
	return { fields: vec4( height, rough, occ, 0 ), albedo: vec4( rust, scratch, grime, 0 ) };

}

// three-strand rope, tile = one circumference long x full circumference around
const ropeHeight = Fn( ( [ q ] ) => {

	const phi = q.x.add( q.y ).mul( 3.0 );
	const sf = fract( phi );
	const prof = sqrt( max( float( 1 ).sub( sf.mul( 2.0 ).sub( 1.0 ).pow( 2.0 ) ), 0.0 ) );
	const yarn = fract( sf.mul( 3.0 ).add( q.x.sub( q.y ).mul( 6.0 ) ) );
	const yprof = sqrt( max( float( 1 ).sub( yarn.mul( 2.0 ).sub( 1.0 ).pow( 2.0 ) ), 0.0 ) );
	const fuzz = pnoise( q.mul( 64.0 ), vec2( 64.0 ), float( 801 ) );
	return prof.mul( 0.8 ).add( yprof.mul( prof ).mul( 0.2 ) ).add( fuzz.mul( 0.03 ) );

} ).setLayout( { name: 'vlgRopeH', type: 'float', inputs: [ { name: 'q', type: 'vec2' } ] } );

function ropeGen( X, Y ) {

	const q = vec2( X, Y );
	const e = 1 / 256;
	const hx = ropeHeight( q.add( vec2( e, 0 ) ) ).sub( ropeHeight( q.sub( vec2( e, 0 ) ) ) );
	const hy = ropeHeight( q.add( vec2( 0, e ) ) ).sub( ropeHeight( q.sub( vec2( 0, e ) ) ) );
	const n = normalize( vec3( hx.mul( - 0.09 / ( 2 * e ) ), hy.mul( - 0.09 / ( 2 * e ) ), 1.0 ) );
	const h = ropeHeight( q );
	const fib = n01( pnoise( vec2( X.add( Y ).mul( 40 ), X.sub( Y ).mul( 160 ) ), vec2( 40, 160 ), float( 803 ) ) );
	const lum = float( 0.45 ).add( h.mul( 0.4 ) ).add( fib.mul( 0.15 ) );
	const ao = float( 0.3 ).add( smoothstep( 0.0, 0.7, h ).mul( 0.7 ) );
	return vec4( lum, ao, n.x.mul( 0.5 ).add( 0.5 ), n.y.mul( 0.5 ).add( 0.5 ) );

}

// knotted diamond fishing net, tile = 0.2 m (4 x 4 meshes)
const netHeight = Fn( ( [ q ] ) => {

	const a = q.x.add( q.y ).mul( 4.0 );
	const b = q.x.sub( q.y ).mul( 4.0 );
	const da = float( 0.5 ).sub( abs( fract( a ).sub( 0.5 ) ) );
	const db = float( 0.5 ).sub( abs( fract( b ).sub( 0.5 ) ) );
	const w = float( 0.055 );
	const la = sqrt( max( float( 1 ).sub( da.div( w ).pow( 2.0 ) ), 0.0 ) ).mul( fract( b.mul( 14.0 ) ).sub( 0.5 ).abs().mul( 0.25 ).add( 0.8 ) );
	const lb = sqrt( max( float( 1 ).sub( db.div( w ).pow( 2.0 ) ), 0.0 ) ).mul( fract( a.mul( 14.0 ) ).sub( 0.5 ).abs().mul( 0.25 ).add( 0.8 ) );
	const knot = sqrt( max( float( 1 ).sub( length( vec2( da, db ) ).div( 0.1 ).pow( 2.0 ) ), 0.0 ) ).mul( 1.35 );
	return max( max( la, lb ), knot );

} ).setLayout( { name: 'vlgNetH', type: 'float', inputs: [ { name: 'q', type: 'vec2' } ] } );

function netGen( X, Y ) {

	const q = vec2( X, Y );
	const e = 1 / 256;
	const h = netHeight( q );
	const hx = netHeight( q.add( vec2( e, 0 ) ) ).sub( netHeight( q.sub( vec2( e, 0 ) ) ) );
	const hy = netHeight( q.add( vec2( 0, e ) ) ).sub( netHeight( q.sub( vec2( 0, e ) ) ) );
	const n = normalize( vec3( hx.mul( - 0.012 / ( 2 * e ) ), hy.mul( - 0.012 / ( 2 * e ) ), 1.0 ) );
	const alpha = smoothstep( 0.0, 0.18, h );
	const shade = float( 0.45 ).add( h.min( 1.0 ).mul( 0.55 ) );
	return vec4( alpha, shade, n.x.mul( 0.5 ).add( 0.5 ), n.y.mul( 0.5 ).add( 0.5 ) );

}

// ---------------------------------------------------------------------------
// Derivation pass: Sobel normal + horizon AO from the height field.

function makeDeriveMaterial( fieldsTex, uTexel, uSlope, uAO ) {

	const m = new THREE.NodeMaterial();
	m.name = 'VillageBakeDerive';
	// all taps are created eagerly (Fn bodies run lazily at build time) so bake() can re-point
	// every one of them at the current set's field texture before the first draw
	const c = uv();
	const texNodes = [];
	const tap = ( dx, dy ) => {

		const n = texture( fieldsTex, dx === 0 && dy === 0 ? c : c.add( uTexel.mul( vec2( dx, dy ) ) ) );
		texNodes.push( n );
		return n;

	};

	const tl = tap( - 1, 1 ).x, t = tap( 0, 1 ).x, tr = tap( 1, 1 ).x;
	const l = tap( - 1, 0 ).x, r = tap( 1, 0 ).x;
	const bl = tap( - 1, - 1 ).x, b = tap( 0, - 1 ).x, br = tap( 1, - 1 ).x;
	const f = tap( 0, 0 );
	const hor = [];
	for ( let i = 0; i < 8; i ++ ) {

		const a = i / 8 * TAU + 0.3;
		for ( const rad of [ 3, 9 ] ) hor.push( [ tap( Math.cos( a ) * rad, Math.sin( a ) * rad ).x, rad ] );

	}

	m.userData.texNodes = texNodes;
	m.fragmentNode = Fn( () => {

		const dX = tr.add( r.mul( 2.0 ) ).add( br ).sub( tl.add( l.mul( 2.0 ) ).add( bl ) ).div( 8.0 );
		const dY = tl.add( t.mul( 2.0 ) ).add( tr ).sub( bl.add( b.mul( 2.0 ) ).add( br ) ).div( 8.0 );
		const n = normalize( vec3( dX.mul( uSlope.x ).negate(), dY.mul( uSlope.y ).negate(), 1.0 ) );
		// horizon based occlusion: 8 directions x 2 radii
		let occ = float( 0 );
		for ( const [ hs, rad ] of hor ) occ = occ.add( clamp( hs.sub( f.x ).mul( uAO.y ).div( rad ).mul( 1.6 ), 0.0, 1.0 ) );
		const ao = clamp( float( 1 ).sub( occ.div( 16.0 ).mul( uAO.x ) ), 0.0, 1.0 ).mul( f.z );
		return vec4( n.x.mul( 0.5 ).add( 0.5 ), n.y.mul( 0.5 ).add( 0.5 ), f.y, ao );

	} )();
	return m;

}

// ---------------------------------------------------------------------------
// Bake orchestration. All sets render with ONE uber fields shader (generator selected by a
// uniform) into identically formatted MRT targets, followed by ONE shared derivation shader:
// two pipelines in total, which keeps cold (uncached) shader compilation short.

function finalTexture( tex ) {

	tex.type = THREE.UnsignedByteType;
	tex.generateMipmaps = true;
	tex.minFilter = THREE.LinearMipmapLinearFilter;
	tex.magFilter = THREE.LinearFilter;
	tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
	tex.anisotropy = 8;
	tex.colorSpace = THREE.NoColorSpace;

}

// out: name of the albedo / mask map, nra: name of the normal-roughness-AO map
const SETS = [
	{ name: 'wood', gen: woodGen, w: 1024, h: 1024, mx: 2.0, my: 1.0, hs: 0.0022, ao: 1.2, out: 'woodA', nra: 'woodN' },
	{ name: 'paint', gen: paintGen, w: 512, h: 512, mx: 1.0, my: 0.5, hs: 0.0016, ao: 0.6, out: null, nra: 'paintN' },
	{ name: 'roof', gen: roofGen, w: 512, h: 1024, mx: 0.84, my: 1.68, hs: 0.022, ao: 0.9, out: 'roofA', nra: 'roofN' },
	{ name: 'thatch', gen: thatchGen, w: 1024, h: 1024, mx: 1.0, my: 1.0, hs: 0.035, ao: 1.3, out: 'thatchA', nra: 'thatchN' },
	{ name: 'stone', gen: stoneGen, w: 1024, h: 1024, mx: 2.0, my: 2.0, hs: 0.02, ao: 1.1, out: 'stoneA', nra: 'stoneN' },
	{ name: 'hard', gen: hardGen, w: 512, h: 512, mx: 1.0, my: 1.0, hs: 0.0012, ao: 0.8, out: 'hardA', nra: 'hardN' },
	{ name: 'grime', gen: grimeGen, w: 512, h: 512, out: 'grime', nra: null },
	{ name: 'rope', gen: ropeGen, w: 256, h: 256, out: 'rope', nra: null },
	{ name: 'net', gen: netGen, w: 256, h: 256, out: 'net', nra: null },
];

export class VillageTextures {

	constructor() {

		this.baked = false;
		this.bakeMs = 0;
		this.textures = {};
		this.jobs = [];
		this.quad = new THREE.QuadMesh();
		this._bytes = 0;

		// uber fields shader
		this.uJob = uniform( 0, 'int' );
		const outProp = property( 'vec4', 'vlgBakeOut' );
		const fieldsNode = Fn( () => {

			const t = uv();
			const fields = vec4( 0 ).toVar();
			let chain = null;
			SETS.forEach( ( set, k ) => {

				const body = () => {

					const r = set.gen( t.x, t.y );
					if ( r.isNode ) {

						outProp.assign( r );

					} else {

						fields.assign( r.fields );
						if ( r.albedo ) outProp.assign( r.albedo );

					}

				};

				chain = chain === null ? If( this.uJob.equal( k ), body ) : chain.ElseIf( this.uJob.equal( k ), body );

			} );
			return fields;

		} )();
		this.fieldsMat = new THREE.NodeMaterial();
		this.fieldsMat.name = 'VillageBakeFields';
		this.fieldsMat.fragmentNode = mrt( { fields: fieldsNode, albedo: outProp } );

		// shared derivation shader
		this.uTexel = uniform( new THREE.Vector2() );
		this.uSlope = uniform( new THREE.Vector2() );
		this.uAO = uniform( new THREE.Vector2() );
		// one shared texture object: its source image is re-pointed per job (see bake())
		this.deriveTex = new THREE.Texture();
		this.deriveMat = makeDeriveMaterial( this.deriveTex, this.uTexel, this.uSlope, this.uAO );

		SETS.forEach( ( set, k ) => {

			const rt = new THREE.RenderTarget( set.w, set.h, {
				count: 2, type: THREE.HalfFloatType, depthBuffer: false,
				minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
				wrapS: THREE.RepeatWrapping, wrapT: THREE.RepeatWrapping, generateMipmaps: false,
			} );
			rt.textures[ 0 ].name = 'fields';
			rt.textures[ 1 ].name = 'albedo';
			finalTexture( rt.textures[ 1 ] );
			if ( ! set.out ) rt.textures[ 1 ].generateMipmaps = false;
			if ( set.out ) {

				this.textures[ set.out ] = rt.textures[ 1 ];
				this._bytes += set.w * set.h * 4 * 4 / 3;

			}

			let nra = null;
			if ( set.nra ) {

				nra = new THREE.RenderTarget( set.w, set.h, { type: THREE.UnsignedByteType, depthBuffer: false } );
				finalTexture( nra.texture );
				this.textures[ set.nra ] = nra.texture;
				this._bytes += set.w * set.h * 4 * 4 / 3;

			}

			this.jobs.push( { set, index: k, rt, nra } );

		} );

		this.trigger = nodeObject( new VillageBakeNode( this ) );

	}

	// approximate GPU memory of the final (mipmapped) maps in bytes
	get bytes() {

		return this._bytes;

	}

	bake( renderer ) {

		if ( this.baked ) return;
		this.baked = true;
		const t0 = performance.now();
		const state = resetRendererState( renderer, {} );
		for ( const job of this.jobs ) {

			const { set } = job;
			this.uJob.value = job.index;
			renderer.setRenderTarget( job.rt );
			this.quad.material = this.fieldsMat;
			this.quad.render( renderer );
			if ( job.nra ) {

				for ( const n of this.deriveMat.userData.texNodes ) n.value = job.rt.textures[ 0 ];
				this.uTexel.value.set( 1 / set.w, 1 / set.h );
				this.uSlope.value.set( set.hs / ( set.mx / set.w ), set.hs / ( set.my / set.h ) );
				this.uAO.value.set( set.ao, set.hs / ( ( set.mx / set.w + set.my / set.h ) * 0.5 ) );
				renderer.setRenderTarget( job.nra );
				this.quad.material = this.deriveMat;
				this.quad.render( renderer );

			}

			// the passes above are already submitted: free the transient maps right away so
			// the peak GPU memory stays at the final maps + one field map
			job.rt.textures[ 0 ].dispose();
			if ( ! set.out ) job.rt.textures[ 1 ].dispose();

		}

		restoreRendererState( renderer, state );
		// every bake pass has been submitted: the bake pipelines are no longer needed
		this.fieldsMat.dispose();
		this.deriveMat.dispose();
		this.bakeMs = performance.now() - t0;

	}

	dispose() {

		for ( const job of this.jobs ) {

			job.rt.dispose();
			if ( job.nra ) job.nra.dispose();

		}

	}

}

// Zero-valued node embedded in the village materials. Its updateBefore hook receives the
// renderer on the first frame a village material is drawn and bakes all texture sets.
class VillageBakeNode extends THREE.Node {

	static get type() {

		return 'VillageBakeNode';

	}

	constructor( baker ) {

		super( 'float' );
		this.baker = baker;
		this.updateBeforeType = THREE.NodeUpdateType.FRAME;

	}

	updateBefore( frame ) {

		if ( ! this.baker.baked && frame.renderer ) this.baker.bake( frame.renderer );

	}

	generate( builder ) {

		return builder.generateConst( 'float', 0 );

	}

}
