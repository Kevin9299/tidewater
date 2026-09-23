import * as THREE from 'three/webgpu';
import {
	Fn, If, float, vec2, vec3, vec4, attribute, positionLocal, normalLocal, positionView, positionWorld, varyingProperty,
	sin, cos, mix, smoothstep, abs, fract, floor, length, max, min, select, mrt, cameraProjectionMatrix, cameraViewMatrix,
	uniformArray, int, fwidth, dot, normalView, positionViewDirection, sign, clamp, pow, luminance, atan, step,
	interleavedGradientNoise, screenCoordinate, Discard,
} from 'three/tsl';
import { G } from '../../core/Globals.js';
import { bayer4 } from '../../materials/LODFade.js';
import { physical } from '../../materials/Materials.js';
import { staticVelocity, staticVelocityMRT } from '../../post/CameraVelocity.js';
import { rotateQ, bumpNormal } from '../reef/ReefMaterials.js';
import { SPECIES, SKIN, PATTERN } from './FishSpecies.js';
import { PART } from './FishGeometry.js';

// The fish material (TSL), shared by the swimming fish (FishSchools) and the fish on the
// market stall, drying racks and cleaning tables (FishProps).
//
// Everything is procedural: a per-species table of colours and anatomy landmarks (eye, gill
// cover, lateral line, jaw hinge) and the species' markings in the fragment stage. Body skin:
// counter-shading from the dark back to the pale belly, overlapping scales in rows (colour and
// relief, faded out when smaller than a pixel), the lateral line, the edge of the gill cover,
// silvery guanine reflection with an iridescent sheen, wet specular. Fins: ray-striped
// membranes, darker and thinner toward the edge, lit through from behind. Eyes: pupil, iris
// with radial streaks and a glossy cornea (a dome at the nearest level of detail, painted
// further away). The vertex stage poses the model: a travelling swimming wave with pectoral
// sculling (swimming fish, with true motion vectors), or an open jaw and a curled / sagging
// body (props: fish lying on ice or hanging from a hook).

const aData = attribute( 'aData', 'vec4' );
const ROWS = 8; // vec4 rows per species in the table

// species order = pattern ids
const NAMES = Object.keys( PATTERN ).sort( ( a, b ) => PATTERN[ a ] - PATTERN[ b ] );

function buildTable() {

	const rows = [];
	const lin = ( hex ) => new THREE.Color( hex );
	for ( const name of NAMES ) {

		const S = SPECIES[ name ], K = SKIN[ name ];
		const b = lin( K.back ), f = lin( K.flank ), be = lin( K.belly ), fi = lin( K.fin ), e = lin( K.edge ), ir = lin( S.iris );
		const L = S.body;
		rows.push(
			// guanine reflection: a metal-like specular layer, kept moderate so that silvery fish
			// stay bright in the dim underwater environment lighting
			new THREE.Vector4( b.r, b.g, b.b, S.metal * 0.55 ),
			new THREE.Vector4( f.r, f.g, f.b, S.irid ),
			new THREE.Vector4( be.r, be.g, be.b, K.rough ),
			new THREE.Vector4( fi.r, fi.g, fi.b, S.mouth.tip ),
			new THREE.Vector4( e.r, e.g, e.b, S.scales ),
			new THREE.Vector4( ir.r, ir.g, ir.b, S.scaleVis ),
			new THREE.Vector4( 0.5 - S.eye.u * L, S.eye.y, S.eye.r, 0.5 - S.opercle * L ),
			new THREE.Vector4( S.lateral, S.arch, 0.5 - S.mouth.corner * L, S.mouth.y ),
		);

	}

	return uniformArray( rows, 'vec4' );

}

let _table = null;
export const skinTable = () => _table || ( _table = buildTable() );

// varyings shared by the vertex and fragment stages
function varyings() {

	return {
		local: varyingProperty( 'vec3', 'vFishLocal' ), // rest pose position (model units)
		data: varyingProperty( 'vec4', 'vFishData' ), // aData
		info: varyingProperty( 'vec4', 'vFishInfo' ), // pattern, seed, length (m), 0
		flags: varyingProperty( 'vec4', 'vFishFlags' ), // props: cloudy eye, wet, dried, blood
		delta: varyingProperty( 'vec3', 'vFishDelta' ), // swimming fish: world motion since the last frame
		fade: varyingProperty( 'vec2', 'vFishFade' ), // level-of-detail cross-fade: share, outgoing
	};

}

const partOf = ( d ) => floor( d.y.add( 0.01 ) );

// Level-of-detail cross-fade (see ReefBatch fade channel, materials/LODFade.js): the incoming level
// keeps the Bayer cells below the fade, the outgoing one the others.
const fadeDiscard = ( f ) => {

	const t = bayer4();
	Discard( select( f.y.greaterThan( 0.5 ), t.lessThan( f.x ), t.greaterThanEqual( f.x ) ) );

};

// record index of this vertex's instance (main or fade channel)
const recordOf = ( batch, V, fade ) => {

	if ( ! fade ) return batch.recordIndex();
	const e = batch.fadeEntry();
	V.fade.assign( vec2( e.fade, e.outgoing ) );
	return e.index;

};
const jawOf = ( d ) => max( fract( d.y.add( 0.01 ) ).sub( 0.01 ), 0 ).div( 0.9 );

// ---------------------------------------------------------------------------
// vertex stages

// Swimming fish. Record: r0 = ( position, length ), r1 = orientation, r2 = ( wave phase,
// amplitude, turning bend, pattern + seed * 0.9 ), r3 = ( world motion since the last frame,
// phase change )
function swimVertex( batch, V, fade ) {

	return Fn( () => {

		const rec = batch.record( recordOf( batch, V, fade ) );
		const r0 = rec[ 0 ], q = rec[ 1 ], r2 = rec[ 2 ], r3 = rec[ 3 ];
		const u = aData.x;
		const part = partOf( aData );
		const p = positionLocal.toVar();
		V.local.assign( p );
		V.data.assign( aData );
		V.info.assign( vec4( floor( r2.w ), fract( r2.w ), r0.w, 0 ) );
		V.flags.assign( vec4( 0, 0, 0, 0 ) );
		// Deformation as a function of the phase (evaluated for this and the previous frame):
		// fish: travelling body wave (amplitude grows toward the tail) plus the turning bend,
		// sculling pectorals; rays: the disc margins undulate (stingray) or flap (eagle ray);
		// turtle: the front flippers stroke, the hind ones paddle.
		const pattern = floor( r2.w );
		const env = u.mul( u ).mul( 0.85 ).add( 0.08 ).mul( smoothstep( 0.0, 0.25, u ).mul( 0.7 ).add( 0.3 ) );
		const bend = r2.z.mul( u.mul( u ) );
		const isDisc = part.equal( PART.DISC ), isFlip = part.equal( PART.FLIPPER );
		const turtle = part.greaterThan( PART.WHIP + 0.5 );
		const eagle = pattern.equal( PATTERN.eagleRay );
		const side = aData.z; // rays: distance from the midline; flippers: along the flipper
		const offset = ( ph ) => {

			const lat = sin( ph.sub( u.mul( 5.6 ) ) ).mul( env ).mul( r2.y ).add( bend );
			const flap = select( part.equal( PART.PECTORAL ), sin( ph.mul( 0.7 ).add( 1.3 ) ).mul( aData.z ).mul( 0.035 ), 0 );
			const fish = vec3( lat.add( flap.mul( sign( p.x ) ) ), 0, 0 );
			const k = select( eagle, 1.2, 8.0 );
			const disc = vec3( 0, sin( ph.sub( u.mul( k ) ) ).mul( pow( side, 1.6 ) ).mul( r2.y ), 0 );
			const front = aData.w.lessThan( 1.5 );
			const stroke = vec3( 0, sin( ph ).mul( select( front, 0.3, 0.07 ) ), cos( ph ).mul( select( front, 0.14, 0.0 ) ) ).mul( side );
			return select( isDisc, disc, select( isFlip, stroke, select( turtle, vec3( 0 ), fish ) ) );

		};

		const off = offset( r2.x );
		p.addAssign( off );
		V.delta.assign( rotateQ( q, off.sub( offset( r2.x.sub( r3.w ) ) ) ).mul( r0.w ).add( r3.xyz ) );
		normalLocal.assign( rotateQ( q, normalLocal ) );
		return rotateQ( q, p.mul( r0.w ) ).add( r0.xyz );

	} )();

}

// Fish props. Record: r0 = ( position, length ), r1 = orientation, r2 = ( pattern + seed * 0.9,
// lateral curl, dorso-ventral sag (1 / body length), jaw opening (rad) ), r3 = ( cloudy eye, wet,
// dried, blood )
function propVertex( batch, V, table, fade ) {

	return Fn( () => {

		const rec = batch.record( recordOf( batch, V, fade ) );
		const r0 = rec[ 0 ], q = rec[ 1 ], r2 = rec[ 2 ], r3 = rec[ 3 ];
		const pattern = floor( r2.x );
		const p = positionLocal.toVar(), n = normalLocal.toVar();
		V.local.assign( p );
		V.data.assign( aData );
		V.info.assign( vec4( pattern, fract( r2.x ), r0.w, 0 ) );
		V.flags.assign( r3 );
		V.delta.assign( vec3( 0 ) );

		// lower jaw: rotates down about the hinge at the corner of the mouth
		const hinge = table.element( int( pattern ).mul( ROWS ).add( 7 ) ).zw;
		const a = r2.w.mul( jawOf( aData ) );
		const ca = cos( a ), sa = sin( a );
		const dy = p.y.sub( hinge.y ).toVar(), dz = p.z.sub( hinge.x ).toVar();
		p.y.assign( hinge.y.add( dy.mul( ca ) ).sub( dz.mul( sa ) ) );
		p.z.assign( hinge.x.add( dy.mul( sa ) ).add( dz.mul( ca ) ) );
		const ny = n.y.mul( ca ).sub( n.z.mul( sa ) ).toVar(), nz = n.y.mul( sa ).add( n.z.mul( ca ) ).toVar();
		n.y.assign( ny );
		n.z.assign( nz );

		// body bent along circular arcs about its middle: sideways (curl), then up / down (sag)
		const k1 = r2.y.add( select( r2.y.greaterThanEqual( 0 ), 1e-4, - 1e-4 ) );
		const t1 = k1.mul( p.z );
		const c1 = cos( t1 ), s1 = sin( t1 ), h1 = sin( t1.mul( 0.5 ) );
		const x1 = h1.mul( h1 ).mul( 2 ).div( k1 ).add( p.x.mul( c1 ) );
		const z1 = s1.div( k1 ).sub( p.x.mul( s1 ) );
		const nx1 = n.x.mul( c1 ).add( n.z.mul( s1 ) ), nz1 = n.z.mul( c1 ).sub( n.x.mul( s1 ) );
		const k2 = r2.z.add( select( r2.z.greaterThanEqual( 0 ), 1e-4, - 1e-4 ) );
		const t2 = k2.mul( z1 );
		const c2 = cos( t2 ), s2 = sin( t2 ), h2 = sin( t2.mul( 0.5 ) );
		const y2 = h2.mul( h2 ).mul( 2 ).div( k2 ).add( p.y.mul( c2 ) );
		const z2 = s2.div( k2 ).sub( p.y.mul( s2 ) );
		const ny2 = n.y.mul( c2 ).add( nz1.mul( s2 ) ), nz2 = nz1.mul( c2 ).sub( n.y.mul( s2 ) );
		normalLocal.assign( rotateQ( q, vec3( nx1, ny2, nz2 ) ) );
		return rotateQ( q, vec3( x1, y2, z2 ).mul( r0.w ) ).add( r0.xyz );

	} )();

}

// ---------------------------------------------------------------------------
// fragment stage

const hash = ( p ) => fract( sin( dot( p, vec2( 127.1, 311.7 ) ) ).mul( 43758.5453 ) );

// value noise 2D (cheap, for blotches and skin variation)
const vnoise = ( p ) => {

	const i = floor( p ), f = fract( p );
	const w = f.mul( f ).mul( float( 3 ).sub( f.mul( 2 ) ) );
	const a = hash( i ), b = hash( i.add( vec2( 1, 0 ) ) ), c = hash( i.add( vec2( 0, 1 ) ) ), d = hash( i.add( vec2( 1, 1 ) ) );
	return mix( mix( a, b, w.x ), mix( c, d, w.x ), w.y );

};

// posterior edge of the gill cover (local z) at height fraction h: convex backward, sweeping
// forward under the throat
const opercleEdge = ( zOp, h ) => zOp.sub( float( 0.028 ).mul( float( 1 ).sub( h.mul( h ) ) ) ).add( smoothstep( - 0.35, - 1.0, h ).mul( 0.07 ) );
const opercleMask = ( h ) => smoothstep( - 0.98, - 0.9, h ).mul( float( 1 ).sub( smoothstep( 0.45, 0.62, h ) ) );

const band = ( x, center, width, soft ) => float( 1 ).sub( smoothstep( width, width.add( soft ), abs( x.sub( center ) ) ) );

function surface( V, table, prop, lodFade ) {

	const albedo = vec3( 0.5 ).toVar( 'fishAlbedo' );
	const rough = float( 0.4 ).toVar( 'fishRough' );
	const metal = float( 0 ).toVar( 'fishMetal' );
	const transl = float( 0 ).toVar( 'fishTransl' );
	const coat = float( 0 ).toVar( 'fishCoat' );
	const spec = float( 0.5 ).toVar( 'fishSpec' );
	const row = ( pattern, k ) => table.element( int( pattern ).mul( ROWS ).add( k ) );

	// shared by colour and relief
	const common = () => {

		const D = V.data, Lp = V.local, I = V.info;
		const pattern = I.x.add( 0.5 ).floor();
		const part = partOf( D );
		const scaleSize = row( pattern, 4 ).w, scaleVis = row( pattern, 5 ).w;
		// scale rows: posterior margins are arcs, rows offset by half a scale
		const ss = max( scaleSize, 0.004 );
		const warp = sin( Lp.z.mul( 23 ).add( D.z.mul( 31 ) ) ).mul( 0.35 ).add( sin( Lp.z.mul( 41 ).sub( D.z.mul( 17 ) ) ).mul( 0.25 ) );
		const a = float( 0.5 ).sub( Lp.z ).div( ss ).add( warp ), b = D.z.div( ss.mul( 0.8 ) ).add( warp.mul( 0.6 ) );
		const rowI = floor( b );
		const fb = fract( b ).mul( 2 ).sub( 1 );
		const f = fract( a.add( rowI.mul( 0.5 ) ).add( fb.mul( fb ).mul( 0.32 ) ) );
		// pixel footprint (m) against the scale size (m): fade out sub-pixel detail
		const px = length( fwidth( positionView ) );
		const fade = float( 1 ).sub( smoothstep( 0.25, 0.7, px.div( ss.mul( I.z ) ) ) ).mul( select( scaleSize.greaterThan( 0.001 ), 1, 0 ) );
		return { D, Lp, I, pattern, part, a, b, f, px, fade, scaleVis };

	};

	const height = Fn( () => {

		const { D, Lp, I, pattern, part, f, fade, scaleVis } = common();
		const isBody = part.equal( PART.BODY );
		const L = I.z;
		// scales: each rises toward its free posterior margin
		const sc = smoothstep( 0.0, 0.9, f ).mul( float( 1 ).sub( smoothstep( 0.9, 1.0, f ) ) ).mul( fade ).mul( scaleVis );
		// gill cover: raised in front of its edge
		const eyeOp = row( pattern, 6 );
		const h = D.w;
		const zE = opercleEdge( eyeOp.w, h );
		const onOp = opercleMask( h );
		const op = smoothstep( - 0.004, 0.004, Lp.z.sub( zE ) ).mul( onOp );
		const bodyH = sc.mul( 0.0007 ).add( op.mul( 0.0025 ) );
		// fin rays: ridges
		const isFin = part.greaterThan( 0.5 ).and( part.lessThan( 7.5 ) );
		const rd = abs( fract( D.w.add( 0.5 ) ).sub( 0.5 ) );
		const ray = float( 1 ).sub( smoothstep( 0.0, 0.25, rd ) ).mul( 0.0004 );
		return select( isBody, bodyH, select( isFin, ray, 0 ) ).mul( L );

	} )();

	const color = Fn( () => {

		if ( lodFade ) fadeDiscard( V.fade );
		const { D, Lp, I, pattern, part, a, b, f, px, fade, scaleVis } = common();
		const seed = I.y.toVar(), L = I.z.toVar();
		const pat = pattern.toVar();
		const P = part.toVar();
		const u = D.x.toVar(), h = D.w.toVar(), s = D.z.toVar();
		const z = Lp.z.toVar(), y = Lp.y.toVar();
		const t = D.z.toVar(), w = D.w.toVar(); // fins: along / across the rays
		const isBody = P.equal( PART.BODY ).toVar();
		const isFin = P.greaterThan( 0.5 ).and( P.lessThan( 7.5 ) ).toVar();
		const bodyK = select( isBody, 1, 0 ).toVar();
		const r0 = row( pat, 0 ), r1 = row( pat, 1 ), r2 = row( pat, 2 ), r3 = row( pat, 3 ), r4 = row( pat, 4 ), r5 = row( pat, 5 ), r6 = row( pat, 6 ), r7 = row( pat, 7 );
		const back = r0.xyz.toVar(), flank = r1.xyz.toVar(), belly = r2.xyz.toVar();
		const finC = r3.xyz.toVar(), edgeC = r4.xyz.toVar(), irisC = r5.xyz.toVar();
		const eye = r6.toVar(), lat = r7.toVar();
		const flags = V.flags.toVar();
		const n1 = vnoise( vec2( z, y ).mul( 38 ).add( seed.mul( 17 ) ) ).toVar();
		const n2 = vnoise( vec2( z, s ).mul( 11 ).add( seed.mul( 5 ) ) ).toVar();
		const fwW = fwidth( w ).toVar(), fwH = fwidth( h ).toVar();

		// ---- counter-shading
		const tBack = smoothstep( 0.2, 0.75, h ).toVar();
		const tBelly = float( 1 ).sub( smoothstep( - 0.7, - 0.1, h ) ).toVar();
		const c = mix( mix( flank, back, tBack ), belly, tBelly ).toVar();
		const silver = float( 1 ).sub( tBack.mul( 0.75 ) ).toVar(); // guanine reflection weight
		metal.assign( r0.w.mul( silver ).mul( bodyK ) );
		rough.assign( r2.w );
		// scales: a thin shadow line under each free margin, the exposed field slightly brighter
		// toward the margin
		const scaleShade = smoothstep( 0.3, 0.9, f ).mul( fade ).mul( scaleVis ).toVar();
		const pocket = smoothstep( 0.88, 0.97, f ).mul( float( 1 ).sub( smoothstep( 0.97, 1.0, f ) ) ).mul( fade ).mul( scaleVis );
		const cellK = hash( vec2( floor( a.add( floor( b ).mul( 0.5 ) ) ), floor( b ) ) ).sub( 0.5 ).mul( 0.1 ).mul( fade ).mul( scaleVis );
		c.mulAssign( mix( float( 1 ), float( 0.96 ).add( scaleShade.mul( 0.07 ) ).sub( pocket.mul( 0.14 ) ).add( cellK ), bodyK ) );
		c.mulAssign( mix( 1, n1.mul( 0.14 ).add( 0.93 ), bodyK ) );

		// ---- fins: ray-striped membranes, darker and thinner toward the edge
		If( isFin.and( P.notEqual( PART.FINLET ) ), () => {

			const fin = mix( finC, edgeC, smoothstep( 0.4, 1.0, t ) ).toVar();
			const rd = abs( fract( w.add( 0.5 ) ).sub( 0.5 ) );
			const rayW = select( P.equal( PART.DORSAL1 ), 0.1, 0.06 );
			const ray = float( 1 ).sub( smoothstep( rayW, fwW.mul( 1.2 ).add( rayW ).add( 0.04 ), rd ) ).mul( float( 1 ).sub( smoothstep( 0.2, 0.6, fwW ) ) );
			fin.mulAssign( mix( 0.9, 1.06, ray ) );
			// thicker and darker where the fin joins the body, thinnest at the edge
			fin.mulAssign( smoothstep( 0.0, 0.15, t ).mul( 0.2 ).add( 0.8 ) );
			c.assign( fin );
			transl.assign( mix( 0.8, 0.55, ray ).mul( smoothstep( 0.0, 0.3, t ).mul( 0.4 ).add( 0.6 ) ) );
			// paired fins are pale and nearly clear
			const paired = P.equal( PART.PECTORAL ).or( P.equal( PART.PELVIC ) );
			c.assign( select( paired, mix( c, flank.mul( 1.1 ).add( 0.05 ), 0.45 ), c ) );
			rough.assign( 0.4 );

		} );

		// ---- species markings (body; some on fins)
		If( pat.equal( PATTERN.silverside ), () => {

			// silver lateral band with a dark upper edge; translucent green back
			const bandK = band( h, float( 0.02 ), float( 0.1 ), fwH.add( 0.05 ) ).mul( bodyK );
			c.assign( mix( c, vec3( 0.78, 0.82, 0.84 ), bandK.mul( 0.8 ) ) );
			c.assign( mix( c, vec3( 0.12, 0.2, 0.2 ), band( h, float( 0.14 ), float( 0.015 ), fwH.add( 0.02 ) ).mul( bodyK ).mul( 0.6 ) ) );
			metal.addAssign( bandK.mul( 0.25 ) );

		} ).ElseIf( pat.equal( PATTERN.chromis ), () => {

			// dark margins on the tail lobes, azure line from the snout through the eye
			const lobe = select( P.equal( PART.CAUDAL ), smoothstep( 5.5, 7.5, abs( w.sub( 8 ) ) ), 0 );
			c.assign( mix( c, vec3( 0.01, 0.015, 0.03 ), lobe ) );
			const lineK = band( y.sub( z.sub( eye.x ).mul( 0.35 ) ), eye.y.add( 0.015 ), float( 0.004 ), float( 0.003 ) ).mul( smoothstep( eye.x.sub( 0.02 ), eye.x.add( 0.05 ), z ) ).mul( bodyK );
			c.assign( mix( c, vec3( 0.3, 0.6, 0.95 ), lineK.mul( 0.7 ) ) );

		} ).ElseIf( pat.equal( PATTERN.grunt ), () => {

			// French grunt: yellow with oblique blue-silver stripes (straight above the lateral
			// line); bluestriped grunt: straight blue stripes. Red mouth.
			const blue = fract( seed.mul( 3.7 ) ).lessThan( 0.4 );
			const above = smoothstep( 0.35, 0.45, h );
			const slope = select( blue, 0, mix( 0.45, 0.0, above ) );
			const sv = sin( y.sub( z.mul( slope ) ).mul( select( blue, 190, 150 ) ) );
			const stripe = smoothstep( 0.45, 0.8, sv ).mul( bodyK ).mul( float( 1 ).sub( tBelly.mul( 0.7 ) ) );
			const lineC = select( blue, vec3( 0.12, 0.26, 0.55 ), vec3( 0.52, 0.6, 0.7 ) );
			c.assign( mix( c, lineC, stripe.mul( select( blue, 0.9, 0.7 ) ) ) );

		} ).ElseIf( pat.equal( PATTERN.yellowtail ), () => {

			// yellow stripe from the snout widening into the yellow tail; yellow spots on the back
			const wS = mix( 0.006, 0.035, smoothstep( 0.1, - 0.25, z ) );
			const stripe = band( y.sub( 0.004 ), float( 0 ), wS, float( 0.004 ) ).mul( bodyK );
			const q = vec2( z, y ).mul( 70 );
			const cell = floor( q );
			const j = vec2( hash( cell.add( 3.1 ) ), hash( cell.add( 7.7 ) ) ).sub( 0.5 ).mul( 0.5 );
			const rr = hash( cell.add( 1.3 ) ).mul( 0.14 ).add( 0.1 );
			const spots = float( 1 ).sub( smoothstep( rr, rr.add( 0.12 ), length( fract( q ).sub( 0.5 ).sub( j ) ) ) ).mul( step( 0.4, hash( cell.add( seed ) ) ) ).mul( tBack ).mul( bodyK );
			c.assign( mix( c, vec3( 0.85, 0.62, 0.05 ), max( stripe, spots.mul( 0.8 ) ) ) );
			c.assign( mix( c, vec3( 0.86, 0.66, 0.06 ), select( P.equal( PART.CAUDAL ), 1, 0 ) ) );

		} ).ElseIf( pat.equal( PATTERN.tang ), () => {

			// fine dark wavy lines, pale scalpel at the tail base
			const lines = smoothstep( 0.75, 0.95, sin( y.mul( 170 ).add( z.mul( 30 ) ).add( n1.mul( 3 ) ) ) ).mul( 0.3 ).mul( bodyK );
			c.mulAssign( float( 1 ).sub( lines ) );
			const spine = float( 1 ).sub( smoothstep( 0.01, 0.02, length( vec2( z.add( 0.27 ), y.mul( 1.5 ) ) ) ) ).mul( bodyK );
			c.assign( mix( c, vec3( 0.85, 0.8, 0.55 ), spine ) );
			c.assign( mix( c, edgeC, select( isFin, smoothstep( 0.8, 1.0, t ), 0 ) ) );

		} ).ElseIf( pat.equal( PATTERN.sergeant ), () => {

			// five black bars
			const bars = smoothstep( 0.35, 0.6, sin( z.sub( 0.28 ).mul( 32 ) ) ).mul( smoothstep( - 0.33, - 0.26, z ) ).mul( float( 1 ).sub( smoothstep( 0.24, 0.3, z ) ) ).mul( float( 1 ).sub( tBelly.mul( 0.8 ) ) );
			c.assign( mix( c, vec3( 0.02, 0.02, 0.025 ), bars.mul( select( isBody, 0.92, select( isFin, 0.4, 0 ) ) ) ) );

		} ).ElseIf( pat.equal( PATTERN.wrasse ), () => {

			// bluehead wrasse: yellow initial phase with a dark midlateral stripe; blue-headed males
			const male = fract( seed.mul( 7.1 ) ).lessThan( 0.15 );
			const stripe = band( h, float( 0.05 ), float( 0.1 ), fwH.add( 0.04 ) ).mul( bodyK ).mul( smoothstep( 0.25, 0.1, z ) );
			const female = mix( c, vec3( 0.04, 0.04, 0.03 ), stripe.mul( 0.9 ) );
			const head = smoothstep( 0.12, 0.17, z );
			const collar = band( z, float( 0.13 ), float( 0.012 ), float( 0.006 ) );
			const maleC = mix( mix( vec3( 0.1, 0.42, 0.28 ), vec3( 0.05, 0.14, 0.62 ), head ), vec3( 0.02, 0.02, 0.02 ), collar.mul( bodyK ) );
			c.assign( select( male, maleC, female ) );

		} ).ElseIf( pat.equal( PATTERN.parrot ), () => {

			// stoplight (terminal phase: green, pink / orange marks, yellow spot on the gill cover)
			// or queen parrotfish (blue-green, orange-pink marks around the mouth)
			const queen = fract( seed.mul( 4.3 ) ).lessThan( 0.4 );
			const base = select( queen, vec3( 0.06, 0.34, 0.42 ), vec3( 0.1, 0.42, 0.26 ) );
			c.assign( mix( c, base.mul( mix( 0.8, 1.1, scaleShade ) ), bodyK.mul( 0.75 ) ) );
			const mark = band( y.sub( z.sub( 0.3 ).mul( 0.4 ) ), float( - 0.03 ), float( 0.008 ), float( 0.008 ) ).mul( smoothstep( 0.18, 0.35, z ) ).mul( bodyK );
			c.assign( mix( c, select( queen, vec3( 0.75, 0.42, 0.28 ), vec3( 0.85, 0.45, 0.32 ) ), mark ) );
			const spot = float( 1 ).sub( smoothstep( 0.01, 0.02, length( vec2( z.sub( eye.w ).sub( 0.02 ), y.sub( 0.05 ) ) ) ) ).mul( bodyK );
			c.assign( mix( c, vec3( 0.88, 0.72, 0.12 ), spot.mul( select( queen, 0, 1 ) ) ) );

		} ).ElseIf( pat.equal( PATTERN.angel ), () => {

			// French angelfish: black, yellow rims on the scales, yellow face and eye ring
			const rims = smoothstep( 0.72, 0.95, f ).mul( fade.max( 0.35 ) ).mul( bodyK );
			c.assign( mix( c, vec3( 0.62, 0.48, 0.06 ), rims.mul( 0.6 ) ) );
			const face = smoothstep( 0.4, 0.43, z ).mul( bodyK );
			c.assign( mix( c, vec3( 0.55, 0.45, 0.2 ), face.mul( 0.6 ) ) );

		} ).ElseIf( pat.equal( PATTERN.barracuda ), () => {

			// dark oblique bars on the upper flank, black blotches on the lower rear flank
			const bars = smoothstep( 0.35, 0.8, sin( z.mul( 58 ).add( h.mul( 1.5 ) ).add( n1 ) ) ).mul( smoothstep( 0.2, 0.55, h ) ).mul( bodyK );
			c.mulAssign( float( 1 ).sub( bars.mul( 0.45 ) ) );
			const bl = smoothstep( 0.6, 0.78, n2 ).mul( smoothstep( 0.1, - 0.25, z ) ).mul( float( 1 ).sub( smoothstep( - 0.3, 0.2, h ) ) ).mul( bodyK );
			c.assign( mix( c, vec3( 0.03, 0.03, 0.035 ), bl.mul( 0.9 ) ) );
			c.assign( mix( c, vec3( 0.75, 0.78, 0.8 ), select( P.equal( PART.CAUDAL ), smoothstep( 0.85, 1.0, t ).mul( smoothstep( 5, 7, abs( w.sub( 8 ) ) ) ), 0 ) ) );

		} ).ElseIf( pat.equal( PATTERN.redSnapper ), () => {

			// rose red back fading to a silvery pink belly; rows of scales show as fine oblique lines
			const rows = smoothstep( 0.6, 0.95, sin( y.mul( 210 ).add( z.mul( 120 ) ) ) ).mul( fade.max( 0.3 ) ).mul( bodyK ).mul( 0.15 );
			c.mulAssign( float( 1 ).sub( rows ) );

		} ).ElseIf( pat.equal( PATTERN.grouper ), () => {

			// Nassau grouper: dark brown bars, a band from the snout through the eye, a black saddle
			// on the tail stalk, dark spots around the eye
			const zz = float( 0.5 ).sub( z );
			const wob = n2.sub( 0.5 ).mul( 0.03 );
			const bars = smoothstep( 0.2, 0.6, sin( zz.add( wob ).mul( 34 ).sub( 1.2 ) ) ).mul( smoothstep( 0.3, 0.38, zz ) ).mul( smoothstep( 0.86, 0.78, zz ) ).mul( float( 1 ).sub( tBelly.mul( 0.85 ) ) );
			const stripe = band( y.sub( eye.y ).sub( z.sub( eye.x ).mul( 0.25 ) ), float( 0 ), float( 0.008 ), float( 0.006 ) ).mul( smoothstep( eye.x.sub( 0.08 ), eye.x, z ) ).mul( smoothstep( 0.5, 0.45, z ) );
			const saddle = smoothstep( 0.4, 0.7, h ).mul( band( zz, float( 0.8 ), float( 0.025 ), float( 0.01 ) ) );
			const spots = smoothstep( 0.72, 0.85, vnoise( vec2( z, y ).mul( 160 ).add( seed.mul( 3 ) ) ) ).mul( smoothstep( eye.x.sub( 0.12 ), eye.x, z ) );
			const dark = max( max( bars.mul( 0.75 ), stripe.mul( 0.8 ) ), max( saddle, spots.mul( 0.7 ) ) ).mul( bodyK );
			c.assign( mix( c, vec3( 0.2, 0.13, 0.08 ), dark ) );
			const pale = smoothstep( 0.86, 0.93, vnoise( vec2( z, y ).mul( 150 ).add( 9 ) ) ).mul( bodyK ).mul( 0.2 );
			c.assign( mix( c, vec3( 0.85, 0.8, 0.72 ), pale ) );

		} ).ElseIf( pat.equal( PATTERN.tuna ), () => {

			// blackfin tuna: sharp dark back, bronze band, pale bars on the belly, dusky yellow finlets
			const bronze = band( h, float( 0.28 ), float( 0.05 ), fwH.add( 0.06 ) ).mul( smoothstep( 0.3, 0.2, z ) ).mul( bodyK );
			c.assign( mix( c, vec3( 0.42, 0.34, 0.14 ), bronze.mul( 0.6 ) ) );
			c.assign( mix( c, back, smoothstep( 0.28, 0.4, h ).mul( bodyK ) ) );
			const bars = smoothstep( 0.6, 0.9, sin( z.mul( 95 ) ) ).mul( smoothstep( 0.0, - 0.3, h ) ).mul( smoothstep( 0.2, 0.1, z ) ).mul( bodyK );
			c.assign( mix( c, vec3( 0.85, 0.88, 0.9 ), bars.mul( 0.35 ) ) );
			c.assign( mix( c, vec3( 0.55, 0.48, 0.16 ), select( P.equal( PART.FINLET ), 0.85, 0 ) ) );

		} ).ElseIf( pat.equal( PATTERN.mahi ), () => {

			// mahi-mahi: blue-green back, golden flanks with scattered blue spots
			const cell = floor( vec2( z, y ).mul( 55 ) );
			const fc = fract( vec2( z, y ).mul( 55 ) ).sub( 0.5 );
			const spots = float( 1 ).sub( smoothstep( 0.16, 0.3, length( fc ) ) ).mul( step( 0.5, hash( cell.add( seed.mul( 7 ) ) ) ) ).mul( bodyK ).mul( float( 1 ).sub( tBelly ) );
			c.assign( mix( c, vec3( 0.08, 0.22, 0.5 ), spots.mul( 0.75 ) ) );
			c.assign( mix( c, vec3( 0.2, 0.5, 0.3 ), smoothstep( 0.0, 0.5, h ).mul( bodyK ).mul( 0.35 ) ) );

		} ).ElseIf( pat.equal( PATTERN.mullet ), () => {

			// faint dark stripes along the scale rows of the upper flank
			const lines = smoothstep( 0.7, 0.95, sin( s.mul( 280 ) ) ).mul( smoothstep( - 0.1, 0.3, h ) ).mul( bodyK ).mul( 0.25 );
			c.mulAssign( float( 1 ).sub( lines ) );

		} ).ElseIf( pat.equal( PATTERN.needlefish ), () => {

			// dark blue lateral stripe, dark beak
			const stripe = band( h, float( 0.0 ), float( 0.06 ), fwH.add( 0.05 ) ).mul( bodyK );
			c.assign( mix( c, vec3( 0.12, 0.25, 0.45 ), stripe.mul( 0.6 ) ) );
			c.assign( mix( c, vec3( 0.12, 0.16, 0.16 ), smoothstep( 0.32, 0.36, z ).mul( bodyK ).mul( 0.7 ) ) );

		} ).ElseIf( pat.equal( PATTERN.jack ), () => {

			// bar jack: black stripe along the base of the dorsal fin into the lower tail lobe,
			// electric blue below it
			const top = band( h, float( 0.82 ), float( 0.06 ), fwH.add( 0.05 ) ).mul( smoothstep( 0.2, 0.05, z ) ).mul( bodyK );
			const blue = band( h, float( 0.68 ), float( 0.05 ), fwH.add( 0.05 ) ).mul( smoothstep( 0.2, 0.05, z ) ).mul( bodyK );
			c.assign( mix( c, vec3( 0.15, 0.45, 0.9 ), blue.mul( 0.5 ) ) );
			c.assign( mix( c, vec3( 0.02, 0.03, 0.05 ), top.mul( 0.85 ) ) );
			const lobe = select( P.equal( PART.CAUDAL ), smoothstep( 7.5, 5.5, w ).mul( smoothstep( 0.1, 0.3, t ) ), 0 );
			c.assign( mix( c, vec3( 0.02, 0.03, 0.05 ), lobe.mul( 0.8 ) ) );

		} ).ElseIf( pat.equal( PATTERN.tarpon ), () => {

			// huge scales with dark edges
			const rims = smoothstep( 0.8, 0.97, f ).mul( fade ).mul( bodyK );
			c.mulAssign( float( 1 ).sub( rims.mul( 0.35 ) ) );

		} );

		// ---- lateral line (a row of pores along a dark line)
		const hl = lat.x.add( lat.y.mul( float( 1 ).sub( smoothstep( 0.12, 0.55, u ) ) ) );
		const lineK = band( h, hl, fwH.mul( 0.5 ).add( 0.012 ), fwH.add( 0.008 ) ).mul( bodyK ).mul( smoothstep( 0.18, 0.25, u ) ).mul( smoothstep( 0.9, 0.8, u ) );
		c.mulAssign( float( 1 ).sub( lineK.mul( 0.3 ) ) );

		// ---- gill cover edge and the preopercle (dark creases); gills show red on dead fish
		const zE = opercleEdge( eye.w, h );
		const dOp = z.sub( zE );
		const onOp = opercleMask( h ).mul( bodyK );
		const crease = float( 1 ).sub( smoothstep( 0.0015, 0.004, abs( dOp ) ) ).mul( onOp );
		c.mulAssign( float( 1 ).sub( crease.mul( 0.45 ) ) );
		const pre = float( 1 ).sub( smoothstep( 0.001, 0.003, abs( dOp.sub( 0.04 ) ) ) ).mul( onOp ).mul( smoothstep( 0.6, 0.2, h ) );
		c.mulAssign( float( 1 ).sub( pre.mul( 0.2 ) ) );
		const gill = smoothstep( 0.0, - 0.0015, dOp ).mul( smoothstep( - 0.007, - 0.003, dOp ) ).mul( onOp ).mul( smoothstep( 0.0, - 0.5, h ) ).mul( flags.w );
		c.assign( mix( c, vec3( 0.3, 0.03, 0.035 ), gill.mul( 0.8 ) ) );

		// ---- lips (the mouth line from the snout to the corner), the edge of the upper jaw bone
		// and the nostrils
		const hz = lat.z, hy = lat.w, tipY = r3.w;
		const mt = clamp( z.sub( hz ).div( float( 0.5 ).sub( hz ) ), 0, 1 );
		const yLip = mix( hy, tipY, mt );
		const lips = float( 1 ).sub( smoothstep( 0.0015, 0.0035, abs( y.sub( yLip ) ) ) ).mul( step( hz.sub( 0.004 ), z ) ).mul( bodyK );
		c.mulAssign( float( 1 ).sub( lips.mul( 0.55 ) ) );
		const maxZ = hz.add( 0.006 ).sub( y.sub( hy ).mul( 0.35 ) );
		const maxilla = float( 1 ).sub( smoothstep( 0.001, 0.0025, abs( z.sub( maxZ ) ) ) ).mul( smoothstep( hy.sub( 0.002 ), hy.add( 0.002 ), y ) ).mul( smoothstep( hy.add( 0.04 ), hy.add( 0.025 ), y ) ).mul( bodyK );
		c.mulAssign( float( 1 ).sub( maxilla.mul( 0.3 ) ) );
		const nostril = float( 1 ).sub( smoothstep( 0.1, 0.2, length( vec2( z.sub( eye.x ).sub( eye.z.mul( 1.7 ) ), y.sub( eye.y ).sub( eye.z.mul( 0.25 ) ) ) ).div( eye.z ) ) ).mul( bodyK );
		c.mulAssign( float( 1 ).sub( nostril.mul( 0.6 ) ) );

		// ---- painted eye (under the dome where there is one)
		const er = length( vec2( z.sub( eye.x ), y.sub( eye.y ) ) ).div( eye.z );
		const eyeCol = ( r, ang ) => {

			const streak = sin( ang.mul( 26 ) ).mul( 0.5 ).add( 0.5 );
			const iris = irisC.mul( mix( 0.65, 1.15, streak ) ).mul( smoothstep( 0.4, 0.6, r ).mul( 0.5 ).add( 0.5 ) );
			const ring = smoothstep( 0.78, 0.95, r );
			const e = mix( iris, vec3( 0.03, 0.03, 0.03 ), ring ).toVar();
			e.assign( mix( e, vec3( 0.005, 0.006, 0.008 ), float( 1 ).sub( smoothstep( 0.4, 0.46, r ) ) ) );
			// cloudy eyes of fish out of the water for a while
			e.assign( mix( e, vec3( 0.42, 0.44, 0.46 ), flags.x.mul( 0.55 ).mul( float( 1 ).sub( smoothstep( 0.7, 1.0, r ) ) ) ) );
			return e;

		};

		const painted = float( 1 ).sub( smoothstep( 0.95, 1.1, er ) ).mul( bodyK );
		c.assign( mix( c, eyeCol( er, atan( y.sub( eye.y ), z.sub( eye.x ) ) ), painted ) );
		metal.mulAssign( float( 1 ).sub( painted ) );

		// ---- eye dome: pupil, iris, glossy cornea
		If( P.equal( PART.EYE ), () => {

			const r = length( vec2( t, w ) );
			c.assign( eyeCol( r, atan( w, t ) ) );
			c.assign( mix( c, flank.mul( 0.6 ), smoothstep( 0.93, 1.0, r ) ) );
			rough.assign( mix( 0.04, 0.3, flags.x ) );
			spec.assign( 1.0 );
			metal.assign( 0 );

		} ).ElseIf( P.equal( PART.MOUTH ), () => {

			// inside of the mouth: pale pink lips to a dark throat (grunts are red inside)
			const lip = select( pat.equal( PATTERN.grunt ), vec3( 0.6, 0.08, 0.06 ), vec3( 0.5, 0.3, 0.3 ) );
			c.assign( mix( lip, vec3( 0.03, 0.012, 0.012 ), smoothstep( 0.05, 0.85, t ) ) );
			metal.assign( 0 );
			rough.assign( 0.35 );

		} ).ElseIf( P.equal( PART.FLESH ), () => {

			// cut face: muscle rings around the backbone, bone and blood at the centre
			const r = length( vec2( t, w.mul( 1.2 ) ) );
			const dark = select( pat.equal( PATTERN.tuna ), 1, 0 );
			const meat = mix( vec3( 0.62, 0.36, 0.32 ), vec3( 0.3, 0.035, 0.035 ), dark );
			const rings = smoothstep( 0.6, 0.95, sin( r.mul( 520 ).add( atan( w, abs( t ) ).mul( 2 ) ) ) ).mul( 0.12 );
			const m = meat.mul( float( 1 ).sub( rings ) ).toVar();
			const bone = float( 1 ).sub( smoothstep( 0.006, 0.009, length( vec2( t, w.sub( 0.004 ) ) ) ) );
			m.assign( mix( m, vec3( 0.75, 0.68, 0.58 ), bone ) );
			m.assign( mix( m, vec3( 0.3, 0.02, 0.02 ), float( 1 ).sub( smoothstep( 0.01, 0.03, r ) ).mul( 0.5 ).mul( float( 1 ).sub( bone ) ) ) );
			// skin rim
			c.assign( m );
			metal.assign( 0 );
			rough.assign( 0.3 );
			transl.assign( 0.3 );

		} ).ElseIf( P.equal( PART.FILLET ), () => {

			// salted, sun dried flesh: pale and translucent at the thin edges, muscle chevrons,
			// salt crystals
			const ax = abs( t );
			const chev = smoothstep( 0.55, 0.9, sin( w.add( ax.mul( 0.35 ) ).mul( 160 ) ) ).mul( 0.1 );
			const m = mix( vec3( 0.36, 0.26, 0.13 ), vec3( 0.52, 0.42, 0.26 ), smoothstep( 0.35, 1.0, ax ) ).mul( float( 1 ).sub( chev ) ).toVar();
			const salt = step( 0.94, hash( floor( vec2( t, w ).mul( 900 ) ) ) );
			m.assign( mix( m, vec3( 0.8, 0.8, 0.78 ), salt.mul( 0.6 ) ) );
			m.mulAssign( mix( 0.9, 1.05, n1 ) );
			c.assign( m );
			metal.assign( 0 );
			rough.assign( 0.6 );
			transl.assign( mix( 0.25, 0.8, smoothstep( 0.5, 1.0, ax ) ) );

		} ).ElseIf( P.equal( PART.ICE ), () => {

			// glassy crushed ice: dim albedo (light passes into it), sharp glints
			c.assign( vec3( 0.3, 0.4, 0.46 ).mul( mix( 0.8, 1.15, fract( t.mul( 7.3 ) ) ) ) );
			rough.assign( mix( 0.04, 0.2, fract( w.mul( 5.1 ) ) ) );
			metal.assign( 0 );
			transl.assign( 0.9 );
			spec.assign( 1.0 );

		} ).ElseIf( P.equal( PART.LEAF ), () => {

			// banana leaf: glossy green, pale midrib, fine parallel veins
			const ax = abs( t );
			const veins = smoothstep( 0.6, 0.95, sin( w.mul( 420 ).add( ax.mul( 60 ) ) ) ).mul( 0.12 );
			const lc = mix( vec3( 0.025, 0.08, 0.015 ), vec3( 0.05, 0.13, 0.025 ), n2 ).mul( float( 1 ).sub( veins ) ).toVar();
			lc.assign( mix( lc, vec3( 0.25, 0.3, 0.1 ), float( 1 ).sub( smoothstep( 0.015, 0.03, ax ) ) ) );
			lc.assign( mix( lc, vec3( 0.25, 0.22, 0.08 ), smoothstep( 0.9, 1.0, ax ).mul( 0.6 ) ) );
			c.assign( lc );
			metal.assign( 0 );
			rough.assign( 0.28 );
			transl.assign( 0.25 );

		} ).ElseIf( P.equal( PART.DISC ).or( P.equal( PART.WHIP ) ), () => {

			// rays: sandy, finely mottled back (stingray) or black with white rings (eagle ray);
			// white belly
			const top = w.greaterThan( 0 );
			const eagleK = select( pat.equal( PATTERN.eagleRay ), 1, 0 );
			const q = vec2( Lp.x, Lp.z ).mul( 20 );
			const cell = floor( q );
			const jit = vec2( hash( cell.add( 1.7 ) ), hash( cell.add( 5.3 ) ) ).sub( 0.5 ).mul( 0.4 );
			const rad = hash( cell.add( 9.1 ) ).mul( 0.14 ).add( 0.12 );
			const ring = abs( length( fract( q ).sub( 0.5 ).sub( jit ) ).sub( rad ) );
			const spots = float( 1 ).sub( smoothstep( 0.035, 0.075, ring ) ).mul( step( 0.45, hash( cell ) ) );
			const mottle = vnoise( vec2( Lp.x, Lp.z ).mul( 60 ) ).mul( 0.25 ).add( n2.mul( 0.2 ) ).add( 0.7 );
			const dorsal = mix( back.mul( mottle ), mix( back, vec3( 0.75, 0.78, 0.8 ), spots.mul( 0.85 ) ), eagleK ).toVar();
			dorsal.assign( mix( dorsal, edgeC, smoothstep( 0.8, 1.0, t ).mul( 0.4 ).mul( float( 1 ).sub( eagleK ) ) ) );
			c.assign( select( top, dorsal, belly ) );
			c.assign( select( P.equal( PART.WHIP ), finC, c ) );
			metal.assign( 0 );
			rough.assign( r2.w );

		} ).ElseIf( P.equal( PART.CARAPACE ), () => {

			// green turtle shell: scutes (vertebral row, costals, marginals) with dark seams and
			// radiating olive / brown / amber streaks
			const X = t, Y = w;
			const ax = abs( X );
			const rr = length( vec2( X, Y ) );
			const vert = ax.lessThan( 0.3 );
			const ySeams = select( vert, vec4( - 0.58, - 0.22, 0.14, 0.5 ), vec4( - 0.42, - 0.02, 0.36, 2.0 ) );
			const yc = Y.add( ax.mul( ax ).mul( 0.25 ) );
			const dY = min( min( abs( yc.sub( ySeams.x ) ), abs( yc.sub( ySeams.y ) ) ), min( abs( yc.sub( ySeams.z ) ), abs( yc.sub( ySeams.w ) ) ) );
			const dX = abs( ax.sub( 0.3 ) );
			const marg = rr.greaterThan( 0.84 );
			const angle = atan( Y, X );
			const dM = min( abs( rr.sub( 0.84 ) ), abs( fract( angle.mul( 12 / Math.PI ) ).sub( 0.5 ) ).mul( 0.25 ) );
			const seam = min( select( marg, dM, min( dY, dX ) ), abs( rr.sub( 0.84 ) ) );
			const streak = sin( atan( yc.sub( floor( yc.mul( 2.8 ) ).add( 0.5 ).div( 2.8 ) ), X.sub( sign( X ).mul( 0.55 ) ) ).mul( 11 ).add( n2.mul( 6 ) ) ).mul( 0.5 ).add( 0.5 );
			const blotch = smoothstep( 0.45, 0.8, vnoise( vec2( X, Y ).mul( 9 ) ) );
			const shell = mix( back, flank, streak.mul( 0.55 ).add( blotch.mul( 0.45 ) ) ).toVar();
			shell.assign( mix( shell, vec3( 0.2, 0.14, 0.06 ), smoothstep( 0.6, 0.9, n1 ).mul( 0.4 ) ) );
			shell.mulAssign( mix( 0.45, 1, smoothstep( 0.003, 0.012, seam ) ) );
			c.assign( shell );
			metal.assign( 0 );
			rough.assign( 0.35 );

		} ).ElseIf( P.equal( PART.SKIN ).or( P.equal( PART.FLIPPER ) ), () => {

			// scaly grey-brown skin with pale scale margins; pale yellow plastron
			const q = vec2( Lp.x.add( Lp.y ), Lp.z ).mul( 55 );
			const rowS = floor( q.y );
			const qq = q.add( vec2( rowS.mul( 0.5 ), 0 ) );
			const f = fract( qq ).sub( 0.5 );
			const scale = smoothstep( 0.32, 0.47, max( abs( f.x ), abs( f.y ) ) );
			const tone = hash( floor( qq ) ).mul( 0.35 ).add( 0.8 );
			const skin = mix( finC.mul( tone ), edgeC, scale.mul( 0.45 ) );
			c.assign( select( w.greaterThan( 1.5 ).and( P.equal( PART.SKIN ) ), belly.mul( mix( 0.85, 1.05, n1 ) ), skin ) );
			metal.assign( 0 );
			rough.assign( 0.5 );

		} ).ElseIf( P.equal( PART.SHELL ), () => {

			// spiny lobster: red-brown carapace with cream spots, banded legs
			const sp = vec2( t, w ).mul( 12 );
			const spots = float( 1 ).sub( smoothstep( 0.18, 0.3, length( fract( sp ).sub( 0.5 ) ) ) ).mul( step( 0.6, hash( floor( sp ) ) ) );
			const sh = mix( vec3( 0.16, 0.045, 0.03 ), vec3( 0.32, 0.1, 0.04 ), n1 ).toVar();
			sh.assign( mix( sh, vec3( 0.7, 0.55, 0.22 ), spots.mul( 0.85 ) ) );
			c.assign( sh );
			metal.assign( 0 );
			rough.assign( 0.35 );

		} );

		// ---- props: dull, drying skin; wet sheen; salted skin
		if ( prop ) {

			c.assign( mix( c, vec3( luminance( c ) ), flags.z.mul( 0.55 ).mul( bodyK ) ) );
			c.mulAssign( mix( 1, 0.85, flags.z.mul( bodyK ) ) );
			metal.mulAssign( float( 1 ).sub( flags.z ) );
			rough.assign( mix( rough, rough.mul( 0.55 ), flags.y ) );
			coat.assign( flags.y.mul( select( isBody.or( isFin ), 0.9, 0 ) ) );

		}

		// iridescent sheen on silvery skin at grazing angles
		const cosV = abs( dot( normalView, positionViewDirection ) );
		const irid = r1.w.mul( bodyK ).mul( silver ).mul( float( 1 ).sub( cosV ) );
		const hueA = vec3( 0.55, 0.95, 0.8 ), hueB = vec3( 0.95, 0.6, 1.0 );
		c.assign( mix( c, c.mul( mix( hueA, hueB, cosV ) ).mul( 1.25 ), irid.mul( 0.6 ) ) );
		albedo.assign( c );
		return c;

	} )();

	return { color, height, albedo, rough, metal, transl, coat, spec };

}

// ---------------------------------------------------------------------------

// Material of the swimming fish (batch: ReefBatch with the fish kinds).
// fade: the material of the batch's level-of-detail cross-fade channel
export function createSwimMaterial( batch, { fade = false } = {} ) {

	const V = varyings();
	const table = skinTable();
	const mat = physical( { roughness: 0.4, metalness: 0 } );
	mat.name = fade ? 'FishFade' : 'Fish';
	mat.positionNode = swimVertex( batch, V, fade );
	const S = surface( V, table, false, fade );
	mat.colorNode = S.color;
	mat.roughnessNode = S.rough;
	mat.metalnessNode = S.metal;
	mat.normalNode = bumpNormal( S.height );
	mat.specularIntensityNode = S.spec;
	mat.translucencyNode = ( lightColor ) => lightColor.mul( S.albedo ).mul( S.transl.mul( 0.5 ) );
	// motion vectors: camera motion (as for static geometry) plus the fish's own motion
	const own = Fn( () => {

		const vp = cameraProjectionMatrix.mul( cameraViewMatrix );
		const c = vp.mul( vec4( positionWorld, 1 ) );
		const q = vp.mul( vec4( positionWorld.sub( V.delta ), 1 ) );
		return c.xy.div( c.w ).sub( q.xy.div( q.w ) );

	} )();
	mat.mrtNode = mrt( { velocity: staticVelocity.add( own ) } );
	return mat;

}

// Material of the fish props: wet, glossy (clear coat), static.
export function createPropMaterial( batch, { fade = false } = {} ) {

	const V = varyings();
	const table = skinTable();
	const mat = physical( { roughness: 0.4, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.12 } );
	mat.name = fade ? 'FishPropsFade' : 'FishProps';
	mat.positionNode = propVertex( batch, V, table, fade );
	const S = surface( V, table, true, fade );
	mat.colorNode = S.color;
	mat.roughnessNode = S.rough;
	mat.metalnessNode = S.metal;
	mat.normalNode = bumpNormal( S.height );
	mat.specularIntensityNode = S.spec;
	mat.clearcoatNode = S.coat;
	mat.clearcoatRoughnessNode = float( 0.12 );
	mat.translucencyNode = ( lightColor ) => lightColor.mul( S.albedo ).mul( S.transl.mul( 0.5 ) );
	mat.mrtNode = staticVelocityMRT;
	// thin fin membranes let some light through: stochastic transparency, resolved by the
	// temporal anti-aliasing (the rays stay opaque)
	mat.maskNode = Fn( () => {

		const D = V.data;
		const part = partOf( D );
		const fin = part.greaterThan( 0.5 ).and( part.lessThan( 6.5 ) );
		const ray = float( 1 ).sub( smoothstep( 0.07, 0.14, abs( fract( D.w.add( 0.5 ) ).sub( 0.5 ) ) ) );
		const alpha = select( fin, max( mix( 0.9, 0.55, smoothstep( 0.25, 1.0, D.z ) ), ray ), 1 );
		const dither = interleavedGradientNoise( screenCoordinate.xy.add( fract( G.time.mul( 7.3 ) ).mul( 97 ) ) );
		return dither.lessThan( alpha );

	} )();
	return mat;

}
