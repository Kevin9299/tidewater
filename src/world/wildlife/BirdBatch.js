import * as THREE from 'three/webgpu';
import {
	Fn, If, float, uint, vec2, vec3, vec4, storage, uniformArray, vertexIndex, varyingProperty, normalLocal, select,
	mix, smoothstep, abs, sign, sin, cos, normalize, cross, length, max, fract, mx_noise_float, dot, saturate, normalWorld,
} from 'three/tsl';
import { G } from '../../core/Globals.js';
import { standard } from '../../materials/Materials.js';
import { SPECIES, buildBird } from './BirdShapes.js';
import { InstanceRecords, instancedMesh, rotateQ, motionVelocity } from './Kit.js';

// One instanced draw for every bird (gulls, terns, pelicans, frigatebirds, sanderlings), plus
// the near shadow cascade.
//
// Instance record (16 vec4), written by the simulations through write():
//   0 position, scale          1 body orientation     2-4 left wing: shoulder, elbow, wrist
//   5 head rotation            6 head offset, fold    7 left knee, tail pitch
//   8 left foot, tail spread   9 right knee, species  10 right foot, seed
//   11 previous position, scale   12 previous orientation   13-15 previous wing rotations
// Wing rotations are cumulative (shoulder, shoulder * elbow, shoulder * elbow * wrist) in the
// body frame; the right wing mirrors them. Knees / feet are in the body frame.

const REC = 16;

const srgb = ( r, g, b ) => vec3( Math.pow( r, 2.2 ), Math.pow( g, 2.2 ), Math.pow( b, 2.2 ) );

export class BirdBatch {

	constructor( { csm = null, capacity = 160 } = {} ) {

		// ---- species rest shapes, one storage buffer (species-major)
		const built = SPECIES.map( ( sp ) => buildBird( sp ) );
		const NV = built[ 0 ].count;
		this.NV = NV;
		const sp = new Float32Array( SPECIES.length * NV * 16 );
		built.forEach( ( b, s ) => {

			for ( let i = 0; i < NV; i ++ ) {

				const o = ( s * NV + i ) * 16;
				sp.set( b.A.subarray( i * 4, i * 4 + 4 ), o );
				sp.set( b.B.subarray( i * 4, i * 4 + 4 ), o + 4 );
				sp.set( b.C.subarray( i * 4, i * 4 + 4 ), o + 8 );
				sp.set( b.D.subarray( i * 4, i * 4 + 4 ), o + 12 );

			}

		} );
		this.speciesNode = storage( new THREE.StorageBufferAttribute( sp, 4 ), 'vec4', SPECIES.length * NV * 4 ).toReadOnly().setName( 'birdSpecies' );

		// joints per species: shoulder (+ leg radius), elbow, wrist, neck pivot, tail base, hip, eye
		const J = [];
		SPECIES.forEach( ( s, k ) => {

			const j = built[ k ].joints;
			J.push( new THREE.Vector4( ...j.S, s.legs.r ) );
			J.push( new THREE.Vector4( ...j.E, 0 ) );
			J.push( new THREE.Vector4( ...j.W, 0 ) );
			J.push( new THREE.Vector4( ...s.pivot, 0 ) );
			J.push( new THREE.Vector4( 0, s.tail.y, s.tail.z, 0 ) );
			J.push( new THREE.Vector4( ...s.legs.hip, 0 ) );
			J.push( new THREE.Vector4( 0, s.eye[ 1 ], s.eye[ 0 ], s.eye[ 2 ] ) );
			J.push( new THREE.Vector4() );

		} );
		this.joints = uniformArray( J, 'vec4' ).setName( 'birdJoints' );
		this.built = built;

		// ---- template geometry (species 0 as the nominal attributes)
		const g = new THREE.BufferGeometry();
		const p = new Float32Array( NV * 3 ), n = new Float32Array( NV * 3 );
		for ( let i = 0; i < NV; i ++ ) {

			p.set( built[ 0 ].C.subarray( i * 4, i * 4 + 3 ), i * 3 );
			n.set( [ 0, 1, 0 ], i * 3 );

		}

		g.setAttribute( 'position', new THREE.BufferAttribute( p, 3 ) );
		g.setAttribute( 'normal', new THREE.BufferAttribute( n, 3 ) );
		g.setIndex( built[ 0 ].index );

		this.records = new InstanceRecords( 'birdInstances', capacity, REC );
		this.material = this.createMaterial();
		this.mesh = instancedMesh( 'Birds', g, this.material, this.records, { csm, castShadow: true } );
		this.triangles = built[ 0 ].index.length / 3;

	}

	begin() {

		this.records.begin();

	}

	// b: pose (see Flight / pose helpers): pos, scale, q, QA, QB, QC, qH, headOff, fold, kneeL, footL,
	// kneeR, footR, tailPitch, tailSpread, species, seed, and the previous pos / scale / q / QA / QB / QC
	write( b ) {

		const o = this.records.push();
		if ( o < 0 ) return;
		const d = this.records.data;
		const v4 = ( k, a, w ) => {

			const i = o + k * 4;
			d[ i ] = a[ 0 ]; d[ i + 1 ] = a[ 1 ]; d[ i + 2 ] = a[ 2 ]; d[ i + 3 ] = w === undefined ? a[ 3 ] : w;

		};

		v4( 0, b.pos, b.scale );
		v4( 1, b.q );
		v4( 2, b.QA );
		v4( 3, b.QB );
		v4( 4, b.QC );
		v4( 5, b.qH );
		v4( 6, b.headOff, b.fold );
		v4( 7, b.kneeL, b.tailPitch );
		v4( 8, b.footL, b.tailSpread );
		v4( 9, b.kneeR, b.species );
		v4( 10, b.footR, b.seed );
		v4( 11, b.pPos, b.pScale );
		v4( 12, b.pQ );
		v4( 13, b.pQA );
		v4( 14, b.pQB );
		v4( 15, b.pQC );

	}

	commit() {

		this.records.commit();

	}

	get count() {

		return this.records.count;

	}

	createMaterial() {

		const R = this.records, NV = this.NV, SPB = this.speciesNode, JT = this.joints;
		const vRest = varyingProperty( 'vec3', 'vBirdRest' );
		const vRestN = varyingProperty( 'vec3', 'vBirdRestN' );
		const vInfo = varyingProperty( 'vec4', 'vBirdInfo' ); // part, u, v, species + seed
		const vDelta = varyingProperty( 'vec3', 'vBirdDelta' );
		const vFold = varyingProperty( 'float', 'vBirdFold' );

		const mat = standard( { roughness: 0.72, metalness: 0 } );
		mat.name = 'Birds';
		mat.underwaterLighting = 'none';

		mat.positionNode = Fn( () => {

			const r0 = R.field( 0 ), q = R.field( 1 ), QA = R.field( 2 ), QB = R.field( 3 ), QC = R.field( 4 );
			const qH = R.field( 5 ), r6 = R.field( 6 ), r7 = R.field( 7 ), r8 = R.field( 8 ), r9 = R.field( 9 ), r10 = R.field( 10 );
			const p11 = R.field( 11 ), pq = R.field( 12 ), pQA = R.field( 13 ), pQB = R.field( 14 ), pQC = R.field( 15 );
			const si = uint( r9.w.add( 0.5 ) ).toVar();
			const base = si.mul( NV ).add( vertexIndex ).mul( 4 ).toVar();
			const A = SPB.element( base ).toVar(), B = SPB.element( base.add( 1 ) ).toVar();
			const C = SPB.element( base.add( 2 ) ).toVar(), D = SPB.element( base.add( 3 ) ).toVar();
			const J = ( k ) => JT.element( si.mul( 8 ).add( k ) );
			const part = A.w.toVar();
			const fold = r6.w;

			const pl = vec3( 0 ).toVar(), pp = vec3( 0 ).toVar(), nl = vec3( 0, 1, 0 ).toVar();

			If( part.lessThan( 3.5 ), () => {

				// body, neck, head and bill: the head bone turns about the neck pivot
				const piv = J( 3 ).xyz;
				const hp = piv.add( r6.xyz ).add( rotateQ( qH, A.xyz.sub( piv ) ) );
				const w = select( part.lessThan( 2.5 ), B.w, float( 0 ) );
				pl.assign( mix( hp, A.xyz, w ) );
				nl.assign( mix( rotateQ( qH, B.xyz ), B.xyz, w ) );
				pp.assign( pl );

			} ).ElseIf( part.lessThan( 4.5 ), () => {

				// tail: spread about the centre line, pitched about the tail base
				const T = J( 4 ).xyz;
				const d = A.xyz.sub( T ).mul( vec3( r8.w, 1, 1 ) );
				const c = cos( r7.w ), s = sin( r7.w );
				pl.assign( T.add( vec3( d.x, d.y.mul( c ).add( d.z.mul( s ) ), d.z.mul( c ).sub( d.y.mul( s ) ) ) ) );
				nl.assign( vec3( B.x, B.y.mul( c ).add( B.z.mul( s ) ), B.z.mul( c ).sub( B.y.mul( s ) ) ) );
				pp.assign( pl );

			} ).ElseIf( part.lessThan( 7.5 ), () => {

				// wings: three bones, mirrored for the right wing, blended at the joints, and morphed
				// into the folded shape
				const side = sign( A.x );
				const mir = vec4( 1, side, side, 1 );
				const sx = vec3( side, 1, 1 );
				const S = J( 0 ).xyz.mul( sx ), E = J( 1 ).xyz.mul( sx ), Wj = J( 2 ).xyz.mul( sx );
				const seg = part.sub( 5 ).toVar();
				const wing = ( qa, qb, qc, withNormal ) => {

					const a = qa.mul( mir ), b = qb.mul( mir ), c = qc.mul( mir );
					const Ep = S.add( rotateQ( a, E.sub( S ) ) );
					const Wp = Ep.add( rotateQ( b, Wj.sub( E ) ) );
					const pa = S.add( rotateQ( a, A.xyz.sub( S ) ) );
					const pb = Ep.add( rotateQ( b, A.xyz.sub( E ) ) );
					const pc = Wp.add( rotateQ( c, A.xyz.sub( Wj ) ) );
					const p = select( seg.lessThan( 0.5 ), pa, select( seg.lessThan( 1.5 ), mix( pb, pa, B.w ), mix( pc, pb, B.w ) ) );
					let n = null;
					if ( withNormal ) {

						const na = rotateQ( a, B.xyz ), nb = rotateQ( b, B.xyz ), nc = rotateQ( c, B.xyz );
						n = select( seg.lessThan( 0.5 ), na, select( seg.lessThan( 1.5 ), mix( nb, na, B.w ), mix( nc, nb, B.w ) ) );

					}

					return { p: mix( p, C.xyz, fold ), n };

				};

				const cur = wing( QA, QB, QC, true );
				pl.assign( cur.p );
				nl.assign( mix( cur.n, D.xyz, fold ) );
				pp.assign( wing( pQA, pQB, pQC, false ).p );

			} ).Else( () => {

				// legs: tube hip -> knee -> ankle, the foot at the ankle
				const side = A.x, seg = A.y;
				const hip = J( 5 ).xyz.mul( vec3( side, 1, 1 ) );
				const left = side.greaterThan( 0 );
				const knee = select( left, r7.xyz, r9.xyz );
				const foot = select( left, r8.xyz, r10.xyz );
				const c = select( seg.lessThan( 0.5 ), hip, select( seg.lessThan( 1.5 ), knee, foot ) );
				const d0 = select( seg.lessThan( 0.5 ), knee.sub( hip ), select( seg.lessThan( 1.5 ), foot.sub( hip ), foot.sub( knee ) ) );
				const dir = d0.div( max( length( d0 ), 1e-5 ) );
				const ax = normalize( cross( dir, vec3( 1, 0, 0.001 ) ) );
				const ay = cross( dir, ax );
				const radial = ax.mul( B.x ).add( ay.mul( B.y ) );
				// collapsed legs (tucked in flight) vanish: the radius follows the leg's length
				const r = J( 0 ).w.mul( B.z ).mul( smoothstep( 0.0, 0.01, length( knee.sub( hip ) ).add( length( foot.sub( knee ) ) ) ) );
				const ring = c.add( radial.mul( r ) );
				const isFoot = seg.greaterThan( 2.5 );
				pl.assign( select( isFoot, foot.add( C.xyz.mul( smoothstep( 0.0, 0.01, length( foot.sub( knee ) ) ) ) ), ring ) );
				nl.assign( select( isFoot, vec3( 0, 1, 0 ), radial ) );
				pp.assign( pl );

			} );

			vRest.assign( select( part.greaterThan( 7.5 ), C.xyz, A.xyz ) );
			vRestN.assign( B.xyz );
			vInfo.assign( vec4( part, C.w, D.w, r9.w.add( fract( r10.w ).mul( 0.9 ) ) ) );
			vFold.assign( fold );

			const world = r0.xyz.add( rotateQ( q, pl.mul( r0.w ) ) ).toVar();
			const prev = p11.xyz.add( rotateQ( pq, pp.mul( p11.w ) ) );
			vDelta.assign( world.sub( prev ) );
			normalLocal.assign( rotateQ( q, nl ) );
			return world;

		} )();

		mat.mrtNode = motionVelocity( vDelta );

		// ---- plumage, bills, legs, eyes
		const albedo = vec3( 0 ).toVar( 'birdAlbedo' );
		const rough = float( 0.72 ).toVar( 'birdRough' );
		const trans = float( 0 ).toVar( 'birdTrans' );
		mat.colorNode = Fn( () => {

			const part = vInfo.x.add( 0.5 ).floor().toVar();
			const u = vInfo.y.toVar(), v = abs( vInfo.z ).toVar();
			const species = vInfo.w.floor().toVar(), seed = fract( vInfo.w ).div( 0.9 ).toVar();
			const P = vRest.toVar(), N = vRestN.toVar();
			const fold = vFold.toVar();
			const isBody = part.lessThan( 2.5 ), isBill = part.equal( 3 ), isTail = part.equal( 4 );
			const isWing = part.greaterThan( 4.5 ).and( part.lessThan( 7.5 ) ), isLeg = part.greaterThan( 7.5 );
			const top = N.y.greaterThan( 0 );
			const c = vec3( 0.5 ).toVar();
			rough.assign( 0.72 );
			trans.assign( select( isWing.or( isTail ), float( 0.35 ), float( 0 ) ) );
			// feather texture: fine noise in the rest frame (scaled to the bird's size)
			const fn = mx_noise_float( P.mul( 160 ) ).toVar();

			If( species.equal( 0 ), () => {

				// laughing gull: slate mantle and upperwing, black primaries with a white trailing
				// edge, white body and tail; breeding birds have a black hood, winter birds a grey smudge
				const white = srgb( 0.93, 0.93, 0.92 ), slate = srgb( 0.38, 0.4, 0.43 ), black = srgb( 0.05, 0.05, 0.055 );
				const hooded = seed.lessThan( 0.5 );
				const mantle = smoothstep( 0.1, 0.45, N.y ).mul( smoothstep( - 0.1, - 0.07, P.z ) ).mul( smoothstep( 0.1, 0.075, P.z ) );
				const head = smoothstep( 0.1, 0.115, P.z ).toVar();
				const eyeArc = smoothstep( 0.006, 0.004, length( vec2( P.z.sub( 0.139 ), P.y.sub( 0.037 ) ) ) ).mul( smoothstep( 0.002, 0.004, length( vec2( P.z.sub( 0.141 ), P.y.sub( 0.032 ) ) ) ) );
				const hood = select( hooded, head.mul( float( 1 ).sub( eyeArc ) ), head.mul( smoothstep( 0.012, 0.004, length( vec2( P.z.sub( 0.126 ), P.y.sub( 0.03 ) ) ) ) ).mul( 0.45 ) );
				const body = mix( mix( white, slate, mantle ), select( hooded, black, srgb( 0.45, 0.45, 0.47 ) ), hood );
				const tipK = smoothstep( 0.68, 0.76, u ).toVar();
				const edge = smoothstep( 0.9, 0.97, v ).mul( float( 1 ).sub( tipK ) ).mul( float( 1 ).sub( fold ) );
				const wingTop = mix( mix( slate, white, edge ), black, tipK );
				const wingBot = mix( mix( white, srgb( 0.72, 0.73, 0.75 ), smoothstep( 0.3, 0.9, u ).mul( 0.6 ) ), black, smoothstep( 0.78, 0.9, u ) );
				c.assign( select( isWing, select( top, wingTop, wingBot ), select( isTail, white, body ) ) );
				c.assign( select( isBill, select( hooded, srgb( 0.42, 0.07, 0.07 ), srgb( 0.12, 0.08, 0.08 ) ), c ) );
				c.assign( select( isLeg, srgb( 0.16, 0.07, 0.07 ), c ) );

			} ).ElseIf( species.equal( 1 ), () => {

				// royal tern: pale grey mantle and upperwing with a dusky primary wedge, white body,
				// shaggy black crest behind a white forehead, orange bill, black legs
				const white = srgb( 0.95, 0.95, 0.94 ), grey = srgb( 0.74, 0.77, 0.8 );
				const mantle = smoothstep( 0.15, 0.5, N.y ).mul( smoothstep( - 0.095, - 0.07, P.z ) ).mul( smoothstep( 0.1, 0.075, P.z ) );
				const cap = smoothstep( 0.108, 0.12, P.z ).mul( smoothstep( 0.158, 0.145, P.z ) ).mul( smoothstep( 0.028, 0.033, P.y ) );
				const body = mix( mix( white, grey, mantle ), srgb( 0.04, 0.04, 0.045 ), cap );
				const wedge = smoothstep( 0.66, 0.8, u ).mul( smoothstep( 0.2, 0.0, v ).mul( 0.5 ).add( 0.5 ) );
				const wingTop = mix( grey, srgb( 0.32, 0.33, 0.35 ), wedge );
				const wingBot = mix( white, srgb( 0.55, 0.56, 0.58 ), smoothstep( 0.82, 0.95, u ).mul( smoothstep( 0.4, 1.0, v ) ) );
				c.assign( select( isWing, select( top, wingTop, wingBot ), select( isTail, mix( white, grey, 0.3 ), body ) ) );
				c.assign( select( isBill, srgb( 0.95, 0.45, 0.1 ), c ) );
				c.assign( select( isLeg, srgb( 0.05, 0.05, 0.05 ), c ) );

			} ).ElseIf( species.equal( 2 ), () => {

				// brown pelican: silvery grey-brown back, dark belly, cream-white head and neck
				// (chestnut hind neck in breeding birds), dark flight feathers, grey bill, dark pouch
				const back = mix( srgb( 0.44, 0.42, 0.38 ), srgb( 0.68, 0.67, 0.63 ), smoothstep( 0.0, 0.6, fn ).mul( smoothstep( 0.0, 0.6, N.y ) ) );
				const belly = srgb( 0.24, 0.22, 0.2 );
				const neckZone = smoothstep( 0.12, 0.17, P.z ).toVar();
				const head = mix( srgb( 0.92, 0.9, 0.82 ), srgb( 0.95, 0.86, 0.55 ), smoothstep( 0.27, 0.3, P.z ).mul( smoothstep( 0.13, 0.15, P.y ) ) );
				const hindNeck = select( seed.lessThan( 0.4 ), srgb( 0.28, 0.14, 0.09 ), srgb( 0.9, 0.88, 0.82 ) );
				const neckCol = mix( hindNeck, head, smoothstep( 0.2, 0.235, P.z ) );
				const body = mix( mix( belly, back, smoothstep( - 0.35, 0.25, N.y ) ), mix( neckCol, head, smoothstep( 0.22, 0.25, P.z ) ), neckZone );
				const flight = smoothstep( 0.52, 0.6, u ).max( smoothstep( 0.45, 0.62, v ).mul( float( 1 ).sub( fold ) ) ).toVar();
				const wingTop = mix( mix( srgb( 0.56, 0.55, 0.51 ), srgb( 0.72, 0.71, 0.67 ), fn.mul( 0.5 ).add( 0.5 ).mul( 0.6 ) ), srgb( 0.1, 0.09, 0.085 ), flight );
				const wingBot = mix( srgb( 0.2, 0.19, 0.18 ), srgb( 0.45, 0.44, 0.42 ), smoothstep( 0.2, 0.35, v ).mul( smoothstep( 0.6, 0.4, v ) ).mul( smoothstep( 0.6, 0.3, u ) ) );
				c.assign( select( isWing, select( top, wingTop, wingBot ), select( isTail, srgb( 0.2, 0.19, 0.18 ), body ) ) );
				const pouch = smoothstep( 0.3, 0.38, v ).mul( smoothstep( 0.7, 0.62, v ) );
				c.assign( select( isBill, mix( mix( srgb( 0.58, 0.54, 0.5 ), srgb( 0.75, 0.55, 0.45 ), smoothstep( 0.6, 1.0, u ) ), srgb( 0.22, 0.21, 0.2 ), pouch ), c ) );
				c.assign( select( isLeg, srgb( 0.12, 0.12, 0.12 ), c ) );
				trans.mulAssign( 0.5 );

			} ).ElseIf( species.equal( 3 ), () => {

				// magnificent frigatebird: black with a faint gloss; females a white breast and a
				// brown bar on the upperwing, juveniles a white head and breast; males a red throat
				const black = srgb( 0.035, 0.035, 0.04 );
				const female = seed.lessThan( 0.45 ), juvenile = seed.greaterThan( 0.85 );
				const breast = smoothstep( 0.2, - 0.3, N.y ).mul( smoothstep( - 0.06, 0.0, P.z ) ).mul( smoothstep( 0.14, 0.1, P.z ) );
				const headW = smoothstep( 0.13, 0.16, P.z );
				const white = srgb( 0.9, 0.9, 0.88 );
				const body = mix( black, white, select( juvenile, breast.max( headW ), select( female, breast, float( 0 ) ) ) );
				const throat = smoothstep( 0.13, 0.16, P.z ).mul( smoothstep( 0.0, - 0.5, N.y ) ).mul( select( female.or( juvenile ), 0, 1 ) );
				const bar = smoothstep( 0.08, 0.14, u ).mul( smoothstep( 0.45, 0.38, u ) ).mul( smoothstep( 0.12, 0.22, v ) ).mul( smoothstep( 0.5, 0.4, v ) ).mul( select( female.or( juvenile ), 1, 0.4 ) );
				const wingTop = mix( black, srgb( 0.32, 0.25, 0.18 ), bar );
				c.assign( select( isWing, select( top, wingTop, black ), select( isTail, black, mix( body, srgb( 0.55, 0.06, 0.05 ), throat ) ) ) );
				c.assign( select( isBill, srgb( 0.5, 0.52, 0.56 ), c ) );
				c.assign( select( isLeg, srgb( 0.35, 0.28, 0.28 ), c ) );
				rough.assign( 0.55 );
				trans.mulAssign( 0.3 );

			} ).Else( () => {

				// sanderling (winter): pale grey above with dark feather centres, white below, dark
				// shoulder and primaries, bold white wing bar, black bill and legs
				const white = srgb( 0.95, 0.95, 0.94 ), grey = mix( srgb( 0.62, 0.62, 0.6 ), srgb( 0.48, 0.48, 0.47 ), smoothstep( 0.2, 0.7, fn ) );
				const upper = smoothstep( - 0.05, 0.35, N.y ).mul( smoothstep( 0.092, 0.08, P.z ).max( smoothstep( 0.2, 0.6, N.y ) ) );
				const body = mix( white, grey, upper );
				const bar = smoothstep( 0.18, 0.26, u ).mul( smoothstep( 0.78, 0.7, u ) ).mul( smoothstep( 0.42, 0.48, v ) ).mul( smoothstep( 0.66, 0.6, v ) ).mul( float( 1 ).sub( fold ) );
				const shoulder = smoothstep( 0.2, 0.3, u ).mul( smoothstep( 0.55, 0.45, u ) ).mul( smoothstep( 0.3, 0.1, v ) );
				const wingTop = mix( mix( mix( grey, srgb( 0.12, 0.12, 0.12 ), smoothstep( 0.62, 0.72, u ) ), white, bar ), srgb( 0.1, 0.1, 0.1 ), shoulder.mul( 0.8 ) );
				c.assign( select( isWing, select( top, wingTop, white ), select( isTail, mix( grey, srgb( 0.2, 0.2, 0.2 ), smoothstep( 0.25, 0.0, abs( u.sub( 0.5 ) ) ) ), body ) ) );
				c.assign( select( isBill.or( isLeg ), srgb( 0.04, 0.04, 0.04 ), c ) );

			} );

			// eye: dark, with a pale iris for pelicans
			const eye = JT.element( uint( species ).mul( 8 ).add( 6 ) );
			const ed = length( vec2( P.z.sub( eye.z ), P.y.sub( eye.y ) ) );
			const onHead = part.lessThan( 2.5 ).and( abs( P.x ).greaterThan( 0.004 ) );
			const iris = smoothstep( eye.w, eye.w.mul( 0.8 ), ed ).mul( select( onHead, 1, 0 ) ).toVar();
			const pupil = smoothstep( eye.w.mul( 0.6 ), eye.w.mul( 0.45 ), ed ).mul( select( onHead, 1, 0 ) );
			c.assign( mix( mix( c, select( species.equal( 2 ), srgb( 0.8, 0.75, 0.55 ), srgb( 0.05, 0.03, 0.02 ) ), iris ), srgb( 0.01, 0.01, 0.01 ), pupil ) );
			rough.assign( mix( select( isBill.or( isLeg ), float( 0.4 ), rough ), float( 0.15 ), iris ) );
			// fine feather texture, darker feather bases toward the trailing edge of the wings
			c.mulAssign( fn.mul( 0.06 ).add( 1 ).mul( select( isWing, mix( 1.0, 0.92, smoothstep( 0.5, 1.0, v ).mul( float( 1 ).sub( fold ) ) ), 1 ) ) );
			albedo.assign( c );
			return c;

		} )();

		mat.roughnessNode = rough;
		// light through the thin wing and tail feathers when they are between the sun and the eye
		mat.translucencyNode = ( lightColor ) => {

			const back = saturate( dot( normalWorld, G.sunDir ).negate() );
			return lightColor.mul( albedo ).mul( trans ).mul( back.mul( 0.8 ).add( 0.2 ) );

		};

		return mat;

	}

}
