import * as THREE from 'three/webgpu';
import {
	Fn, If, float, int, uint, vec2, vec3, vec4, storage, uniformArray, vertexIndex, varyingProperty, normalLocal, attribute,
	select, mix, smoothstep, abs, sin, cos, asin, clamp, normalize, cross, length, max, fract, mx_noise_float, Discard,
	interleavedGradientNoise, screenCoordinate, atan,
} from 'three/tsl';
import { G } from '../../core/Globals.js';
import { standard } from '../../materials/Materials.js';
import { buildCritters, BODY_VERTS, LIMBS, CRITTER } from './CritterShapes.js';
import { InstanceRecords, instancedMesh, rotateQ, motionVelocity } from './Kit.js';

// One instanced draw for the beach critters: ghost crabs, hermit crabs and burrows.
//
// Instance record (6 vec4), written through write():
//   0 position (ground), scale     1 orientation
//   2 species, gait phase, stride (0 still .. 1 full), body lift
//   3 out (ghost crab: 0 down the burrow .. 1 out; hermit crab: 0 withdrawn .. 1 out), eyes up,
//     claws raised, seed
//   4 previous position, previous gait phase     5 previous orientation

const REC = 6;
const srgb = ( r, g, b ) => vec3( Math.pow( r, 2.2 ), Math.pow( g, 2.2 ), Math.pow( b, 2.2 ) );

// tetrapod gait: legs 0, 2, 5, 7 swing together, 1, 3, 4, 6 half a cycle later
const GAIT = [ 0, Math.PI, 0, Math.PI, Math.PI, 0, Math.PI, 0 ];

export class CritterBatch {

	constructor( { capacity = 160 } = {} ) {

		const T = buildCritters();
		this.template = T;
		this.bodyNode = storage( new THREE.StorageBufferAttribute( T.bodyData, 4 ), 'vec4', T.bodyData.length / 4 ).toReadOnly().setName( 'critterBodies' );
		this.limbs = uniformArray( T.limbs, 'vec4' ).setName( 'critterLimbs' );
		this.records = new InstanceRecords( 'critterInstances', capacity, REC );
		this.material = this.createMaterial();
		this.mesh = instancedMesh( 'Critters', T.geometry, this.material, this.records, { castShadow: false } );
		this.triangles = T.triangles;

	}

	begin() {

		this.records.begin();

	}

	// c: { x, y, z, scale, q, species, phase, stride, lift, out, eyes, claws, seed, px, py, pz, pPhase, pq }
	write( c ) {

		const o = this.records.push();
		if ( o < 0 ) return;
		const d = this.records.data;
		d[ o ] = c.x; d[ o + 1 ] = c.y; d[ o + 2 ] = c.z; d[ o + 3 ] = c.scale;
		d[ o + 4 ] = c.q[ 0 ]; d[ o + 5 ] = c.q[ 1 ]; d[ o + 6 ] = c.q[ 2 ]; d[ o + 7 ] = c.q[ 3 ];
		d[ o + 8 ] = c.species; d[ o + 9 ] = c.phase; d[ o + 10 ] = c.stride; d[ o + 11 ] = c.lift;
		d[ o + 12 ] = c.out; d[ o + 13 ] = c.eyes; d[ o + 14 ] = c.claws; d[ o + 15 ] = c.seed;
		d[ o + 16 ] = c.px; d[ o + 17 ] = c.py; d[ o + 18 ] = c.pz; d[ o + 19 ] = c.pPhase;
		d[ o + 20 ] = c.pq[ 0 ]; d[ o + 21 ] = c.pq[ 1 ]; d[ o + 22 ] = c.pq[ 2 ]; d[ o + 23 ] = c.pq[ 3 ];

	}

	commit() {

		this.records.commit();

	}

	get count() {

		return this.records.count;

	}

	createMaterial() {

		const R = this.records, BODY = this.bodyNode, LT = this.limbs;
		const aLimb = attribute( 'aLimb', 'vec4' ), aRing = attribute( 'aRing', 'vec2' );
		const vInfo = varyingProperty( 'vec4', 'vCritInfo' ); // part (0 body, 1 leg, 2 claw, 3 eye), u, v, species + seed
		const vLocal = varyingProperty( 'vec3', 'vCritLocal' );
		const vDelta = varyingProperty( 'vec3', 'vCritDelta' );
		const gait = uniformArray( GAIT.map( ( g ) => new THREE.Vector4( g, 0, 0, 0 ) ), 'vec4' );

		const mat = standard( { roughness: 0.6, metalness: 0 } );
		mat.name = 'Critters';
		mat.underwaterLighting = 'none';

		// joints of limb k for gait phase ph: j0 (at the body) .. j3 (tip)
		const joints = ( si, k, ph, stride, lift, out, eyes, claws ) => {

			const a = LT.element( si.mul( LIMBS * 2 ).add( k.mul( 2 ) ) );
			const L = LT.element( si.mul( LIMBS * 2 ).add( k.mul( 2 ) ).add( 1 ) );
			const dir = ( az, el ) => vec3( cos( az ).mul( cos( el ) ), sin( el ), sin( az ).mul( cos( el ) ) );
			const kf = float( k );
			const isLeg = kf.lessThan( 7.5 ), isClaw = kf.greaterThan( 7.5 ).and( kf.lessThan( 9.5 ) );
			const side = select( a.x.greaterThan( 0 ), float( 1 ), float( - 1 ) );
			const p = ph.add( gait.element( k.min( uint( 7 ) ) ).x );
			const swing = max( sin( p ), 0 ).mul( stride );
			const az = a.w.add( cos( p ).mul( stride ).mul( 0.28 ).mul( select( isLeg, 1, 0 ) ) );
			const j0 = vec3( a.x, a.y.add( lift ), a.z ).toVar();
			const j1 = vec3( 0 ).toVar(), j2 = vec3( 0 ).toVar(), j3 = vec3( 0 ).toVar();
			If( isLeg, () => {

				// merus up and out, carpus level, dactyl reaching down to the ground
				j1.assign( j0.add( dir( az, float( 0.62 ).add( swing.mul( 0.22 ) ) ).mul( L.x ) ) );
				j2.assign( j1.add( dir( az, float( - 0.15 ).add( swing.mul( 0.15 ) ) ).mul( L.y ) ) );
				const e3 = asin( clamp( j2.y.negate().div( max( L.z, 1e-4 ) ), - 0.99, - 0.25 ) ).add( swing.mul( 0.25 ) );
				j3.assign( j2.add( dir( az, e3 ).mul( L.z ) ) );

			} ).ElseIf( isClaw, () => {

				// chelipeds folded in front of the mouth, lifted when feeding / threatening
				const inward = side.mul( 0.95 );
				j1.assign( j0.add( dir( az, float( - 0.35 ).add( claws.mul( 0.5 ) ) ).mul( L.x ) ) );
				j2.assign( j1.add( dir( az.add( inward ), float( 0.25 ).add( claws.mul( 0.6 ) ) ).mul( L.y ) ) );
				j3.assign( j2.add( dir( az.add( inward.mul( 1.55 ) ), float( - 0.25 ).add( claws.mul( 0.4 ) ) ).mul( L.z ) ) );

			} ).Else( () => {

				// eyestalks: up when alert, folded along the front edge when running for the burrow
				const el = mix( float( 0.05 ), float( 1.3 ), eyes );
				j1.assign( j0.add( dir( az, el.mul( 0.8 ) ).mul( L.x ) ) );
				j2.assign( j1.add( dir( az, el ).mul( L.y ) ) );
				j3.assign( j2 );

			} );

			// hermit crabs pull everything back into the aperture
			const hole = vec3( 0, lift.add( 0.32 ), 0.5 );
			const k2 = select( si.equal( uint( CRITTER.HERMIT ) ), out, float( 1 ) );
			return {
				j0: mix( hole, j0, k2 ), j1: mix( hole, j1, k2 ), j2: mix( hole, j2, k2 ), j3: mix( hole, j3, k2 ),
				r: L.w.mul( select( si.equal( uint( CRITTER.HERMIT ) ), out.mul( 0.7 ).add( 0.3 ), float( 1 ) ) ),
			};

		};

		// a vertex of limb k at ( segment, t, radius ) around the segment
		const limbVertex = ( J, seg, t, rs, ring ) => {

			const a = select( seg.lessThan( 0.5 ), J.j0, select( seg.lessThan( 1.5 ), J.j1, J.j2 ) );
			const b = select( seg.lessThan( 0.5 ), J.j1, select( seg.lessThan( 1.5 ), J.j2, J.j3 ) );
			const d0 = b.sub( a );
			const d = d0.div( max( length( d0 ), 1e-5 ) );
			const ref = select( abs( d.y ).greaterThan( 0.9 ), vec3( 1, 0, 0 ), vec3( 0, 1, 0 ) );
			const ax = normalize( cross( d, ref ) );
			const ay = cross( ax, d );
			const radial = ax.mul( ring.x ).add( ay.mul( ring.y ) );
			return { p: mix( a, b, t ).add( radial.mul( J.r.mul( rs ) ) ), n: normalize( radial.add( d.mul( select( rs.lessThan( 0.01 ), 1, 0 ) ) ) ) };

		};

		mat.positionNode = Fn( () => {

			const r0 = R.field( 0 ), q = R.field( 1 ), r2 = R.field( 2 ), r3 = R.field( 3 ), p4 = R.field( 4 ), pq = R.field( 5 );
			const si = uint( r2.x.add( 0.5 ) ).toVar();
			const lift = r2.w, out = r3.x;
			const pl = vec3( 0 ).toVar(), pp = vec3( 0 ).toVar(), nl = vec3( 0, 1, 0 ).toVar();
			const info = vec4( 0 ).toVar();
			const k = aLimb.x;

			If( k.lessThan( 0 ), () => {

				// body grid: carapace / shell (lifted on the legs) or the burrow mound
				const base = si.mul( BODY_VERTS ).add( vertexIndex ).mul( 2 );
				const A = BODY.element( base ), B = BODY.element( base.add( 1 ) );
				const up = select( si.equal( uint( CRITTER.BURROW ) ), float( 0 ), lift );
				pl.assign( A.xyz.add( vec3( 0, up, 0 ) ) );
				nl.assign( B.xyz );
				pp.assign( pl );
				info.assign( vec4( 0, A.w, B.w, 0 ) );

			} ).Else( () => {

				const ki = uint( k );
				const J = joints( si, ki, r2.y, r2.z, lift, out, r3.y, r3.z );
				const v = limbVertex( J, aLimb.y, aLimb.z, aLimb.w, aRing );
				pl.assign( v.p );
				nl.assign( v.n );
				const Jp = joints( si, ki, p4.w, r2.z, lift, out, r3.y, r3.z );
				pp.assign( limbVertex( Jp, aLimb.y, aLimb.z, aLimb.w, aRing ).p );
				const part = select( k.lessThan( 7.5 ), float( 1 ), select( k.lessThan( 9.5 ), float( 2 ), float( 3 ) ) );
				info.assign( vec4( part, aLimb.y.add( aLimb.z ), aLimb.w, 0 ) );

			} );

			// ghost crabs sink into their burrow (the sand hides what is below)
			const sink = select( si.equal( uint( CRITTER.GHOST ) ), float( 1 ).sub( out ).mul( 1.3 ), float( 0 ) );
			pl.y.subAssign( sink );
			pp.y.subAssign( sink );
			vLocal.assign( pl );
			vInfo.assign( vec4( info.xyz, r2.x.add( fract( r3.w ).mul( 0.9 ) ) ) );
			const world = r0.xyz.add( rotateQ( q, pl.mul( r0.w ) ) ).toVar();
			const prev = p4.xyz.add( rotateQ( pq, pp.mul( r0.w ) ) );
			vDelta.assign( world.sub( prev ) );
			normalLocal.assign( rotateQ( q, nl ) );
			return world;

		} )();

		mat.mrtNode = motionVelocity( vDelta );

		const rough = float( 0.6 ).toVar( 'critRough' );
		mat.colorNode = Fn( () => {

			const part = vInfo.x.add( 0.5 ).floor().toVar();
			const u = vInfo.y.toVar(), v = vInfo.z.toVar();
			const species = vInfo.w.floor().toVar(), seed = fract( vInfo.w ).div( 0.9 ).toVar();
			const P = vLocal.toVar();
			const n = mx_noise_float( P.mul( 14 ).add( seed.mul( 17 ) ) ).toVar();
			const c = vec3( 0.5 ).toVar();
			rough.assign( 0.6 );

			If( species.equal( CRITTER.GHOST ), () => {

				// ghost crab: pale straw carapace with fine granules, whitish legs and claws, black
				// club-shaped eyes on the stalks
				const straw = mix( srgb( 0.8, 0.73, 0.58 ), srgb( 0.88, 0.83, 0.7 ), n.mul( 0.5 ).add( 0.5 ) );
				const body = straw.mul( mix( 0.86, 1.04, smoothstep( 0.0, 0.18, P.y ) ) );
				const legs = mix( srgb( 0.86, 0.82, 0.72 ), srgb( 0.7, 0.62, 0.5 ), smoothstep( 0.8, 1.0, fract( u ) ).mul( 0.5 ) );
				const claw = mix( srgb( 0.9, 0.87, 0.8 ), srgb( 0.78, 0.7, 0.75 ), smoothstep( 1.6, 2.3, u ) );
				const eye = mix( srgb( 0.8, 0.75, 0.62 ), srgb( 0.04, 0.04, 0.045 ), smoothstep( 1.25, 1.4, u ) );
				c.assign( select( part.equal( 0 ), body, select( part.equal( 1 ), legs, select( part.equal( 2 ), claw, eye ) ) ) );
				rough.assign( select( part.equal( 3 ).and( u.greaterThan( 1.3 ) ), float( 0.15 ), float( 0.55 ) ) );

			} ).ElseIf( species.equal( CRITTER.HERMIT ), () => {

				// hermit crab: turban shell (banded / mottled / chequered by seed), red-orange legs
				// with pale tips, a purple claw; the aperture shows the crab's dark body
				const turns = u.add( v.mul( 3.2 ) ); // spiral: sutures along u + v * turns
				const suture = smoothstep( 0.9, 0.97, fract( turns ) ).mul( smoothstep( 0.05, 0.3, v ) );
				const kind = fract( seed.mul( 3.7 ) ).mul( 3 ).floor();
				const c1 = select( kind.equal( 0 ), srgb( 0.9, 0.86, 0.78 ), select( kind.equal( 1 ), srgb( 0.72, 0.42, 0.2 ), srgb( 0.85, 0.8, 0.7 ) ) );
				const c2 = select( kind.equal( 0 ), srgb( 0.12, 0.11, 0.1 ), select( kind.equal( 1 ), srgb( 0.35, 0.18, 0.08 ), srgb( 0.55, 0.35, 0.22 ) ) );
				const bands = select( kind.equal( 0 ),
					smoothstep( 0.3, 0.7, sin( u.mul( 6.2832 * 7 ).add( sin( v.mul( 40 ) ).mul( 1.5 ) ) ) ), // zigzag
					smoothstep( 0.2, 0.8, sin( v.mul( 31 ).add( n.mul( 2 ) ) ) ) );
				const shellC = mix( c1, c2, bands.mul( 0.8 ) ).mul( float( 1 ).sub( suture.mul( 0.5 ) ) ).mul( n.mul( 0.12 ).add( 0.94 ) );
				const aperture = smoothstep( 0.86, 0.93, v );
				const body = mix( shellC, srgb( 0.35, 0.12, 0.08 ), aperture );
				const legs = mix( srgb( 0.72, 0.28, 0.12 ), srgb( 0.9, 0.78, 0.6 ), smoothstep( 2.6, 3.0, u ) );
				const claw = mix( srgb( 0.42, 0.14, 0.4 ), srgb( 0.85, 0.5, 0.2 ), smoothstep( 2.5, 3.1, u ) );
				c.assign( select( part.equal( 0 ), body, select( part.equal( 1 ), legs, select( part.equal( 2 ), claw, srgb( 0.1, 0.08, 0.06 ) ) ) ) );
				rough.assign( select( part.equal( 0 ), float( 0.45 ), float( 0.5 ) ) );

			} ).Else( () => {

				// burrow: dark shaft, damp dug-out sand around the lip, loose clumps fanned out; the
				// rim fades into the beach
				const dry = srgb( 0.86, 0.79, 0.64 ).mul( n.mul( 0.1 ).add( 0.95 ) );
				const damp = srgb( 0.66, 0.58, 0.45 );
				const clumps = smoothstep( 0.35, 0.55, mx_noise_float( P.mul( 38 ).add( seed.mul( 5 ) ) ) );
				const sand = mix( mix( damp, dry, smoothstep( 0.4, 0.75, v ) ), dry.mul( 1.06 ), clumps.mul( 0.5 ) );
				const hole = smoothstep( 0.36, 0.2, v );
				c.assign( mix( sand, srgb( 0.035, 0.03, 0.025 ), hole ) );
				rough.assign( 0.9 );
				// dithered fade at the outer edge (resolved by the temporal filter)
				const fade = smoothstep( 1.0, 0.72, v );
				const noise = interleavedGradientNoise( screenCoordinate.xy.add( fract( G.time.mul( 7.13 ) ).mul( 97 ) ) );
				If( noise.greaterThan( fade ), () => {

					Discard();

				} );

			} );

			return c;

		} )();

		mat.roughnessNode = rough;
		return mat;

	}

}
