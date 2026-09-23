import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, uint, vec2, vec3, vec4, ivec2, storage, instancedArray, instanceIndex, atomicAdd, If, Loop,
	Break, Discard, max, min, abs, clamp, saturate, mix, smoothstep, length, normalize, dot, cross, exp, pow, sqrt, select, sin,
	cos, fract, positionGeometry, positionWorld, cameraPosition, cameraProjectionMatrix, varyingProperty, screenUV,
	screenSize, textureLoad, textureSize, texture, perspectiveDepthToViewZ, cameraNear, cameraFar, positionView, mrt,
} from 'three/tsl';
import { G, GRAVITY } from '../core/Globals.js';
import { LAYERS } from '../core/SceneRenderer.js';
import { staticVelocity } from '../post/CameraVelocity.js';

// GPU spray particles: drops, ligaments, dense spray and mist.
//
// One storage ring buffer. The first part is written by GPU emitters (breaking waves) through an
// atomic head; the tail is owned by the CPU emit API (boat bow spray, splashes), whose requests
// are expanded by the update kernel into the slots the CPU reserved. Particles are integrated with
// gravity + drag toward the air around them (the wind; for mist first the air pushed along by the
// wave) and die when they fall back into the water, where they leave foam (ShoreSim deposit).
//
// Rendering (soft camera-facing sprites, premultiplied alpha):
//  * drops and ligaments are clear water: sub-pixel drops are drawn as motion-blurred streaks whose
//    opacity conserves the drop's cross-section (a drop covering 1/10 of a pixel for 1/5 of the
//    exposure adds 1/50 of its colour), so their visual weight is the amount of water; they show the
//    bright sky they refract, a tiny sun glint, and light up strongly when backlit (forward lobe)
//  * dense spray (clouds of drops) is white from every side (multiple scattering), with a forward
//    lobe and its own shadow on the far side
//  * mist is a thin, strongly forward-scattering, sky-tinted veil
//  * clear sheets (a boat's bow sheet) are thin water: translucent, showing the sky, glowing when
//    backlit, torn into strands and drop clusters as they age
//  * all of it is darkened in the shadow of the wave that made it, of the clouds, and at night
//
// Particles collide with one moving body (the boat, setBody): its hull (plan outline with an
// elliptic bow, up to the sheer) and one box (the wheelhouse). They are pushed out through the
// nearest face, lose the velocity into it (relative to the body), and drops that hit it run off.

export const SPRAY = { DROPLET: 0, MIST: 1, LIGAMENT: 2, SPRAY: 3, SHEET: 4 };

// per-kind constants: droplet (a drop of a few mm, ballistic), mist (fine drops that follow the air),
// ligament (an elongated blob of water, ~1-2 cm, torn off a sheet or jet), spray (a cloud of drops:
// the white of splashes), sheet (a thin clear sheet of water peeling off a hull, spreading and tearing)
const KIND = {
	tau: [ 3.0, 0.35, 5.0, 1.8, 2.5 ], // drag time constant toward the air velocity (s): ~ drop size^2
	grav: [ GRAVITY, 0.3, GRAVITY, 8.5, GRAVITY ], // effective gravity (mist barely settles; torn sheets fall back)
	turb: [ 0, 0.5, 0, 0.3, 0.1 ], // turbulent wander (m/s^2, zero mean)
	grow: [ 0, 0.22, 0, 0.3, 0.7 ], // size growth (1/s)
	dies: [ 1, 0, 1, 1, 1 ], // killed when falling into the water (else skims over it)
	deposit: [ 1, 0, 3, 4, 2 ], // foam left where it falls into the water (drops' worth)
	stretch: [ 1 / 40, 0, 1 / 40, 1 / 30, 1 / 60 ], // motion blur (s of travel)
	alpha: [ 0.55, 0.06, 0.8, 0.66, 0.22 ], // (dense spray: see-through, streaked by its motion, never cotton wool)
	fadeIn: [ 0.02, 0.2, 0.02, 0.03, 0.02 ],
	fadeOut: [ 0.8, 0.45, 0.8, 0.7, 0.5 ], // fraction of life when it starts to fade
};

// select a per-kind constant
const byKind = ( kind, arr ) => select( kind.lessThan( 0.5 ), float( arr[ 0 ] ), select( kind.lessThan( 1.5 ), float( arr[ 1 ] ), select( kind.lessThan( 2.5 ), float( arr[ 2 ] ), select( kind.lessThan( 3.5 ), float( arr[ 3 ] ), float( arr[ 4 ] ) ) ) ) );

const MAX_REQUESTS = 32;

// PCG hash of a uint -> [0, 1)
const hashU = Fn( ( [ seed ] ) => {

	const state = seed.mul( 747796405 ).add( 2891336453 );
	const word = state.shiftRight( state.shiftRight( 28 ).add( 4 ) ).bitXor( state ).mul( 277803737 );
	return word.shiftRight( 22 ).bitXor( word ).toFloat().mul( 1 / 4294967296 );

} ).setLayout( { name: 'sprayHash', type: 'float', inputs: [ { name: 'seed', type: 'uint' } ] } );

// Henyey-Greenstein phase (1/sr)
const phaseHG = ( cosT, g ) => {

	const g2 = g * g;
	return float( ( 1 - g2 ) / ( 4 * Math.PI ) ).div( pow( max( float( 1 + g2 ).sub( cosT.mul( 2 * g ) ), 1e-4 ), 1.5 ) );

};

export class Spray {

	constructor( renderer, { query, terrain, sceneCopy, clouds = null, gpuCapacity = 32768, cpuCapacity = 8192 } ) {

		this.renderer = renderer;
		this.query = query;
		this.terrain = terrain;
		this.clouds = clouds;
		// optional hooks, resolved when the shaders are first built:
		//   shoreSim: drops falling into the surf zone leave foam there (ShoreSim.depositAt)
		//   waveShadow( p, tag ) -> sun visibility for particles made by a breaking wave (Breakers)
		this.shoreSim = null;
		this.waveShadow = null;
		this.NG = gpuCapacity; // power of two (ring index wraps with the uint head)
		this.NC = cpuCapacity;
		this.N = gpuCapacity + cpuCapacity;

		const N = this.N;
		// pos: xyz, age (s) | vel: xyz, radius (m) | info: kind, life (s, 0 = dead), water height, tag
		// (tag: fraction = random seed, integer part = which breaking crest made it, 0 = none)
		this.pos = instancedArray( N, 'vec4' ).setName( 'sprayPos' );
		this.vel = instancedArray( N, 'vec4' ).setName( 'sprayVel' );
		this.info = instancedArray( N, 'vec4' ).setName( 'sprayInfo' );
		this.head = storage( new THREE.StorageBufferAttribute( new Uint32Array( 1 ), 1 ), 'uint', 1 ).toAtomic().setName( 'sprayHead' );

		// CPU emit requests (4 vec4 each): (a.xyz, prefix end) (b.xyz, size) (vel.xyz, kind) (velSpread, posSpread, life, sizeJitter)
		this.reqData = new Float32Array( MAX_REQUESTS * 16 );
		this.reqAttr = new THREE.StorageBufferAttribute( this.reqData, 4 );
		this.reqNode = storage( this.reqAttr, 'vec4', MAX_REQUESTS * 4 ).toReadOnly().setName( 'sprayReq' );
		this.nReq = 0;
		this.nReqParticles = 0;
		this.cpuHead = 0;
		this.cpuStart = uniform( 0, 'uint' ).setName( 'sprayCpuStart' );
		this.cpuCount = uniform( 0, 'uint' ).setName( 'sprayCpuCount' );
		this.reqCount = uniform( 0, 'uint' ).setName( 'sprayReqCount' );
		this.frameSeed = uniform( 0, 'uint' ).setName( 'sprayFrame' );
		this._frame = 0;

		this.params = {
			intensity: uniform( 1 ).setName( 'sprayI' ),
			maxDistance: uniform( 320 ).setName( 'sprayMaxDist' ),
		};

		// one moving body the particles collide with (setBody / setBodyShape)
		this.body = {
			on: uniform( 0 ).setName( 'sprayBodyOn' ),
			mat: uniform( new THREE.Matrix4() ).setName( 'sprayBodyMat' ), // body -> world
			inv: uniform( new THREE.Matrix4() ).setName( 'sprayBodyInv' ), // world -> body
			vel: uniform( new THREE.Vector3() ).setName( 'sprayBodyVel' ),
			hull: uniform( new THREE.Vector4( 0, 0, 1, 0 ) ).setName( 'sprayBodyHull' ), // at the sheer: z aft, z shoulder, z stem, half beam
			hullWL: uniform( new THREE.Vector4( 0, 1, 0, 0 ) ).setName( 'sprayBodyHullWL' ), // at y = 0: z shoulder, z stem, half beam
			sheer: uniform( new THREE.Vector4( 0, 0, - 1, 0 ) ).setName( 'sprayBodySheer' ), // y sheer aft, y sheer stem, y bottom
			boxMin: uniform( new THREE.Vector3() ).setName( 'sprayBodyBoxMin' ),
			boxMax: uniform( new THREE.Vector3() ).setName( 'sprayBodyBoxMax' ),
		};

		this._buildUpdate();
		this._buildMesh( sceneCopy );

	}

	// ------------------------------------------------------------------ TSL helpers for GPU emitters

	// Reserve `n` (uint) ring slots and call init( i, slot ) for each. Use inside a compute Fn.
	emitNode( n, init ) {

		If( n.greaterThan( uint( 0 ) ), () => {

			const base = atomicAdd( this.head.element( 0 ), n ).toVar();
			Loop( { start: uint( 0 ), end: n, type: 'uint', condition: '<' }, ( { i } ) => {

				init( i, base.add( i ).bitAnd( uint( this.NG - 1 ) ) );

			} );

		} );

	}

	writeNode( slot, p, v, size, kind, life, seed ) {

		this.pos.element( slot ).assign( vec4( p, 0 ) );
		this.vel.element( slot ).assign( vec4( v, size ) );
		this.info.element( slot ).assign( vec4( float( kind ), life, p.y, seed ) );

	}

	rand( a, b = 0 ) {

		return hashU( uint( a ).add( uint( b ).mul( 1664525 ) ).add( this.frameSeed.mul( 2654435761 ) ) );

	}

	// ------------------------------------------------------------------ CPU API

	// Emit `count` particles at `position` (Vector3) with base `velocity` (Vector3, m/s). size: radius (m),
	// kind: SPRAY.DROPLET | SPRAY.LIGAMENT | SPRAY.SPRAY | SPRAY.MIST. Up to 32 calls and 8192 particles per
	// frame; nothing is allocated. opts: spread (velocity jitter, m/s), jitter (position jitter, m),
	// life (s), to (Vector3: emit along the segment position -> to), sizeJitter (0..1)
	emit( position, velocity, count, size = 0.04, kind = SPRAY.DROPLET, opts = {} ) {

		if ( this.nReq >= MAX_REQUESTS || count <= 0 ) return;
		count = Math.min( Math.round( count ), this.NC - this.nReqParticles );
		if ( count <= 0 ) return;
		const { spread = 0.6, jitter = 0.05, life = kind === SPRAY.MIST ? 2.5 : 1.6, to = null, sizeJitter = 0.5 } = opts;
		// requests are expanded on the GPU by the update kernel (see _buildUpdate)
		const b = to || position;
		const d = this.reqData;
		const o = this.nReq * 16;
		this.nReqParticles += count;
		d[ o ] = position.x; d[ o + 1 ] = position.y; d[ o + 2 ] = position.z; d[ o + 3 ] = this.nReqParticles;
		d[ o + 4 ] = b.x; d[ o + 5 ] = b.y; d[ o + 6 ] = b.z; d[ o + 7 ] = size;
		d[ o + 8 ] = velocity.x; d[ o + 9 ] = velocity.y; d[ o + 10 ] = velocity.z; d[ o + 11 ] = kind;
		d[ o + 12 ] = spread; d[ o + 13 ] = jitter; d[ o + 14 ] = life; d[ o + 15 ] = sizeJitter;
		this.nReq ++;

	}

	// Emit along a polyline: countPerSegment particles spread over each segment. velocity is a Vector3,
	// or an array with one per point (each segment uses the velocity of its start point).
	emitAlongPoints( points, velocity, countPerSegment, size = 0.04, kind = SPRAY.DROPLET, opts = {} ) {

		const o = Object.assign( this._segOpts || ( this._segOpts = {} ), opts );
		for ( let i = 0; i + 1 < points.length; i ++ ) {

			o.to = points[ i + 1 ];
			this.emit( points[ i ], Array.isArray( velocity ) ? velocity[ i ] : velocity, countPerSegment, size, kind, o );

		}

		o.to = null;

	}

	// Collision shape of the body, in its own frame (+Z forward, +Y up): the hull's plan outline
	// (full half beam aft of the shoulder, an elliptic bow to the stem), given at the waterline
	// (y = 0: wl*) and at the sheer, blended with height (flare, raked stem), from yBottom up to the
	// sheer (linear from ySheerAft to ySheerStem); and one box (e.g. the wheelhouse).
	setBodyShape( { zAft, zShoulder, zStem, halfBeam, wlShoulder, wlStem, wlHalfBeam, ySheerAft, ySheerStem, yBottom, boxMin, boxMax } ) {

		const b = this.body;
		b.hull.value.set( zAft, zShoulder, zStem, halfBeam );
		b.hullWL.value.set( wlShoulder, wlStem, wlHalfBeam, 0 );
		b.sheer.value.set( ySheerAft, ySheerStem, yBottom, 0 );
		b.boxMin.value.copy( boxMin );
		b.boxMax.value.copy( boxMax );

	}

	// The body's pose (matrixWorld) and velocity this frame; null disables the collisions.
	setBody( matrixWorld, velocity ) {

		const b = this.body;
		b.on.value = matrixWorld ? 1 : 0;
		if ( ! matrixWorld ) return;
		b.mat.value.copy( matrixWorld );
		b.inv.value.copy( matrixWorld ).invert();
		b.vel.value.copy( velocity );

	}

	// ------------------------------------------------------------------ simulation

	_buildUpdate() {

		const NG = this.NG, NC = this.NC;
		const query = this.query;
		const terrain = this.terrain;

		this.updateKernel = Fn( () => {

			const i = instanceIndex;
			const P = this.pos.element( i );
			const Vv = this.vel.element( i );
			const I = this.info.element( i );

			// ---- CPU-owned slots: spawn from this frame's requests
			If( i.greaterThanEqual( uint( NG ) ), () => {

				const k = i.sub( uint( NG ) ).add( uint( NC ) ).sub( this.cpuStart ).mod( uint( NC ) ).toVar();
				If( k.lessThan( this.cpuCount ), () => {

					const r = uint( 0 ).toVar();
					Loop( { start: uint( 0 ), end: this.reqCount, type: 'uint', condition: '<' }, ( { i: j } ) => {

						r.assign( j );
						If( k.toFloat().lessThan( this.reqNode.element( j.mul( 4 ) ).w ), () => {

							Break();

						} );

					} );

					const r0 = this.reqNode.element( r.mul( 4 ) );
					const r1 = this.reqNode.element( r.mul( 4 ).add( 1 ) );
					const r2 = this.reqNode.element( r.mul( 4 ).add( 2 ) );
					const r3 = this.reqNode.element( r.mul( 4 ).add( 3 ) );
					const h0 = this.rand( i, 11 ), h1 = this.rand( i, 12 ), h2 = this.rand( i, 13 );
					const h3 = this.rand( i, 14 ), h4 = this.rand( i, 15 ), h5 = this.rand( i, 16 );
					const h6 = this.rand( i, 17 );
					const along = mix( r0.xyz, r1.xyz, h0 );
					const jit = vec3( h1.sub( 0.5 ), h2.sub( 0.5 ), h3.sub( 0.5 ) ).mul( r3.y.mul( 2 ) );
					const vj = vec3( h4.sub( 0.5 ), h5.sub( 0.5 ), h6.sub( 0.5 ) ).mul( r3.x.mul( 2 ) );
					const size = r1.w.mul( float( 1 ).add( h2.sub( 0.5 ).mul( r3.w ) ) );
					const life = r3.z.mul( h3.mul( 0.5 ).add( 0.75 ) );
					this.writeNode( i, along.add( jit ), r2.xyz.add( vj ), size, r2.w, life, h5 );

				} );

			} );

			// ---- integrate live particles
			const info = I.toVar();
			If( info.y.greaterThan( 0 ), () => {

				const dt = G.dt;
				const p = P.xyz.toVar();
				const v = Vv.xyz.toVar();
				const age = P.w.add( dt ).toVar();
				const kind = info.x;

				// wind near the surface (~70% of the 10 m wind), gusty
				const gust = sin( G.time.mul( 0.7 ).add( p.x.mul( 0.05 ) ) ).mul( 0.25 ).add( 0.85 );
				const wind = vec3( G.windDir.x, 0, G.windDir.y ).mul( G.windSpeed.mul( 0.7 ).mul( gust ) );
				// drag toward the wind: small drops follow the air, mist drifts with it
				// mist first keeps moving with the air the wave pushes ahead of it, then joins the wind
				const isMist = kind.greaterThan( 0.5 ).and( kind.lessThan( 1.5 ) );
				const tau = byKind( kind, KIND.tau ).mul( select( isMist, exp( age.mul( - 1.2 ) ).mul( 4 ).add( 1 ), float( 1 ) ) );
				const grav = byKind( kind, KIND.grav );
				const turb = vec3(
					sin( age.mul( 2.1 ).add( fract( info.w ).mul( 40 ) ) ),
					sin( age.mul( 1.7 ).add( fract( info.w ).mul( 17 ) ) ).mul( 0.5 ),
					cos( age.mul( 1.9 ).add( fract( info.w ).mul( 29 ) ) ) ).mul( byKind( kind, KIND.turb ) );
				v.addAssign( wind.sub( v ).div( tau ).add( turb ).mul( dt ) );
				v.y.subAssign( grav.mul( dt ) );
				p.addAssign( v.mul( dt ) );

				const life = info.y.toVar();
				this._collideBody( p, v, age, life, isMist );

				// water surface and ground below
				const hw = query.heightAtNode( p.xz ).toVar();
				const ground = terrain.heightAt( p.xz ).toVar();
				const top = max( hw, ground );

				// drops die in the water / on the sand; mist and foam skim over it
				If( p.y.lessThan( top ).and( v.y.lessThan( 0 ) ).and( age.greaterThan( 0.04 ) ), () => {

					If( byKind( kind, KIND.dies ).greaterThan( 0.5 ), () => {

						life.assign( 0 );
						// fell into the water (not onto the sand): its bubbles add to the foam there
						if ( this.shoreSim ) If( hw.greaterThan( ground.add( 0.02 ) ), () => {

							this.shoreSim.depositAt( p.xz, uint( byKind( kind, KIND.deposit ) ) );

						} );

					} ).Else( () => {

						p.y.assign( top.add( 0.02 ) );
						v.y.assign( max( v.y, 0 ) );
						v.xz.mulAssign( 0.95 );

					} );

				} );

				If( age.greaterThan( life ), () => {

					life.assign( 0 );

				} );

				// mist and spray clouds grow as they dilute
				const size = Vv.w.mul( float( 1 ).add( dt.mul( byKind( kind, KIND.grow ) ) ) );
				P.assign( vec4( p, age ) );
				Vv.assign( vec4( v, size ) );
				I.assign( vec4( info.x, life, hw, info.w ) );

			} );

		} )().compute( this.N, [ 64 ] ).setName( 'Spray Update' );

	}

	// Push a particle out of the body (hull or box) through the nearest face; the velocity into it
	// (relative to the body) is removed and the rest damped (the water runs off as a film); drops
	// that hit it die soon after. Inside a compute Fn.
	_collideBody( p, v, age, life, isMist ) {

		const B = this.body;
		If( B.on.greaterThan( 0.5 ), () => {

			const q = B.inv.mul( vec4( p, 1 ) ).xyz.toVar();
			const H = B.hull, W = B.hullWL, S = B.sheer;
			// hull: the outline at this height (waterline -> sheer), half breadth at q.z (elliptic bow)
			const sheer = mix( S.x, S.y, saturate( q.z.sub( H.x ).div( max( H.z.sub( H.x ), 0.01 ) ) ) ).toVar();
			const f = saturate( q.y.div( max( sheer, 0.1 ) ) );
			const zSh = mix( W.x, H.y, f ), zSt = mix( W.y, H.z, f ).toVar(), HB = mix( W.z, H.w, f ).toVar();
			const bowL = max( zSt.sub( zSh ), 0.01 ).toVar();
			const e = saturate( q.z.sub( zSh ).div( bowL ) ).toVar();
			const c = sqrt( max( float( 1 ).sub( e.mul( e ) ), 0.02 ) ).toVar();
			const hb = HB.mul( c ).toVar();
			const inHull = q.z.greaterThan( H.x ).and( q.z.lessThan( zSt ) ).and( abs( q.x ).lessThan( hb ) ).and( q.y.lessThan( sheer ) ).and( q.y.greaterThan( S.z ) );
			const b0 = B.boxMin, b1 = B.boxMax;
			const inBox = q.x.greaterThan( b0.x ).and( q.x.lessThan( b1.x ) ).and( q.y.greaterThan( b0.y ) ).and( q.y.lessThan( b1.y ) )
				.and( q.z.greaterThan( b0.z ) ).and( q.z.lessThan( b1.z ) );
			If( inHull.or( inBox ), () => {

				const n = vec3( 0, 1, 0 ).toVar();
				If( inHull, () => {

					const sx = select( q.x.greaterThan( 0 ), float( 1 ), float( - 1 ) );
					If( hb.sub( abs( q.x ) ).lessThan( sheer.sub( q.y ) ), () => {

						// through the side; on the bow the side faces forward too (- d hb / dz)
						n.assign( normalize( vec3( sx, 0, HB.mul( e ).div( c.mul( bowL ) ) ) ) );
						q.x.assign( sx.mul( hb.add( 0.02 ) ) );

					} ).Else( () => {

						q.y.assign( sheer.add( 0.02 ) );

					} );

				} ).Else( () => {

					const dx = min( q.x.sub( b0.x ), b1.x.sub( q.x ) );
					const dz = min( q.z.sub( b0.z ), b1.z.sub( q.z ) );
					const dy = b1.y.sub( q.y );
					If( dy.lessThan( dx ).and( dy.lessThan( dz ) ), () => {

						q.y.assign( b1.y.add( 0.02 ) );

					} ).ElseIf( dx.lessThan( dz ), () => {

						const right = q.x.greaterThan( b0.x.add( b1.x ).mul( 0.5 ) );
						n.assign( vec3( select( right, float( 1 ), float( - 1 ) ), 0, 0 ) );
						q.x.assign( select( right, b1.x.add( 0.02 ), b0.x.sub( 0.02 ) ) );

					} ).Else( () => {

						const front = q.z.greaterThan( b0.z.add( b1.z ).mul( 0.5 ) );
						n.assign( vec3( 0, 0, select( front, float( 1 ), float( - 1 ) ) ) );
						q.z.assign( select( front, b1.z.add( 0.02 ), b0.z.sub( 0.02 ) ) );

					} );

				} );

				const nw = normalize( B.mat.mul( vec4( n, 0 ) ).xyz );
				const vr = v.sub( B.vel ).toVar();
				const vn = dot( vr, nw );
				If( vn.lessThan( 0 ), () => {

					vr.subAssign( nw.mul( vn ) );

				} );
				v.assign( B.vel.add( vr.mul( 0.5 ) ) );
				p.assign( B.mat.mul( vec4( q, 1 ) ).xyz );
				// water that hits it wets it and runs off (gone); mist flows around it and settles
				life.assign( min( life, select( isMist, age.add( 0.25 ), age ) ) );

			} );

		} );

	}

	update() {

		// CPU requests -> ring slots
		const n = this.nReqParticles;
		this.cpuStart.value = this.cpuHead;
		this.cpuCount.value = n;
		this.reqCount.value = this.nReq;
		this.cpuHead = ( this.cpuHead + n ) % this.NC;
		if ( this.nReq > 0 ) this.reqAttr.needsUpdate = true;
		this.frameSeed.value = ( ++ this._frame ) >>> 0;
		this.renderer.compute( this.updateKernel );
		this.nReq = 0;
		this.nReqParticles = 0;

	}

	// ------------------------------------------------------------------ rendering

	_buildMesh( sceneCopy ) {

		const geo = new THREE.BufferGeometry();
		geo.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array( [ - 1, - 1, 0, 1, - 1, 0, 1, 1, 0, - 1, 1, 0 ] ), 3 ) );
		geo.setIndex( [ 0, 1, 2, 0, 2, 3 ] );
		geo.boundingSphere = new THREE.Sphere( new THREE.Vector3(), 1e7 );

		const mat = new THREE.NodeMaterial();
		mat.name = 'Spray';
		mat.transparent = true;
		mat.depthWrite = false;
		mat.depthTest = true;
		mat.side = THREE.DoubleSide;
		mat.forceSinglePass = true;
		mat.blending = THREE.CustomBlending;
		mat.blendEquation = THREE.AddEquation;
		mat.blendSrc = THREE.OneFactor;
		mat.blendDst = THREE.OneMinusSrcAlphaFactor;
		mat.blendSrcAlpha = THREE.OneFactor;
		mat.blendDstAlpha = THREE.OneMinusSrcAlphaFactor;
		mat.lights = false;
		mat.fog = false;

		const posA = this.pos.toAttribute();
		const velA = this.vel.toAttribute();
		const infoA = this.info.toAttribute();

		const vUV = varyingProperty( 'vec2', 'vSprayUV' );
		const vCol = varyingProperty( 'vec4', 'vSprayCol' ); // premultiplied-ready radiance, opacity
		const vMisc = varyingProperty( 'vec4', 'vSprayMisc' ); // kind, water height, softness, seed
		const vFwd = varyingProperty( 'vec3', 'vSprayFwd' ); // forward-scattered sun (thin parts glow with it)

		const P = this.params;
		const L = G.sunDir;

		mat.positionNode = Fn( () => {

			const p = posA.xyz;
			const age = posA.w;
			const v = velA.xyz;
			const info = infoA;
			const kind = info.x;
			const life = info.y;
			const isDrop = kind.lessThan( 0.5 );
			const isLig = kind.greaterThan( 1.5 ).and( kind.lessThan( 2.5 ) );
			const isMist = kind.greaterThan( 0.5 ).and( kind.lessThan( 1.5 ) );
			const water = isDrop.or( isLig ); // clear water: drops and ligaments
			const alive = life.greaterThan( 0 ).and( age.lessThan( life ) );

			const toCam = cameraPosition.sub( p );
			const dist = max( length( toCam ), 0.05 );
			const Vd = toCam.div( dist );

			// pixel footprint at this distance: drops are drawn at least ~1.3 px wide
			const p11 = cameraProjectionMatrix.element( 1 ).element( 1 );
			const pixel = dist.mul( 2 ).div( p11.mul( screenSize.y ) );
			const r = velA.w; // radius (m)
			const size = max( r, pixel.mul( 1.3 ) );

			// motion blur along the velocity projected on the view plane; ligaments are elongated anyway
			const vPerp = v.sub( Vd.mul( dot( v, Vd ) ) );
			const speed = length( vPerp );
			const stretchLen = speed.mul( byKind( kind, KIND.stretch ) );
			const elong = select( isLig, r.mul( 2.0 ), float( 0 ) );
			const up = select( speed.greaterThan( 1e-3 ), vPerp.div( max( speed, 1e-3 ) ), vec3( 0, 1, 0 ) );
			// clouds: random rotation that slowly turns
			const rot = fract( info.w ).mul( 6.283 ).add( age.mul( fract( info.w ).sub( 0.5 ) ) );
			const camRight = normalize( cross( vec3( 0, 1, 0 ), Vd ).add( vec3( 1e-5, 0, 0 ) ) );
			const camUp = cross( Vd, camRight );
			const mRight = camRight.mul( cos( rot ) ).add( camUp.mul( sin( rot ) ) );
			const mUp = camUp.mul( cos( rot ) ).sub( camRight.mul( sin( rot ) ) );
			// torn sheets (dense spray) are drawn along their motion too, tilted a little at random and
			// stretched by their speed: fibrous, streaked silhouettes instead of round puffs
			const isSheet = kind.greaterThan( 2.5 );
			const isClear = kind.greaterThan( 3.5 ); // clear sheet (bow sheet)
			const alongV = water.or( isSheet );
			const side = normalize( cross( Vd, up ) );
			const tilt = fract( info.w.mul( 7.31 ) ).sub( 0.5 ).mul( 0.7 );
			const sUp = up.mul( cos( tilt ) ).add( side.mul( sin( tilt ) ) );
			const sSide = side.mul( cos( tilt ) ).sub( up.mul( sin( tilt ) ) );
			const axisY = select( water, up, select( isSheet, sUp, mUp ) );
			const axisX = select( water, side, select( isSheet, sSide, mRight ) );
			const sheetLen = select( isSheet, size.mul( clamp( speed.mul( 0.08 ), 0, 0.8 ) ), float( 0 ) );
			const halfY = size.add( stretchLen.mul( 0.5 ) ).add( elong ).add( sheetLen );
			const halfX = select( isSheet, size.mul( 1.05 ), size );
			const corner = positionGeometry.xy;
			const world = p.add( axisX.mul( corner.x.mul( halfX ) ) ).add( axisY.mul( corner.y.mul( halfY ) ) );

			// coverage that conserves the water's cross-section: the drop's projected area (and the time it
			// spends on each pixel of its streak) spread over the drawn footprint. For the gaussian
			// footprint exp( -k d^2 ) the peak is k r (r + elong) / ( halfX halfY ). Clouds: lost to the clamp.
			const kShape = select( isLig, float( 4.5 ), float( 3.5 ) );
			const cover = select( water, r.mul( r.add( elong ) ).mul( kShape ).div( halfX.mul( halfY ) ).min( 1 ), saturate( r.div( size ) ) );

			// ---- lighting (per particle)
			const cosT = dot( Vd.negate(), L ); // 1 = looking toward the sun through the particle
			const cosA = dot( Vd, L );
			const sunVis = ( this.clouds ? this.clouds.shadow( p.xz ) : float( 1 ) ).mul( this.waveShadow ? this.waveShadow( p, info.w ) : float( 1 ) );
			const sun = G.sunColor.mul( sunVis ).toVar();
			// clear water (drop, ligament): the bright sky it refracts and reflects, a strong forward lobe
			// (diffraction + refraction) when backlit, a small glint from any side
			const cWater = G.skyIrradiance.mul( 0.9 ).add( sun.mul( phaseHG( cosT, 0.85 ).mul( 1.2 ).add( 0.12 ) ) );
			// dense spray: multiply scattered, white from any side (a diffuse sphere: its far side is in its
			// own shadow), plus a forward lobe
			const lambert = sqrt( max( float( 1 ).sub( cosA.mul( cosA ) ), 0 ) ).add( float( Math.PI ).sub( cosA.acos() ).mul( cosA ) ).div( Math.PI );
			// (the forward lobe is passed on separately: thin, torn parts glow with it, thick parts shade it)
			const cSpray = sun.mul( lambert.mul( 0.65 ).add( 0.35 ).div( Math.PI ) ).add( G.skyIrradiance.mul( 1.15 ) );
			const fSpray = sun.mul( phaseHG( cosT, 0.6 ).mul( 0.9 ) );
			// mist: a thin veil of fine drops, strongly forward scattering, tinted by the sky
			const cMist = sun.mul( 0.25 / Math.PI ).add( G.skyIrradiance.mul( 0.9 ) );
			const fMist = sun.mul( phaseHG( cosT, 0.75 ) );
			// clear sheet: thin water, the sky it shows and a little sun off its surface; the sun shining
			// through it (forward lobe) is passed on: the thin torn parts glow when backlit
			const cClear = G.skyIrradiance.mul( 0.95 ).add( sun.mul( 0.05 ) );
			const fClear = sun.mul( phaseHG( cosT, 0.8 ).mul( 1.1 ) );
			const col = select( water, cWater, select( isMist, cMist, select( isClear, cClear, cSpray ) ) );
			vFwd.assign( select( water, vec3( 0 ), select( isMist, fMist, select( isClear, fClear, fSpray ) ) ) );

			// opacity over the particle's life
			const t = age.div( max( life, 1e-3 ) );
			const fadeIn = smoothstep( 0.0, byKind( kind, KIND.fadeIn ), t );
			const fadeOut = float( 1 ).sub( smoothstep( byKind( kind, KIND.fadeOut ), 1.0, t ) );
			const baseA = byKind( kind, KIND.alpha );
			// far: fade out; very near the eye: sheets and mist would fill the screen (and cost a lot of overdraw)
			const distFade = smoothstep( P.maxDistance, P.maxDistance.mul( 0.55 ), dist ).mul( select( water, float( 1 ), smoothstep( 0.6, 3.0, dist ) ) );
			const a = baseA.mul( fadeIn ).mul( fadeOut ).mul( cover ).mul( distFade ).mul( P.intensity );

			vUV.assign( corner );
			vCol.assign( vec4( col, a ) );
			// x: kind + 0.45 * life fraction (the kind tests below have 0.5 of margin)
			vMisc.assign( vec4( kind.add( saturate( t ).mul( 0.45 ) ), info.z, size, fract( info.w ) ) );

			// dead particles collapse off-screen
			return select( alive.and( a.greaterThan( 1e-4 ) ), world, vec3( 0, - 1e5, 0 ) );

		} )();

		// ---- fragment
		const depthTex = sceneCopy.depthTexture;
		const sceneDepthAt = ( uv ) => {

			const sz = textureSize( textureLoad( depthTex ), 0 );
			const q = ivec2( clamp( uv, 0, 0.9999 ).mul( vec2( sz ) ) );
			return textureLoad( depthTex, q ).x;

		};

		const noiseTex = makePuffTexture();
		const dotsTex = makeDotsTexture();

		const fragment = Fn( () => {

			const uv = vUV;
			const kind = vMisc.x;
			const isDrop = kind.lessThan( 0.5 );
			const isLig = kind.greaterThan( 1.5 ).and( kind.lessThan( 2.5 ) );
			const isSheet = kind.greaterThan( 2.5 );
			const isClear = kind.greaterThan( 3.5 );
			const t = saturate( fract( kind ).div( 0.45 ) ); // life fraction
			const r2 = dot( uv, uv );
			const sd = vec2( vMisc.w, vMisc.w.mul( 1.7 ) );
			// drop: gaussian streak; ligament: a slightly sharper, beaded blob
			const drop = exp( r2.mul( - 3.5 ) ).sub( 0.03 ).max( 0 );
			const lig = exp( r2.mul( - 4.5 ) ).mul( sin( uv.y.mul( 5.0 ).add( vMisc.w.mul( 40 ) ) ).mul( 0.2 ).add( 0.9 ) ).sub( 0.03 ).max( 0 );
			// torn sheet: noise streaked along the motion (uv.y), eroded from its edges inward and more and
			// more as it ages, so it tears into strands and fragments instead of shrinking. Thick parts are
			// dense white water, the torn edges thin and translucent.
			const env = saturate( float( 1 ).sub( r2 ) );
			const fib = texture( noiseTex, vec2( uv.x.mul( 0.5 ), uv.y.mul( 0.26 ) ).add( sd ) ).x;
			const fine = texture( noiseTex, vec2( uv.x.mul( 1.2 ), uv.y.mul( 0.6 ) ).add( sd.mul( 2.3 ) ) ).x;
			const field = fib.mul( 0.6 ).add( fine.mul( 0.4 ) ).add( env.sub( 0.55 ).mul( 0.75 ) ).sub( smoothstep( 0.8, 1.0, r2 ) );
			const erode = mix( float( 0.26 ), float( 0.7 ), t );
			const dens = saturate( field.sub( erode ).mul( 3.0 ) );
			// torn edges are thin, translucent water: soft and see-through, the core dense white
			const torn = smoothstep( erode.sub( 0.04 ), erode.add( 0.2 ), field ).mul( dens.mul( 0.45 ).add( 0.55 ) );
			// ... which breaks up into a cluster of drops: many small dots in a ragged envelope that thins
			// out as it ages (sub-pixel drops average out through the mipmaps: never a solid blob)
			const dots = texture( dotsTex, uv.mul( vec2( 0.5, 0.32 ) ).add( sd.mul( 3.7 ) ) ).x;
			const swarm = dots.mul( smoothstep( 0.25, 0.55, fib.add( env.mul( 0.45 ) ).sub( t.mul( 0.2 ) ) ) ).mul( float( 1 ).sub( t.mul( 0.5 ) ) );
			const sheet = max( torn.mul( float( 1 ).sub( smoothstep( 0.15, 0.6, t ) ) ), swarm );
			// mist: a soft veil of low-frequency noise that drifts and thins, never a disc
			const m1 = texture( noiseTex, uv.mul( 0.2 ).add( sd ) ).x;
			const m2 = texture( noiseTex, uv.mul( 0.55 ).add( sd.mul( 3.1 ) ) ).x;
			const veil = smoothstep( 0.28, 0.8, m1.mul( 0.7 ).add( m2.mul( 0.3 ) ).add( env.mul( 0.3 ) ).sub( 0.12 ) ).mul( env.sqrt() ).mul( float( 1 ).sub( t.mul( 0.4 ) ) );
			const shape = select( isDrop, drop, select( isLig, lig, select( isSheet, sheet, veil ) ) );
			// self-shadowing inside thick sheets; the forward-scattered sun lights up the thin parts
			const shade = select( isSheet.and( isClear.not() ), float( 1 ).sub( dens.mul( 0.15 ) ), float( 1 ) );
			const glow = select( isClear, float( 1 ).sub( dens.mul( 0.3 ) ), select( isSheet, float( 1 ).sub( dens.mul( 0.6 ) ), float( 1 ) ) );

			// soft intersections: opaque scene depth and the water surface under the particle
			const sceneZ = perspectiveDepthToViewZ( sceneDepthAt( screenUV ), cameraNear, cameraFar );
			const soft = vMisc.z.mul( 1.5 ).add( 0.03 );
			const fadeScene = saturate( positionView.z.sub( sceneZ ).div( soft ) );
			const fadeWater = saturate( positionWorld.y.sub( vMisc.y ).div( soft.mul( 0.6 ) ).add( 0.15 ) );
			const aOut = vCol.w.mul( shape ).mul( fadeScene ).mul( fadeWater ).toVar();
			If( aOut.lessThan( 0.002 ), () => {

				Discard();

			} );

			return vec4( vCol.rgb.mul( shade ).add( vFwd.mul( glow ) ).mul( aOut ), aOut );

		} )();

		mat.outputNode = fragment;
		// motion vectors weighted by coverage (premultiplied, blended like the color)
		const velOut = vec4( staticVelocity.mul( fragment.w ), 0, fragment.w );
		mat.mrtNode = mrt( { velocity: velOut } );
		mat.mrtNode.setBlendMode( 'velocity', new THREE.BlendMode( THREE.MaterialBlending ) );

		const mesh = this.mesh = new THREE.Mesh( geo, mat );
		mesh.count = this.N;
		mesh.frustumCulled = false;
		mesh.castShadow = false;
		mesh.receiveShadow = false;
		mesh.renderOrder = 20;
		mesh.layers.set( LAYERS.TRANSPARENT );
		mesh.name = 'Spray';

	}

}

// Small tileable value-noise texture for mist puffs (generated once on the CPU).
function makePuffTexture( size = 64 ) {

	const data = new Uint8Array( size * size * 4 );
	const rnd = ( i, j, o ) => {

		const s = Math.sin( ( i % o ) * 127.1 + ( j % o ) * 311.7 + o * 17.3 ) * 43758.5453;
		return s - Math.floor( s );

	};

	const vnoise = ( x, y, o ) => {

		const i = Math.floor( x ), j = Math.floor( y );
		const fx = x - i, fy = y - j;
		const ux = fx * fx * ( 3 - 2 * fx ), uy = fy * fy * ( 3 - 2 * fy );
		const a = rnd( i, j, o ), b = rnd( i + 1, j, o ), c = rnd( i, j + 1, o ), d = rnd( i + 1, j + 1, o );
		return ( a * ( 1 - ux ) + b * ux ) * ( 1 - uy ) + ( c * ( 1 - ux ) + d * ux ) * uy;

	};

	for ( let j = 0; j < size; j ++ ) for ( let i = 0; i < size; i ++ ) {

		let s = 0, a = 0.5, n = 0;
		for ( let o = 4; o <= 32; o *= 2 ) {

			s += vnoise( i / size * o, j / size * o, o ) * a;
			n += a;
			a *= 0.55;

		}

		const v = Math.max( 0, Math.min( 1, ( s / n - 0.2 ) * 1.6 ) );
		const k = ( j * size + i ) * 4;
		data[ k ] = data[ k + 1 ] = data[ k + 2 ] = Math.round( v * 255 );
		data[ k + 3 ] = 255;

	}

	const tex = new THREE.DataTexture( data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType );
	tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
	tex.magFilter = THREE.LinearFilter;
	tex.minFilter = THREE.LinearMipmapLinearFilter;
	tex.generateMipmaps = true;
	tex.needsUpdate = true;
	return tex;

}

// Tileable texture of scattered drops (r: coverage), for clusters of drops. Heavy-tailed radii: many
// tiny drops, a few large ones.
function makeDotsTexture( size = 128, count = 150 ) {

	const data = new Uint8Array( size * size * 4 );
	let seed = 12345;
	const rnd = () => ( ( seed = ( seed * 1664525 + 1013904223 ) >>> 0 ) / 4294967296 );
	const cov = new Float32Array( size * size );
	for ( let k = 0; k < count; k ++ ) {

		const cx = rnd() * size, cy = rnd() * size;
		const r = Math.min( 0.9 * Math.pow( 1 - rnd() * 0.97, - 0.55 ), 6 );
		const R = Math.ceil( r + 1.5 );
		for ( let dy = - R; dy <= R; dy ++ ) for ( let dx = - R; dx <= R; dx ++ ) {

			const x = ( ( Math.floor( cx ) + dx ) % size + size ) % size, y = ( ( Math.floor( cy ) + dy ) % size + size ) % size;
			const d = Math.hypot( Math.floor( cx ) + dx + 0.5 - cx, Math.floor( cy ) + dy + 0.5 - cy );
			const c = Math.max( 0, Math.min( 1, r + 0.5 - d ) );
			cov[ y * size + x ] = Math.max( cov[ y * size + x ], c );

		}

	}

	for ( let i = 0; i < size * size; i ++ ) {

		const v = Math.round( cov[ i ] * 255 );
		data[ i * 4 ] = data[ i * 4 + 1 ] = data[ i * 4 + 2 ] = v;
		data[ i * 4 + 3 ] = 255;

	}

	const tex = new THREE.DataTexture( data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType );
	tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
	tex.magFilter = THREE.LinearFilter;
	tex.minFilter = THREE.LinearMipmapLinearFilter;
	tex.generateMipmaps = true;
	tex.needsUpdate = true;
	return tex;

}
