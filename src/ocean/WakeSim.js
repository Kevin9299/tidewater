import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, int, uint, vec2, vec3, vec4, uvec2, tanh, instancedArray, workgroupArray, workgroupBarrier,
	localId, workgroupId, If, select, sqrt, exp, max, min, saturate, abs, smoothstep, sin, cos, length, floor, fract, ivec2,
	mix, dot, texture, textureStore, clamp,
} from 'three/tsl';
import { G, GRAVITY } from '../core/Globals.js';

const N = 512; // grid cells per side
const LOG2N = 9;
const HALF = N / 2;
const MASK = N - 1;
const CELL = 0.4; // m
const SIZE = N * CELL; // window size (m)
// reference depths of the four spectral operators (m); a cell blends the two bracketing its depth
const DEPTHS = [ Infinity, 6, 1.8, 0.6 ];
const SPONGE = 22; // absorbing band at the window edges (cells)
// boat-frame box of the near-field template (steady wave pattern under the hull)
const TW = 32, TH = 88; // 0.1 m texels (steep pressure near the stem: keep resampling errors small)
const TX0 = - 1.6, TX1 = 1.6, TZ0 = - 4.4, TZ1 = 4.4;

// Interactive boat wake: a linear free-surface wave simulation with exact dispersion.
//
// Height h and vertical velocity w on a 512^2 grid (0.4 m cells, 205 m) in a window that follows
// the boat. The window is addressed toroidally (world cell I lives in texel I mod N): it scrolls
// by whole cells without moving any data, and the FFT's periodicity is the addressing itself.
// Each step is 3 dispatches in one compute pass (~0.13 ms on an M5 Pro):
//   1. rows: per-cell physics (hull pressure, dissipation, breaking, foam, obstacles), display
//      texture, forward FFT along x
//   2. columns: forward FFT along z, spectral operators, inverse FFT along z (spare workgroups
//      update the boat's near-field template)
//   3. rows: inverse FFT along x, symplectic Euler step, absorbing sponge at the window edges
//
// Waves: dw/dt = -g L[h + P], dh/dt = w. L = |k| tanh(|k| d) is the Dirichlet-to-Neumann operator of
// linear water waves; P is the pressure head of the hull (the boat is a moving pressure
// distribution, after Havelock): its immersion, a dynamic part ~U^2/2g where the hull enters the
// water and the hollow behind the transom at speed. Deep water gives the Kelvin wake (transverse
// and divergent waves inside 19.5 deg, narrowing at high Froude numbers) for any course and speed.
// Variable depth: four spectral operators (d = inf, 6, 1.8, 0.6 m) are blended per cell in the
// symmetric form sum_i s_i L_i[s_i u] (the two levels bracketing the local depth; <= 5 % phase
// speed error for 1-100 m waves down to 0.5 m depth). It is self-adjoint, so wake waves slow
// down, shorten, refract and grow over the shoaling beach until they break in the swash zone.
// Dissipation is a turbulent eddy viscosity in flux form (conserves mass): breaking crests, the
// surf and the propeller wash feed the turbulence, which flattens the waves it passes through.
// Dry land and the pier piles are rigid (weak reflection, scattering around the piles).
//
// Output (display texture, rgba16f): (h, dh/dx, dh/dz, foam), read by WaterSurface through
// displacement() / fragment() (one sampler per shader stage), plus the aeration (bubbles in the
// water column, a slower field than the surface foam: fragment().aeration, 0..aerOut), read with
// textureLoad (no sampler).
export class WakeSim {

	constructor( renderer, { terrainGPU, boat, colliders = null } ) {

		this.renderer = renderer;
		this.terrain = terrainGPU;
		this.boat = boat;
		this.lines = boat.model.lines;

		// ---- window (cell indices), output fade
		this.uOrigin = uniform( new THREE.Vector2() ).setName( 'wkOrigin' ); // min corner, cells
		this.uPrev = uniform( new THREE.Vector2() ).setName( 'wkPrev' ); // min corner last step
		this.uReset = uniform( 1 ).setName( 'wkReset' );
		this.uDt = uniform( 1 / 60 ).setName( 'wkDt' );
		this.uCenter = uniform( new THREE.Vector2() ).setName( 'wkCenter' ); // world centre (m)
		this.uAmount = uniform( 0 ).setName( 'wkAmount' );

		// ---- boat
		this.uBoatPos = uniform( new THREE.Vector2() ).setName( 'wkBoatPos' );
		this.uBoatRot = uniform( new THREE.Vector2( 1, 0 ) ).setName( 'wkBoatRot' ); // cos, sin yaw
		this.uTrim = uniform( new THREE.Vector3() ).setName( 'wkTrim' ); // immersion change: c + a x + b z
		this.uSource = uniform( 1 ).setName( 'wkSource' ); // pressure gain
		this.uWash = uniform( 0 ).setName( 'wkWash' ); // propeller wash / transom turbulence
		this.uBow = uniform( 0 ).setName( 'wkBow' ); // breaking bow wave
		this.uSpeed = uniform( 0 ).setName( 'wkSpeed' );
		this.uWashW = uniform( 0.6 ).setName( 'wkWashW' ); // half width of the wash at the transom
		this.uBoil = uniform( 1.2 ).setName( 'wkBoil' ); // turbulent stirring of the surface (boils)
		this.uVisc = uniform( 0.006 ).setName( 'wkVisc' ); // m^2/s, damps grid-scale waves (~k^2)
		this.uDyn = uniform( 0 ).setName( 'wkDyn' ); // dynamic pressure head at the entry, K U^2 / 2g (m)
		this.dynamicPressure = 0.35; // K
		this.uHollow = uniform( 1 ).setName( 'wkHollow' ); // length of the transom hollow (m)
		this.uHollowK = uniform( 0 ).setName( 'wkHollowK' );

		// ---- tuning
		this.amplitude = uniform( 1 ).setName( 'wkAmp' );
		this.foamGain = uniform( 0.35 ).setName( 'wkFoamGain' ); // thick churn at the transom, patchy lace behind
		this.sourceGain = 0.72; // near-field waves ~0.2-0.4 m at cruise

		this.state = instancedArray( N * N, 'vec4' ).setName( 'wakeState' ); // h, w, foam, turbulence
		this.scratch = instancedArray( N * N, 'vec4' ).setName( 'wakeScratch' ); // h, w', foam', turb'
		this.spec = instancedArray( N * N, 'vec4' ).setName( 'wakeSpectrum' ); // two packed complex fields
		// aeration: bubbles mixed into the water column by the propeller race and breaking (the
		// turquoise-milky water under and around the foam); slower than the surface foam
		this.aerA = instancedArray( N * N, 'float' ).setName( 'wakeAer' );
		this.aerB = instancedArray( N * N, 'float' ).setName( 'wakeAerNext' );
		this.aerGain = uniform( 1 ).setName( 'wkAerGain' );
		this.aerOut = uniform( 0.3 ).setName( 'wkAerOut' ); // output aeration at saturation (0..1)

		const t = new THREE.StorageTexture( N, N );
		t.type = THREE.HalfFloatType;
		t.format = THREE.RGBAFormat;
		t.magFilter = t.minFilter = THREE.LinearFilter;
		t.wrapS = t.wrapT = THREE.RepeatWrapping;
		t.generateMipmaps = false;
		t.name = 'wakeDisplay';
		this.display = t;
		const at = new THREE.StorageTexture( N, N );
		at.type = THREE.HalfFloatType;
		at.format = THREE.RGBAFormat;
		at.magFilter = at.minFilter = THREE.NearestFilter; // read with textureLoad: no sampler binding
		at.wrapS = at.wrapT = THREE.RepeatWrapping;
		at.generateMipmaps = false;
		at.name = 'wakeAeration';
		this.aerTex = at;

		// The boat must not feel its own steady wave pattern through the water queries (the
		// controller models its hydrodynamics already, and the 1-3 frame query latency turns that
		// self-coupling into porpoising). A boat-frame running mean of the wake height under the
		// hull is subtracted inside the footprint: waves moving relative to the hull (crossing an
		// old wake) still get through.
		this.nearBuf = instancedArray( TW * TH, 'float' ).setName( 'wakeNearMean' );
		const nt = new THREE.StorageTexture( TW, TH );
		nt.type = THREE.HalfFloatType;
		nt.format = THREE.RGBAFormat;
		nt.magFilter = nt.minFilter = THREE.NearestFilter; // read with textureLoad: no sampler binding
		nt.wrapS = nt.wrapT = THREE.ClampToEdgeWrapping;
		nt.generateMipmaps = false;
		nt.name = 'wakeNearField';
		this.nearTex = nt;
		this.uNearRate = uniform( 1 ).setName( 'wkNearRate' );
		this.uDead = uniform( new THREE.Vector2( 0.03, 0.08 ) ).setName( 'wkDead' ); // residual dead zone (m)

		this._bakeHull();
		this._bakePiles( colliders );
		this._buildKernels();
		this._buildReaders();

		this.center = new THREE.Vector2(); // window centre (cells, integer)
		this.hasWindow = false;
		this.sleeping = true;
		this.idleTime = 1e9;
		this.stepCount = 0;
		this.primed = false;
		this.settleTime = 40; // s of calm before the simulation goes to sleep

	}

	// ---------------------------------------------------------------- setup

	// Hull immersion at the design waterline over the boat frame (x = |port|, z = forward),
	// blurred to the grid scale so the moving pressure field does not excite grid noise.
	_bakeHull() {

		const L = this.lines;
		const NX = 24, NZ = 112;
		const X1 = 1.9, Z0 = L.zAft - 0.8, Z1 = L.wlEnd + 0.8;
		const R = 0.55;
		const data = new Uint16Array( NX * NZ );
		for ( let iz = 0; iz < NZ; iz ++ ) {

			for ( let ix = 0; ix < NX; ix ++ ) {

				const x = ( ix + 0.5 ) / NX * X1, z = Z0 + ( iz + 0.5 ) / NZ * ( Z1 - Z0 );
				let sum = 0, wsum = 0;
				for ( let u = - 3; u <= 3; u ++ ) {

					for ( let v = - 3; v <= 3; v ++ ) {

						const dx = u / 3 * R, dz = v / 3 * R;
						const wgt = Math.exp( - ( dx * dx + dz * dz ) / ( R * R * 0.4 ) );
						const y = L.bottomAt( x + dx, z + dz );
						sum += wgt * ( Number.isFinite( y ) ? Math.max( 0, - y ) : 0 );
						wsum += wgt;

					}

				}

				data[ iz * NX + ix ] = THREE.DataUtils.toHalfFloat( ix === NX - 1 || iz === 0 || iz === NZ - 1 ? 0 : sum / wsum );

			}

		}

		const tex = new THREE.DataTexture( data, NX, NZ, THREE.RedFormat, THREE.HalfFloatType );
		tex.magFilter = tex.minFilter = THREE.LinearFilter;
		tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
		tex.generateMipmaps = false;
		tex.needsUpdate = true;
		this.hullTex = tex;
		this.hullBox = { X1, Z0, Z1 };

	}

	// Fraction of each grid cell blocked by the pier piles standing in the water (cells are made
	// rigid). Land, cliffs and the sea stacks come from the terrain.
	_bakePiles( colliders ) {

		const piles = colliders ? colliders.cylinders.filter( ( c ) => c.tag === 'pile' && c.yMin < - 0.1 && c.yMax > 0.2 ) : [];
		let x0 = 0, z0 = 0, x1 = 1, z1 = 1;
		if ( piles.length ) {

			x0 = Math.min( ...piles.map( ( c ) => c.x - c.radius ) ) - 1;
			x1 = Math.max( ...piles.map( ( c ) => c.x + c.radius ) ) + 1;
			z0 = Math.min( ...piles.map( ( c ) => c.z - c.radius ) ) - 1;
			z1 = Math.max( ...piles.map( ( c ) => c.z + c.radius ) ) + 1;

		}

		const res = Math.max( 0.1, ( x1 - x0 ) / 2048, ( z1 - z0 ) / 2048 );
		const W = Math.max( 2, Math.ceil( ( x1 - x0 ) / res ) ), H = Math.max( 2, Math.ceil( ( z1 - z0 ) / res ) );
		const cover = new Float32Array( W * H );
		const sub = 4;
		for ( const c of piles ) {

			const ia = Math.floor( ( c.x - c.radius - x0 ) / res ), ib = Math.ceil( ( c.x + c.radius - x0 ) / res );
			const ja = Math.floor( ( c.z - c.radius - z0 ) / res ), jb = Math.ceil( ( c.z + c.radius - z0 ) / res );
			for ( let j = Math.max( 0, ja ); j <= Math.min( H - 1, jb ); j ++ ) {

				for ( let i = Math.max( 0, ia ); i <= Math.min( W - 1, ib ); i ++ ) {

					let n = 0;
					for ( let a = 0; a < sub; a ++ ) for ( let b = 0; b < sub; b ++ ) {

						const px = x0 + ( i + ( a + 0.5 ) / sub ) * res, pz = z0 + ( j + ( b + 0.5 ) / sub ) * res;
						if ( ( px - c.x ) ** 2 + ( pz - c.z ) ** 2 < c.radius * c.radius ) n ++;

					}

					cover[ j * W + i ] = Math.min( 1, cover[ j * W + i ] + n / ( sub * sub ) );

				}

			}

		}

		// box filter over one grid cell -> blocked fraction of a cell centred at each texel
		const k = Math.max( 1, Math.round( CELL / res / 2 ) );
		const data = new Uint16Array( W * H );
		for ( let j = 0; j < H; j ++ ) {

			for ( let i = 0; i < W; i ++ ) {

				let s = 0, n = 0;
				for ( let b = - k; b < k; b ++ ) for ( let a = - k; a < k; a ++ ) {

					const ii = i + a, jj = j + b;
					if ( ii >= 0 && jj >= 0 && ii < W && jj < H ) s += cover[ jj * W + ii ];
					n ++;

				}

				data[ j * W + i ] = THREE.DataUtils.toHalfFloat( s / n );

			}

		}

		const tex = new THREE.DataTexture( data, W, H, THREE.RedFormat, THREE.HalfFloatType );
		tex.magFilter = tex.minFilter = THREE.LinearFilter;
		tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
		tex.generateMipmaps = false;
		tex.needsUpdate = true;
		this.pileTex = tex;
		this.pileCount = piles.length;
		this.uPileBox = uniform( new THREE.Vector4( x0, z0, x1 - x0, z1 - z0 ) ).setName( 'wkPileBox' );

	}

	// ---------------------------------------------------------------- kernels

	_buildKernels() {

		const { state, scratch, spec } = this;
		const terrain = this.terrain;
		const shRow = workgroupArray( 'vec4', N );
		const shCol = workgroupArray( 'vec4', N * 2 );

		const brev = ( v ) => {

			let r = v.bitAnd( uint( 1 ) ).shiftLeft( uint( LOG2N - 1 ) );
			for ( let b = 1; b < LOG2N; b ++ ) r = r.bitOr( v.shiftRight( uint( b ) ).bitAnd( uint( 1 ) ).shiftLeft( uint( LOG2N - 1 - b ) ) );
			return r;

		};

		// radix-2 DIT stages over one or two N-point sequences of complex pairs (vec4) in shared
		// memory (bit-reversed in, natural order out). dir = -1 forward, +1 inverse (unnormalized).
		const stages = ( sh, t, bases, dir ) => {

			for ( let s = 0; s < LOG2N; s ++ ) {

				const half = 1 << s;
				const pos = t.bitAnd( uint( half - 1 ) );
				const i0 = t.shiftRight( uint( s ) ).shiftLeft( uint( s + 1 ) ).bitOr( pos ).toVar();
				const ang = float( pos ).mul( dir * Math.PI / half );
				const w = vec2( cos( ang ), sin( ang ) ).toVar();
				for ( const base of bases ) {

					const i = base ? i0.add( uint( base ) ) : i0;
					const j = i.add( uint( half ) );
					const a = sh.element( i ).toVar();
					const b = sh.element( j ).toVar();
					const bw = vec4(
						b.x.mul( w.x ).sub( b.y.mul( w.y ) ), b.x.mul( w.y ).add( b.y.mul( w.x ) ),
						b.z.mul( w.x ).sub( b.w.mul( w.y ) ), b.z.mul( w.y ).add( b.w.mul( w.x ) )
					).toVar();
					sh.element( i ).assign( a.add( bw ) );
					sh.element( j ).assign( a.sub( bw ) );

				}

				workgroupBarrier();

			}

		};

		// toroidal texel -> world cell index in the window starting at o
		const worldIndex = ( i, o ) => o.add( i.sub( o ).bitAnd( int( MASK ) ) );
		const inWindow = ( I, J, o ) => I.greaterThanEqual( int( o.x ) ).and( I.lessThan( int( o.x ).add( N ) ) )
			.and( J.greaterThanEqual( int( o.y ) ) ).and( J.lessThan( int( o.y ).add( N ) ) );

		// state of texel (c, r) if it held the same world cell last step, else zero
		const loadState = ( c, r ) => {

			const I = worldIndex( int( c ), int( this.uOrigin.x ) );
			const J = worldIndex( int( r ), int( this.uOrigin.y ) );
			const keep = inWindow( I, J, this.uPrev ).and( this.uReset.lessThan( 0.5 ) );
			return select( keep, state.element( r.mul( uint( N ) ).add( c ) ), vec4( 0 ) );

		};

		const loadAer = ( c, r ) => {

			const I = worldIndex( int( c ), int( this.uOrigin.x ) );
			const J = worldIndex( int( r ), int( this.uOrigin.y ) );
			const keep = inWindow( I, J, this.uPrev ).and( this.uReset.lessThan( 0.5 ) );
			return select( keep, this.aerA.element( r.mul( uint( N ) ).add( c ) ), float( 0 ) );

		};

		const cellPos = ( c, r ) => {

			const I = worldIndex( int( c ), int( this.uOrigin.x ) );
			const J = worldIndex( int( r ), int( this.uOrigin.y ) );
			return vec2( float( I ).add( 0.5 ), float( J ).add( 0.5 ) ).mul( CELL );

		};

		// sqrt of the blend weights of the four operators (deep, 6, 1.8, 0.6 m) for local depth d:
		// the two levels bracketing d share the weight (fitted for <= 5 % phase-speed error over
		// 1-100 m wavelengths down to 0.5 m depth); below 0.6 m the shallowest one fades out
		const depthWeights = ( d ) => {

			const t0 = float( 1 ).sub( exp( d.sub( 6 ).div( - 9 ) ) );
			const t1 = saturate( d.sub( 1.8 ).div( 4.2 ) ).pow( 0.8 );
			const t2 = saturate( d.sub( 0.6 ).div( 1.2 ) ).pow( 0.85 );
			const w0 = select( d.greaterThan( 6 ), t0, float( 0 ) );
			const w1 = select( d.greaterThan( 6 ), float( 1 ).sub( t0 ), select( d.greaterThan( 1.8 ), t1, float( 0 ) ) );
			const w2 = select( d.greaterThan( 6 ), float( 0 ), select( d.greaterThan( 1.8 ), float( 1 ).sub( t1 ), select( d.greaterThan( 0.6 ), t2, float( 0 ) ) ) );
			const w3 = select( d.greaterThan( 1.8 ), float( 0 ), select( d.greaterThan( 0.6 ), float( 1 ).sub( t2 ), saturate( d.div( 0.6 ) ) ) );
			return sqrt( vec4( w0, w1, w2, w3 ) );

		};

		const dt = this.uDt;
		const HB = this.hullBox;
		// one texture node per texture: every lookup shares a single binding / sampler
		const hullT = texture( this.hullTex );
		const pileT = texture( this.pileTex );
		const displayT = texture( this.display );

		// hull immersion (m) at boat-frame (x, z): design immersion + scheduled heave / trim
		const immersionAt = ( x, z ) => {

			const huv = vec2( abs( x ).div( HB.X1 ), z.sub( HB.Z0 ).div( HB.Z1 - HB.Z0 ) );
			const D = hullT.sample( huv ).level( 0 ).x;
			return max( D.add( this.uTrim.x ).add( this.uTrim.y.mul( x ) ).add( this.uTrim.z.mul( z ) ), 0 ).mul( smoothstep( 0.0, 0.04, D ) );

		};

		// integer lattice hash -> [0, 1), smooth value noise
		const hash = ( ix, iy ) => {

			let v = uint( ix ).mul( uint( 0x8da6b343 ) ).bitXor( uint( iy ).mul( uint( 0xd8163841 ) ) );
			v = v.bitXor( v.shiftRight( uint( 13 ) ) ).mul( uint( 0x5bd1e995 ) );
			v = v.bitXor( v.shiftRight( uint( 15 ) ) );
			return float( v.shiftRight( uint( 8 ) ) ).mul( 1 / 16777216 );

		};

		const noise = ( q ) => {

			const i = floor( q ).toVar();
			const f = fract( q );
			const u = f.mul( f ).mul( float( 3 ).sub( f.mul( 2 ) ) ).toVar();
			const ix = int( i.x ), iy = int( i.y );
			const a = hash( ix, iy ), b = hash( ix.add( 1 ), iy );
			const c = hash( ix, iy.add( 1 ) ), d = hash( ix.add( 1 ), iy.add( 1 ) );
			return mix( mix( a, b, u.x ), mix( c, d, u.x ), u.y );

		};

		// ---- pass 1: per-cell physics + forward FFT along rows
		const cellPhysics = Fn( ( [ col, row ] ) => {

			const idx = row.mul( uint( N ) ).add( col );
			const p = cellPos( col, row ).toVar();
			const s = loadState( col, row ).toVar();
			const cL = col.add( uint( MASK ) ).bitAnd( uint( MASK ) ), cR = col.add( uint( 1 ) ).bitAnd( uint( MASK ) );
			const rD = row.add( uint( MASK ) ).bitAnd( uint( MASK ) ), rU = row.add( uint( 1 ) ).bitAnd( uint( MASK ) );
			const sL = loadState( cL, row ).toVar(), sR = loadState( cR, row ).toVar();
			const sD = loadState( col, rD ).toVar(), sU = loadState( col, rU ).toVar();

			const d = G.seaLevel.sub( terrain.heightAt( p ) ).toVar();

			// boat frame of the cell
			const rel = p.sub( this.uBoatPos );
			const cs = this.uBoatRot.x, sn = this.uBoatRot.y;
			const bx = rel.x.mul( cs ).sub( rel.y.mul( sn ) ).toVar();
			const bz = rel.x.mul( sn ).add( rel.y.mul( cs ) ).toVar();
			// hull pressure head (m): hydrostatic (immersion, with the speed-scheduled trim) and a
			// dynamic part where the hull enters the water: ~ K U^2/2g times the rise of the bottom
			const immersion = ( z ) => immersionAt( bx, z );

			const e = 0.7;
			const Ph = immersion( bz ).toVar();
			const entry = immersion( bz.sub( e ) ).sub( immersion( bz.add( e ) ) ).div( 2 * e ).max( 0 );
			// transom stern at speed: the flow separates at the transom and leaves a hollow behind it
			// (then the rooster tail and the quarter waves)
			const aft = float( this.lines.zAft ).sub( bz );
			const hollow = immersion( float( this.lines.zAft + 0.25 ) ).mul( exp( aft.max( 0 ).div( this.uHollow ).negate() ) ).mul( smoothstep( - 0.3, 0.3, aft ) );
			const P = Ph.max( hollow.mul( this.uHollowK ) ).add( entry.mul( this.uDyn ) ).mul( this.uSource ).toVar();

			// on a (re)start the water under the hull is already displaced: no transient
			const h = select( this.uReset.greaterThan( 0.5 ), P.negate(), s.x ).toVar();
			const w = s.y;

			const gx = sR.x.sub( sL.x ).div( 2 * CELL ), gz = sU.x.sub( sD.x ).div( 2 * CELL );
			const slope = length( vec2( gx, gz ) );

			// obstacles: dry land, pier piles
			const puv = p.sub( this.uPileBox.xy ).div( this.uPileBox.zw );
			const pile = pileT.sample( puv ).level( 0 ).x.mul( select( puv.x.greaterThan( 0 ).and( puv.x.lessThan( 1 ) ).and( puv.y.greaterThan( 0 ) ).and( puv.y.lessThan( 1 ) ), float( 1 ), float( 0 ) ) );
			const wet = smoothstep( 0.0, 0.05, d );
			const keep = wet.mul( saturate( float( 1 ).sub( pile.mul( 2.5 ) ) ) );

			// breaking: steep crests (deep water) and depth-limited (surf zone); free waves only, the
			// forced depression under the hull is steep but does not break
			const brk = saturate( slope.sub( 0.2 ).div( 0.2 ) ).mul( float( 1 ).sub( smoothstep( 0.0, 0.03, P ) ) );
			const surf = saturate( h.sub( d.mul( 0.42 ) ).div( max( d.mul( 0.3 ), 0.03 ) ) ).mul( wet );
			const turb = s.w;

			// propeller wash / transom wake (boat frame): widens behind the transom, streaky
			// (lateral noise that changes as the boat moves on) and patchy
			const time = G.time;
			const behind = float( this.lines.zAft ).sub( bz ).toVar();
			const washW = behind.max( 0 ).mul( 0.1 ).add( this.uWashW );
			const inWash = smoothstep( - 0.6, 0.3, behind ).mul( smoothstep( 7.0, 1.5, behind ) ).mul( smoothstep( washW, washW.mul( 0.3 ), abs( bx ) ) ).toVar();
			const streak = noise( vec2( bx.mul( 1.7 ), time.mul( 0.9 ) ) );
			const patch = noise( p.mul( 0.45 ).add( time.mul( 0.13 ) ) );
			// churning white right behind the transom (the propeller race), streaks and patches further aft
			const race = smoothstep( 7.0, 0.0, behind ).mul( 0.9 );
			// streaks: noise stretched along the track (fixed in the water), deposited with the wash so
			// the band is laid down as long ragged streaks, not a uniform ribbon
			const along = p.x.mul( sn ).add( p.y.mul( cs ) ), across = p.x.mul( cs ).sub( p.y.mul( sn ) );
			const streaks = noise( vec2( along.mul( 0.12 ), across.mul( 0.9 ) ) ).mul( 0.7 ).add( noise( vec2( along.mul( 0.3 ), across.mul( 1.9 ) ).add( 5.3 ) ).mul( 0.3 ) );
			const mottle = clamp( streaks.sub( 0.5 ).mul( 3.2 ), - 0.85, 0.85 ).add( 1 ).toVar();
			const washGen = inWash.mul( this.uWash ).mul( this.uSpeed.add( 1.5 ) ).mul( streak.mul( patch ).mul( 0.6 ).add( race ).add( 0.08 ) ).mul( mix( mottle, float( 1 ), smoothstep( 3.0, 0.0, behind ).mul( 0.7 ) ) );

			// bow: the spray thrown off the forward hull falls back in a band just outside the
			// waterline (up to ~1.3 m out), where the bow wave rolls away from the hull
			const outside = float( 1 ).sub( smoothstep( 0.0, 0.03, Ph ) );
			const near = smoothstep( 0.0, 0.03, immersionAt( abs( bx ).sub( 0.9 ).max( 0 ), bz ) );
			const bowBand = outside.mul( near ).mul( smoothstep( - 0.5, 1.2, bz ) ).mul( smoothstep( 4.2, 3.2, bz ) );
			const bowGen = bowBand.mul( this.uBow ).mul( this.uSpeed.add( 1 ) ).mul( noise( vec2( bz.mul( 1.5 ), time.mul( 3.0 ) ) ).mul( 1.4 ).add( 0.1 ) ).mul( 0.35 );

			// Dissipation is a turbulent eddy viscosity acting on w in flux form: it conserves mass
			// exactly, damps short (breaking, choppy) waves ~k^2 and leaves the refilling flow behind
			// the hull alone. Breaking crests and the surf feed the turbulence; the turbulent wash
			// also stirs the surface (boils).
			const nu = ( tb ) => tb.mul( 0.6 ).min( 1.0 ).add( this.uVisc );
			const nuC = nu( turb ).toVar();
			const flux = nuC.add( nu( sL.w ) ).mul( sL.y.sub( w ) )
				.add( nuC.add( nu( sR.w ) ).mul( sR.y.sub( w ) ) )
				.add( nuC.add( nu( sD.w ) ).mul( sD.y.sub( w ) ) )
				.add( nuC.add( nu( sU.w ) ).mul( sU.y.sub( w ) ) ).mul( 0.5 / ( CELL * CELL ) );
			// boils: Laplacian of (turbulence x noise), so the stirring sums to zero (conserves mass)
			const boil = float( 0 ).toVar();
			If( turb.add( sL.w ).add( sR.w ).add( sD.w ).add( sU.w ).greaterThan( 0.02 ), () => {

				const drift = vec2( time.mul( 0.5 ), time.mul( - 0.35 ) );
				const stir = ( q, tb ) => noise( q.mul( 0.45 ).add( drift ) ).sub( 0.5 ).mul( tb.min( 1.5 ) );
				const c0 = stir( p, turb );
				boil.assign( stir( p.sub( vec2( CELL, 0 ) ), sL.w ).add( stir( p.add( vec2( CELL, 0 ) ), sR.w ) )
					.add( stir( p.sub( vec2( 0, CELL ) ), sD.w ) ).add( stir( p.add( vec2( 0, CELL ) ), sU.w ) ).sub( c0.mul( 4 ) ) );

			} );
			const w1 = w.add( flux.mul( dt ) ).add( boil.mul( dt.mul( this.uBoil ) ) ).mul( exp( dt.mul( - 0.006 ) ) ).mul( keep );

			// foam: thick foam thins quickly (bubbles rise and burst), the lace lingers
			const lapF = sL.z.add( sR.z ).add( sD.z ).add( sU.z ).sub( s.z.mul( 4 ) ).div( CELL * CELL );
			// foam streaks widen: molecular-ish spreading plus turbulent mixing in the wash
			const f0 = s.z.add( lapF.mul( dt.mul( turb.mul( 0.3 ).add( 0.04 ) ).min( 0.03 ) ) ).max( 0 );
			const gen = washGen.add( bowGen ).add( brk.mul( 3 ) ).add( surf.mul( 4 ) ).mul( this.foamGain );
			// the lace breaks up as it ages: it lingers in some patches and streaks and clears fast in
			// others (a fixed noise in the water: the pattern stays put while the foam thins)
			const breakup = noise( p.mul( 0.17 ).add( vec2( 13.1, 7.3 ) ) ).mul( 0.65 ).add( noise( p.mul( 0.55 ).add( vec2( 3.7, 1.9 ) ) ).mul( 0.35 ) ).toVar();
			const clear = smoothstep( 0.3, 0.72, breakup ).mul( 2.6 ).add( 0.2 );
			const foam = min( f0.mul( exp( dt.negate().mul( f0.mul( f0 ).mul( 0.6 ).add( clear.div( 35 ) ) ) ) ).add( gen.mul( dt ) ), 3 );
			const turb1 = turb.mul( exp( dt.mul( - 1 / 6 ) ) ).add( inWash.mul( this.uWash ).mul( this.uSpeed.add( 1.5 ) ).mul( 0.5 ).add( brk.mul( 4 ) ).add( surf.mul( 6 ) ).mul( dt ) ).min( 3 );

			// aeration (0..~3): injected broadly by the race (not only where the surface foam is), by
			// breaking and the surf; mixed sideways by the turbulence; the bubbles rise out over ~25 s
			const aC = loadAer( col, row ).toVar();
			const lapA = loadAer( cL, row ).add( loadAer( cR, row ) ).add( loadAer( col, rD ) ).add( loadAer( col, rU ) ).sub( aC.mul( 4 ) );
			const a0 = aC.add( lapA.mul( turb.mul( 0.35 ).add( 0.07 ).mul( dt ).div( CELL * CELL ).min( 0.2 ) ) ).max( 0 );
			// the bubble cloud is wider than the white race and its edge wanders (soft, irregular)
			const wA = washW.mul( noise( p.mul( 0.22 ).add( time.mul( 0.05 ) ) ).mul( 0.8 ).add( 0.8 ) );
			const aerBand = smoothstep( - 0.6, 0.3, behind ).mul( smoothstep( 9.0, 2.0, behind ) ).mul( smoothstep( wA, wA.mul( 0.2 ), abs( bx ) ) );
			const aerGen = aerBand.mul( this.uWash ).mul( this.uSpeed.add( 1.5 ) ).mul( race.mul( 1.2 ).add( 0.15 ).mul( patch.mul( 1.3 ).add( streak.mul( 0.5 ) ).add( 0.1 ) ) )
				.mul( mottle.mul( mottle ).mul( 0.55 ) ).add( bowGen.mul( 0.6 ) ).add( brk.mul( 1.5 ) ).add( surf.mul( 2 ) ).mul( this.aerGain );
			// bubbles rise out over ~25 s, faster in some patches (mottled as it ages)
			const rise = smoothstep( 0.25, 0.75, breakup.mul( 0.6 ).add( patch.mul( 0.4 ) ) ).mul( 1.3 ).add( 0.45 );
			const aer = min( a0.mul( exp( dt.negate().mul( a0.mul( 0.06 ).add( rise.div( 25 ) ) ) ) ).add( aerGen.mul( dt ) ), 3 ).mul( wet );
			this.aerB.element( idx ).assign( aer );
			textureStore( this.aerTex, uvec2( col, row ), vec4( aer, 0, 0, 0 ) );

			scratch.element( idx ).assign( vec4( h, w1, foam, turb1 ) );
			// the free surface is h (pushed down under the hull, the hollow behind the transom)
			textureStore( this.display, uvec2( col, row ), vec4( h, gx, gz, foam ) );

			return depthWeights( d ).mul( h.add( P ) );

		} ).setLayout( { name: 'wakeCell', type: 'vec4', inputs: [ { name: 'col', type: 'uint' }, { name: 'row', type: 'uint' } ] } );

		this.rowKernel = Fn( () => {

			const t = localId.x;
			const row = workgroupId.x;
			for ( let e = 0; e < 2; e ++ ) {

				const col = t.add( uint( e * HALF ) );
				shRow.element( brev( col ) ).assign( cellPhysics( col, row ) );

			}

			workgroupBarrier();
			stages( shRow, t, [ 0 ], - 1 );
			for ( let e = 0; e < 2; e ++ ) {

				const col = t.add( uint( e * HALF ) );
				spec.element( row.mul( uint( N ) ).add( col ) ).assign( shRow.element( col ) );

			}

		} )().computeKernel( [ HALF, 1, 1 ] ).setName( 'Wake Rows' );

		// ---- pass 2: columns kx and -kx together (separates the two packed real fields)
		const dk = 2 * Math.PI / SIZE;
		const norm = 1 / ( N * N );
		this.colKernel = Fn( () => {

			const t = localId.x;
			const wg = workgroupId.x;
			If( wg.lessThanEqual( uint( HALF ) ), () => {

				const cA = wg;
				const cB = uint( N ).sub( wg ).bitAnd( uint( MASK ) );
				for ( let e = 0; e < 2; e ++ ) {

					const r = t.add( uint( e * HALF ) );
					const rb = brev( r ).toVar();
					shCol.element( rb ).assign( spec.element( r.mul( uint( N ) ).add( cA ) ) );
					shCol.element( rb.add( uint( N ) ) ).assign( spec.element( r.mul( uint( N ) ).add( cB ) ) );

				}

				workgroupBarrier();
				stages( shCol, t, [ 0, N ], - 1 );

				// Z = FFT(a) + i FFT(b) for each packed pair of real fields (a, b) = (s_i u, s_j u);
				// Y = L_i FFT(a) + i L_j FFT(b) = Z (L_i + L_j) / 2 + conj( Z(-k) ) (L_i - L_j) / 2
				const fx = float( select( cA.lessThan( uint( HALF ) ), int( cA ), int( cA ).sub( N ) ) ).mul( dk );
				const apply = ( Z, Zm, p, m ) => vec4(
					Z.x.mul( p.x ).add( Zm.x.mul( m.x ) ), Z.y.mul( p.x ).sub( Zm.y.mul( m.x ) ),
					Z.z.mul( p.y ).add( Zm.z.mul( m.y ) ), Z.w.mul( p.y ).sub( Zm.w.mul( m.y ) )
				);
				const out = [];
				for ( let e = 0; e < 2; e ++ ) {

					const r = t.add( uint( e * HALF ) );
					const rm = uint( N ).sub( r ).bitAnd( uint( MASK ) );
					const fz = float( select( r.lessThan( uint( HALF ) ), int( r ), int( r ).sub( N ) ) ).mul( dk );
					const k = length( vec2( fx, fz ) ).toVar();
					const L = DEPTHS.map( ( D ) => ( D === Infinity ? k : k.mul( tanh( min( k.mul( D ), 12 ) ) ) ).mul( norm ) ); // exp-based tanh overflows
					const p = vec2( L[ 0 ].add( L[ 1 ] ), L[ 2 ].add( L[ 3 ] ) ).mul( 0.5 ).toVar();
					const m = vec2( L[ 0 ].sub( L[ 1 ] ), L[ 2 ].sub( L[ 3 ] ) ).mul( 0.5 ).toVar();
					out.push( apply( shCol.element( r ), shCol.element( rm.add( uint( N ) ) ), p, m ).toVar() );
					out.push( apply( shCol.element( r.add( uint( N ) ) ), shCol.element( rm ), p, m ).toVar() );

				}

				workgroupBarrier();
				for ( let e = 0; e < 2; e ++ ) {

					const rb = brev( t.add( uint( e * HALF ) ) ).toVar();
					shCol.element( rb ).assign( out[ e * 2 ] );
					shCol.element( rb.add( uint( N ) ) ).assign( out[ e * 2 + 1 ] );

				}

				workgroupBarrier();
				stages( shCol, t, [ 0, N ], 1 );
				for ( let e = 0; e < 2; e ++ ) {

					const r = t.add( uint( e * HALF ) );
					spec.element( r.mul( uint( N ) ).add( cA ) ).assign( shCol.element( r ) );
					If( cB.notEqual( cA ), () => {

						spec.element( r.mul( uint( N ) ).add( cB ) ).assign( shCol.element( r.add( uint( N ) ) ) );

					} );

				}

			} ).ElseIf( wg.greaterThan( uint( HALF ) ).and( wg.lessThanEqual( uint( HALF + Math.ceil( TW * TH / HALF ) ) ) ), () => {

				// near-field template (spare workgroups): running mean of the wake height at
				// boat-frame points
				const k = wg.sub( uint( HALF + 1 ) ).mul( uint( HALF ) ).add( t );
				If( k.lessThan( uint( TW * TH ) ), () => {

					const tx = k.mod( uint( TW ) ), tz = k.div( uint( TW ) );
					const bx = float( tx ).add( 0.5 ).mul( ( TX1 - TX0 ) / TW ).add( TX0 );
					const bz = float( tz ).add( 0.5 ).mul( ( TZ1 - TZ0 ) / TH ).add( TZ0 );
					const cs = this.uBoatRot.x, sn = this.uBoatRot.y;
					const pw = this.uBoatPos.add( vec2( bx.mul( cs ).add( bz.mul( sn ) ), bz.mul( cs ).sub( bx.mul( sn ) ) ) );
					const eta = displayT.sample( pw.div( SIZE ) ).level( 0 ).x;
					const prev = this.nearBuf.element( k );
					const mean = prev.add( eta.sub( prev ).mul( this.uNearRate ) ).toVar();
					prev.assign( mean );
					const border = tx.equal( uint( 0 ) ).or( tx.equal( uint( TW - 1 ) ) ).or( tz.equal( uint( 0 ) ) ).or( tz.equal( uint( TH - 1 ) ) );
					const D = hullT.sample( vec2( abs( bx ).div( HB.X1 ), bz.sub( HB.Z0 ).div( HB.Z1 - HB.Z0 ) ) ).level( 0 ).x;
					const m = select( border, float( 0 ), smoothstep( 0.0, 0.03, D ) );
					textureStore( this.nearTex, uvec2( tx, tz ), vec4( mean, m, 0, 0 ) );

				} );

			} );

		} )().computeKernel( [ HALF, 1, 1 ] ).setName( 'Wake Columns' );

		// ---- pass 3: inverse FFT along rows, time step, sponge
		this.invKernel = Fn( () => {

			const t = localId.x;
			const row = workgroupId.x;
			for ( let e = 0; e < 2; e ++ ) {

				const col = t.add( uint( e * HALF ) );
				shRow.element( brev( col ) ).assign( spec.element( row.mul( uint( N ) ).add( col ) ) );

			}

			workgroupBarrier();
			stages( shRow, t, [ 0 ], 1 );

			const rl = float( int( row ).sub( int( this.uOrigin.y ) ).bitAnd( int( MASK ) ) ).add( 0.5 );
			const ez = min( rl, float( N ).sub( rl ) );
			for ( let e = 0; e < 2; e ++ ) {

				const col = t.add( uint( e * HALF ) );
				const idx = row.mul( uint( N ) ).add( col );
				const Y = shRow.element( col );
				const sc = scratch.element( idx ).toVar();
				const p = cellPos( col, row );
				const d = G.seaLevel.sub( terrain.heightAt( p ) );
				const Lu = dot( depthWeights( d ), vec4( Y.x, Y.y, Y.z, Y.w ) );
				const w1 = sc.y.sub( Lu.mul( dt.mul( GRAVITY ) ) );
				const h1 = sc.x.add( w1.mul( dt ) );
				const cl = float( int( col ).sub( int( this.uOrigin.x ) ).bitAnd( int( MASK ) ) ).add( 0.5 );
				const edge = min( min( cl, float( N ).sub( cl ) ), ez );
				const sig = smoothstep( float( SPONGE ), float( 0 ), edge );
				const k = exp( dt.mul( sig.mul( sig ).mul( - 6 ) ) );
				state.element( idx ).assign( vec4( h1.mul( k ), w1.mul( k ), sc.z, sc.w ) );
				this.aerA.element( idx ).assign( this.aerB.element( idx ) );

			}

		} )().computeKernel( [ HALF, 1, 1 ] ).setName( 'Wake Inverse' );

	}

	// ---------------------------------------------------------------- readers (TSL)

	_buildReaders() {

		// one sampler per shader stage for the whole wake: the display texture (height, slopes,
		// foam); the near-field template is read with textureLoad (nearest filter, no sampler)
		const display = texture( this.display );
		const edgeDist = ( xz ) => {

			const rel = abs( xz.sub( this.uCenter ) );
			return float( SIZE / 2 ).sub( max( rel.x, rel.y ) );

		};
		const fade = ( xz ) => smoothstep( 4.0, 16.0, edgeDist( xz ) ).mul( this.uAmount );
		// foam and aeration: a long fade toward the window edge, no visible end of the track
		const fadeLong = ( xz ) => smoothstep( 4.0, 45.0, edgeDist( xz ) ).mul( this.uAmount );

		const near = texture( this.nearTex );
		// template texel coordinates of world xz (boat frame)
		const nearCoord = ( xz ) => {

			const rel = xz.sub( this.uBoatPos );
			const cs = this.uBoatRot.x, sn = this.uBoatRot.y;
			const bx = rel.x.mul( cs ).sub( rel.y.mul( sn ) ), bz = rel.x.mul( sn ).add( rel.y.mul( cs ) );
			return vec2( bx.sub( TX0 ).mul( TW / ( TX1 - TX0 ) ), bz.sub( TZ0 ).mul( TH / ( TZ1 - TZ0 ) ) ).sub( 0.5 );

		};

		const inTemplate = ( tc ) => tc.x.greaterThan( 0 ).and( tc.y.greaterThan( 0 ) ).and( tc.x.lessThan( TW - 1 ) ).and( tc.y.lessThan( TH - 1 ) );

		this._sampleFn = Fn( ( [ xz ] ) => {

			const s = display.sample( xz.div( SIZE ) ).level( 0 ).mul( vec4( vec3( fade( xz ) ), fadeLong( xz ) ) ).toVar();
			// the forced depression under the hull is covered by it: keep its edge out of the normals
			const tc = nearCoord( xz ).toVar();
			If( inTemplate( tc ), () => {

				const m = near.load( ivec2( tc.add( 0.5 ) ) ).y;
				s.assign( vec4( s.x, s.yz.mul( float( 1 ).sub( m ) ), s.w ) );

			} );
			return s;

		} ).setLayout( { name: 'wakeSample', type: 'vec4', inputs: [ { name: 'xz', type: 'vec2' } ] } );

		// aeration (0..aerOut): the aeration texture is read with textureLoad (bilinear by hand), then
		// mottled: a noise fixed in the water sets how milky each patch is and where the band's edge
		// falls (soft, irregular), so it breaks into clouds and streaks instead of a uniform ribbon
		const aerT = texture( this.aerTex );
		const vhash = ( ix, iy ) => {

			let v = uint( ix ).mul( uint( 0x8da6b343 ) ).bitXor( uint( iy ).mul( uint( 0xd8163841 ) ) );
			v = v.bitXor( v.shiftRight( uint( 13 ) ) ).mul( uint( 0x5bd1e995 ) );
			v = v.bitXor( v.shiftRight( uint( 15 ) ) );
			return float( v.shiftRight( uint( 8 ) ) ).mul( 1 / 16777216 );

		};
		const vnoise = ( q ) => {

			const i = floor( q ).toVar();
			const f = fract( q );
			const u = f.mul( f ).mul( float( 3 ).sub( f.mul( 2 ) ) ).toVar();
			const ix = int( i.x ), iy = int( i.y );
			return mix( mix( vhash( ix, iy ), vhash( ix.add( 1 ), iy ), u.x ), mix( vhash( ix, iy.add( 1 ) ), vhash( ix.add( 1 ), iy.add( 1 ) ), u.x ), u.y );

		};
		this._aerFn = Fn( ( [ xz ] ) => {

			const out = float( 0 ).toVar();
			const tc = xz.div( CELL ).sub( 0.5 ).toVar();
			const i = ivec2( floor( tc ) ).toVar();
			const f = fract( tc );
			const ld = ( o ) => aerT.load( i.add( o ).bitAnd( ivec2( MASK ) ) ).x;
			const a = mix( mix( ld( ivec2( 0, 0 ) ), ld( ivec2( 1, 0 ) ), f.x ), mix( ld( ivec2( 0, 1 ) ), ld( ivec2( 1, 1 ) ), f.x ), f.y ).toVar();
			If( a.greaterThan( 0.01 ), () => {

				const m = vnoise( xz.mul( 0.3 ).add( vec2( G.time.mul( 0.02 ), 0 ) ) ).mul( 0.55 ).add( vnoise( xz.mul( 0.85 ).add( 17.3 ) ).mul( 0.45 ) ).toVar();
				const aN = float( 1 ).sub( exp( a.mul( - 0.5 ) ) ).mul( m.mul( 0.8 ).add( 0.6 ) );
				out.assign( smoothstep( m.mul( 0.2 ).add( 0.03 ), m.mul( 0.2 ).add( 0.55 ), aN ).mul( this.aerOut ).mul( fadeLong( xz ) ) );

			} );
			return out;

		} ).setLayout( { name: 'wakeAeration', type: 'float', inputs: [ { name: 'xz', type: 'vec2' } ] } );

		this._heightFn = Fn( ( [ xz ] ) => {

			const h = display.sample( xz.div( SIZE ) ).level( 0 ).x.toVar();
			const tc = nearCoord( xz ).toVar();
			If( inTemplate( tc ), () => {

				// under the hull only waves moving relative to it remain (an old wake being crossed);
				// small residuals are sampling jitter of the steep near field and are dropped
				// (bilinear by hand: the template has no sampler)
				const i = ivec2( floor( tc ) );
				const f = fract( tc );
				const a = near.load( i ), b = near.load( i.add( ivec2( 1, 0 ) ) );
				const c = near.load( i.add( ivec2( 0, 1 ) ) ), d = near.load( i.add( ivec2( 1, 1 ) ) );
				const t = mix( mix( a, b, f.x ), mix( c, d, f.x ), f.y );
				const r = h.sub( t.x );
				h.assign( mix( h, r.mul( smoothstep( this.uDead.x, this.uDead.y, abs( r ) ) ), t.y ) );

			} );

			return h.mul( fade( xz ) ).mul( this.amplitude );

		} ).setLayout( { name: 'wakeHeight', type: 'float', inputs: [ { name: 'xz', type: 'vec2' } ] } );

	}

	// Displacement (vec3) of the wake surface at world xz (Lagrangian point of the ocean grid).
	displacement( xz /*, depth */ ) {

		return vec3( 0, this._heightFn( xz ), 0 );

	}

	// { slopes: vec2 (dh/dx, dh/dz), foam: float, aeration: float (0..1, bubbles in the water column:
	// the prop race and breaking; lingers ~25 s, widening) }
	fragment( xz /*, depth */ ) {

		const s = this._sampleFn( xz ).toVar();
		return { slopes: s.yz.mul( this.amplitude ), foam: s.w, aeration: this._aerFn( xz ) };

	}

	// ---------------------------------------------------------------- per frame

	update( dt ) {

		const b = this.boat;
		const speed = Math.hypot( b.velocity.x, b.velocity.z );
		const moving = speed > 0.5 || ( b.driven && Math.abs( b.throttle ) > 0.04 );
		this.idleTime = moving ? 0 : this.idleTime + dt;

		if ( this.idleTime > this.settleTime ) {

			// asleep: nothing is dispatched; the (faded out) output is ignored by the shaders
			this.sleeping = true;
			this.uAmount.value = 0;
			// one step on the first frame compiles the pipelines behind the loading screen, not
			// when the player first opens the throttle
			if ( ! this.primed ) this._prime();
			return;

		}

		if ( this.sleeping ) {

			this.sleeping = false;
			this.hasWindow = false;
			this.uReset.value = 1;

		}

		this.uAmount.value = Math.min( 1, ( this.settleTime - this.idleTime ) / 4 );
		const h = Math.min( Math.max( dt, 1 / 240 ), 1 / 30 );
		this.uDt.value = h;

		// ---- window: keep the boat inside a box around the centre (moves by whole cells)
		const bx = b.position.x / CELL, bz = b.position.z / CELL;
		const box = 170; // the boat runs up to 68 m off centre: ~170 m of track behind it, ~25 m ahead
		const c = this.center;
		if ( ! this.hasWindow || Math.abs( bx - c.x ) > N || Math.abs( bz - c.y ) > N ) {

			c.set( Math.round( bx ), Math.round( bz ) );
			this.hasWindow = true;
			this.uReset.value = 1;

		}

		if ( bx - c.x > box ) c.x = Math.ceil( bx - box );
		if ( c.x - bx > box ) c.x = Math.floor( bx + box );
		if ( bz - c.y > box ) c.y = Math.ceil( bz - box );
		if ( c.y - bz > box ) c.y = Math.floor( bz + box );
		this.uPrev.value.copy( this.uOrigin.value );
		this.uOrigin.value.set( c.x - HALF, c.y - HALF );
		if ( this.uReset.value > 0.5 ) this.uPrev.value.copy( this.uOrigin.value );
		this.uCenter.value.set( c.x * CELL, c.y * CELL );

		// ---- boat
		const f = b.forward( _fwd );
		const yaw = Math.atan2( f.x, f.z );
		this.uBoatPos.value.set( b.position.x, b.position.z );
		this.uBoatRot.value.set( Math.cos( yaw ), Math.sin( yaw ) );
		this._scheduleTrim( speed );
		this.uSpeed.value = speed;
		this.uDyn.value = this.dynamicPressure * speed * speed / ( 2 * GRAVITY );
		const plane = THREE.MathUtils.smoothstep( speed, 2, 9 );
		this.uHollow.value = 0.35 + 0.015 * speed * speed;
		this.uHollowK.value = 0.3 * plane;
		this.uSource.value = this.sourceGain * ( 1 + 0.2 * plane );
		const prop = b.driven ? Math.abs( b.throttle ) * b.rpm : 0;
		this.uWash.value = prop * 1.0 + THREE.MathUtils.smoothstep( speed, 1.5, 7 ) * 0.5;
		this.uWashW.value = 0.7 + 0.6 * THREE.MathUtils.smoothstep( speed, 2, 9 );
		this.uBow.value = THREE.MathUtils.smoothstep( speed, 3.5, 9 );

		this.uNearRate.value = this.uReset.value > 0.5 ? 1 : 1 - Math.exp( - h / 0.25 );
		this.renderer.compute( [ this.rowKernel, this.colKernel, this.invKernel ], [ N, 1, 1 ] );
		this.primed = true;
		this.uReset.value = 0;
		this.stepCount ++;

	}

	_prime() {

		this.primed = true;
		this.uReset.value = 1;
		this.uPrev.value.copy( this.uOrigin.value );
		this.renderer.compute( [ this.rowKernel, this.colKernel, this.invKernel ], [ N, 1, 1 ] );

	}

	// Immersion change of the hull relative to its design waterline, c + a x + b z (m), scheduled
	// from speed rather than fitted to the boat's pose: the boat feels its own pressure field
	// through the water queries 1-3 frames late, and any dependence of the source on the boat's
	// heave / pitch / roll closes a delayed feedback loop (porpoising). Coming onto the plane the
	// hull rises and the bow lifts out, so the pressure footprint moves aft.
	_scheduleTrim( speed ) {

		const plane = THREE.MathUtils.smoothstep( speed, 3, 10 );
		this.uTrim.value.set( - 0.08 * plane, 0, - 0.035 * plane );

	}

}

const _fwd = new THREE.Vector3();
