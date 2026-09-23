import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, uint, vec2, vec4, uvec2, globalId, texture, textureStore, max, min, exp, clamp, saturate,
	smoothstep, select, length, mix, fract, dot, If, floor, ivec2, textureLoad, fwidth, storage, atomicAdd,
} from 'three/tsl';
import { G } from '../core/Globals.js';
import { makeLaceTexture, LACE_TILE } from './SurfFoam.js';

// Eulerian state over the main beach, updated every frame on the GPU:
//   r = foam carried by the water (made by the bore roller, the plunge point, the swash front and
//       the spray falling back; advected with the actual flow: bores, uprush, backwash; thinned
//       where the flow spreads it out)
//   g = sand wetness (1 while covered, dries over ~half a minute)
//   b = foam stranded on the sand when the water drains away (pops over a few seconds)
//   a = depth-averaged flow speed along the local wave direction (m/s): carries the foam pattern
//       (SurfFoam flow map) and gives the divergence that thins the foam
export class ShoreSim {

	constructor( renderer, { terrainGPU, shore, center = new THREE.Vector2( 10, - 25 ), size = 380, res = 768 } ) {

		this.renderer = renderer;
		this.terrain = terrainGPU;
		this.shore = shore;
		this.res = res;
		this.uMin = uniform( new THREE.Vector2( center.x - size / 2, center.y - size / 2 ) ).setName( 'ssMin' );
		this.uSize = uniform( size ).setName( 'ssSize' );
		this.dryTime = uniform( 28 ).setName( 'ssDry' );
		this.foamLife = uniform( 4.5 ).setName( 'ssFoamLife' ); // on the thin swash sheet
		this.surfFoamLife = uniform( 2.6 ).setName( 'ssSurfLife' ); // in the turbulent surf zone
		this.residueLife = uniform( 5.0 ).setName( 'ssResidue' );
		this.foamGen = uniform( 1.0 ).setName( 'ssFoamGen' );

		// tileable lace (bubble strands, bubbles, mottling, per-cell random), generated once on the CPU
		this.lace = makeLaceTexture();
		// cheap filtered wave direction over the region (lace motion, surf zone turbidity)
		shore.buildDirTexture( { min: new THREE.Vector2( center.x - size / 2, center.y - size / 2 ), size } );

		const make = ( name ) => {

			const t = new THREE.StorageTexture( res, res );
			t.type = THREE.HalfFloatType;
			t.format = THREE.RGBAFormat;
			// nearest + read with loads everywhere (manual bilinear): no sampler binding in any material
			t.magFilter = t.minFilter = THREE.NearestFilter;
			t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
			t.generateMipmaps = false;
			t.name = name;
			return t;

		};

		this.stateA = make( 'shoreStateA' ); // read by materials
		this.stateB = make( 'shoreStateB' ); // written by the sim

		// foam deposited by spray falling back into the water (drops per texel, atomic adds from the
		// spray update, consumed and cleared here)
		this.depositAttr = new THREE.StorageBufferAttribute( new Uint32Array( res * res ), 1 );
		this.deposit = storage( this.depositAttr, 'uint', res * res ).setName( 'shoreDeposit' );
		this.depositAtomic = storage( this.depositAttr, 'uint', res * res ).toAtomic().setName( 'shoreDepositA' );
		this.depositGain = uniform( 0.02 ).setName( 'ssDepositGain' ); // foam per drop

		this.kernel = Fn( () => {

			const ij = vec2( globalId.xy );
			const uv = ij.add( 0.5 ).div( res );
			const p = this.uMin.add( uv.mul( this.uSize ) ).toVar();
			const ground = terrainGPU.heightAt( p ).toVar();
			const depth = G.seaLevel.sub( ground ).toVar();
			const dt = G.dt;

			const here = textureLoad( this.stateA, ivec2( globalId.xy ) );
			const di = globalId.y.mul( res ).add( globalId.x );
			const drops = this.deposit.element( di ).toFloat().toVar();
			If( drops.greaterThan( 0 ), () => {

				this.deposit.element( di ).assign( uint( 0 ) );

			} );

			// far from the surf and swash zone nothing happens: just let everything decay
			If( depth.greaterThan( 7 ).or( ground.greaterThan( 3.2 ) ), () => {

				const k = exp( dt.negate().div( 2.0 ) );
				textureStore( this.stateB, uvec2( globalId.xy ), vec4( here.x.mul( k ), here.y.mul( exp( dt.negate().div( this.dryTime ) ) ), here.z.mul( k ), 0 ) );

			} ).Else( () => {

				const sw = shore.evaluate( p, depth, ground, { withNormal: false, world: true } );

				// fraction of this texel covered by water: open water, or the swash sheet up to its leading
				// edge (soft over one texel, so no field stored here shows the texel grid)
				const cov = select( depth.greaterThan( 0.03 ), float( 1 ), saturate( sw.runup.sub( sw.inland ).div( size / res ).add( 0.5 ) ) ).toVar();
				const covered = cov.greaterThan( 0.5 );
				const vel = select( covered, sw.flow, vec2( 0 ) ).toVar();

				// semi-Lagrangian advection (backtrace)
				const back = uv.sub( vel.mul( dt ).div( this.uSize ) );
				const prev = this.stateAt( back );

				// foam: made where the bore roller / plunge point / swash front pass, torn into patches by
				// the mottling of the lace texture, then it decays (bubbles rising and popping) and drains
				// into the sand once the water has gone
				const mott = texture( this.lace, p.div( LACE_TILE * 4.3 ) ).level( 2 ).z;
				const patch = smoothstep( 0.25, 0.75, mott ).mul( 0.8 ).add( 0.35 );
				const swashy = smoothstep( 0.35, 0.05, depth );
				// dense foam collapses within a second or two (big bubbles burst first), the lace it leaves lingers
				const lace = mix( this.surfFoamLife, this.foamLife, swashy );
				const life = mix( float( 0.7 ), mix( lace, float( 0.8 ), smoothstep( 0.3, 0.8, prev.x ) ), cov );
				const gen = sw.foam.mul( 2.2 ).add( sw.swashFoam.mul( 1.4 ) ).mul( patch ).mul( cov );
				// the foam is diluted where the flow spreads it out (the uprush thinning as it climbs, the
				// backwash draining): d(foam)/dt = - foam * du/ds along the flow
				const h = size / res;
				const dirH = sw.dir.mul( h / size );
				const uAhead = this.stateAt( uv.add( dirH ) ).w, uBehind = this.stateAt( uv.sub( dirH ) ).w;
				const spreadRate = max( uAhead.sub( uBehind ).div( 2 * h ), 0 );
				const splash = drops.mul( this.depositGain ).mul( cov );
				const foam = min( prev.x.mul( exp( dt.negate().mul( float( 1 ).div( life ).add( spreadRate ) ) ) ).add( gen.mul( this.foamGen ).mul( dt ) ).add( splash ), 1.0 );

				// wetness: saturated while covered, then dries
				const wet = max( here.y.mul( exp( dt.negate().div( this.dryTime ) ) ), cov );

				// residue: foam stranded on the sand when the water leaves (the draining film gathers its
				// bubbles into lines, so it concentrates); washed away by the next uprush
				const stranded = min( here.x.mul( 2.4 ), 1 ).mul( float( 1 ).sub( cov ) );
				const residue = max( here.z.mul( mix( exp( dt.negate().div( this.residueLife ) ), float( 0.85 ), cov ) ), stranded );

				textureStore( this.stateB, uvec2( globalId.xy ), vec4( foam, wet, residue, dot( vel, sw.dir ) ) );

			} );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Shore Sim' );

	}

	update() {

		const r = this.renderer;
		r.compute( this.kernel, [ this.res / 8, this.res / 8, 1 ] );
		r.copyTextureToTexture( this.stateB, this.stateA );

	}

	// ---- TSL readers

	uvOf( xz ) {

		return xz.sub( this.uMin ).div( this.uSize );

	}

	inside( uv ) {

		return smoothstep( 0.0, 0.02, uv.x ).mul( smoothstep( 1.0, 0.98, uv.x ) ).mul( smoothstep( 0.0, 0.02, uv.y ) ).mul( smoothstep( 1.0, 0.98, uv.y ) );

	}

	// bilinear state at texture coordinate uv, from 4 loads
	stateAt( uv ) {

		const res = this.res;
		const f = clamp( uv.mul( res ).sub( 0.5 ), 0, res - 1.001 );
		const i = ivec2( floor( f ) );
		const t = fract( f );
		const tex = this.stateA;
		const a = textureLoad( tex, i ), b = textureLoad( tex, i.add( ivec2( 1, 0 ) ) );
		const c = textureLoad( tex, i.add( ivec2( 0, 1 ) ) ), d = textureLoad( tex, i.add( ivec2( 1, 1 ) ) );
		return mix( mix( a, b, t.x ), mix( c, d, t.x ), t.y );

	}

	// raw state (foam, wetness, residue amount, lace offset), faded out at the region border
	state( xz ) {

		const uv = this.uvOf( xz ).toVar();
		const out = vec4( 0 ).toVar();
		If( uv.x.greaterThan( 0 ).and( uv.x.lessThan( 1 ) ).and( uv.y.greaterThan( 0 ) ).and( uv.y.lessThan( 1 ) ), () => {

			out.assign( this.stateAt( uv ).mul( this.inside( uv ) ) );

		} );
		return out;

	}

	// vec4( foam amount on the water, sand wetness, foam amount left on the sand, flow speed ).
	// Used by the water (x, w), the underwater lighting (x) and the terrain (y, z; for the lacy look
	// of the foam left on the sand use sandFoam()). No sampler bindings.
	sample( xz ) {

		return this.state( xz );

	}

	// bilinear lace lookup from 4 loads of the nearest-filtered copy (no sampler binding needed)
	laceLoad( q ) {

		const tex = this.lace.userData.nearest, n = this.lace.userData.size;
		const f = q.div( LACE_TILE ).mul( n ).sub( 0.5 );
		const i = ivec2( floor( f ) );
		const t = fract( f );
		const m = ivec2( n - 1, n - 1 );
		const a = textureLoad( tex, i.bitAnd( m ) ), b = textureLoad( tex, i.add( ivec2( 1, 0 ) ).bitAnd( m ) );
		const c = textureLoad( tex, i.add( ivec2( 0, 1 ) ).bitAnd( m ) ), d = textureLoad( tex, i.add( ivec2( 1, 1 ) ).bitAnd( m ) );
		return mix( mix( a, b, t.x ), mix( c, d, t.x ), t.y );

	}

	// Foam left on the sand (0..1): thin bubble lines and single bubbles where the draining water
	// left its foam, popping patch by patch as it dries. Static on the sand (world space). s:
	// sample( xz ) if the caller already has it.
	sandFoam( xz, s = null ) {

		if ( ! s ) s = this.state( xz );
		const r = s.z;
		const out = float( 0 ).toVar();
		If( r.greaterThan( 0.01 ), () => {

			const lace = this.laceLoad( xz.add( 11.3 ) ).toVar();
			// fade to the average where a pixel covers several strands (no mipmaps on the load path)
			const fp = length( fwidth( xz ) ).div( LACE_TILE ).mul( this.lace.userData.size );
			const near = smoothstep( 3.0, 1.2, fp );
			const keep = smoothstep( lace.w.mul( 0.55 ), lace.w.mul( 0.55 ).add( 0.08 ), r ); // staggered popping
			// thin bubble lines where the strands were, a little wider where more foam was left
			const lw = r.mul( 0.1 ).add( 0.06 );
			const strand = float( 1 ).sub( smoothstep( lw, lw.add( 0.07 ), lace.x ) ).mul( lace.z.mul( 0.5 ).add( 0.6 ) );
			const lines = max( strand.mul( smoothstep( 0.02, 0.25, r ) ).mul( keep ), lace.y.mul( keep ).mul( 0.8 ) );
			out.assign( mix( smoothstep( 0.08, 0.6, r ).mul( 0.12 ), lines, near ).mul( 0.85 ) );

		} );
		return out;

	}

	// Deposit foam where spray falls back (TSL, compute): n drops at world xz
	depositAt( xz, n ) {

		const uv = this.uvOf( xz );
		If( uv.x.greaterThan( 0 ).and( uv.x.lessThan( 1 ) ).and( uv.y.greaterThan( 0 ) ).and( uv.y.lessThan( 1 ) ), () => {

			const ij = uvec2( uv.mul( this.res ) );
			atomicAdd( this.depositAtomic.element( ij.y.mul( this.res ).add( ij.x ) ), n );

		} );

	}

}
