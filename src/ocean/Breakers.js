import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, int, uint, vec2, vec3, vec4, storage, instanceIndex, attribute, If, Loop, max, clamp,
	saturate, mix, smoothstep, length, normalize, dot, pow, sqrt, abs, select, sin, cos, floor, positionWorld,
	cameraPosition, varyingProperty, frontFacing, reflect, texture, mrt, fwidth, dFdx, dFdy, cross, sign,
} from 'three/tsl';
import { G, GRAVITY } from '../core/Globals.js';
import { LAYERS } from '../core/SceneRenderer.js';
import { fresnelDielectric } from './WaterMaterial.js';
import { staticVelocity } from '../post/CameraVelocity.js';
import { SPRAY } from '../fx/Spray.js';
import { makeLaceTexture, LACE_TILE } from './SurfFoam.js';

// Plunging breakers along the main beach.
//
// Stations are laid out every ~0.6 m along the shoreline (CPU, once). Each frame a compute pass
// marches every station's transect through the surf zone and finds the crests of the breaking
// waves as crossings of the wave phase s = (t - T) / period + wobble with integers (the same
// analytic field that drives the water surface, so the crest found here is exactly the crest of
// the rendered surface). For every crest it stores the lip root, the breaking progress and the
// wave height, and emits spray where a real breaker makes it (rates ~ the energy released there):
//  * nothing while the face steepens: the face and the lip are clean, glassy water
//  * a few drops and ligaments torn off the leading edge of the falling lip, falling with it
//  * the splash-up where the lip hits the trough (the plunge point): a burst of drops, ligaments and
//    dense spray thrown up (up to ~H) and forward, falling back within 1-2 s, a puff blown out of
//    the collapsing barrel, and a mist that drifts off with the wind
//  * the turbulent front of the roller: small drops and a low mist, all the way to the shore, and
//    a splash where the bore runs into the backwash of the previous wave
//  * spindrift (fine drops blown back over the crest) only in a strong offshore wind
// Drops falling back into the water leave foam there (ShoreSim).
//
// The thrown lip is a separate ribbon mesh extruded along the crest (Thuerey et al. 2007): a
// ballistic curtain leaving the crest horizontally and falling in front of the concave face,
// shaded like the water (Fresnel sky reflection, light through the thin sheet, aerated streaks
// that grow toward the tip) and blended over the water with premultiplied alpha. Its root lies
// on the crest of the water surface and fades in there, so there is no visible seam.

const NV = 20; // profile vertices across the lip (2 on the back of the crest + 18 along the curtain)

export class Breakers {

	constructor( renderer, { surface, shore, terrainData, sky, spray = null, clouds = null } ) {

		this.renderer = renderer;
		this.surface = surface;
		this.shore = shore;
		this.sky = sky;
		this.spray = spray;
		this.clouds = clouds;

		this.params = {
			spray: uniform( 1 ).setName( 'brkSpray' ), // emission multiplier
			sheet: uniform( 1 ).setName( 'brkSheet' ), // lip opacity multiplier
			emitRange: uniform( 240 ).setName( 'brkEmitRange' ), // no emission beyond this camera distance
		};
		this.cameraPos = uniform( new THREE.Vector3() ).setName( 'brkCam' );

		const t0 = performance.now();
		const st = buildStations( terrainData );
		this.NS = st.count;
		this.spacing = st.spacing;
		this.stationData = st.data;
		this.stations = storage( new THREE.StorageBufferAttribute( st.data, 4 ), 'vec4', this.NS ).toReadOnly().setName( 'brkStations' );
		// per station and slot (wave parity): 3 x vec4
		//   (root.xyz, b) (back.xyz, H) (dir.xz, trough y, wave id 1..1024 or 0 = none)
		this.crestAttr = new THREE.StorageBufferAttribute( new Float32Array( this.NS * 2 * 3 * 4 ), 4 );
		this.crest = storage( this.crestAttr, 'vec4', this.NS * 6 ).setName( 'brkCrest' );
		this.crestRead = storage( this.crestAttr, 'vec4', this.NS * 6 ).toReadOnly().setName( 'brkCrestR' );
		this.setupMs = performance.now() - t0;

		if ( spray ) {

			// the spray shader asks the crests for the wave's shadow on the particles they made, and the
			// drops falling back into the water leave foam in the shore simulation
			spray.waveShadow = ( p, tag ) => this.sprayShadow( p, tag );
			if ( ! spray.shoreSim && surface.shoreSim ) spray.shoreSim = surface.shoreSim;

		}

		this._buildKernel();
		this._buildMesh();

	}

	update( camera ) {

		this.cameraPos.value.copy( camera.position );
		this.renderer.compute( this.kernel );

	}

	// ------------------------------------------------------------------ crest finder + emitters

	_buildKernel() {

		const shore = this.shore;
		const S = this.surface;
		const NS = this.NS;
		const STEP = 2.0, K = 64; // transects: 128 m seaward from the shoreline
		const fft = S.fft;

		// FFT displacement at a Lagrangian point (the short cascades that survive in the surf zone)
		const fftDisp = ( p, depth ) => {

			const d = vec3( 0 ).toVar();
			for ( let c = 1; c < fft.cascades; c ++ ) {

				const L = fft.sizes[ c ];
				const texel = L / 256;
				const level = Math.max( Math.log2( 0.35 / texel ) + 0.7, 0 );
				d.addAssign( texture( fft.displacementTexture, p.div( L ) ).depth( c ).level( level ).xyz.mul( S.cascadeAttenuation( c, depth ) ) );

			}

			return d.mul( S.amplitude );

		};

		this.kernel = Fn( () => {

			const i = instanceIndex;
			If( i.lessThan( uint( NS ) ), () => {

				const st = this.stations.element( i );
				const o = st.xy;
				const n = st.zw; // toward the sea
				const base = i.mul( 6 );
				// clear both slots
				this.crest.element( base.add( 2 ) ).assign( vec4( 0 ) );
				this.crest.element( base.add( 5 ) ).assign( vec4( 0 ) );

				const sPrev = float( 0 ).toVar();
				Loop( { start: int( 0 ), end: int( K ), type: 'int', condition: '<' }, ( { i: k } ) => {

					const dist = float( k ).mul( STEP );
					const s = shore.phaseAt( o.add( n.mul( dist ) ) ).s.toVar();
					// s grows seaward: a crest (integer phase) lies between this sample and the previous one
					If( k.greaterThan( int( 0 ) ).and( floor( s ).greaterThan( floor( sPrev ) ) ), () => {

						const m = floor( s ).toVar();
						// secant refinement on the exact phase
						const lo = dist.sub( STEP ).toVar(), hi = dist.toVar();
						const sLo = sPrev.toVar(), sHi = s.toVar();
						const x = lo.add( m.sub( sLo ).div( max( sHi.sub( sLo ), 1e-5 ) ).mul( STEP ) ).toVar();
						for ( let it = 0; it < 2; it ++ ) {

							const sx = shore.phaseAt( o.add( n.mul( x ) ) ).s;
							If( sx.lessThan( m ), () => {

								lo.assign( x );
								sLo.assign( sx );

							} ).Else( () => {

								hi.assign( x );
								sHi.assign( sx );

							} );
							x.assign( lo.add( m.sub( sLo ).div( max( sHi.sub( sLo ), 1e-5 ) ).mul( hi.sub( lo ) ) ) );

						}

						const pc = o.add( n.mul( x ) ).toVar();
						this._processCrest( i, pc, m, fftDisp );

					} );
					sPrev.assign( s );

				} );

			} );

		} )().compute( NS, [ 64 ] ).setName( 'Surf Crests' );

	}

	// One crest of wave m at Lagrangian point pc (station i): store the lip frame and emit spray.
	_processCrest( i, pc, m, fftDisp ) {

		const shore = this.shore;
		const S = this.surface;
		const spray = this.spray;
		const ph = shore.phaseAt( pc );
		const dir = ph.dir.toVar();
		const along = ph.along;
		const ground = S.terrain.heightAt( pc );
		const depth = G.seaLevel.sub( ground ).toVar();
		const A = shore.waveAmp( m, along ).toVar();
		const cr = shore.crestParams( A, depth );
		const b = cr.b.toVar();
		const env = smoothstep( 26, 13, depth ).mul( saturate( ph.exposure.mul( 1.4 ) ) ).mul( shore.enabled ).toVar();

		// followed from before it breaks until the bore reaches the shore (the lip sheet only uses b < 1.25)
		If( b.greaterThan( - 0.6 ).and( env.greaterThan( 0.3 ) ).and( depth.greaterThan( 0.12 ) ), () => {

			const c = sqrt( clamp( depth, 0.3, 25 ).mul( GRAVITY ) );
			const lam = c.mul( shore.period );
			const s0 = shore.shape( float( 0 ), A, depth, lam );
			const ub = float( 0.4 ).div( lam );
			const s1 = shore.shape( ub, A, depth, lam );
			const fd = fftDisp( pc, depth ).toVar();
			const d3 = vec3( dir.x, 0, dir.y );
			const root = vec3( pc.x, G.seaLevel, pc.y ).add( d3.mul( s0.x.mul( env ) ) ).add( vec3( 0, s0.y.mul( env ), 0 ) ).add( fd ).toVar();
			const back = vec3( pc.x, G.seaLevel, pc.y ).add( d3.mul( s1.x.mul( env ).sub( 0.4 ) ) ).add( vec3( 0, s1.y.mul( env ), 0 ) ).add( fd );
			const H = cr.H.mul( env ).toVar();
			const trough = G.seaLevel.add( cr.trough.mul( env ) ).add( fd.y ).toVar();
			// slot by wave parity (neighbouring stations agree on it), id = wave index mod 1024, + 1 (0 = none)
			const slot = uint( m.sub( floor( m.mul( 0.5 ) ).mul( 2 ) ) );
			const id = m.sub( floor( m.div( 1024 ) ).mul( 1024 ) ).add( 1 );
			const k = i.mul( 6 ).add( slot.mul( 3 ) );
			this.crest.element( k ).assign( vec4( root, b ) );
			this.crest.element( k.add( 1 ) ).assign( vec4( back, H ) );
			this.crest.element( k.add( 2 ) ).assign( vec4( dir, trough, id ) );

			if ( spray ) this._emit( i, slot, root, dir, b, H, trough, c, m, depth, shore.boreParams( A, depth ) );

		} );

	}

	// Spray from one crest (station i, crest slot `slot`), per frame. Positions and velocities come from
	// the analytic wave state; rates are per station (0.6 m of crest) and scale with the energy the
	// breaker releases (~ H^2.5 for the plunge, ~ Hb^2.5 for the roller).
	_emit( i, slot, root, dir, b, H, trough, c, m, depth, bore ) {

		const spray = this.spray;
		const P = this.params;
		const camFade = smoothstep( P.emitRange, P.emitRange.mul( 0.35 ), length( root.sub( this.cameraPos ) ) );
		const gain = P.spray.mul( camFade ).mul( G.dt ).toVar();
		const d3 = vec3( dir.x, 0, dir.y );
		const tg = vec3( dir.y.negate(), 0, dir.x );
		const up = vec3( 0, 1, 0 );
		const Hc = H.clamp( 0.15, 3.0 ).toVar();
		const E = pow( Hc, 2.5 ).toVar();
		const Hb = bore.Hb.max( 0 ).toVar();
		const Eb = pow( Hb.min( 2 ).div( 0.6 ), 2.5 ).toVar();
		const Wt = bore.Wt;
		const q = clamp( b.div( 0.9 ), 0, 1 );
		const Xi = H.mul( 0.8 );
		const Yi = max( root.y.sub( trough ), 0.05 ).toVar();
		const seed = i.mul( 7919 ).add( uint( m.sub( floor( m.div( 64 ) ).mul( 64 ) ) ).mul( 31 ) );
		// integer part: which crest made the particle (its wave shadow, see sprayShadow); fraction: random
		const tag = float( i.mul( 2 ).add( slot ).add( 1 ) );
		const offshore = max( dot( G.windDir, dir ).negate(), 0 ).mul( G.windSpeed );

		// ---- stages of the breaker (weights 0..1)
		const lipShed = smoothstep( 0.5, 0.75, b ).mul( float( 1 ).sub( smoothstep( 0.86, 0.93, b ) ) );
		// the splash-up is a fast burst (~0.15-0.2 s) as the lip hits the trough; its mist lingers
		const impact = smoothstep( 0.87, 0.92, b ).mul( float( 1 ).sub( smoothstep( 1.0, 1.14, b ) ) );
		const haze = smoothstep( 0.9, 1.0, b ).mul( float( 1 ).sub( smoothstep( 1.2, 1.5, b ) ) );
		const spit = smoothstep( 0.98, 1.08, b ).mul( float( 1 ).sub( smoothstep( 1.15, 1.3, b ) ) );
		const roller = smoothstep( 1.05, 1.3, b ).mul( smoothstep( 0.06, 0.25, Hb ) );
		const clash = roller.mul( smoothstep( 0.1, 0.2, depth ) ).mul( smoothstep( 0.7, 0.35, depth ) );
		const drift = smoothstep( 9, 15, offshore ).mul( smoothstep( - 0.4, 0.1, b ) ).mul( float( 1 ).sub( smoothstep( 0.75, 0.9, b ) ) );

		const count = ( rate, salt ) => uint( floor( rate.mul( gain ).add( spray.rand( seed, salt ) ) ) );
		const along = ( r ) => tg.mul( r.sub( 0.5 ).mul( this.spacing * 1.15 ) );
		// the leading edge of the falling lip
		const tipAt = ( r ) => root.add( d3.mul( Xi.mul( q ).mul( r.mul( 0.12 ).add( 0.88 ) ) ) ).sub( up.mul( Yi.mul( q ).mul( q ) ) );
		// A point on the front of the wave, on the water surface (the same curve as ShoreWaves' profile:
		// concave tube face while plunging, convex roller front of the bore), and its outward normal.
		// s: 0 = crest top .. 1 = foot. Everything the breaker throws starts just outside it.
		const wB = bore.wBore;
		const facePoint = ( sp ) => {

			const th = sp.mul( Math.PI / 2 );
			const ct = cos( th ), st = sin( th );
			const fx = mix( float( 1 ).sub( ct ), st, wB ), fy = mix( float( 1 ).sub( st ), ct, wB );
			const n = normalize( d3.mul( Yi.mul( mix( ct, st, wB ) ) ).add( up.mul( Wt.mul( mix( st, ct, wB ) ).add( 0.02 ) ) ) );
			return { p: root.add( d3.mul( Wt.mul( fx ) ) ).setY( trough.add( Yi.mul( fy ) ) ), n };

		};

		// the plunge point: in the trough ahead of the face while the tube is open; once the bore front
		// has formed over it, on the lower part of that front
		const xr = Xi.sub( bore.Xc ).toVar(); // horizontal distance from the crest top
		const plunge = ( r, r2 ) => {

			const f = facePoint( r2.mul( 0.45 ).add( 0.55 ) );
			const inTrough = root.add( d3.mul( xr.add( r.sub( 0.5 ).mul( 0.5 ) ) ) ).setY( trough.add( 0.03 ) );
			const open = xr.greaterThan( Wt.add( 0.1 ) );
			return { p: select( open, inTrough, f.p.add( f.n.mul( 0.04 ) ) ), n: select( open, up, f.n ) };

		};

		// ---- drops and ligaments (ballistic)
		const n0 = count( lipShed.mul( 14 ).mul( E ), 1 ).toVar(); // drops off the lip
		const n1 = n0.add( count( lipShed.mul( 5 ).mul( E ), 2 ) ).toVar(); // ligaments off the lip
		const n2 = n1.add( count( impact.mul( 520 ).mul( E ), 3 ) ).toVar(); // splash-up drops
		const n3 = n2.add( count( impact.mul( 90 ).mul( E ), 4 ) ).toVar(); // splash-up ligaments (torn strands)
		const n4 = n3.add( count( roller.mul( 90 ).mul( Eb ), 5 ) ).toVar(); // roller front (breaks up its silhouette)
		const n5 = n4.add( count( clash.mul( 40 ).mul( Eb ), 6 ) ).toVar(); // bore / backwash collision
		const n6 = n5.add( count( drift.mul( 24 ).mul( Hc ), 7 ) ).toVar(); // spindrift
		spray.emitNode( n6, ( j, ring ) => {

			const h = seed.add( j.mul( 13 ) );
			const r0 = spray.rand( h, 11 ), r1 = spray.rand( h, 12 ), r2 = spray.rand( h, 13 ), r3 = spray.rand( h, 14 ), r4 = spray.rand( h, 15 );
			const isLip = j.lessThan( n1 ), isImp = j.lessThan( n3 ), isRol = j.lessThan( n4 ), isCl = j.lessThan( n5 );
			const lig = j.greaterThanEqual( n0 ).and( j.lessThan( n1 ) ).or( j.greaterThanEqual( n2 ).and( j.lessThan( n3 ) ) );
			// lip: moving with the jet (thrown forward a little faster than the wave, falling)
			const vLip = d3.mul( c.mul( r2.mul( 0.2 ).add( 1.05 ) ) ).sub( up.mul( q.mul( sqrt( Yi.mul( 2 * GRAVITY ) ) ).mul( r3.mul( 0.3 ).add( 0.7 ) ) ) ).add( tg.mul( r4.sub( 0.5 ).mul( 0.6 ) ) );
			// splash-up: most drops stay low, some reach ~1.3 H; thrown up and forward, out of the surface
			const pl = plunge( r1, r2 );
			const vUp = sqrt( Hc.mul( r3.mul( r3 ).mul( 1.0 ).add( 0.3 ) ).mul( 2 * GRAVITY ) );
			const vImp = up.mul( vUp ).add( pl.n.mul( r4.mul( 1.5 ) ) ).add( d3.mul( c.mul( r2.mul( 0.6 ).add( 0.15 ) ).sub( 0.3 ) ) ).add( tg.mul( r4.sub( 0.5 ).mul( 2.0 ) ) );
			// roller: tossed forward and up by the tumbling front (from its upper part)
			const fr = facePoint( r1.mul( r1 ).mul( 0.7 ) ); // mostly from the tumbling top
			const vRol = d3.mul( c.mul( r2.mul( 0.3 ).add( 0.95 ) ) ).add( up.mul( r3.mul( 1.6 ).add( 0.6 ).mul( sqrt( Hb.div( 0.5 ) ) ) ) ).add( fr.n.mul( 0.5 ) ).add( tg.mul( r4.sub( 0.5 ).mul( 1.2 ) ) );
			// bore running into the backwash: thrown straight up from its front
			const fc = facePoint( r1.mul( 0.5 ).add( 0.3 ) );
			const vCl = up.mul( r3.mul( 1.6 ).add( 0.8 ).mul( sqrt( Hb.div( 0.4 ) ) ) ).add( d3.mul( r2.mul( 2 ).sub( 0.6 ) ) ).add( tg.mul( r4.sub( 0.5 ).mul( 1.5 ) ) );
			// spindrift: fine drops blown back over the crest
			const vDr = d3.mul( offshore.mul( r2.mul( 0.3 ).add( 0.35 ) ).negate() ).add( up.mul( r3.mul( 1.5 ).add( 0.8 ) ) ).add( tg.mul( r4.sub( 0.5 ).mul( 0.5 ) ) );
			const p = select( isLip, tipAt( r1 ), select( isImp, pl.p, select( isRol, fr.p.add( fr.n.mul( 0.03 ) ), select( isCl, fc.p.add( fc.n.mul( 0.03 ) ), root.add( up.mul( 0.04 ) ) ) ) ) ).add( along( r0 ) );
			const v = select( isLip, vLip, select( isImp, vImp, select( isRol, vRol, select( isCl, vCl, vDr ) ) ) );
			// radius (m): drops of a few mm (smaller off the roller, finest in the spindrift), ligaments ~1-2 cm
			// heavy-tailed drop sizes (many fine drops, a few big ones): r = r0 (1 - u)^-0.7
			const tail = pow( float( 1 ).sub( r4.mul( 0.98 ) ), - 0.7 );
			const rDrop = select( isImp.or( isCl ), float( 0.001 ), select( isRol, float( 0.0008 ), select( isLip, float( 0.001 ), float( 0.0005 ) ) ) ).mul( tail ).min( 0.012 );
			const rLig = float( 0.005 ).add( r4.mul( r4 ).mul( select( isImp, float( 0.016 ), float( 0.008 ) ) ) );
			const kind = select( lig, float( SPRAY.LIGAMENT ), float( SPRAY.DROPLET ) );
			spray.writeNode( ring, p, v, select( lig, rLig, rDrop ), kind, float( 2.5 ), tag.add( r0.mul( 0.999 ) ) );

		} );

		// ---- dense spray (clouds of drops: the white of the splash-up), a puff out of the barrel
		const c0 = count( impact.mul( 110 ).mul( E ), 21 ).toVar();
		const c1 = c0.add( count( spit.mul( 10 ).mul( E ), 22 ) ).toVar();
		const c2 = c1.add( count( roller.mul( 0.6 ).mul( Eb ), 23 ) ).toVar(); // (rare: a row of them reads as cotton puffs)
		const c3 = c2.add( count( clash.mul( 2 ).mul( Eb ), 24 ) ).toVar();
		spray.emitNode( c3, ( j, ring ) => {

			const h = seed.add( j.mul( 29 ) );
			const r0 = spray.rand( h, 41 ), r1 = spray.rand( h, 42 ), r2 = spray.rand( h, 43 ), r3 = spray.rand( h, 44 );
			const isImp = j.lessThan( c0 ), isSpit = j.lessThan( c1 ), isRol = j.lessThan( c2 );
			// torn sheets of the splash-up (half-width): many small ones, a few big
			const sz = select( isImp, sqrt( Hc ).mul( r2.mul( r2 ).mul( 0.14 ).add( 0.08 ) ), select( isSpit, sqrt( Hc ).mul( 0.14 ), select( isRol, sqrt( Hb.div( 0.5 ) ).mul( r2.mul( 0.04 ).add( 0.05 ) ), float( 0.08 ) ) ) ).toVar();
			// splash-up: sheets thrown up and forward out of the plunge line, rising up to ~1.3 H and
			// falling back as curtains
			const pl = plunge( r1, r3 );
			const pImp = pl.p.add( pl.n.mul( sz.mul( 0.4 ) ) ).add( up.mul( r2.mul( Hc ).mul( 0.1 ) ) );
			const vImp = up.mul( sqrt( Hc.mul( r3.mul( r3 ).mul( 0.95 ).add( 0.35 ) ).mul( 2 * GRAVITY ) ) ).add( pl.n.mul( 0.8 ) ).add( d3.mul( c.mul( r2.mul( 0.5 ).add( 0.2 ) ) ) ).add( tg.mul( r1.sub( 0.5 ).mul( 1.2 ) ) );
			// the puff blown out of the collapsing barrel: out of the middle of the front, along the crest
			const fs = facePoint( r1.mul( 0.3 ).add( 0.35 ) );
			const pSpit = fs.p.add( fs.n.mul( sz.mul( 0.5 ) ) );
			const vSpit = d3.mul( c.mul( 0.7 ) ).add( fs.n.mul( 1.2 ) ).add( tg.mul( r1.sub( 0.5 ).mul( 4 ) ) );
			// the tumbling top of the roller
			const fr = facePoint( r1.mul( 0.4 ) );
			const pRol = fr.p.add( fr.n.mul( sz.mul( 0.4 ) ) );
			const vRol = d3.mul( c.mul( 0.85 ) ).add( up.mul( r3.mul( 0.5 ).add( 0.3 ) ) );
			const fc = facePoint( r1.mul( 0.5 ).add( 0.3 ) );
			const pCl = fc.p.add( fc.n.mul( sz.mul( 0.5 ) ) );
			const vCl = up.mul( r3.mul( 1.2 ).add( 0.8 ).mul( sqrt( Hb.div( 0.4 ) ) ) ).add( d3.mul( r2.sub( 0.3 ) ) );
			const p = select( isImp, pImp, select( isSpit, pSpit, select( isRol, pRol, pCl ) ) ).add( along( r0 ) );
			const v = select( isImp, vImp, select( isSpit, vSpit, select( isRol, vRol, vCl ) ) );
			const life = select( isImp, r3.mul( 0.5 ).add( 1.0 ), select( isSpit, float( 0.9 ), r3.mul( 0.3 ).add( 0.5 ) ) );
			spray.writeNode( ring, p, v, sz, SPRAY.SPRAY, life, tag.add( r0.mul( 0.999 ) ) );

		} );

		// ---- mist: the fine spray that drifts off with the wind (and the air pushed by the wave)
		// (few, large, faint sprites: mist is the biggest overdraw of the spray)
		const m0 = count( haze.mul( 5 ).mul( E ).add( spit.mul( 3 ).mul( E ) ), 31 ).toVar();
		const m1 = m0.add( count( roller.mul( 2.5 ).mul( Eb ), 32 ) ).toVar();
		spray.emitNode( m1, ( j, ring ) => {

			const h = seed.add( j.mul( 17 ) );
			const r0 = spray.rand( h, 51 ), r1 = spray.rand( h, 52 ), r2 = spray.rand( h, 53 ), r3 = spray.rand( h, 54 );
			const isImp = j.lessThan( m0 );
			const pl = plunge( r1, r3 );
			const ft = facePoint( r1.mul( 0.3 ) );
			const p = select( isImp, pl.p.add( pl.n.mul( 0.2 ) ).add( up.mul( r2.mul( Hc ).mul( 0.4 ) ) ), ft.p.add( ft.n.mul( 0.15 ) ) ).add( along( r0 ) );
			const v = select( isImp, d3.mul( c.mul( r1.mul( 0.3 ).add( 0.4 ) ) ).add( up.mul( r3.mul( 0.9 ).add( 0.4 ) ) ), d3.mul( c.mul( 0.8 ) ).add( up.mul( 0.2 ) ) );
			const size = select( isImp, sqrt( Hc ).mul( r3.mul( 0.3 ).add( 0.45 ) ), r3.mul( 0.15 ).add( 0.3 ) );
			const life = select( isImp, r3.mul( 1.5 ).add( 2.5 ), r3.mul( 1.0 ).add( 1.5 ) );
			spray.writeNode( ring, p, v, size, SPRAY.MIST, life, tag.add( r0.mul( 0.999 ) ) );

		} );

	}

	// Sun visibility (0..1) for a spray particle made by one of the crests: in front of the wave with
	// the sun behind it (a beach view into the sun) the particles below the crest line are in the
	// shadow of the wave (and of the overhanging lip while it plunges). seedTag: the particle's tag.
	sprayShadow( p, seedTag ) {

		const C = this.crestRead;
		const out = float( 1 ).toVar();
		const idx = floor( seedTag ).sub( 1 );
		If( idx.greaterThanEqual( 0 ).and( idx.lessThan( this.NS * 2 ) ), () => {

			const k = uint( idx ).mul( 3 );
			const c0 = C.element( k ), c1 = C.element( k.add( 1 ) ), c2 = C.element( k.add( 2 ) );
			const L = G.sunDir;
			const d2 = c2.xy;
			const Ld = dot( L.xz, d2 ); // < 0: the sun is on the sea side of the wave
			If( c2.w.greaterThan( 0.5 ).and( Ld.lessThan( - 0.02 ) ), () => {

				const root = c0.xyz, b = c0.w, H = c1.w;
				const q = clamp( b.div( 0.9 ), 0, 1 ).mul( float( 1 ).sub( smoothstep( 1.0, 1.3, b ) ) );
				// vertical plane through the crest (moved forward under the overhanging lip)
				const plane = root.xz.add( d2.mul( H.mul( 0.8 ).mul( q ).mul( 0.6 ) ) );
				const s = dot( p.xz.sub( plane ), d2 ); // > 0: in front of it
				const tau = s.div( Ld.negate() );
				const yRay = p.y.add( L.y.mul( tau ) ); // height of the ray toward the sun where it crosses the plane
				const top = root.y.add( 0.05 );
				const shade = smoothstep( top, top.sub( 0.4 ), yRay ).mul( smoothstep( - 0.1, 0.1, s ) ).mul( smoothstep( 14, 6, s ) );
				out.assign( float( 1 ).sub( shade.mul( 0.55 ) ) );

			} );

		} );
		return out;

	}

	// ------------------------------------------------------------------ lip sheet

	_buildMesh() {

		const NS = this.NS;
		const nSeg = NS - 1;
		const vertsPerStrip = NV * 2;
		const nStrips = nSeg * 2;
		const ids = new Float32Array( nStrips * vertsPerStrip * 4 );
		const pos = new Float32Array( nStrips * vertsPerStrip * 3 );
		const index = new Uint32Array( nStrips * ( NV - 1 ) * 6 );
		let p = 0, q = 0;
		for ( let seg = 0; seg < nSeg; seg ++ ) for ( let slot = 0; slot < 2; slot ++ ) {

			const v0 = ( seg * 2 + slot ) * vertsPerStrip;
			for ( let side = 0; side < 2; side ++ ) for ( let k = 0; k < NV; k ++ ) {

				ids[ p ++ ] = seg; ids[ p ++ ] = slot; ids[ p ++ ] = side; ids[ p ++ ] = k;

			}

			for ( let k = 0; k < NV - 1; k ++ ) {

				const a = v0 + k, b = v0 + k + 1, c = v0 + NV + k, d = v0 + NV + k + 1;
				index[ q ++ ] = a; index[ q ++ ] = c; index[ q ++ ] = b;
				index[ q ++ ] = b; index[ q ++ ] = c; index[ q ++ ] = d;

			}

		}

		const geo = new THREE.BufferGeometry();
		geo.setAttribute( 'position', new THREE.BufferAttribute( pos, 3 ) );
		geo.setAttribute( 'sheetId', new THREE.BufferAttribute( ids, 4 ) );
		geo.setIndex( new THREE.BufferAttribute( index, 1 ) );
		geo.boundingSphere = new THREE.Sphere( new THREE.Vector3(), 1e7 );

		const mat = new THREE.NodeMaterial();
		mat.name = 'BreakerLip';
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

		const vN = varyingProperty( 'vec3', 'vLipN' );
		const vLip = varyingProperty( 'vec4', 'vLip' ); // v, b, along, curtain length
		const vFade = varyingProperty( 'float', 'vLipFade' );
		const C = this.crestRead;
		const spacing = this.spacing;

		mat.positionNode = Fn( () => {

			const id = attribute( 'sheetId', 'vec4' );
			const seg = uint( id.x ), slot = uint( id.y ), side = uint( id.z );
			const k = id.w;
			const st = seg.add( side );
			const ot = seg.add( uint( 1 ).sub( side ) );
			const e = st.mul( 6 ).add( slot.mul( 3 ) );
			const eo = ot.mul( 6 ).add( slot.mul( 3 ) );
			const c0 = C.element( e ), c1 = C.element( e.add( 1 ) ), c2 = C.element( e.add( 2 ) );
			const mOther = C.element( eo.add( 2 ) ).w;
			// (the crests are followed until the bore reaches the shore; the sheet is gone after the plunge)
			const valid = c2.w.greaterThan( 0.5 ).and( abs( mOther.sub( c2.w ) ).lessThan( 0.5 ) ).and( c0.w.lessThan( 1.25 ) );

			const root = c0.xyz, b = c0.w, back = c1.xyz, H = c1.w;
			const d3 = vec3( c2.x, 0, c2.y );
			const trough = c2.z;
			const q = clamp( b.div( 0.9 ), 0, 1.35 );
			const Xi = max( H.mul( 0.8 ), 0.05 );
			const Yi = max( root.y.sub( trough ), 0.05 );
			// profile parameter: k = 0, 1 on the back of the crest (-1, -0.45), then 0..1 along the curtain
			const v = select( k.lessThan( 0.5 ), float( - 1 ), select( k.lessThan( 1.5 ), float( - 0.45 ), k.sub( 2 ).div( NV - 3 ) ) );
			const xl = Xi.mul( q ).mul( max( v, 0 ) );
			const f = xl.div( Xi );
			const yl = Yi.negate().mul( f.mul( f ) ); // ballistic: the jet leaves the crest horizontally
			const onLip = root.add( d3.mul( xl ) ).add( vec3( 0, yl, 0 ) );
			const onCap = mix( root, back, v.negate().max( 0 ) ).add( vec3( 0, 0.012, 0 ) );
			const P = select( v.lessThan( 0 ), onCap, onLip );
			// outward normal of the curtain (upper surface of the lip)
			const slope = Yi.mul( 2 ).mul( xl ).div( Xi.mul( Xi ) );
			vN.assign( normalize( vec3( 0, 1, 0 ).add( d3.mul( slope ) ) ) );
			vLip.assign( vec4( v, b, float( st ).mul( spacing ), Xi.mul( q ).add( Yi.mul( q ).mul( q ) ) ) ); // w: curtain length (m)
			// the cap fades in over the crest; the whole lip fades out once it has become whitewater
			const capA = smoothstep( - 1, - 0.1, v );
			// once the jet has re-entered the water the curtain is gone (the splash and the roller take over)
			vFade.assign( capA.mul( smoothstep( 0.02, 0.12, q ) ).mul( float( 1 ).sub( smoothstep( 1.0, 1.2, b ) ) ) );
			return select( valid, P, vec3( 0, - 1e5, 0 ) );

		} )();

		const sky = this.sky;
		const lace = makeLaceTexture();
		const fragment = Fn( () => {

			const v = vLip.x, b = vLip.y, a = vLip.z, len = vLip.w;
			const pos = positionWorld;
			const V = normalize( cameraPosition.sub( pos ) ).toVar();
			const Nw = normalize( vN );
			const N0 = select( frontFacing, Nw, Nw.negate() ).toVar();
			// The water of the jet is stretched along the flow: streaks and ripples running down the
			// curtain (the lace pattern stretched ~8x along the jet and moving with it) in its surface
			// (normal, from the screen-space gradient of a small relief) and in its thickness
			const flowS = v.mul( len ).sub( G.time.mul( 1.1 ) );
			// along-crest coordinate warped by low-frequency noise (three incommensurate scales, slope kept
			// below 1 so it never folds): the streak spacing drifts along the crest, no fixed period, and the
			// tiles of every pattern below never line up into a comb
			const aw = a.add( sin( a.mul( 0.23 ).add( 1.7 ) ).mul( 1.6 ) ).add( sin( a.mul( 0.61 ).add( 4.2 ) ).mul( 0.4 ) ).add( sin( a.mul( 1.37 ).add( 0.4 ) ).mul( 0.12 ) ).toVar();
			const sv = texture( lace, vec2( aw.div( 0.83 ), flowS.div( 3.0 ) ) ).toVar();
			// and broad bands (sections of the lip thicker or thinner than others), visible from afar
			const sbv = texture( lace, vec2( aw.div( 2.9 ), flowS.div( 9.0 ) ).add( vec2( 0.37, 0.61 ) ) ).toVar();
			const sb = sbv.z;
			const hS = sv.x.mul( 0.02 ).add( sv.z.mul( 0.012 ) );
			const dpdx = dFdx( pos ), dpdy = dFdy( pos );
			const r1 = cross( dpdy, N0 ), r2 = cross( N0, dpdx );
			const det = dot( dpdx, r1 );
			const grad = r1.mul( dFdx( hS ) ).add( r2.mul( dFdy( hS ) ) ).mul( sign( det ) );
			const N = normalize( N0.mul( abs( det ) ).sub( grad ).add( N0.mul( 1e-9 ) ) ).toVar();
			const NdV = max( dot( N, V ), 1e-3 );
			const F = fresnelDielectric( NdV, 1.333 ).toVar();
			const L = G.sunDir;
			const sunVis = this.clouds ? this.clouds.shadow( pos.xz ) : float( 1 );
			const sun = G.sunColor.mul( sunVis ).toVar();

			// reflection: sky + sun glint
			const Rr = reflect( V.negate(), N );
			const R = normalize( vec3( Rr.x, max( Rr.y, 0.004 ), Rr.z ) );
			const refl = sky.reflectionRadiance( R );
			const Hh = normalize( L.add( V ) );
			const spec = sun.mul( pow( max( dot( N, Hh ), 0 ), 180 ).mul( 12 ) ).mul( F );

			// light through the thin sheet: turquoise when backlit
			const tint = vec3( 0.16, 0.62, 0.56 );
			const back = pow( saturate( dot( V.negate(), L ).mul( 0.5 ).add( 0.5 ) ), 4.0 );
			// thick and deep green at the root, thin, bright and clear toward the tip, uneven along the streaks
			const thin = mix( float( 1.5 ), float( 0.7 ), v ).mul( sv.z.mul( 0.3 ).add( 0.8 ) ).mul( sb.mul( 0.8 ).add( 0.6 ) );
			const glow = sun.mul( tint ).mul( back.mul( 0.9 ).add( 0.08 ) ).add( G.skyIrradiance.mul( tint ).mul( 1.4 ) ).mul( thin );
			const aW = F.add( float( 1 ).sub( F ).mul( mix( float( 0.42 ), float( 0.16 ), v ) ).mul( sv.x.mul( 0.5 ).add( 0.75 ) ).mul( sb.mul( 0.7 ).add( 0.65 ) ).min( 0.85 ) );
			const cW = refl.mul( F ).add( spec ).add( glow.mul( float( 1 ).sub( F ) ).mul( 0.42 ) );

			// The jet stays clear, glassy water while it is in the air: only its leading edge tears into
			// aerated fingers (filaments of the lace pattern stretched along the flow, merging into a
			// ragged white rim). Sections of the lip differ a little.
			const flowM = v.mul( len ).sub( G.time.mul( 1.1 ) ); // metres down the curtain, moving with the jet
			const fuv = vec2( aw.div( 1.9 ).add( 0.53 ), flowM.div( 1.8 ) );
			const fl = texture( lace, fuv ).toVar();
			const sect = fl.z; // slowly varying along the crest
			const vary = sect.sub( 0.5 ).add( sin( aw.mul( 0.29 ).add( b.mul( 2.1 ) ) ).mul( 0.15 ) );
			const reach = v.add( vary.mul( 0.3 ) );
			// slightly irregular leading edge (sections of the lip reach a little further than others)
			const tipN = sin( aw.mul( 1.3 ).add( G.time.mul( 0.3 ) ) ).mul( 0.6 ).add( sin( aw.mul( 4.7 ).add( 1.3 ) ).mul( 0.4 ) );
			const edge = v.add( tipN.mul( 0.03 ) ).add( vary.mul( 0.04 ) );
			const thrown = smoothstep( 0.35, 0.8, b );
			// a thin, translucent aerated rim along the leading edge (it tears into the drops and
			// ligaments the spray system throws off it)
			const rim = smoothstep( 0.9, 0.96, edge ).mul( fl.z.mul( 0.2 ).add( 0.2 ) ).mul( thrown );
			// once the lip has landed the curtain is a falling mass of whitewater that dissolves (blotchy)
			// into the splash-up and the roller within a fraction of a second
			// (in streaks: white fingers run down the curtain ahead of the rest, so along a peeling crest the
			// broken section feathers into the clear one instead of ending on a vertical line)
			// (each finger its own length and brightness: the fine strands' per-cell random, gated and grouped
			// by the broad pattern so fingers cluster, merge and leave gaps instead of a regular row)
			const fing = sv.w.mul( 0.55 ).add( sv.z.mul( 0.2 ) ).mul( smoothstep( 0.25, 0.75, sbv.w.mul( 0.6 ).add( sb.mul( 0.4 ) ) ).mul( 0.8 ).add( 0.2 ) ).mul( 0.34 );
			const wh = smoothstep( float( 1 ).sub( v ).mul( 0.18 ).add( 0.9 ).sub( fing ), float( 1 ).sub( v ).mul( 0.18 ).add( 1.0 ).sub( fing ), b ).toVar(); // from the tip up
			const lc = texture( lace, vec2( aw.mul( 0.83 ).add( 1.9 ), v.mul( len ).sub( G.time.mul( 1.3 ) ) ).div( LACE_TILE * 0.8 ) );
			const blot = lc.z.mul( 0.7 ).add( lc.y.mul( 0.3 ) );
			const gone = smoothstep( 1.0, 1.25, b );
			const clumpW = smoothstep( gone.sub( 0.12 ), gone.add( 0.12 ), blot );
			const clumps = saturate( blot.sub( gone ).mul( 2.5 ) ); // thicker inside the blotches
			// thin aerated filaments stretched down the curtain (foam of the previous wave drawn up the
			// face and thrown out with the lip), more of them toward the tip
			const fil = float( 1 ).sub( smoothstep( 0.02, 0.12, sv.x ) ).mul( smoothstep( 0.15, 0.8, v ) ).mul( sv.w.mul( 0.5 ).add( 0.25 ) ).mul( sb.mul( 0.9 ).add( 0.3 ) ).mul( thrown );
			const aer = mix( max( rim, fil ), float( 0.95 ), wh ).toVar();
			// aerated water is a dense scatterer: bright from every side, glowing when backlit
			const foamLit = sun.mul( max( dot( N, L ), 0 ).mul( 0.5 ).add( 0.5 ).add( back.mul( 1.2 ) ) ).div( Math.PI ).add( G.skyIrradiance ).mul( 0.9 );

			const tip = float( 1 ).sub( smoothstep( 0.93, 1.0, edge ) );
			const alpha = vFade.mul( tip ).mul( this.params.sheet ).mul( mix( float( 1 ), clumpW, wh ) ).toVar();
			const aF = aer.mul( 0.92 );
			const col = foamLit.mul( mix( float( 1 ), clumps.mul( 0.4 ).add( 0.75 ), wh ) ).mul( aF ).add( cW.mul( float( 1 ).sub( aF ) ) );
			const aOut = aF.add( aW.mul( float( 1 ).sub( aF ) ) ).mul( alpha );
			return vec4( col.mul( alpha ), aOut );

		} )();

		mat.outputNode = fragment;
		mat.mrtNode = mrt( { velocity: vec4( staticVelocity.mul( fragment.w ), 0, fragment.w ) } );
		mat.mrtNode.setBlendMode( 'velocity', new THREE.BlendMode( THREE.MaterialBlending ) );

		const mesh = this.mesh = new THREE.Mesh( geo, mat );
		mesh.frustumCulled = false;
		mesh.castShadow = false;
		mesh.receiveShadow = false;
		mesh.renderOrder = 10;
		mesh.layers.set( LAYERS.TRANSPARENT );
		mesh.name = 'BreakerLips';

	}

}

// Shoreline stations along the main beach: evenly spaced along the (smoothed) shoreline, each with
// the unit direction toward the sea. Returns { data: Float32Array(n * 4) (x, z, nx, nz), count, spacing }.
export function buildStations( terrain, { x0 = - 175, x1 = 195, spacing = 0.6 } = {} ) {

	// shoreline z(x): first land found marching north from the water
	const pts = [];
	for ( let x = x0; x <= x1; x += 0.5 ) {

		if ( terrain.heightAt( x, 12 ) > - 0.3 ) continue; // not open water in front
		let z = 12, zs = null;
		for ( ; z > - 110; z -= 0.5 ) {

			if ( terrain.heightAt( x, z ) > 0 ) {

				let lo = z, hi = z + 0.5; // land at lo, water at hi
				for ( let k = 0; k < 16; k ++ ) {

					const mid = ( lo + hi ) * 0.5;
					if ( terrain.heightAt( x, mid ) > 0 ) lo = mid; else hi = mid;

				}

				zs = ( lo + hi ) * 0.5;
				break;

			}

		}

		if ( zs !== null ) pts.push( [ x, zs ] );

	}

	// break into continuous runs, keep the longest (the main beach)
	let best = [], run = [];
	for ( let k = 0; k < pts.length; k ++ ) {

		if ( run.length && ( Math.abs( pts[ k ][ 1 ] - run[ run.length - 1 ][ 1 ] ) > 3 || pts[ k ][ 0 ] - run[ run.length - 1 ][ 0 ] > 1.01 ) ) {

			if ( run.length > best.length ) best = run;
			run = [];

		}

		run.push( pts[ k ] );

	}

	if ( run.length > best.length ) best = run;

	// smooth the polyline (the transects should not follow every wiggle of the waterline)
	let sm = best.map( ( p ) => [ p[ 0 ], p[ 1 ] ] );
	for ( let it = 0; it < 30; it ++ ) {

		const nx = sm.map( ( p, k ) => {

			if ( k === 0 || k === sm.length - 1 ) return p;
			return [ p[ 0 ], ( sm[ k - 1 ][ 1 ] + 2 * p[ 1 ] + sm[ k + 1 ][ 1 ] ) * 0.25 ];

		} );
		sm = nx;

	}

	// resample by arc length
	const out = [];
	let acc = 0, next = 0;
	for ( let k = 1; k < sm.length; k ++ ) {

		const [ ax, az ] = sm[ k - 1 ], [ bx, bz ] = sm[ k ];
		const seg = Math.hypot( bx - ax, bz - az );
		while ( next <= acc + seg ) {

			const t = ( next - acc ) / seg;
			const x = ax + ( bx - ax ) * t, z = az + ( bz - az ) * t;
			// normal toward the sea (+z side of a west->east polyline)
			const tx = ( bx - ax ) / seg, tz = ( bz - az ) / seg;
			out.push( x, z, - tz, tx );
			next += spacing;

		}

		acc += seg;

	}

	return { data: new Float32Array( out ), count: out.length / 4, spacing };

}
