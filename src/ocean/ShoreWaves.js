import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, vec2, vec3, vec4, normalize, length, dot, cross, mix, clamp, saturate, smoothstep,
	sin, cos, exp, pow, sqrt, max, min, abs, floor, fract, select, If, sign, Loop, Break, textureLoad, ivec2, mat4,
} from 'three/tsl';
import { G, GRAVITY } from '../core/Globals.js';

// Depth-aware shoreline waves.
//
// Wave phase comes from the precomputed travel-time field (refraction around headlands, fronts
// aligning with depth contours). Each individual wave m has its own height (sets + along-shore
// variation). Height grows in shallow water (Green's law) until H > gamma * depth. The wave then
// plunges: the front face turns into a vertical, concave wall under the crest (the thrown lip is a
// separate sheet, see Breakers), the tube collapses where the lip lands and the wave continues
// as a turbulent bore, which finally runs up the beach as a thin swash sheet whose leading edge
// advances and retreats (vertical run-up R(t) compared against the sand height).

const hash1 = ( x ) => fract( sin( x.mul( 127.1 ).add( 311.7 ) ).mul( 43758.5453 ) );
const TAU = Math.PI * 2;
const BEACH_SLOPE = 0.066; // run-up is converted to a horizontal excursion with this slope
const SWASH_UP = 0.4, SWASH_DOWN = 0.55; // fractions of the period: uprush, backwash
const SWASH_OVERSHOOT = 0.6; // the mesh sheet reaches this far (m) past the leading edge

export class ShoreWaves {

	constructor( terrainGPU ) {

		this.terrain = terrainGPU;
		this.period = uniform( 9.0 ).setName( 'swPeriod' );
		this.amplitude = uniform( 0.34 ).setName( 'swAmp' ); // offshore amplitude (H/2)
		this.variation = uniform( 0.55 ).setName( 'swVar' );
		this.gamma = uniform( 0.78 ).setName( 'swGamma' );
		// fraction of the break depth over which the lip plunges. Also sets how fast the breaking state
		// changes along a crest whose height varies: wide enough that a broken section joins the clean
		// face through a shoulder where the lip is still falling (peeling), not a vertical cut
		this.breakSpan = uniform( 0.13 ).setName( 'swBreakSpan' );
		this.curl = uniform( 1.0 ).setName( 'swCurl' );
		this.runup = uniform( 1.0 ).setName( 'swRunup' );
		this.enabled = uniform( 1.0 ).setName( 'swOn' );
		this.turbidity = uniform( 0.16 ).setName( 'swTurbidity' ); // sediment + bubble scattering in the surf zone (1/m)
		this.time = G.time;

		// Emitted as real WGSL functions (not inlined at every call site): keeps the vertex,
		// simulation and query shaders small, which matters a lot for compile times.
		this._waveAmpFn = Fn( ( [ m, along ] ) => this._waveAmpImpl( m, along ) ).setLayout( {
			name: 'shoreWaveAmp', type: 'float',
			inputs: [ { name: 'm', type: 'float' }, { name: 'along', type: 'float' } ],
		} );
		this._shapeFn = Fn( ( [ u, A, d, lam ] ) => {

			const s = this._shapeImpl( u, A, d, lam );
			return vec4( s.x, s.y, s.foam, s.b );

		} ).setLayout( {
			name: 'shoreShape', type: 'vec4',
			inputs: [ { name: 'u', type: 'float' }, { name: 'A', type: 'float' }, { name: 'd', type: 'float' }, { name: 'lam', type: 'float' } ],
		} );
		// the cross-section at u and at u + du (for the surface normal) in one call:
		// ( x0, y0, foam, b ), ( x1, y1, face, roller )
		this._shapePairFn = Fn( ( [ u, du, A, d, lam ] ) => {

			const P = this._breakParams( A, d );
			const s0 = this._profile( u, lam, P, true );
			const s1 = this._profile( u.add( du ), lam, P, false );
			return mat4( vec4( s0.x, s0.y, s0.foam, s0.b ), vec4( s1.x, s1.y, s0.face, s0.roller ), vec4( 0 ), vec4( 0 ) );

		} ).setLayout( {
			name: 'shoreShapePair', type: 'mat4',
			inputs: [ { name: 'u', type: 'float' }, { name: 'du', type: 'float' }, { name: 'A', type: 'float' }, { name: 'd', type: 'float' }, { name: 'lam', type: 'float' } ],
		} );
		// the surface at a fixed WORLD point (for the Eulerian shore simulation): whitewater there and
		// the water height there (the parcel shown at this point rests ~x up-wave: first-order inverse
		// of the horizontal displacement, which is large on a breaking wave)
		// ( foam, height, displacement of the parcel resting here, b )
		this._worldFn = Fn( ( [ u, A, d, lam, env ] ) => {

			const P = this._breakParams( A, d );
			const s0 = this._profile( u, lam, P, false );
			const s1 = this._profile( u.add( s0.x.mul( env ).div( lam ) ), lam, P, false );
			return vec4( this._whitewater( u.mul( lam ).negate(), lam, P ), s1.y, s0.x, P.b );

		} ).setLayout( {
			name: 'shoreWorld', type: 'vec4',
			inputs: [ { name: 'u', type: 'float' }, { name: 'A', type: 'float' }, { name: 'd', type: 'float' }, { name: 'lam', type: 'float' }, { name: 'env', type: 'float' } ],
		} );
		this._boreFn = Fn( ( [ A, d ] ) => {

			const P = this._breakParams( A, d );
			return vec4( P.Hb, P.Wt, P.Xc, P.wBore );

		} ).setLayout( {
			name: 'shoreBore', type: 'vec4',
			inputs: [ { name: 'A', type: 'float' }, { name: 'd', type: 'float' } ],
		} );
		this._crestFn = Fn( ( [ A, d ] ) => {

			const P = this._breakParams( A, d );
			return vec4( P.b, P.yc.sub( P.yt ), P.ytB, P.Xi );

		} ).setLayout( {
			name: 'shoreCrest', type: 'vec4',
			inputs: [ { name: 'A', type: 'float' }, { name: 'd', type: 'float' } ],
		} );

	}

	waveAmp( m, along ) {

		return this._waveAmpFn( m, along );

	}

	shape( u, A, d, lam ) {

		const r = this._shapeFn( u, A, d, lam ).toVar();
		return { x: r.x, y: r.y, foam: r.z, b: r.w };

	}

	// breaking progress b, wave height H, trough level (relative to mean) and how far the lip is thrown
	crestParams( A, d ) {

		const r = this._crestFn( A, d ).toVar();
		return { b: r.x, H: r.y, trough: r.z, lipThrow: r.w };

	}

	// Depth that sets the breaking state of the wave a parcel belongs to: the depth under that wave's
	// crest, not under the parcel. Breaking is a property of the wave: a parcel ahead of the crest is
	// in shallower water and would otherwise "break" first (whitewater creeping up the foot of a still
	// glassy face). Near the troughs it hands over to the local depth, where the neighbouring wave
	// takes over (the profile stays continuous at u = +-0.5). u: local phase, lam: local wavelength.
	breakDepth( xz, dir, u, lam, d ) {

		const pc = xz.add( dir.mul( u.mul( lam ) ) );
		const dc = G.seaLevel.sub( this.terrain.heightAt( pc ) );
		return mix( dc, d, smoothstep( 0.3, 0.5, abs( u ) ) );

	}

	// the bore that follows the plunge: roller height Hb, horizontal extent of its front Wt, forward
	// shift of the crest Xc (to where the lip landed), bore weight (0 while plunging .. 1)
	boreParams( A, d ) {

		const r = this._boreFn( A, d ).toVar();
		return { Hb: r.x, Wt: r.y, Xc: r.z, wBore: r.w };

	}

	// along-shore phase wobble (in periods): crests bend over the uneven bottom
	wobble( along ) {

		return sin( along.mul( 0.029 ).add( 0.7 ) ).mul( 0.07 ).add( sin( along.mul( 0.083 ).add( 2.1 ) ).mul( 0.035 ) );

	}

	// ------------------------------------------------------------ per-wave height

	_waveAmpImpl( m, along ) {

		// sets: groups of ~7 waves with larger ones in the middle, plus per-wave randomness
		const set = sin( m.mul( Math.PI / 7 ) ).abs().mul( 0.6 ).add( 0.55 );
		const rnd = hash1( m ).sub( 0.5 ).mul( 2 );
		// along-shore variation so waves peel instead of closing out
		// peaks ~40-50 m wide with lower shoulders between them, different for every wave (the warp keeps
		// them from repeating along the beach); each peak breaks first and peels outward from it
		const warp = sin( along.mul( 0.016 ).add( m.mul( 0.9 ) ) ).mul( 1.6 );
		const a1 = sin( along.mul( 0.062 ).add( m.mul( 1.7 ) ).add( warp ) );
		const a2 = sin( along.mul( 0.13 ).add( m.mul( 4.1 ) ).add( 1.3 ).sub( warp.mul( 0.7 ) ) );
		const alongV = a1.mul( 0.6 ).add( a2.mul( 0.4 ) );
		return this.amplitude.mul( set ).mul( float( 1 ).add( rnd.mul( this.variation ).mul( 0.5 ) ).add( alongV.mul( this.variation ).mul( 0.7 ) ) ).max( 0.02 );

	}

	// ------------------------------------------------------------ cross-section shape

	_breakParams( A, d ) {

		// break depth for this wave (Green's law shoaling, H = gamma d)
		const db = pow( A.mul( 3.556 ).div( this.gamma ), 0.8 );
		const b = db.sub( d ).div( db.mul( this.breakSpan ) ).toVar(); // <0 shoaling, 0..1 plunging, >1 bore
		const shoal = pow( float( 10 ).div( clamp( d, 0.35, 10 ) ), 0.25 );
		const Ash = A.mul( shoal ).toVar();
		const p = mix( float( 1.0 ), float( 3.0 ), smoothstep( - 2.5, 0.0, b ) ).toVar();
		const meanP = float( 1 ).div( sqrt( p.add( 0.25 ).mul( Math.PI ) ) ).toVar();
		const crestPeak = float( 1 ).add( smoothstep( - 1.0, 0.3, b ).mul( 0.22 ) );
		const yc = Ash.mul( 2 ).mul( float( 1 ).sub( meanP ) ).mul( crestPeak ).toVar(); // crest height
		const yt = Ash.mul( - 2 ).mul( meanP ).mul( crestPeak ).toVar(); // trough level
		const H = yc.sub( yt );
		const wBore = smoothstep( 0.85, 1.35, b ).toVar();
		const Xi = H.mul( 0.8 ); // lip throw at impact
		// bore height, limited by the depth and fading out in the last few decimetres
		const Hb = min( H.mul( 0.6 ), max( d, 0 ).mul( 0.75 ) ).mul( smoothstep( 0.0, 0.3, d ).mul( 0.7 ).add( 0.3 ) );
		const ycB = mix( yc, yt.mul( 0.6 ).add( Hb ), wBore ); // crest / roller top
		const ytB = mix( yt, yt.mul( 0.6 ), wBore ); // trough
		// the collapsing crest moves to where the lip landed (no shift once the bore has run out of height)
		const Xc = mix( float( 0 ), Xi.sub( Hb.mul( 0.55 ) ), wBore ).mul( smoothstep( 0.03, 0.3, Hb ) );
		// horizontal extent of the face: concave tube face while plunging, short convex roller front on the bore
		const Wt = mix( H.mul( mix( 0.25, 0.55, smoothstep( 0.1, 0.9, b ) ) ), Hb.mul( 0.6 ).add( 0.08 ), wBore );
		return { db, b, Ash, p, meanP, crestPeak, yc, yt, H, wBore, Xi, Hb, ycB, ytB, Xc, Wt };

	}

	// u: local phase in [-0.5, 0.5], crest at 0, u < 0 in front (shoreward) of the crest
	// Returns { x: shoreward displacement, y: height above mean, foam, b: breaking progress }
	_shapeImpl( u, A, d, lam ) {

		return this._profile( u, lam, this._breakParams( A, d ), true );

	}

	// the cross-section for break parameters P (see _breakParams)
	_profile( u, lam, P, withFoam ) {

		const { b, Ash, p, meanP, crestPeak, wBore, ycB, ytB, Xc, Wt, Xi } = P;

		// --- shoaling: peaked (cnoidal-like) crest, the front compressed by a phase skew
		const skew = smoothstep( - 3.0, 0.0, b ).mul( 0.55 );
		const phi = u.sub( skew.mul( float( 1 ).sub( cos( u.mul( TAU ) ) ) ).div( TAU ) );
		const c = max( cos( phi.mul( TAU ) ).add( 1 ).mul( 0.5 ), 0 ); // pow() of a rounding-negative base is NaN
		const yPre = Ash.mul( 2 ).mul( pow( c, p ).sub( meanP ) ).mul( crestPeak );
		const Q = smoothstep( - 3.0, 0.0, b ).mul( 0.25 ).add( 0.1 );
		const xPre = sin( u.mul( TAU ) ).mul( Ash ).mul( Q );

		// --- plunging / bore profile. Front: the upper uf of the phase is an elliptic arc from the
		// crest down to the trough (concave tube face -> convex roller front), the rest of the
		// front is trough, stretched to meet the next wave. Back: the shoaling back, decaying
		// exponentially behind the bore.
		const uf = 0.09;
		const inFace = u.greaterThan( - uf );
		const th = clamp( u.negate().div( uf ), 0, 1 ).mul( Math.PI / 2 );
		const fx = mix( float( 1 ).sub( cos( th ) ), sin( th ), wBore );
		const fy = mix( float( 1 ).sub( sin( th ) ), cos( th ), wBore );
		const sTr = clamp( u.negate().sub( uf ).div( 0.5 - uf ), 0, 1 );
		const ul = u.mul( lam );
		const xFront = select( inFace, Xc.add( Wt.mul( fx ) ), mix( Xc.add( Wt ), lam.mul( 0.5 ), sTr ) ).add( ul );
		const yFront = select( inFace, ytB.add( ycB.sub( ytB ).mul( fy ) ), ytB );
		const cb = pow( max( cos( u.mul( TAU * 0.85 ) ).add( 1 ).mul( 0.5 ), 0 ), p );
		const yBack = ytB.add( ycB.sub( ytB ).mul( mix( cb, exp( u.mul( - 7 ) ), wBore ) ) );
		const xBack = Xc.mul( exp( u.mul( - 6 ) ) ).mul( smoothstep( 0.5, 0.35, u ) ).add( xPre.mul( float( 1 ).sub( wBore ) ) );
		const front = u.lessThan( 0 );
		const wC = smoothstep( - 0.35, 0.25, b ).mul( this.curl );
		const x = mix( xPre, select( front, xFront, xBack ), wC );
		const y = mix( yPre, select( front, yFront, yBack ), wC );
		if ( ! withFoam ) return { x, y, b };

		// --- whitewater (a function of where the parcel is now, see _whitewater) and, per parcel, the
		// clear face of the plunging wave and the relief of the roller
		const foam = this._whitewater( x.sub( u.mul( lam ) ), lam, P );
		const onFace = front.and( inFace );
		const faceFill = this._faceFill( b, fx );
		// the clear, concave face of a plunging wave (WaterSurface keeps the foam carried by the water off it)
		const face = select( onFace, float( 1 ), float( 0 ) ).mul( smoothstep( - 0.4, 0.0, b ) ).mul( float( 1 ).sub( faceFill ) );
		// turbulent relief of the whitewater roller (m): its front and the top it tumbles over
		const rollerAmp = select( front, select( inFace, float( 1 ), float( 0 ) ), exp( u.mul( - 25 ) ) ).mul( wBore ).mul( P.Hb ).mul( wC );

		return { x, y, foam, b, face, roller: rollerAmp };

	}

	// Whitewater made by the breaking wave at xi (m): the horizontal position relative to the crest's
	// rest position, positive shoreward (a parcel's own position, or a fixed world point: the shore
	// simulation deposits it where the water actually is, not where the parcel rests).
	//  * nothing at all while the lip is in the air: the face and the lip are clear, glassy water
	//  * the lip lands in the trough at the plunge point (b ~ 0.9, xi = Xi): whitewater appears there
	//    and spreads out from it while the tube collapses behind it
	//  * the collapsed tube becomes the roller: whitewater over the front and top of the bore,
	//    shedding foam behind it (carried on by ShoreSim)
	// whitewater over the face after the plunge: s = 0 at the crest .. 1 at the foot of the face
	_faceFill( b, s ) {

		// the foot turns white first (where the lip lands), the top of the face last: along a peeling
		// crest the edge of the broken section slants down and forward instead of standing vertical
		const k = float( 1 ).sub( s ).mul( 0.45 );
		return smoothstep( k.add( 0.9 ), k.add( 1.08 ), b );

	}

	_whitewater( xi, lam, P ) {

		const { b, Xi, Xc, Wt, wBore } = P;
		const landed = smoothstep( 0.86, 0.99, b );
		const spread = saturate( b.sub( 0.9 ).div( 0.45 ) );
		const reach = Xi.mul( 0.25 ).add( spread.mul( Xi.mul( 0.9 ).add( 1.0 ) ) ); // radius around the plunge point
		const impact = landed.mul( smoothstep( reach, reach.mul( 0.6 ), xi.sub( Xi ).abs() ) ).mul( float( 1 ).sub( smoothstep( 1.4, 1.9, b ) ) );
		const toe = Xc.add( Wt );
		// the collapsing tube turns white from its foot (next to the plunge point) up to the crest
		const faceFill = this._faceFill( b, saturate( xi.sub( Xc ).div( max( Wt, 0.05 ) ) ) );
		const ahead = exp( xi.sub( toe ).mul( - 3 ) ).mul( wBore );
		const behind = exp( Xc.sub( xi ).div( lam ).mul( - 16 ) ).mul( wBore );
		const roller = select( xi.greaterThan( toe ), ahead, select( xi.lessThan( Xc ), behind, max( faceFill, wBore ) ) );
		return saturate( max( impact, roller ) );

	}

	// ------------------------------------------------------------ cheap wave direction lookup

	// Low-resolution filtered texture of the wave direction and exposure over a region (one bilinear
	// fetch instead of the 4 exact loads of the shore field) for per-pixel effects near the beach.
	// region: { min: Vector2, size }. Returns vec3( dir.x, dir.z, exposure ) in TSL via dirAt().
	buildDirTexture( { min, size, res = 128 } ) {

		const tex = this.terrain.shoreTexture;
		const data = tex.image.data, fres = tex.image.width;
		const origin = this.terrain.origin, tsize = this.terrain.size;
		const out = new Float32Array( res * res * 4 );
		for ( let j = 0; j < res; j ++ ) for ( let i = 0; i < res; i ++ ) {

			const x = min.x + ( i + 0.5 ) / res * size, z = min.y + ( j + 0.5 ) / res * size;
			const fx = Math.min( Math.max( ( x - origin ) / tsize * fres - 0.5, 0 ), fres - 1.001 );
			const fz = Math.min( Math.max( ( z - origin ) / tsize * fres - 0.5, 0 ), fres - 1.001 );
			const i0 = Math.floor( fx ), j0 = Math.floor( fz ), tx = fx - i0, tz = fz - j0;
			const at = ( c ) => {

				const a = data[ ( j0 * fres + i0 ) * 4 + c ], b = data[ ( j0 * fres + i0 + 1 ) * 4 + c ];
				const cc = data[ ( ( j0 + 1 ) * fres + i0 ) * 4 + c ], d = data[ ( ( j0 + 1 ) * fres + i0 + 1 ) * 4 + c ];
				return ( a * ( 1 - tx ) + b * tx ) * ( 1 - tz ) + ( cc * ( 1 - tx ) + d * tx ) * tz;

			};

			const dx = at( 1 ), dz = at( 2 );
			const e = Math.hypot( dx, dz ) || 1e-4;
			const k = ( j * res + i ) * 4;
			out[ k ] = dx / e;
			out[ k + 1 ] = dz / e;
			out[ k + 2 ] = Math.min( 1, e * 1.4 );
			out[ k + 3 ] = 1;

		}

		// float data, read with exact loads: no sampler binding (fragment shaders are at their sampler limit)
		const t = new THREE.DataTexture( out, res, res, THREE.RGBAFormat, THREE.FloatType );
		t.magFilter = t.minFilter = THREE.NearestFilter;
		t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
		t.generateMipmaps = false;
		t.colorSpace = THREE.NoColorSpace;
		t.name = 'shoreDir';
		t.needsUpdate = true;
		this.dirTexture = t;
		this.dirMin = uniform( min.clone() ).setName( 'swDirMin' );
		this.dirSize = uniform( size ).setName( 'swDirSize' );
		this.dirRes = res;
		return t;

	}

	// vec3( dir.x, dir.z, exposure ) from the direction texture (bilinear from 4 loads)
	dirAt( xz ) {

		const res = this.dirRes;
		const f = xz.sub( this.dirMin ).div( this.dirSize ).mul( res ).sub( 0.5 );
		const fc = clamp( f, 0, res - 1.001 );
		const i = ivec2( floor( fc ) );
		const t = fract( fc );
		const tex = this.dirTexture;
		const a = textureLoad( tex, i ), b = textureLoad( tex, i.add( ivec2( 1, 0 ) ) );
		const c = textureLoad( tex, i.add( ivec2( 0, 1 ) ) ), d = textureLoad( tex, i.add( ivec2( 1, 1 ) ) );
		return mix( mix( a, b, t.x ), mix( c, d, t.x ), t.y ).xyz;

	}

	// ------------------------------------------------------------ surf zone water

	// Optical properties of the water stirred up by breaking waves: suspended sand and fine bubbles
	// scatter light (milky, luminous), fine sediment and dissolved matter absorb blue. Returns the
	// extra { scatter, absorb } coefficients (1/m) that make the surf zone turquoise, not ocean-clear.
	surfMedium( xz, depth ) {

		const k = float( 0 ).toVar();
		If( depth.lessThan( 4.5 ).and( depth.greaterThan( - 0.2 ) ), () => {

			const expo = this.dirTexture ? this.dirAt( xz ).z : saturate( length( this.terrain.shoreSample( xz ).yz ).mul( 1.4 ) );
			k.assign( smoothstep( 4.5, 1.2, depth ).mul( smoothstep( - 0.2, 0.15, depth ) ).mul( expo ).mul( this.turbidity ).mul( this.enabled ) );

		} );
		return { scatter: vec3( 0.9, 1.0, 0.85 ).mul( k ), absorb: vec3( 0.1, 0.2, 0.62 ).mul( k ) };

	}

	// ------------------------------------------------------------ light through thin crests

	// Water path (m) along the refracted view ray Tv (unit, inside the water) from the surface point
	// with rest position lagXZ until the ray leaves through the other side of the wave, or 1e4 if
	// it doesn't within a few metres. The upper part of a steep wave is only a few metres thick
	// horizontally: the view ray crosses it and exits into the sky behind, which is what makes
	// breaking faces and crests glow turquoise. Marches the analytic cross-section (up to 3 steps).
	crestPath( lagXZ, depth, Tv ) {

		if ( ! this._crestPathFn ) {

			this._crestPathFn = Fn( ( [ p, d, T ] ) => {

				const out = float( 1e4 ).toVar();
				const ph = this.phaseAt( p );
				const tXi = dot( T.xz, ph.dir ).toVar(); // shoreward component of the ray
				const env = smoothstep( 26, 13, d ).mul( saturate( ph.exposure.mul( 1.4 ) ) ).mul( this.enabled ).toVar();
				// rays heading out through the back of the wave (a view from the beach side), near breakers
				If( env.greaterThan( 0.05 ).and( tXi.lessThan( - 0.05 ) ).and( d.lessThan( 6 ) ), () => {

					const lam = sqrt( clamp( d, 0.3, 25 ).mul( GRAVITY ) ).mul( this.period ).toVar();
					const m = floor( ph.s.add( 0.5 ) );
					const u = ph.s.sub( m ).toVar();
					const A = this.waveAmp( m, ph.along ).toVar();
					const P = this._breakParams( A, this.breakDepth( p, ph.dir, u, lam, d ) );
					const s0 = this._profile( u, lam, P, false );
					const y0 = s0.y.mul( env ).toVar();
					// only the upper part of steep (nearly breaking) waves is thin enough to see through
					If( y0.greaterThan( A.mul( 0.2 ) ).and( P.b.greaterThan( - 1.5 ) ), () => {

						const xi0 = u.negate().mul( lam ).add( s0.x.mul( env ) ).toVar();
						const slope = T.y.div( tXi.negate() ); // ray rise per metre of horizontal travel
						const prevGap = float( 0 ).toVar();
						const prevDist = float( 0 ).toVar();
						const off = float( 1.1 ).toVar();
						Loop( 3, () => {

							const uk = min( u.add( off.div( lam ) ), 0.5 );
							const sk = this._profile( uk, lam, P, false );
							const dist = xi0.sub( uk.negate().mul( lam ).add( sk.x.mul( env ) ) ).abs();
							// ray height above the surface there (> 0: the ray has left the water)
							const gap = y0.add( slope.mul( dist ) ).sub( sk.y.mul( env ) ).toVar();
							If( gap.greaterThan( 0 ), () => {

								const f = prevGap.negate().div( max( gap.sub( prevGap ), 1e-4 ) );
								out.assign( mix( prevDist, dist, saturate( f ) ).div( tXi.negate() ) );
								Break();

							} );
							prevGap.assign( gap );
							prevDist.assign( dist );
							off.mulAssign( 2.6 );

						} );

					} );

				} );
				return out;

			} ).setLayout( {
				name: 'shoreCrestPath', type: 'float',
				inputs: [ { name: 'p', type: 'vec2' }, { name: 'd', type: 'float' }, { name: 'T', type: 'vec3' } ],
			} );

		}

		return this._crestPathFn( lagXZ, depth, Tv );

	}

	// ------------------------------------------------------------ swash

	// Run-up of the most recent wave at a point on the beach (distances in metres up the beach face).
	// The run-up is compared with the height of the sand (converted with the nominal beach slope), so
	// the front is exact at the waterline and follows the contours of the sand. sh: shoreSample( xz ).
	_swashRunup( sh, along, groundH ) {

		const Tp = this.period;
		const exposure = length( vec2( sh.y, sh.z ) );
		const Ts = sh.w;
		const inland = max( groundH.sub( G.seaLevel ), 0 ).div( BEACH_SLOPE ).toVar();
		const ss = this.time.sub( Ts ).div( Tp ).add( this.wobble( along ) );
		const ms = floor( ss );
		const tau = ss.sub( ms ).toVar(); // 0..1 time since that wave's bore reached the shoreline
		const Am = this.waveAmp( ms, along );
		// vertical run-up ~ H on this gentle beach, converted to a horizontal excursion
		const RhMax = Am.mul( 2.1 ).mul( this.runup ).mul( saturate( exposure.mul( 1.4 ) ) ).div( BEACH_SLOPE ).toVar();
		// decelerating uprush, then a backwash that starts slowly and accelerates as the sheet drains
		const su = saturate( tau.div( SWASH_UP ) ), sb = saturate( tau.sub( SWASH_UP ).div( SWASH_DOWN ) );
		const isUp = tau.lessThan( SWASH_UP );
		const Rh = select( isUp, float( 1 ).sub( pow( float( 1 ).sub( su ), 1.5 ) ), float( 1 ).sub( pow( sb, 1.6 ) ) ).mul( RhMax ).sub( 0.3 );
		// the front is lobed, not a straight line: each wave runs up a little differently along the beach
		const lobes = sin( along.mul( 0.61 ).add( ms.mul( 2.3 ) ) ).mul( 0.5 ).add( sin( along.mul( 1.73 ).add( ms.mul( 5.1 ) ) ).mul( 0.3 ) ).add( sin( along.mul( 4.3 ).add( ms.mul( 1.7 ) ) ).mul( 0.2 ) );
		// the backwash never quite exposes the lower beach face: a film of water always covers the
		// first decimetres past the shoreline, so the sea never meets the sand along mesh triangles
		const Rt = max( Rh.add( lobes.mul( Rh.max( 0 ).mul( 0.07 ).add( 0.35 ) ) ), 0.35 ).mul( this.enabled ).toVar();
		return { tau, Rt, inland, RhMax, su, sb, isUp };

	}

	// Water film thickness clipped at the leading edge of the swash sheet, for the water shader's
	// edge fade: min( thickness, distance to the front (m) * 0.08 ). Only evaluated where the film is
	// thin, so the sheet ends on the analytic front instead of the mesh triangles at ~no cost.
	swashClip( xz, thickness ) {

		if ( ! this._swashClipFn ) {

			this._swashClipFn = Fn( ( [ p, t ] ) => {

				const out = t.toVar();
				If( t.lessThan( 0.2 ), () => {

					const g = this.terrain.heightAt( p ).toVar();
					If( g.greaterThan( G.seaLevel ), () => {

						const ph = this.phaseAt( p );
						const r = this._swashRunup( ph.sh, ph.along, g );
						out.assign( min( t, r.Rt.sub( r.inland ).mul( 0.08 ) ) );

					} );

				} );
				return out;

			} ).setLayout( { name: 'shoreSwashClip', type: 'float', inputs: [ { name: 'p', type: 'vec2' }, { name: 't', type: 'float' } ] } );

		}

		return this._swashClipFn( xz, thickness );

	}

	// ------------------------------------------------------------ evaluation at a point

	// Local wave phase data at a (Lagrangian) point: shared by evaluate() and the crest finder.
	phaseAt( xz ) {

		const sh = this.terrain.shoreSample( xz );
		const T = sh.x;
		const dirE = vec2( sh.y, sh.z );
		const exposure = length( dirE );
		const dir = dirE.div( max( exposure, 1e-4 ) );
		const along = dot( xz, vec2( dir.y.negate(), dir.x ) );
		const s = this.time.sub( T ).div( this.period ).add( this.wobble( along ) );
		return { sh, T, dir, exposure, along, s };

	}

	// Returns displacement relative to (xz, seaLevel), foam, breaking indicator and the
	// swash surface level (absolute height) for this point. With withNormal (the water mesh) also
	// face: 1 on the clear concave face of a plunging wave, roller: relief of the whitewater (m).
	// world: the shore simulation's view (a fixed world point instead of a water parcel): foam and
	// height are those of the water shown at xz, the displacement is not meaningful.
	evaluate( xz, depth, groundH, { withNormal = true, world = false } = {} ) {

		const ph = this.phaseAt( xz );
		const { sh, exposure, along } = ph;
		const dir = ph.dir.toVar();
		const Tp = this.period;

		const d = depth;
		const c = sqrt( clamp( d, 0.3, 25 ).mul( GRAVITY ) );
		const lam = c.mul( Tp ).toVar();

		const s = ph.s;
		const m = floor( s.add( 0.5 ) );
		const u = s.sub( m ).toVar();

		// wave height with smooth hand-over between consecutive waves at the trough
		const A0 = this.waveAmp( m, along );
		const An = this.waveAmp( m.add( sign( u ) ), along );
		const A = mix( A0, An, smoothstep( 0.32, 0.5, abs( u ) ).mul( 0.5 ) ).toVar();
		// the breaking state of the whole wave comes from the depth under its crest
		const dB = this.breakDepth( xz, dir, u, lam, d ).toVar();

		// offshore fade-in (FFT covers deep water) and fade on land (the swash sheet takes over there)
		const env = smoothstep( 26, 13, d ).mul( smoothstep( - 0.25, 0.05, d ) ).mul( saturate( exposure.mul( 1.4 ) ) ).mul( this.enabled ).toVar();

		// finite difference along the propagation direction for the normal
		const e = 0.15;
		let sh0, sh1 = null;
		let face = float( 0 ), roller = float( 0 );
		if ( world ) {

			const r = this._worldFn( u, A, dB, lam, env ).toVar();
			sh0 = { x: r.z, y: r.y, foam: r.x, b: r.w };

		} else if ( withNormal ) {

			const pair = this._shapePairFn( u, float( - e ).div( lam ), A, dB, lam ).toVar();
			const c0 = pair.element( 0 ), c1 = pair.element( 1 );
			sh0 = { x: c0.x, y: c0.y, foam: c0.z, b: c0.w };
			sh1 = { x: c1.x, y: c1.y };
			face = c1.z.mul( env );
			roller = c1.w.mul( env ).toVar();

		} else sh0 = this.shape( u, A, dB, lam );

		const disp = vec3( dir.x.mul( sh0.x ), sh0.y, dir.y.mul( sh0.x ) ).mul( env ).toVar();

		let nShore = vec3( 0, 1, 0 );
		if ( withNormal ) {

			const dX = float( e ).add( sh1.x.sub( sh0.x ).mul( env ) );
			const dY = sh1.y.sub( sh0.y ).mul( env );
			const tAlong = vec3( dir.x.mul( dX ), dY, dir.y.mul( dX ) );
			const tAcross = vec3( dir.y.negate(), 0, dir.x );
			// epsilon: never a zero vector; never facing down (the mesh doesn't overhang, the lip sheet does)
			const n = normalize( cross( tAcross, tAlong ).add( vec3( 0, 1e-4, 0 ) ) ).toVar();

			// the whitewater roller is not a smooth tube: lumps of foam tumble along its front and over
			// its top (relief of a few decimetres, with its slope in the normal)
			const sx = u.mul( lam ); // rest position along the wave direction (m, seaward)
			const t = this.time;
			const a3 = along.mul( 0.61 ).add( t.mul( 0.9 ) );
			const a1 = along.mul( 1.7 ).add( sx.mul( 1.1 ) ).sub( t.mul( 2.3 ) ).add( sin( a3 ).mul( 2 ) );
			const a2 = along.mul( 4.3 ).sub( sx.mul( 2.7 ) ).add( t.mul( 3.7 ) );
			const amp = roller.mul( 0.2 );
			const lump = sin( a1 ).mul( 0.6 ).add( sin( a2 ).mul( 0.4 ) );
			disp.y.addAssign( lump.mul( amp ) );
			const c1a = cos( a1 ), c2a = cos( a2 );
			const dAlong = c1a.mul( cos( a3 ).mul( 2 * 0.61 ).add( 1.7 ) ).mul( 0.6 ).add( c2a.mul( 4.3 * 0.4 ) );
			const dShore = c1a.mul( - 1.1 * 0.6 ).add( c2a.mul( 2.7 * 0.4 ) ); // d/d(shoreward) = - d/dsx
			const g = vec2( dir.y.negate(), dir.x ).mul( dAlong ).add( dir.mul( dShore ) ).mul( amp ).mul( n.y );
			nShore = normalize( vec3( n.x.sub( g.x ), max( n.y, 0.04 ), n.z.sub( g.y ) ) );

		}

		// ---- swash: run-up of the most recent wave on the sand
		const swr = this._swashRunup( sh, along, groundH );
		const { tau, Rt, inland, RhMax, su, sb, isUp } = swr;
		const up = SWASH_UP, dn = SWASH_DOWN, beachSlope = BEACH_SLOPE;
		const front = Rt.sub( inland ).toVar(); // signed distance to the leading edge (m), > 0 under the sheet
		const covered = front.greaterThan( 0 );
		// leading edge velocity along the slope (m/s), positive = uphill
		const dRdt = select( isUp, pow( float( 1 ).sub( su ), 0.5 ).mul( 1.5 / ( up ) ), pow( sb.max( 1e-3 ), 0.6 ).mul( - 1.6 / dn ) ).mul( RhMax ).div( Tp ).toVar();
		// thin sheet (a few cm, thickening behind the leading edge). The mesh sheet overshoots the
		// leading edge a little; the water shader cuts it exactly on the front (swashClip), so the edge
		// doesn't follow the mesh triangles.
		const fm = front.add( SWASH_OVERSHOOT );
		const thick = clamp( min( fm.mul( 0.3 ), max( fm.sub( 0.1 ), 0 ).mul( beachSlope * 0.22 ).add( 0.03 ) ), - 0.1, 0.12 ).toVar();
		const swashLevel = groundH.add( thick );
		// bubbly foam line riding the leading edge all the way up (left behind as the swash mark)
		const uprush = smoothstep( 0.46, 0.32, tau );
		const edge = smoothstep( 0.8, 0.0, front ).mul( smoothstep( - 0.05, 0.05, front ) ).mul( uprush ).mul( smoothstep( 0.0, 1.0, Rt ) );
		const swashFoam = edge.mul( 0.9 );

		// ---- depth-averaged water velocity (along dir), for foam advection
		// waves / bores: shallow-water particle velocity c * eta / h; swash sheet: the tip moves at
		// dR/dt, slower toward the shoreline during uprush, faster there while it drains
		const eta = disp.y;
		const uWave = c.mul( eta ).div( max( d, 0 ).add( max( eta, d.mul( - 0.8 ) ) ).add( 0.15 ) ).clamp( - 2.5, 4.0 );
		const rel = saturate( inland.div( max( Rt, 0.5 ) ) );
		const uSwash = dRdt.mul( select( isUp, clamp( inland.add( 3 ).div( Rt.add( 3 ) ), 0.15, 1 ), rel.oneMinus().mul( 0.6 ).add( 0.9 ) ) );
		const wSwash = smoothstep( 0.3, 0.05, d );
		const uFlow = mix( uWave, uSwash, wSwash ).mul( select( covered.or( d.greaterThan( 0.02 ) ), float( 1 ), float( 0 ) ) );

		return {
			disp, nShore, env, foam: sh0.foam.mul( env ), breaking: sh0.b, u, dir, exposure,
			swashLevel, swashCovered: covered, thick, swashFoam, runup: Rt, inland, dRdt, tau,
			flow: dir.mul( uFlow ), flowSpeed: uFlow, face, roller,
		};

	}

}
