import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, vec2, vec3, vec4, ivec2, textureLoad, textureSize, screenUV, screenCoordinate, rtt, mix,
	smoothstep, dot, max, min, abs, exp, clamp, floor, fract, select, If, Loop, normalize, length,
	interleavedGradientNoise, luminance,
} from 'three/tsl';
import { G } from '../core/Globals.js';
import { SHADOW_SPLITS } from '../materials/SunShadowFilter.js';

// Air above the water, in post (no per-material cost):
//  - aerial perspective + marine haze: two exponential height layers (a thin, dense marine layer at
//    the sea surface and a thin aerosol layer kilometres high), analytic optical depth along the view
//    ray. The haze takes the colour of the sky (Hillaire sky view LUT) just above the horizon in the
//    view direction, so distant land and the far sea fade into the actual horizon sky, sun glow
//    included. Sky pixels get none (the sky already holds its in-scatter).
//  - volumetric sun shafts: sun in-scatter of the haze ray-marched at half resolution with the
//    cascaded shadow maps (palms, pier, rocks), the terrain hill shadow and the cloud shadow
//    (crepuscular rays), jittered per pixel and frame (the TAAU resolves the noise), depth-aware
//    upsampled in the composite. Near geometry gets the lit in-scatter (bright shafts between shadowed
//    air); toward the sky / far geometry it hands over to the shadowed deficit only, so silhouettes
//    against the sky stay consistent and the sky is only ever darkened (crepuscular rays in the sky).
// The camera under water, and pixels whose lens is in water, get nothing (Underwater handles those).

const STEPS = 16;
const MARCH_DIST = 2500; // m: shafts from clouds and hills reach this far
const NEAR = 900; // m: explicit (shadowed) sun single scattering in front of geometry closer than this
const FAR_CLAMP = 60000; // m (fits half float)
// screen-space god rays: radial blur taps per pass, sample decay per pass, overall gain
const SS_TAPS = 8;
const SS_DECAY = [ 0.9, 0.97, 1.0 ];
const SS_GAIN = 3.5;

// haze layers: sea level extinction (1/m) and scale height (m)
const MARINE = { sigma: 1.5e-4, H: 110 };
const AEROSOL = { sigma: 3.2e-5, H: 1400 };

export class AirHaze {

	constructor( { depthTexture, underwater, atmosphere, sky = null, clouds = null, terrain = null, csm = null } ) {

		this.depthTexture = depthTexture;
		this.uw = underwater;
		this.atmosphere = atmosphere;
		this.sky = sky;
		this.clouds = clouds;
		this.terrain = terrain;
		this.csm = csm;

		// haze (1 = ~20 km visibility at sea level). Default: a humid tropical day, ~12 km: the far side
		// of the island and the horizon soften visibly
		this.density = uniform( 1.6 ).setName( 'hzDensity' );
		// sun shaft strength (1 = physical single scattering near the camera; more veils everything in
		// front of a low sun in white)
		this.shafts = uniform( 1.0 ).setName( 'hzShafts' );
		this.enabled = uniform( 1 ).setName( 'hzOn' );
		this.frame = uniform( 0 ).setName( 'hzFrame' );
		this.cascadeMatrices = [];
		this._cascadeLights = [];
		this.mediumTexture = null; // lens medium (set by the post chain): water pixels are skipped

		this.low = rtt( this._marchNode(), null, null, { type: THREE.HalfFloatType, resolutionScale: 0.5 } );
		this.low.setName && this.low.setName( 'hazeShafts' );

		// screen-space god rays (GPU Gems 3 ch. 13 / UE4 light shafts) on top of the volumetric term:
		// sky visibility around the key light (sun disc + aureole, times the cloud transmittance) at
		// quarter resolution, blurred toward the light's screen position in three passes of 8 taps
		// (512 effective samples): crisp streaks through fronds, treelines and cloud gaps
		this.sunUV = uniform( new THREE.Vector2( 0.5, 0.5 ) ).setName( 'hzSunUV' );
		this.ssFade = uniform( 0 ).setName( 'hzSSFade' ); // light in view, low in the sky, camera in air
		const q = () => ( { type: THREE.HalfFloatType, resolutionScale: 0.25 } );
		this.ssPasses = [ rtt( this._sunMaskNode(), null, null, q() ) ];
		for ( let p = 0; p < 3; p ++ ) this.ssPasses.push( rtt( this._radialBlurNode( this.ssPasses[ p ], p ), null, null, q() ) );
		this.ssShafts = this.ssPasses[ 3 ];
		this.godRays = true;

	}

	setScale( s ) {

		this.low.setResolutionScale( 0.5 * s );
		for ( const r of this.ssPasses ) r.setResolutionScale( 0.25 * s );

	}

	// per frame, after the scene render (shadow matrices match the maps' content)
	update() {

		this.frame.value = ( this.frame.value + 1 ) % 1024;
		for ( let i = 0; i < this._cascadeLights.length; i ++ ) this.cascadeMatrices[ i ].value.copy( this._cascadeLights[ i ].shadow.matrix );

		// key light on screen (view space = camera rotation transposed times the light direction)
		const m = this.uw.camWorld.value.elements, L = G.sunDir.value, P = this.uw.proj.value;
		const vx = m[ 0 ] * L.x + m[ 1 ] * L.y + m[ 2 ] * L.z;
		const vy = m[ 4 ] * L.x + m[ 5 ] * L.y + m[ 6 ] * L.z;
		const vz = m[ 8 ] * L.x + m[ 9 ] * L.y + m[ 10 ] * L.z;
		const ss = THREE.MathUtils.smoothstep;
		let fade = 0;
		if ( vz < - 0.02 ) {

			const u = 0.5 + 0.5 * ( vx / - vz ) * P.x, v = 0.5 - 0.5 * ( vy / - vz ) * P.y;
			this.sunUV.value.set( u, v );
			// fades out as the light leaves the frame, and as it climbs (strong at golden hour only)
			fade = ( 1 - ss( Math.max( Math.abs( u - 0.5 ), Math.abs( v - 0.5 ) ), 0.6, 1.15 ) ) * ss( - vz, 0.02, 0.25 ) * ( 1 - ss( L.y, 0.3, 0.75 ) );

		}

		if ( G.cameraUnderwater.value > 0.5 || this.enabled.value < 0.5 || this.shafts.value <= 0 || ! this.godRays ) fade = 0;
		this.ssFade.value = fade;

	}

	// ---------------------------------------------------------------- helpers (TSL)

	// optical depth of one exponential layer from height hc along a ray with direction y component
	// vy over distance d
	static layerDepth( layer, hc, vy, d ) {

		const H = layer.H;
		const base = exp( hc.div( - H ) ).mul( layer.sigma );
		const k = vy.mul( d ).div( H );
		const f = select( abs( k ).lessThan( 1e-3 ), d, float( H ).mul( float( 1 ).sub( exp( k.negate() ) ) ).div( vy ) );
		return base.mul( f );

	}

	// Cornette-Shanks (strong forward lobe) plus a little isotropic scattering
	static phase( cosT ) {

		// (a softer lobe than coastal aerosol's ~0.76: toward the sun the haze glared over everything
		// in front of it and washed distant foliage out to white)
		const g = 0.62, g2 = g * g;
		const cs = float( 3 * ( 1 - g2 ) / ( 8 * Math.PI * ( 2 + g2 ) ) ).mul( cosT.mul( cosT ).add( 1 ) )
			.div( max( float( 1 + g2 ).sub( cosT.mul( 2 * g ) ), 1e-4 ).pow( 1.5 ) );
		return cs.mul( 0.7 ).add( 0.3 / ( 4 * Math.PI ) );

	}

	// view distance of the pixel (reversed-Z depth, sky = 0) and its world direction
	_ray( uv ) {

		const uw = this.uw;
		const size = textureSize( textureLoad( this.depthTexture ), 0 );
		const d = textureLoad( this.depthTexture, ivec2( clamp( uv, 0, 0.9999 ).mul( vec2( size ) ) ) ).x;
		// declared here, unconditionally: TSL would otherwise emit them inside the first branch using them
		const ray = uw.viewRay( uv ).toVar();
		const rayLen = length( ray ).toVar();
		const sky = d.lessThan( 1e-7 ).toVar();
		const dist = select( sky, float( FAR_CLAMP ), min( uw.viewZ( max( d, 1e-9 ) ).negate().mul( rayLen ), FAR_CLAMP ) ).toVar();
		const dir = normalize( uw.camWorld.mul( vec4( ray, 0 ) ).xyz ).toVar();
		return { dist, dir, sky, rayLen };

	}

	// sun visibility at world position P (view depth z): shadow cascades, hills, clouds
	_visibility( P, z ) {

		const v = float( 1 ).toVar();
		const n = this.cascadeMatrices.length;
		if ( n > 0 ) {

			const lookup = ( i ) => {

				const tex = this._cascadeLights[ i ].shadow.map.depthTexture;
				const c = this.cascadeMatrices[ i ].mul( vec4( P, 1 ) );
				const suv = vec2( c.x, c.y.oneMinus() );
				const size = vec2( textureSize( textureLoad( tex ), 0 ) );
				const inside = suv.x.greaterThan( 0 ).and( suv.x.lessThan( 1 ) ).and( suv.y.greaterThan( 0 ) ).and( suv.y.lessThan( 1 ) );
				const dep = textureLoad( tex, ivec2( clamp( suv, 0, 0.9999 ).mul( size ) ) ).x;
				// reversed depth: lit when the point is at least as close to the light as the stored occluder
				v.assign( select( inside.and( c.z.lessThan( dep.sub( 2e-5 ) ) ), float( 0 ), float( 1 ) ) );

			};

			let chain = If( z.lessThan( SHADOW_SPLITS[ 0 ] ), () => lookup( 0 ) );
			for ( let i = 1; i < n; i ++ ) chain = chain.ElseIf( z.lessThan( SHADOW_SPLITS[ Math.min( i, SHADOW_SPLITS.length - 1 ) ] ), () => lookup( i ) );

		}

		const t = this.terrain;
		if ( t && t.sunShadowTexture ) {

			// one nearest texel (the jitter and the TAAU smooth it)
			const N = t.sunShadowTexture.image.width;
			const tuv = t.uvOf( P.xz );
			const s = textureLoad( t.sunShadowTexture, ivec2( clamp( tuv, 0, 0.9999 ).mul( N ) ) );
			const w = s.y.mul( 0.012 ).add( 0.35 );
			const inMap = tuv.x.greaterThan( 0 ).and( tuv.x.lessThan( 1 ) ).and( tuv.y.greaterThan( 0 ) ).and( tuv.y.lessThan( 1 ) );
			const hill = mix( float( 1 ), smoothstep( w.negate(), w, P.y.sub( s.x ) ), t.uSunBaked );
			v.mulAssign( select( inMap, hill, float( 1 ) ) );

		}

		const c = this.clouds;
		if ( c && c.shadowMap ) {

			// the cloud shadow map is the ground's shadow along the key light: follow the light down
			const L = G.sunDir;
			const g = P.xz.sub( L.xz.mul( max( P.y, 0 ).div( max( L.y, 0.08 ) ) ) );
			const cuv = g.sub( c.shadowCenter ).div( c.shadowSize ).add( 0.5 );
			const res = c.shadowMap.image.width;
			const T = textureLoad( c.shadowMap, ivec2( clamp( cuv, 0, 0.9999 ).mul( res ) ) ).x;
			const e = abs( cuv.sub( 0.5 ) );
			v.mulAssign( mix( float( 1 ), T, c.shadowStrength.mul( smoothstep( 0.5, 0.42, max( e.x, e.y ) ) ) ) );

		}

		return v;

	}

	// ---------------------------------------------------------------- half resolution march

	_marchNode() {

		return Fn( () => {

			// cascade lights exist once the scene materials were built (before the post chain is)
			if ( this.csm && this.csm.lights && this.cascadeMatrices.length === 0 ) {

				for ( const l of this.csm.lights ) {

					if ( ! l.shadow || ! l.shadow.map || ! l.shadow.map.depthTexture ) continue;
					this._cascadeLights.push( l );
					this.cascadeMatrices.push( uniform( new THREE.Matrix4().copy( l.shadow.matrix ) ) );

				}

			}

			const uv = screenUV;
			const { dist, dir, rayLen } = this._ray( uv );
			const out = vec4( 0, 0, dist, 1 ).toVar();
			If( this.enabled.greaterThan( 0.5 ).and( G.cameraUnderwater.lessThan( 0.5 ) ).and( this.shafts.greaterThan( 0 ) ), () => {

				const cam = this.uw.camPos;
				const tMax = min( dist, MARCH_DIST ).toVar();
				// interleaved gradient noise, decorrelated per frame (golden ratio sequence)
				const jitter = fract( interleavedGradientNoise( screenCoordinate.xy ).add( float( this.frame ).mul( 0.61803398875 ) ) );
				const sigM = float( MARINE.sigma ).mul( this.density );
				const sigA = float( AEROSOL.sigma ).mul( this.density );
				const lit = float( 0 ).toVar();
				const all = float( 0 ).toVar();
				const tau = float( 0 ).toVar();
				const tPrev = float( 0 ).toVar();
				Loop( STEPS, ( { i } ) => {

					// quadratic spacing: dense near the camera (palm and pier shafts), sparse far out (clouds, hills)
					const u = float( i ).add( jitter ).div( STEPS );
					const t = u.mul( u ).mul( tMax );
					const dt = u.mul( 2 / STEPS ).mul( tMax );
					const P = cam.add( dir.mul( t ) ).toVar();
					const h = max( P.y.sub( G.seaLevel ), 0 );
					const sig = exp( h.div( - MARINE.H ) ).mul( sigM ).add( exp( h.div( - AEROSOL.H ) ).mul( sigA ) );
					tau.addAssign( sig.mul( t.sub( tPrev ) ) );
					tPrev.assign( t );
					const Tr = exp( tau.negate() );
					const w = sig.mul( Tr ).mul( dt );
					lit.addAssign( w.mul( this._visibility( P, t.div( rayLen ) ) ) );
					all.addAssign( w );

				} );

				out.assign( vec4( lit, all, dist, 1 ) );

			} );

			return out;

		} )();

	}

	// ---------------------------------------------------------------- screen-space god rays (quarter res)

	// sources: sky pixels near the key light, weighted by the sun disc and its aureole, times the
	// cloud transmittance in that direction
	_sunMaskNode() {

		return Fn( () => {

			const out = vec4( 0 ).toVar();
			If( this.ssFade.greaterThan( 0.001 ), () => {

				const uv = screenUV;
				const dir = this.uw.worldDir( uv ).toVar();
				const cloudT = ( this.clouds && this.clouds.sampleView ? this.clouds.sampleView( dir ).a : float( 1 ) ).toVar();
				const size = textureSize( textureLoad( this.depthTexture ), 0 );
				const d = textureLoad( this.depthTexture, ivec2( clamp( uv, 0, 0.9999 ).mul( vec2( size ) ) ) ).x;
				const c = dot( dir, G.sunDir );
				const glow = exp( c.sub( 1 ).mul( 600 ) ).add( exp( c.sub( 1 ).mul( 50 ) ).mul( 0.25 ) );
				out.assign( vec4( select( d.lessThan( 1e-7 ), glow.mul( cloudT ), float( 0 ) ), 0, 0, 1 ) );

			} );
			return out;

		} )();

	}

	// one pass of the iterative radial blur toward the light: pass p spans 1 / 8^p of the way, so the
	// three passes together sample the whole segment densely
	_radialBlurNode( src, p ) {

		const span = 0.95 / Math.pow( SS_TAPS, p );
		const decay = SS_DECAY[ p ];
		return Fn( () => {

			const out = vec4( 0 ).toVar();
			If( this.ssFade.greaterThan( 0.001 ), () => {

				const uv = screenUV;
				const step = this.sunUV.sub( uv ).mul( span / SS_TAPS ).toVar();
				const acc = float( 0 ).toVar();
				let wSum = 0;
				for ( let j = 0; j < SS_TAPS; j ++ ) {

					const w = Math.pow( decay, j );
					acc.addAssign( src.sample( uv.add( step.mul( j ) ) ).level( 0 ).r.mul( w ) );
					wSum += w;

				}

				out.assign( vec4( acc.div( wSum ), 0, 0, 1 ) );

			} );
			return out;

		} )();

	}

	// ---------------------------------------------------------------- composite (full resolution)

	// c: the scene colour at uv (vec4). Returns the hazed colour (vec4). Call inside a Fn.
	//   geometry: c T + (1 - T) fog (1 - fSun (1 - h)) + (1 - h) E p(θ) lit - h fSun fog (all - lit)
	//   sky:      c - fSun fog (all - lit)
	// fog: sky radiance just above the horizon in the view direction; fSun: its sunlit share (phase
	// weighted); lit / all: shadowed / unshadowed in-scatter depth from the march; h: 0 near the
	// camera (explicit, shadowed sun single scattering: bright shafts) -> 1 far away and for the sky
	// (only the shadowed share of the sky-coloured haze is removed: consistent with the sky)
	apply( uv, c ) {

		const out = c.rgb.toVar();
		const active = this.enabled.greaterThan( 0.5 ).and( G.cameraUnderwater.lessThan( 0.5 ) );
		If( active, () => {

			const inAir = float( 1 ).toVar();
			if ( this.mediumTexture ) {

				const m = this.mediumTexture;
				const ms = ivec2( m.size() );
				inAir.assign( select( m.load( ivec2( uv.mul( vec2( ms ) ) ).clamp( ivec2( 0 ), ms.sub( 1 ) ) ).r.greaterThan( 0.5 ), float( 0 ), float( 1 ) ) );

			}

			If( inAir.greaterThan( 0.5 ), () => {

				const { dist, dir, sky } = this._ray( uv );
				const camH = max( this.uw.camPos.y.sub( G.seaLevel ), 0 );

				// the haze looks like the sky just above the horizon in this direction
				const vh = normalize( vec3( dir.x, max( dir.y, 0.02 ), dir.z ) );
				let fogN = this.atmosphere.skyLuminance( vh );
				if ( this.sky && this.sky.moonSky ) fogN = fogN.add( this.sky.moonSky( vh ) );
				const fog = fogN.toVar();

				// sun (moon) light scattered toward the eye, and its share of the haze radiance
				const Ep = G.sunColor.mul( AirHaze.phase( dot( dir, G.sunDir ) ) ).toVar();
				const eL = luminance( Ep );
				const fSun = eL.div( eL.add( luminance( G.skyIrradiance ) ).add( 1e-5 ) ).mul( min( this.shafts, 1 ) ).toVar();
				const h = select( sky, float( 1 ), smoothstep( 0, NEAR, dist ) ).toVar();

				// ---- aerial perspective / marine haze on geometry (the water surface included)
				If( sky.not(), () => {

					const tau = AirHaze.layerDepth( MARINE, camH, dir.y, dist ).add( AirHaze.layerDepth( AEROSOL, camH, dir.y, dist ) ).mul( this.density );
					const T = exp( tau.negate() );
					out.assign( out.mul( T ).add( fog.mul( float( 1 ).sub( T ) ).mul( float( 1 ).sub( fSun.mul( float( 1 ).sub( h ) ) ) ) ) );

				} );

				// ---- sun shafts: depth-aware upsample of the half resolution march
				If( this.shafts.greaterThan( 0 ), () => {

					const low = this.low;
					const ls = ivec2( low.size() );
					const pa = uv.mul( vec2( ls ) ).sub( 0.5 );
					const i0 = floor( pa );
					const fr = pa.sub( i0 );
					const acc = vec2( 0 ).toVar();
					const wSum = float( 1e-6 ).toVar();
					for ( const [ ox, oy ] of [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ], [ 1, 1 ] ] ) {

						const s = low.load( ivec2( i0 ).add( ivec2( ox, oy ) ).clamp( ivec2( 0 ), ls.sub( 1 ) ) );
						const wb = ( ox ? fr.x : fr.x.oneMinus() ).mul( oy ? fr.y : fr.y.oneMinus() );
						const rel = abs( s.z.sub( dist ) ).div( max( dist, 0.5 ) );
						const wd = float( 1 ).div( rel.mul( 10 ).add( 1 ).pow2() );
						const wt = wb.mul( wd ).add( 1e-5 );
						acc.addAssign( s.xy.mul( wt ) );
						wSum.addAssign( wt );

					}

					const sh = acc.div( wSum );
					const lit = sh.x, all = max( sh.y, sh.x );
					const near = Ep.mul( lit ).mul( float( 1 ).sub( h ) ).mul( this.shafts );
					const deficit = fog.mul( fSun ).mul( all.sub( lit ) ).mul( h );
					out.assign( max( out.add( near ).sub( deficit ), vec3( 0 ) ) );

				} );

				// ---- screen-space god rays: sun colour x phase x the near air's haze depth
				If( this.ssFade.greaterThan( 0.001 ), () => {

					const rays = this.ssShafts.sample( uv ).level( 0 ).r;
					const k = float( 1 ).sub( exp( float( - MARINE.sigma * 300 ).mul( this.density ) ) );
					out.addAssign( Ep.mul( rays ).mul( k ).mul( this.shafts ).mul( this.ssFade ).mul( SS_GAIN ) );

				} );

			} );

		} );

		return vec4( out, c.a );

	}

}
