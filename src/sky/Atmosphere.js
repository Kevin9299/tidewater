import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, int, vec2, vec3, vec4, uvec2, globalId, If, Loop, select,
	sqrt, exp, max, min, abs, clamp, dot, normalize, length, cos, sin, acos, texture, textureStore,
	mix, smoothstep, pow, instancedArray,
} from 'three/tsl';
import { G } from '../core/Globals.js';

// Physically based sky (Hillaire 2020, "A Scalable and Production Ready Sky and Atmosphere
// Rendering Technique"). All distances in km inside the atmosphere code.

const RG = 6360.0;
const RT = 6460.0;
const PI = Math.PI;

export const SUN_ILLUMINANCE = 11.0; // scene units (sun irradiance outside the atmosphere)
export const SUN_ANGULAR_RADIUS = 0.004675 * 1.15;

const T_W = 256, T_H = 64;
const MS_RES = 32;
const SV_W = 192, SV_H = 108;

function makeLUT( w, h, name ) {

	const t = new THREE.StorageTexture( w, h );
	t.type = THREE.HalfFloatType;
	t.format = THREE.RGBAFormat;
	t.magFilter = t.minFilter = THREE.LinearFilter;
	t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
	t.generateMipmaps = false;
	t.name = name;
	return t;

}

export class Atmosphere {

	constructor( renderer ) {

		this.renderer = renderer;
		this.transmittanceLUT = makeLUT( T_W, T_H, 'atmoTransmittance' );
		this.multiScatLUT = makeLUT( MS_RES, MS_RES, 'atmoMultiScat' );
		this.skyViewLUT = makeLUT( SV_W, SV_H, 'atmoSkyView' );
		this.skyViewLUT.wrapS = THREE.ClampToEdgeWrapping;

		this.rayleighScale = uniform( 1 ).setName( 'atmoRay' );
		this.mieScale = uniform( 1 ).setName( 'atmoMie' );
		this.mieG = uniform( 0.8 ).setName( 'atmoMieG' );
		this.ozoneScale = uniform( 1 ).setName( 'atmoOzone' );
		this.groundAlbedo = uniform( new THREE.Color( 0.06, 0.08, 0.1 ) ).setName( 'atmoAlbedo' );
		this.viewHeight = uniform( RG + 0.002 ).setName( 'atmoViewH' ); // km
		this.sunIlluminance = uniform( new THREE.Color( SUN_ILLUMINANCE, SUN_ILLUMINANCE, SUN_ILLUMINANCE ) ).setName( 'atmoSunE' );
		// the real sun (may be below the horizon: twilight, night). G.sunDir is the key light, which
		// becomes the moon at night; the sky itself is always scattered sunlight.
		this.sunDir = uniform( new THREE.Vector3( 0.3, 0.6, - 0.7 ).normalize() ).setName( 'atmoSunDir' );

		// small buffer for sky irradiance readback
		this.irrBuffer = instancedArray( 4, 'vec4' ).setName( 'atmoIrr' );

		this._build();
		this.needsStatic = true;
		this._irrPending = false;
		this._irrTimer = 0;

	}

	// ---------------------------------------------------------------- medium

	_medium( hKm ) {

		const rayDensity = exp( hKm.div( 8.0 ).negate() );
		const mieDensity = exp( hKm.div( 1.2 ).negate() );
		const ozoneDensity = max( 0, float( 1 ).sub( abs( hKm.sub( 25 ) ).div( 15 ) ) );
		const rayScat = vec3( 5.802e-3, 13.558e-3, 33.1e-3 ).mul( rayDensity ).mul( this.rayleighScale );
		const mieScat = float( 3.996e-3 ).mul( mieDensity ).mul( this.mieScale );
		const mieExt = float( 4.440e-3 ).mul( mieDensity ).mul( this.mieScale );
		const ozoneAbs = vec3( 0.650e-3, 1.881e-3, 0.085e-3 ).mul( ozoneDensity ).mul( this.ozoneScale );
		const extinction = rayScat.add( mieExt ).add( ozoneAbs );
		return { rayScat, mieScat, extinction, scattering: rayScat.add( mieScat ) };

	}

	// ---------------------------------------------------------------- LUT parameterizations

	// (r, mu) -> transmittance LUT uv
	transmittanceUV( r, mu ) {

		const H = Math.sqrt( RT * RT - RG * RG );
		const rho = sqrt( max( r.mul( r ).sub( RG * RG ), 0 ) );
		const disc = r.mul( r ).mul( mu.mul( mu ).sub( 1 ) ).add( RT * RT );
		const d = max( 0, r.mul( mu ).negate().add( sqrt( max( disc, 0 ) ) ) );
		const dMin = float( RT ).sub( r );
		const dMax = rho.add( H );
		const xMu = d.sub( dMin ).div( dMax.sub( dMin ) );
		const xR = rho.div( H );
		// unit -> sub-uv
		return vec2(
			xMu.add( 0.5 / T_W ).mul( T_W / ( T_W + 1 ) ),
			xR.add( 0.5 / T_H ).mul( T_H / ( T_H + 1 ) )
		);

	}

	sampleTransmittance( r, mu ) {

		return texture( this.transmittanceLUT, this.transmittanceUV( r, mu ) ).level( 0 ).rgb;

	}

	sampleMultiScat( r, cosSun ) {

		const uv = vec2( cosSun.mul( 0.5 ).add( 0.5 ), r.sub( RG ).div( RT - RG ) );
		const suv = uv.mul( ( MS_RES - 1 ) / MS_RES ).add( 0.5 / MS_RES );
		return texture( this.multiScatLUT, suv ).level( 0 ).rgb;

	}

	// nearest positive ray-sphere intersection from ro (km, planet centered), -1 if none
	static raySphereNearest( ro, rd, radius ) {

		const b = dot( ro, rd );
		const c = dot( ro, ro ).sub( radius * radius );
		const disc = b.mul( b ).sub( c );
		const sq = sqrt( max( disc, 0 ) );
		const t0 = b.negate().sub( sq );
		const t1 = b.negate().add( sq );
		return select( disc.lessThan( 0 ), float( - 1 ), select( t0.greaterThan( 0 ), t0, select( t1.greaterThan( 0 ), t1, float( - 1 ) ) ) );

	}

	// ---------------------------------------------------------------- kernels

	_build() {

		const self = this;

		// ----- transmittance LUT
		this.transmittanceKernel = Fn( () => {

			const px = globalId.xy;
			const uv = vec2( px ).add( 0.5 ).div( vec2( T_W, T_H ) );
			const xMu = uv.x.sub( 0.5 / T_W ).mul( T_W / ( T_W - 1 ) );
			const xR = uv.y.sub( 0.5 / T_H ).mul( T_H / ( T_H - 1 ) );
			const H = Math.sqrt( RT * RT - RG * RG );
			const rho = xR.mul( H );
			const r = sqrt( rho.mul( rho ).add( RG * RG ) ).toVar();
			const dMin = float( RT ).sub( r );
			const dMax = rho.add( H );
			const d = dMin.add( xMu.mul( dMax.sub( dMin ) ) );
			const mu = clamp( select( d.equal( 0 ), float( 1 ), float( H * H ).sub( rho.mul( rho ) ).sub( d.mul( d ) ).div( r.mul( d ).mul( 2 ) ) ), - 1, 1 ).toVar();

			const ro = vec3( 0, r, 0 );
			const rd = vec3( sqrt( max( float( 1 ).sub( mu.mul( mu ) ), 0 ) ), mu, 0 );
			const tMax = Atmosphere.raySphereNearest( ro, rd, RT ).toVar();
			const steps = 40;
			const dt = tMax.div( steps );
			const od = vec3( 0 ).toVar();

			Loop( steps, ( { i } ) => {

				const t = float( i ).add( 0.5 ).mul( dt );
				const p = ro.add( rd.mul( t ) );
				const h = length( p ).sub( RG );
				od.addAssign( self._medium( h ).extinction.mul( dt ) );

			} );

			textureStore( this.transmittanceLUT, uvec2( px ), vec4( exp( od.negate() ), 1 ) );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Atmosphere Transmittance' );

		// ----- multiple scattering LUT
		this.multiScatKernel = Fn( () => {

			const px = globalId.xy;
			const uv = vec2( px ).add( 0.5 ).div( MS_RES ).sub( 0.5 / MS_RES ).mul( MS_RES / ( MS_RES - 1 ) );
			const cosSun = uv.x.mul( 2 ).sub( 1 );
			const r = float( RG ).add( clamp( uv.y, 0.001, 0.999 ).mul( RT - RG ) ).toVar();
			const sunDir = normalize( vec3( 0, cosSun, sqrt( max( float( 1 ).sub( cosSun.mul( cosSun ) ), 0 ) ).negate() ) ).toVar();
			const ro = vec3( 0, r, 0 ).toVar();

			const Lsum = vec3( 0 ).toVar();
			const fmsSum = vec3( 0 ).toVar();
			const SQ = 8;
			const isoPhase = 1 / ( 4 * PI );

			Loop( SQ * SQ, ( { i } ) => {

				const ii = float( i.mod( SQ ) ).add( 0.5 ).div( SQ );
				const jj = float( i.div( SQ ) ).add( 0.5 ).div( SQ );
				const theta = ii.mul( 2 * PI );
				const phi = acos( float( 1 ).sub( jj.mul( 2 ) ) );
				const rd = vec3( cos( theta ).mul( sin( phi ) ), cos( phi ), sin( theta ).mul( sin( phi ) ) ).toVar();

				const tBottom = Atmosphere.raySphereNearest( ro, rd, RG ).toVar();
				const tTop = Atmosphere.raySphereNearest( ro, rd, RT ).toVar();
				const hitGround = tBottom.greaterThan( 0 );
				const tMax = select( hitGround, tBottom, tTop ).toVar();
				const steps = 20;
				const dt = tMax.div( steps );
				const throughput = vec3( 1 ).toVar();
				const L = vec3( 0 ).toVar();
				const fms = vec3( 0 ).toVar();

				Loop( { start: 0, end: steps, type: 'int', condition: '<', name: 's' }, ( { s } ) => {

					const t = float( s ).add( 0.3 ).mul( dt );
					const p = ro.add( rd.mul( t ) ).toVar();
					const pr = length( p );
					const m = self._medium( pr.sub( RG ) );
					const up = p.div( pr );
					const cosSunP = dot( up, sunDir );
					const Tsun = self.sampleTransmittance( pr, cosSunP );
					const shadowT = Atmosphere.raySphereNearest( p, sunDir, RG );
					const earthShadow = select( shadowT.greaterThan( 0 ), float( 0 ), float( 1 ) );
					const S = Tsun.mul( earthShadow ).mul( m.scattering ).mul( isoPhase );
					const Tstep = exp( m.extinction.mul( dt ).negate() );
					const ext = max( m.extinction, vec3( 1e-6 ) );
					L.addAssign( throughput.mul( S.sub( S.mul( Tstep ) ).div( ext ) ) );
					fms.addAssign( throughput.mul( m.scattering.sub( m.scattering.mul( Tstep ) ).div( ext ) ) );
					throughput.mulAssign( Tstep );

				} );

				If( hitGround, () => {

					const p = ro.add( rd.mul( tMax ) );
					const up = normalize( p );
					const cosS = dot( up, sunDir );
					const Tsun = self.sampleTransmittance( float( RG ), cosS );
					L.addAssign( Tsun.mul( throughput ).mul( max( cosS, 0 ) ).mul( self.groundAlbedo ).div( PI ) );

				} );

				Lsum.addAssign( L.mul( 4 * PI / ( SQ * SQ ) ) );
				fmsSum.addAssign( fms.mul( 4 * PI / ( SQ * SQ ) ) );

			} );

			const Lin = Lsum.mul( isoPhase );
			const fmsAvg = fmsSum.mul( isoPhase );
			const Lms = Lin.div( vec3( 1 ).sub( fmsAvg ) );
			textureStore( this.multiScatLUT, uvec2( px ), vec4( Lms, 1 ) );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Atmosphere MultiScat' );

		// ----- sky view LUT (per frame)
		const sunE = this.sunIlluminance;
		const mieG = this.mieG;

		this.skyViewKernel = Fn( () => {

			const px = globalId.xy;
			const uv = vec2( px ).add( 0.5 ).div( vec2( SV_W, SV_H ) );
			const u = uv.x.sub( 0.5 / SV_W ).mul( SV_W / ( SV_W - 1 ) );
			const v = uv.y.sub( 0.5 / SV_H ).mul( SV_H / ( SV_H - 1 ) );

			const viewH = this.viewHeight;
			const vHorizon = sqrt( max( viewH.mul( viewH ).sub( RG * RG ), 0 ) );
			const beta = acos( vHorizon.div( viewH ) );
			const zenithHorizonAngle = float( PI ).sub( beta );

			const vzA = float( 0 ).toVar();
			If( v.lessThan( 0.5 ), () => {

				let c = v.mul( 2 );
				c = float( 1 ).sub( c );
				c = c.mul( c );
				c = float( 1 ).sub( c );
				vzA.assign( zenithHorizonAngle.mul( c ) );

			} ).Else( () => {

				let c = v.mul( 2 ).sub( 1 );
				c = c.mul( c );
				vzA.assign( zenithHorizonAngle.add( beta.mul( c ) ) );

			} );

			const cosViewZenith = cos( vzA );
			const sinViewZenith = sin( vzA );
			const cu = u.mul( u );
			const lightViewCos = cu.mul( 2 ).sub( 1 ).negate();
			const lightViewSin = sqrt( max( float( 1 ).sub( lightViewCos.mul( lightViewCos ) ), 0 ) );

			// local frame: up = +y, sun azimuth along +x
			const sunDirW = self.sunDir;
			const sunCosZ = sunDirW.y;
			const sunSinZ = sqrt( max( float( 1 ).sub( sunCosZ.mul( sunCosZ ) ), 0 ) );
			const sunDir = vec3( sunSinZ, sunCosZ, 0 );
			const rd = vec3( sinViewZenith.mul( lightViewCos ), cosViewZenith, sinViewZenith.mul( lightViewSin ) ).toVar();
			const ro = vec3( 0, viewH, 0 ).toVar();

			const tBottom = Atmosphere.raySphereNearest( ro, rd, RG ).toVar();
			const tTop = Atmosphere.raySphereNearest( ro, rd, RT ).toVar();
			const tMax = select( tBottom.greaterThan( 0 ), tBottom, tTop ).toVar();
			const steps = 32;
			const cosTheta = dot( rd, sunDir );
			const rayPhase = float( 3 / ( 16 * PI ) ).mul( cosTheta.mul( cosTheta ).add( 1 ) );
			const g = mieG;
			const g2 = g.mul( g );
			// Cornette-Shanks
			const miePhase = float( 3 / ( 8 * PI ) ).mul( float( 1 ).sub( g2 ).mul( cosTheta.mul( cosTheta ).add( 1 ) ) )
				.div( g2.add( 2 ).mul( pow( max( g2.add( 1 ).sub( g.mul( cosTheta ).mul( 2 ) ), 1e-4 ), 1.5 ) ) );

			const throughput = vec3( 1 ).toVar();
			const L = vec3( 0 ).toVar();
			const tPrev = float( 0 ).toVar();

			Loop( steps, ( { i } ) => {

				// quadratic step distribution
				const t0 = float( i ).div( steps );
				const t1 = float( i ).add( 1 ).div( steps );
				const ta = t0.mul( t0 ).mul( tMax );
				const tb = t1.mul( t1 ).mul( tMax );
				const t = mix( ta, tb, 0.3 );
				const dt = tb.sub( ta );
				const p = ro.add( rd.mul( t ) ).toVar();
				const pr = length( p );
				const m = self._medium( pr.sub( RG ) );
				const up = p.div( pr );
				const cosSunP = dot( up, sunDir );
				const Tsun = self.sampleTransmittance( pr, cosSunP );
				const shadowT = Atmosphere.raySphereNearest( p, sunDir, RG );
				const earthShadow = select( shadowT.greaterThan( 0 ), float( 0 ), float( 1 ) );
				const ms = self.sampleMultiScat( pr, cosSunP );
				const phaseScat = m.rayScat.mul( rayPhase ).add( vec3( m.mieScat.mul( miePhase ) ) );
				const S = Tsun.mul( earthShadow ).mul( phaseScat ).add( ms.mul( m.scattering ) );
				const Tstep = exp( m.extinction.mul( dt ).negate() );
				const ext = max( m.extinction, vec3( 1e-6 ) );
				L.addAssign( throughput.mul( S.sub( S.mul( Tstep ) ).div( ext ) ) );
				throughput.mulAssign( Tstep );
				tPrev.assign( tb );

			} );

			textureStore( this.skyViewLUT, uvec2( px ), vec4( L.mul( sunE ), 1 ) );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Atmosphere SkyView' );

		// ----- sky irradiance (cosine weighted hemisphere integral of sky view LUT)
		this.irradianceKernel = Fn( () => {

			const sum = vec3( 0 ).toVar();
			const N = 16;
			Loop( N * N, ( { i } ) => {

				const a = float( i.mod( N ) ).add( 0.5 ).div( N );
				const b = float( i.div( N ) ).add( 0.5 ).div( N );
				// cosine-weighted hemisphere
				const r = sqrt( b );
				const phi = a.mul( 2 * PI );
				const dir = vec3( r.mul( cos( phi ) ), sqrt( max( float( 1 ).sub( b ), 0 ) ), r.mul( sin( phi ) ) );
				sum.addAssign( self.skyLuminance( dir, true ) );

			} );

			// E = PI * mean(L) for cosine-weighted samples; store E/PI (radiance-equivalent irradiance)
			self.irrBuffer.element( 0 ).assign( vec4( sum.div( N * N ), 1 ) );
			// sun transmittance at sea level for the current sun direction
			const Ts = self.sampleTransmittance( float( RG + 0.001 ), self.sunDir.y );
			self.irrBuffer.element( 1 ).assign( vec4( Ts, 1 ) );
			// horizon color (average around the horizon)
			const hs = vec3( 0 ).toVar();
			Loop( 16, ( { i } ) => {

				const phi = float( i ).mul( 2 * PI / 16 );
				hs.addAssign( self.skyLuminance( normalize( vec3( cos( phi ), 0.03, sin( phi ) ) ), true ) );

			} );
			self.irrBuffer.element( 2 ).assign( vec4( hs.div( 16 ), 1 ) );

		} )().compute( 1 ).setName( 'Atmosphere Irradiance' );

	}

	// ---------------------------------------------------------------- sampling (TSL)

	// Sky luminance (no sun disk) for world direction `dir` (normalized).
	skyLuminance( dir, inCompute = false ) {

		const viewH = this.viewHeight;
		const vHorizon = sqrt( max( viewH.mul( viewH ).sub( RG * RG ), 0 ) );
		const beta = acos( vHorizon.div( viewH ) );
		const zenithHorizonAngle = float( PI ).sub( beta );
		const viewZenithAngle = acos( clamp( dir.y, - 1, 1 ) );

		const vCoordA = float( 1 ).sub( sqrt( max( float( 1 ).sub( viewZenithAngle.div( zenithHorizonAngle ) ), 0 ) ) ).mul( 0.5 );
		const vCoordB = sqrt( max( viewZenithAngle.sub( zenithHorizonAngle ).div( beta ), 0 ) ).mul( 0.5 ).add( 0.5 );
		const v = select( viewZenithAngle.lessThan( zenithHorizonAngle ), vCoordA, vCoordB );

		// azimuth relative to sun
		const sunH = normalize( vec2( this.sunDir.x, this.sunDir.z ).add( vec2( 1e-5, 0 ) ) );
		const dirH = normalize( vec2( dir.x, dir.z ).add( vec2( 1e-5, 0 ) ) );
		const lightViewCos = dot( sunH, dirH );
		const u = sqrt( clamp( lightViewCos.mul( - 0.5 ).add( 0.5 ), 0, 1 ) );

		const suv = vec2(
			u.mul( ( SV_W - 1 ) / SV_W ).add( 0.5 / SV_W ),
			v.mul( ( SV_H - 1 ) / SV_H ).add( 0.5 / SV_H )
		);

		const s = texture( this.skyViewLUT, suv );
		return ( inCompute ? s.level( 0 ) : s.level( 0 ) ).rgb;

	}

	// Transmittance from sea level toward direction dir (for sun disk / sun light color).
	transmittanceToSpace( dir ) {

		return this.sampleTransmittance( this.viewHeight, dir.y );

	}

	// ---------------------------------------------------------------- update

	update( dt, cameraY ) {

		const r = this.renderer;
		this.viewHeight.value = RG + Math.max( 0.001, cameraY / 1000 + 0.0005 );

		if ( this.needsStatic ) {

			this.needsStatic = false;
			r.compute( this.transmittanceKernel, [ T_W / 8, T_H / 8, 1 ] );
			r.compute( this.multiScatKernel, [ MS_RES / 8, MS_RES / 8, 1 ] );

		}

		r.compute( this.skyViewKernel, [ SV_W / 8, Math.ceil( SV_H / 8 ), 1 ] );

		// periodically integrate irradiance and read it back for CPU-side uniforms/lights
		this._irrTimer -= dt;
		if ( this._irrTimer <= 0 && ! this._irrPending ) {

			this._irrTimer = 0.25;
			r.compute( this.irradianceKernel );
			this._irrPending = true;
			r.getArrayBufferAsync( this.irrBuffer.value ).then( ( buf ) => {

				const f = new Float32Array( buf );
				this.skyIrradiance = [ f[ 0 ], f[ 1 ], f[ 2 ] ];
				this.sunTransmittance = [ f[ 4 ], f[ 5 ], f[ 6 ] ];
				this.horizon = [ f[ 8 ], f[ 9 ], f[ 10 ] ];
				this._irrPending = false;
				if ( this.onIrradiance ) this.onIrradiance( this );

			} ).catch( () => {

				this._irrPending = false;

			} );

		}

	}

	invalidate() {

		this.needsStatic = true;

	}

}
