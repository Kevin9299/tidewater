import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, vec2, vec3, vec4, normalize, dot, acos, clamp, smoothstep, sqrt, max, mix, exp,
	normalWorldGeometry, fract, floor, sin, pow, length, log2, select, saturate, If,
} from 'three/tsl';
import { G } from '../core/Globals.js';
import { SUN_ANGULAR_RADIUS } from './Atmosphere.js';

// hash without sine (D. Hoskins): keeps full precision for large integer coordinates
const hash13 = ( p ) => {

	const p3 = fract( p.mul( vec3( 0.1031, 0.1030, 0.0973 ) ) ).toVar();
	p3.addAssign( dot( p3, p3.yzx.add( 33.33 ) ) );
	return fract( p3.x.add( p3.y ).mul( p3.z ) );

};

const STAR_CELLS = 160; // cells per half cube face (one star candidate per cell, ~9 px at 25 px/deg)
const STAR_SIGMA = 0.1; // star PSF (cells, ~1 px)
const MW = new THREE.Vector3( 0.3, 0.2, 1 ).normalize();
const MILKY_WAY = vec3( MW.x, MW.y, MW.z ); // pole of the Milky Way band
// reflections spread point sources over rough water: only a trace of the stars survives
const STAR_REFLECTION = 0.08;

export class Sky {

	constructor( atmosphere ) {

		this.atmosphere = atmosphere;
		this.clouds = null; // set later: object with sample(dir) -> vec4(rgb inscatter, a transmittance)
		this.sunDiskIntensity = uniform( 1 ).setName( 'sunDiskI' );
		this.moonDir = uniform( new THREE.Vector3( - 0.3, 0.5, 0.8 ).normalize() ).setName( 'moonDir' );
		this.starIntensity = uniform( 0 ).setName( 'starI' );

	}

	// Sun disk radiance along dir (already includes atmospheric transmittance).
	sunDisk( dir ) {

		const atmo = this.atmosphere;
		const cosA = dot( dir, atmo.sunDir ); // the real sun (G.sunDir is the moon at night)
		const ang = acos( clamp( cosA, - 1, 1 ) );
		const r = ang.div( SUN_ANGULAR_RADIUS );
		const mask = smoothstep( 1.0, 0.9, r );
		const mu = sqrt( max( float( 1 ).sub( r.mul( r ) ), 0 ) );
		const limb = float( 1 ).sub( float( 0.6 ).mul( float( 1 ).sub( mu ) ) );
		const T = atmo.transmittanceToSpace( dir );
		// physically the disk radiance is E / solid angle (~1.6e5); clamp for fp16 targets
		return T.mul( mask ).mul( limb ).mul( 2500 ).mul( this.sunDiskIntensity ).mul( smoothstep( - 0.02, 0.0, dir.y ) );

	}

	// Stars: one candidate per cell of a cube-face grid, jittered inside it, with a power law
	// magnitude distribution (few bright stars, many faint ones), denser along the Milky Way. They
	// come out after civil twilight: the brightest first, the faintest once the sky is fully dark.
	stars( dir ) {

		// cube face coordinates
		const a = dir.abs();
		const onX = a.x.greaterThan( a.y ).and( a.x.greaterThan( a.z ) );
		const onY = a.y.greaterThan( a.z );
		const face = select( onX, dir.x.sign().add( 2 ), select( onY, dir.y.sign().add( 5 ), dir.z.sign().add( 8 ) ) );
		const uv = select( onX, dir.yz.div( a.x ), select( onY, dir.xz.div( a.y ), dir.xy.div( a.z ) ) ).mul( STAR_CELLS );
		const cell = vec3( floor( uv ), face );
		const h = hash13( cell );
		const bx = dot( dir, MILKY_WAY ).mul( 4 );
		const band = exp( bx.mul( bx ).negate() );
		// the cell holds a star with probability P (higher along the Milky Way); an independent
		// uniform u ranks its brightness: N( < m ) ~ 10^( m / 2 ), brightest about magnitude -1
		const has = h.lessThan( band.mul( 0.035 ).add( 0.025 ) );
		const uc = max( hash13( cell.add( 7.7 ) ), 2e-4 );
		const m = log2( uc ).mul( 0.602 ).add( 6.5 );
		// limiting magnitude: -1 when the sun is 6 deg below the horizon, 6.5 below 16 deg
		const dark = float( 1 ).sub( smoothstep( - 0.28, - 0.1, this.atmosphere.sunDir.y ) );
		const vis = smoothstep( m.sub( 0.6 ), m.add( 0.6 ), dark.mul( 7.5 ).sub( 1 ) ).mul( select( has, float( 1 ), float( 0 ) ) );
		// angular distance to the star (isotropic whatever the cube face distortion), in cells
		const sp = floor( uv ).add( vec2( hash13( cell.add( 3.1 ) ), hash13( cell.add( 5.7 ) ) ).mul( 0.4 ).add( 0.3 ) ).div( STAR_CELLS );
		const sdir = normalize( select( onX, vec3( dir.x.sign(), sp ), select( onY, vec3( sp.x, dir.y.sign(), sp.y ), vec3( sp, dir.z.sign() ) ) ) );
		const d = length( dir.sub( sdir ) ).mul( STAR_CELLS );
		// flux relative to a magnitude 6.5 star; bright stars look bigger (glare), not just clipped
		const flux = pow( uc, - 0.8 );
		const size = log2( flux ).mul( 0.08 ).add( 1 );
		const psf = exp( d.mul( d ).div( size.mul( size ) ).mul( - 0.5 / ( STAR_SIGMA * STAR_SIGMA ) ) ).div( size.mul( size ) );
		// subtle scintillation, stronger low in the sky
		const tw = sin( G.time.mul( hash13( cell.add( 13.3 ) ).mul( 9 ).add( 5 ) ).add( h.mul( 60 ) ) )
			.mul( mix( 0.18, 0.06, saturate( dir.y.mul( 2 ) ) ) ).add( 1 );
		const col = mix( vec3( 1.0, 0.8, 0.6 ), vec3( 0.75, 0.85, 1.0 ), hash13( cell.add( 17 ) ) ).mul( 0.5 ).add( 0.5 );
		const star = col.mul( psf.mul( flux ).mul( vis ).mul( tw ).mul( 0.0075 ) );
		// diffuse glow of the Milky Way
		const glow = vec3( 0.55, 0.6, 0.75 ).mul( band.mul( dark ).mul( 0.0035 ) );
		// atmospheric extinction toward the horizon
		return star.add( glow ).mul( this.starIntensity ).mul( smoothstep( 0.0, 0.2, dir.y ) );

	}

	moon( dir ) {

		const cosA = dot( dir, this.moonDir );
		const ang = acos( clamp( cosA, - 1, 1 ) );
		const r = ang.div( 0.0048 );
		const mask = smoothstep( 1.0, 0.92, r );
		return vec3( 0.9, 0.92, 1.0 ).mul( mask ).mul( 3.0 ).mul( this.starIntensity ).mul( smoothstep( - 0.02, 0.02, dir.y ) );

	}

	// Faint blue-grey moonlit sky (a little brighter toward the horizon) and the moon's aureole.
	// Physically dim: about the level of the app's night ambient light.
	moonSky( dir ) {

		const cosA = dot( dir, this.moonDir );
		const ang = acos( clamp( cosA, - 1, 1 ) );
		const aureole = exp( ang.mul( - 14 ) ).mul( 2.4 ).add( exp( ang.mul( - 2.5 ) ).mul( 0.9 ) );
		const grad = mix( float( 1.7 ), float( 1 ), saturate( dir.y.mul( 3 ) ) );
		const up = smoothstep( - 0.05, 0.15, this.moonDir.y );
		return vec3( 0.005, 0.0068, 0.0105 ).mul( grad.add( aureole ) ).mul( G.night ).mul( up );

	}

	// Everything behind the clouds except the sun and moon disks. The night terms are only
	// evaluated at night (uniform branch).
	_background( dir, starK = 1 ) {

		const L = this.atmosphere.skyLuminance( dir ).toVar();
		If( this.starIntensity.greaterThan( 0.001 ), () => {

			L.addAssign( this.moonSky( dir ).add( this.stars( dir ).mul( starK ) ) );

		} );
		return L;

	}

	// Full sky radiance for a direction (no clouds).
	radiance( dir, withSun = true ) {

		let L = this._background( dir ).add( this.moon( dir ) );
		if ( withSun ) L = L.add( this.sunDisk( dir ) );
		return L;

	}

	// Sky with clouds composited (low resolution cloud panorama). withSun = false (environment
	// map): no sun or moon disk, the key light is lit directly.
	radianceWithClouds( dir, withSun = true ) {

		const base = withSun ? this._background( dir ).add( this.moon( dir ) ).add( this.sunDisk( dir ) ) : this._background( dir, STAR_REFLECTION );
		if ( ! this.clouds ) return base;
		const c = this.clouds.sample( dir );
		return base.mul( c.a ).add( c.rgb );

	}

	// Water reflections: no moon disk (the water's specular lobe reflects the key light) and only a
	// trace of the stars, which rough water would spread into flickering sparkles.
	reflectionRadiance( dir ) {

		const base = this._background( dir, STAR_REFLECTION );
		if ( ! this.clouds ) return base;
		const c = this.clouds.sample( dir );
		return base.mul( c.a ).add( c.rgb );

	}

	// Main view background: full resolution clouds for the camera's view.
	backgroundNode() {

		return Fn( () => {

			const dir = normalize( normalWorldGeometry );
			const base = this._background( dir ).add( this.moon( dir ) ).add( this.sunDisk( dir ) );
			if ( ! this.clouds ) return vec4( base, 1 );
			const c = this.clouds.sampleView( dir );
			return vec4( base.mul( c.a ).add( c.rgb ), 1 );

		} )();

	}

}

// Simple solar position model. Returns direction toward the sun (world: +x east, -z north, +y up).
export function sunDirectionFromTime( hours, latitudeDeg = 24, declinationDeg = 6, out = new THREE.Vector3() ) {

	const phi = THREE.MathUtils.degToRad( latitudeDeg );
	const dec = THREE.MathUtils.degToRad( declinationDeg );
	const H = THREE.MathUtils.degToRad( ( hours - 12 ) * 15 );
	const east = - Math.cos( dec ) * Math.sin( H );
	const north = Math.cos( phi ) * Math.sin( dec ) - Math.sin( phi ) * Math.cos( dec ) * Math.cos( H );
	const up = Math.sin( phi ) * Math.sin( dec ) + Math.cos( phi ) * Math.cos( dec ) * Math.cos( H );
	return out.set( east, up, - north ).normalize();

}
