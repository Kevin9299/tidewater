import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, vec2, vec3, ivec2, uint, textureLoad, textureSize, instancedArray, length, atan, cos, abs,
	max, pow, exp, smoothstep, floor, mix, select,
} from 'three/tsl';
import { G } from '../core/Globals.js';

// Camera lens flare for the sun (and the moon, dimly), built from what a real multi-element lens
// does rather than sprites:
//  - ghosts: reflections between lens surfaces, images of the 7-blade aperture strung along the line
//    from the sun through the image centre; each with a bright rim, a soft body and dispersion
//    fringes (every colour images the aperture at a slightly different size)
//  - halo: a faint dispersive ring around the image centre, strongest when the sun nears the edge
//  - starburst: diffraction spikes from the aperture blades (14 for 7 blades) around the sun
//  - veiling glare: a soft wide glow lifting the blacks
// The sun's visibility (fraction of the disc not hidden by the scene, times the cloud
// transmittance) is measured on the GPU from the depth buffer every frame and eased, so leaves and
// masts crossing the sun make the flare flicker naturally without popping.

const BLADES = 7;
const ROT = 0.3; // aperture rotation (rad)
// a: position along the axis (1 = sun, 0 = image centre, < 0 past it), r: radius (fraction of the
// image height), tint, strength
const GHOSTS = [
	{ a: 0.72, r: 0.018, tint: [ 1.0, 0.85, 0.6 ], k: 0.55 },
	{ a: 0.44, r: 0.045, tint: [ 0.55, 0.9, 1.0 ], k: 0.35 },
	{ a: 0.16, r: 0.022, tint: [ 0.8, 1.0, 0.7 ], k: 0.45 },
	{ a: - 0.18, r: 0.07, tint: [ 0.6, 0.75, 1.0 ], k: 0.22 },
	{ a: - 0.42, r: 0.03, tint: [ 1.0, 0.7, 0.9 ], k: 0.4 },
	{ a: - 0.75, r: 0.11, tint: [ 0.7, 1.0, 0.85 ], k: 0.14 },
	{ a: - 1.15, r: 0.05, tint: [ 1.0, 0.8, 0.55 ], k: 0.28 },
];
const VIS_TAPS = 24;

export class LensFlare {

	constructor( { depthTexture, clouds = null, sunDir } ) {

		this.strength = uniform( 1 ).setName( 'flareStrength' );
		this.sunUV = uniform( new THREE.Vector2( 0.5, 0.5 ) ).setName( 'flareSunUV' );
		this.sunDir = sunDir; // true sun direction (atmosphere), a vec3 uniform
		this.inView = uniform( 0 ).setName( 'flareInView' ); // 0..1, fades as the sun leaves the frame
		this.aboveWater = uniform( 1 ).setName( 'flareAboveWater' );
		this.aspect = uniform( 16 / 9 ).setName( 'flareAspect' );
		this.discPx = uniform( 6 ).setName( 'flareDiscPx' ); // sun disc radius in depth-buffer pixels
		this.dt = uniform( 1 / 60 ).setName( 'flareDt' );
		this.visibility = instancedArray( new Float32Array( [ 0 ] ), 'float' ).setName( 'flareVisibility' );

		const depthNode = depthTexture;
		this.kernel = Fn( () => {

			const size = textureSize( textureLoad( depthNode ), 0 );
			const c = this.sunUV.mul( vec2( size ) );
			const sky = float( 0 ).toVar();
			for ( let i = 0; i < VIS_TAPS; i ++ ) {

				// golden-angle spiral over the sun's disc
				const r = Math.sqrt( ( i + 0.5 ) / VIS_TAPS );
				const t = i * 2.39996323;
				const p = ivec2( c.add( vec2( Math.cos( t ) * r, Math.sin( t ) * r ).mul( this.discPx ) ) ).clamp( ivec2( 0 ), ivec2( size ).sub( 1 ) );
				// reversed depth: the sky is at 0
				sky.addAssign( select( textureLoad( depthNode, p ).x.lessThan( 1e-7 ), float( 1 ), float( 0 ) ) );

			}

			const cloudT = clouds ? clouds.sampleView( this.sunDir ).a : float( 1 );
			const up = smoothstep( - 0.02, 0.04, this.sunDir.y );
			const target = sky.div( VIS_TAPS ).mul( cloudT ).mul( up ).mul( this.inView ).mul( this.aboveWater );
			const v = this.visibility.element( uint( 0 ) );
			v.assign( mix( v, target, float( 1 ).sub( exp( this.dt.mul( - 14 ) ) ) ) );

		} )().compute( 1 ).setName( 'Lens Flare Visibility' );

	}

	// per frame, before the post chain: sun position on screen and whether it can flare
	update( camera, dt, { aboveWater = true } = {} ) {

		const d = this.sunDir.value;
		const v = _v.copy( camera.position ).addScaledVector( d, 1000 ).project( camera );
		const ahead = _f.set( 0, 0, - 1 ).applyQuaternion( camera.quaternion ).dot( d ) > 0.05;
		this.sunUV.value.set( v.x * 0.5 + 0.5, 0.5 - v.y * 0.5 );
		// fade out over the last 5% before the frame edge (no pop when the sun leaves the view)
		const edge = Math.max( Math.abs( v.x ), Math.abs( v.y ) );
		this.inView.value = ahead ? THREE.MathUtils.clamp( ( 1.0 - edge ) / 0.1, 0, 1 ) : 0;
		this.aboveWater.value = aboveWater ? 1 : 0;
		this.aspect.value = camera.aspect;
		this.dt.value = Math.min( dt, 0.1 );
		// sun disc radius (0.265°) in pixels of the depth buffer
		const h = this._depthH || 1080;
		this.discPx.value = Math.max( 1.5, 0.004625 / Math.tan( THREE.MathUtils.degToRad( camera.fov ) * 0.5 ) * h * 0.5 );

	}

	setDepthHeight( h ) {

		this._depthH = h;

	}

	// HDR flare light for this screen position (added before exposure / tone mapping)
	node( uv ) {

		const asp = vec2( this.aspect, 1 );
		const p = uv.sub( 0.5 ).mul( asp ).mul( vec2( 1, - 1 ) );
		const s = this.sunUV.sub( 0.5 ).mul( asp ).mul( vec2( 1, - 1 ) );
		const vis = this.visibility.element( uint( 0 ) );
		// light arriving from the sun disc (the key light colour follows the atmosphere's
		// transmittance; at night the moon is the key light and flares only faintly)
		const light = G.sunColor.mul( vis ).mul( this.strength ).mul( 0.02 );
		const seg = ( 2 * Math.PI ) / BLADES;

		// aperture polygon: distance scaled so the polygon edge sits at the given radius
		const polyDist = ( d ) => {

			const a = atan( d.y, d.x ).add( ROT );
			return length( d ).mul( cos( floor( a.div( seg ).add( 0.5 ) ).mul( seg ).sub( a ) ) );

		};

		const ghosts = vec3( 0 ).toVar();
		for ( const g of GHOSTS ) {

			const pd = polyDist( p.sub( s.mul( g.a ) ) );
			const edge = ( rr ) => smoothstep( rr, rr * 0.9, pd );
			const body = vec3( edge( g.r * 0.975 ), edge( g.r ), edge( g.r * 1.025 ) );
			const rim = smoothstep( g.r * 0.55, g.r, pd ).mul( 0.7 ).add( 0.3 );
			// smaller ghosts concentrate the same reflected energy: brighter
			ghosts.addAssign( body.mul( rim ).mul( vec3( ...g.tint ) ).mul( g.k * 0.0028 / ( g.r * g.r ) ) );

		}

		// fade the ghosts as the sun nears the centre (they collapse onto it and vanish)
		const offAxis = smoothstep( 0.02, 0.2, length( s ) );

		// halo: dispersive ring about the image centre
		const rc = length( p );
		const ring = ( R ) => smoothstep( 0.03, 0, abs( rc.sub( R ) ) );
		const halo = vec3( ring( 0.43 ), ring( 0.445 ), ring( 0.46 ) ).mul( smoothstep( 0.25, 0.8, length( s ) ) ).mul( 0.12 );

		// starburst and veiling glare around the sun
		const q = p.sub( s );
		const rq = max( length( q ), 1e-4 );
		const aq = atan( q.y, q.x ).add( ROT );
		const spikes = pow( abs( cos( aq.mul( BLADES ) ) ), 600 ).mul( exp( rq.mul( - 9 ) ) ).mul( 1.2 );
		const fine = pow( abs( cos( aq.mul( 53 ).add( cos( aq.mul( 11 ) ).mul( 2 ) ) ) ), 80 ).mul( exp( rq.mul( - 22 ) ) ).mul( 0.35 );
		const glow = exp( rq.mul( - 5 ) ).mul( 0.05 ).add( exp( rq.mul( - 40 ) ).mul( 0.4 ) );
		const burst = spikes.add( fine ).add( glow );

		return light.mul( ghosts.mul( offAxis ).add( halo ).add( vec3( burst ) ) );

	}

}

const _v = new THREE.Vector3();
const _f = new THREE.Vector3();

