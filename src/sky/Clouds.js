import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, int, uint, vec2, vec3, vec4, ivec2, ivec3, uvec2, uvec3, globalId, textureStore, texture, texture3D, textureLoad, texture3DLoad,
	If, Loop, Break, Return, select, normalize, length, dot, mix, clamp, saturate, smoothstep, exp, pow,
	sqrt, max, min, abs, floor, ceil, fract, sin, cos, acos, atan, mod, log2, exp2,
} from 'three/tsl';
import { G } from '../core/Globals.js';

// Volumetric trade-wind cumulus (Schneider/Nubis density model, Hillaire multiple-scattering
// approximation) under a thin cirrus veil.
//   - view: ray marched in screen space, one pixel of every 4x4 block per frame (two while the camera
//     turns), accumulated at 0.6x the render resolution with reprojection (camera motion and wind drift)
//     and upsampled bicubically, like sky-pro-webgpu's High quality (about 1/44 of the display pixels
//     marched per frame). Empty space is skipped with a coverage dependent 3D distance field, so rays
//     only take small (24 m) steps near clouds.
//   - panorama: a cheaper low resolution version of the whole sky dome for water reflections,
//     the environment cube and Snell's window, refreshed progressively (1/32 per frame).
//   - shadow: transmittance of the cloud layer along the sun around the camera (1/4 per frame).
// Detail (after the user's sky-pro-webgpu renderer): a 3 octave worley erosion volume filtered by the pixel
// footprint (plus a finer fetch at close range), erosion growing with height in the cloud (flat bases,
// cauliflower tops, wispy undersides) and a short detailed light tap that shades the billows. Lighting also
// follows sky-pro: three scattering orders from one exponential with a droplet phase, skylight occlusion
// from two upward probes, darker bases.
// Everything outputs vec4( in-scattered radiance, transmittance ): sky * a + rgb composites it.

const PI = Math.PI;
const EARTH_R = 6360000;
const TILE = 32768; // m, period of the whole cloud field (every noise tiles within it)
const WEATHER_RES = 512;
const SDF_RES = 256, SDF_H = 16; // distance field cells (horizontal over the tile, vertical over the layer)
const SDF_R = 24; // cells searched horizontally
const SHAPE_RES = 128, SHAPE_SIZE = 2048;
const SHEAR = 0.12; // horizontal lean of the clouds per metre of height (downwind)
const EDGE = 12; // density ramp of the raw shape value (softness of the cloud surface: crisp, the detail erosion shapes it)
const ISLAND_NEAR = 2500, ISLAND_FAR = 9000; // m: big clusters only a little away from the island
const DETAIL_RES = 64, DETAIL_SIZE = 480;
// detail erosion (after sky-pro-webgpu / Nubis): three worley fbm octaves per fetch (r: 4 - 16, g: 8 - 32,
// b: 16 - 64 cells per period). An octave fades to the mean once its features shrink under ~2 px (a mip
// filter without mips); `crease` is 0 on a lump, 1 between lumps
const D_MEAN = 0.48;
const D_S1 = DETAIL_SIZE / 4, D_NEAR = 3.7, D_S2 = D_S1 / D_NEAR; // m: coarsest features of the two fetches
const creaseOf = ( f ) => smoothstep( 0.4, 0.72, f );
// erosion grows with the height in the cloud: flat, dense bases, billowy tops; the undersides use the
// inverted field (wisps instead of lumps)
const erosionAmount = ( b ) => mix( 0.3, 1.0, smoothstep( 0.0, 0.5, b.y ) );
const erosionField = ( F, b ) => mix( float( 1 ).sub( F ), F, smoothstep( 0.02, 0.2, b.y ) );
// density with every detail octave at its mean (reflections, shadows, deep light samples): the same
// cloud as the detailed one, seen through a coarse filter
const meanCrease = ( b ) => creaseOf( erosionField( float( D_MEAN ), b ) );
const meanDensity = ( b ) => saturate( b.x.sub( meanCrease( b ).mul( erosionAmount( b ) ) ).sub( 0.012 ).mul( EDGE ) );
const PANO_W = 1024, PANO_H = 320;
// cirrus veil: coverage varying over hundreds of km, fibres (flow line streaks, mipmapped)
const SYN_RES = 256, SYN_SIZE = 409600; // m
const FIB_RES = 1024, FIB_TILE = 40000; // m
const SHADOW_RES = 256;
const AP_DIST = 30000; // m, aerial perspective scale toward the horizon

// 4x4 ordered-dither sequence: one pixel of every block per frame
const ORDER = [ 0, 10, 2, 8, 5, 15, 7, 13, 1, 11, 3, 9, 4, 14, 6, 12 ];
const REBUILD_SLOTS = 4; // slots traced per frame right after a camera cut (sharp again after 16 / 4 frames)
// history resolution relative to the (dynamic) render resolution: like sky-pro-webgpu's High quality (0.5),
// the clouds are reconstructed at reduced width and height and one pixel of every 4x4 block is marched per
// frame; the saved rays buy finer steps and fuller lighting
const HISTORY_SCALE = 0.6;

const hash3 = ( p ) => fract( sin( dot( p, vec3( 127.1, 311.7, 74.7 ) ) ).mul( 43758.5453 ) );
const hash2 = ( p ) => fract( sin( dot( p, vec2( 127.1, 311.7 ) ) ).mul( 43758.5453 ) );

// top of a cloud column (fraction of the layer) from the cell profile (0..1), the cell's top and
// the turret noise: broad rounded domes, uneven
const smallTop = ( cs, top, lump ) => top.mul( pow( cs, 0.6 ) ).mul( lump.mul( 0.8 ).add( 0.65 ) );
const bigTop = ( cb, lump ) => pow( cb, 0.6 ).mul( lump.mul( 0.35 ).add( 0.7 ) );

// cubic B-spline filtered texture lookup in 4 bilinear taps (GPU Gems 2, ch. 20): magnified cirrus fibres
// stay smooth instead of showing the bilinear texel grid (res: texels at mip 0)
const bspline = ( tex, uv, lod, res ) => {

	const size = float( res ).div( exp2( lod ) );
	const st = uv.mul( size ).sub( 0.5 );
	const i = floor( st );
	const f = st.sub( i );
	const f2 = f.mul( f ), f3 = f2.mul( f );
	const w0 = f3.negate().add( f2.mul( 3 ) ).sub( f.mul( 3 ) ).add( 1 ).div( 6 );
	const w1 = f3.mul( 3 ).sub( f2.mul( 6 ) ).add( 4 ).div( 6 );
	const w3 = f3.div( 6 );
	const w2 = float( 1 ).sub( w0 ).sub( w1 ).sub( w3 );
	const g0 = w0.add( w1 ), g1 = w2.add( w3 );
	const p0 = i.sub( 0.5 ).add( w1.div( g0 ) ).div( size ), p1 = i.add( 1.5 ).add( w3.div( g1 ) ).div( size );
	const tap = ( x, y ) => texture( tex, vec2( x, y ) ).level( lod );
	return tap( p0.x, p0.y ).mul( g0.x.mul( g0.y ) ).add( tap( p1.x, p0.y ).mul( g1.x.mul( g0.y ) ) )
		.add( tap( p0.x, p1.y ).mul( g0.x.mul( g1.y ) ) ).add( tap( p1.x, p1.y ).mul( g1.x.mul( g1.y ) ) );

};

// layer a (in-scattered radiance, transmittance) in front of layer b
const over = ( a, b ) => vec4( a.rgb.add( b.rgb.mul( a.a ) ), a.a.mul( b.a ) );

// Henyey-Greenstein phase
const phaseHG = ( c, g ) => float( ( 1 - g * g ) / ( 4 * PI ) ).div( pow( max( float( 1 + g * g ).sub( c.mul( 2 * g ) ), 1e-4 ), 1.5 ) );

// Key light of the clouds: the sun while it is the app's key light (seen from cloud altitude,
// so it keeps lighting the clouds a little after it has set at sea level), else the moon.
// Returns { dir, E, isMoon } (E: illuminance before the earth shadow).
const keyLight = ( atmo, altKm ) => {

	const isMoon = dot( G.sunDir, atmo.sunDir ).lessThan( 0.9999 );
	const sunE = atmo.sampleTransmittance( float( 6360 ).add( altKm ), G.sunDir.y ).mul( atmo.sunIlluminance );
	return { dir: G.sunDir, E: select( isMoon, G.sunColor, sunE ), isMoon };

};

// earth shadow on a point at altitude alt (m) and horizontal offset pxz (m) from the camera: the
// light is below its horizon once mu < -sqrt( 2 alt / R ) (the local vertical tilts with distance)
const earthShadow = ( light, alt, pxz ) => {

	const mu = light.dir.y.add( dot( light.dir.xz, pxz ).div( EARTH_R ) );
	const lit = smoothstep( - 0.006, 0.006, mu.add( sqrt( max( alt, 0 ).mul( 2 / EARTH_R ) ) ) );
	return select( light.isMoon, float( 1 ), lit );

};

// distance along rd from a point at height camY (on the planet axis) to the sphere at altitude H
// (camera below it). Stable form: |oc|^2 - R^2 = (camY - H)(2Re + camY + H)
const shell = ( camY, rd, H ) => {

	const b = rd.y.mul( camY.add( EARTH_R ) );
	const cc = camY.sub( H ).mul( camY.add( H ).add( 2 * EARTH_R ) );
	const disc = b.mul( b ).sub( cc );
	return b.negate().add( sqrt( max( disc, 0 ) ) );

};

export class Clouds {

	constructor( renderer, atmosphere ) {

		this.renderer = renderer;
		this.atmosphere = atmosphere;

		this.coverage = uniform( 0.45 ).setName( 'clCoverage' );
		this.densityScale = uniform( 0.07 ).setName( 'clDensity' ); // extinction (1/m) of the densest cloud
		this.bottom = uniform( 800 ).setName( 'clBottom' );
		this.top = uniform( 2000 ).setName( 'clTop' );
		// m/s; clouds drift along it (by default with the surface wind, as trade winds do)
		this.wind = uniform( new THREE.Vector2().copy( G.windDir.value ).normalize().multiplyScalar( 12 ) ).setName( 'clWind' );
		this.offset = uniform( new THREE.Vector2() ).setName( 'clOffset' ); // accumulated wind offset (m)
		this.shadowCenter = uniform( new THREE.Vector2() ).setName( 'clShadowC' );
		this.shadowSize = uniform( 8000 ).setName( 'clShadowSize' );
		this.shadowStrength = uniform( 0.85 ).setName( 'clShadowK' );
		// cirrus veil: amount 0..1 (0.5: faint, opacity mostly 0.05 - 0.2) and altitude (m)
		this.cirrus = uniform( 0.5 ).setName( 'clCirrus' );
		this.cirrusAlt = uniform( 9000 ).setName( 'clCirrusAlt' );

		// weather map lookups are rotated so cloud streets line up with the (initial) wind
		const w = this.wind.value;
		this._rot = Math.atan2( w.y, w.x );
		this._offsetW = new THREE.Vector2();

		// per frame state (camera relative: the camera sits on the planet axis)
		this.camY = uniform( 1 ).setName( 'clCamY' );
		this.nOrigin = uniform( new THREE.Vector2() ).setName( 'clNOrigin' ); // noise space origin
		this.wOrigin = uniform( new THREE.Vector2() ).setName( 'clWOrigin' ); // weather space origin
		this.camXZ = uniform( new THREE.Vector2() ).setName( 'clCamXZ' );
		this.hOffset = uniform( new THREE.Vector2() ).setName( 'clHOffset' ); // cirrus drift (faster upper wind)
		this._offsetH = new THREE.Vector2();
		this.horizonY = uniform( - 0.001 ).setName( 'clHorizonY' ); // rays below hit the planet
		this.windN = uniform( new THREE.Vector2( 1, 0 ) ).setName( 'clWindN' ); // wind direction (shear lean)

		this._makeNoise();
		this._makeTargets();
		this._buildFunctions();
		this._buildKernels();

		this.frame = 0;
		this.panoWarm = 1;
		// resolution of the view clouds relative to the drawing buffer (e.g. follow a dynamic
		// resolution scale); the sky is reconstructed per pixel from direction, so any size works
		this.resolutionScale = 1;
		this.historyValid = uniform( 0 ).setName( 'clHistValid' );
		this._prevCam = new THREE.Vector3();
		this._prevSun = new THREE.Vector3();
		this._hasPrev = false;
		this._rebuild = 0; // slots traced since the last camera cut (16: done)
		this._since = 0; // frames since the rebuild completed
		this._traces = 0;
		this._pp = 0; // history ping-pong

	}

	// ------------------------------------------------------------ noise volumes

	_makeNoise() {

		const r = this.renderer;
		const make3D = ( size, name ) => {

			const t = new THREE.Storage3DTexture( size, size, size );
			t.format = THREE.RGBAFormat;
			t.type = THREE.UnsignedByteType;
			t.wrapS = t.wrapT = t.wrapR = THREE.RepeatWrapping;
			t.magFilter = t.minFilter = THREE.LinearFilter;
			t.generateMipmaps = false;
			t.name = name;
			return t;

		};

		this.shapeTex = make3D( SHAPE_RES, 'cloudShape' );
		this.detailTex = make3D( DETAIL_RES, 'cloudDetail' );

		const w = new THREE.StorageTexture( WEATHER_RES, WEATHER_RES );
		w.format = THREE.RGBAFormat;
		w.type = THREE.UnsignedByteType;
		w.wrapS = w.wrapT = THREE.RepeatWrapping;
		w.magFilter = w.minFilter = THREE.LinearFilter;
		w.generateMipmaps = false;
		w.name = 'cloudWeather';
		this.weatherTex = w;

		const make2D = ( res, name, mips, type = THREE.UnsignedByteType ) => {

			const t = new THREE.StorageTexture( res, res );
			t.format = THREE.RGBAFormat;
			t.type = type;
			t.wrapS = t.wrapT = THREE.RepeatWrapping;
			t.magFilter = THREE.LinearFilter;
			t.minFilter = mips ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter;
			t.generateMipmaps = mips;
			if ( mips ) t.anisotropy = 8; // sheets are seen at grazing angles
			t.name = name;
			return t;

		};

		// r = cirrus coverage (hundreds of km), g = broad streets along the upper wind, b = patches (5 - 50 km)
		this.synTex = make2D( SYN_RES, 'cloudSynoptic', false );
		// cirrus: r = fibres, g = veil noise
		this.fibTex = make2D( FIB_RES, 'cloudFibres', true );
		// fibre seeds and flow direction (r = seeds, g = flow angle)
		this.auxTex = make2D( FIB_RES, 'cloudFibreFlow', false, THREE.HalfFloatType );

		// tileable 3D worley (F1) with `cells` cells per unit
		const worley3 = ( p, cells ) => {

			const q = p.mul( cells );
			const ip = floor( q );
			const fp = fract( q );
			const d = float( 1e3 ).toVar();
			for ( let z = - 1; z <= 1; z ++ ) for ( let y = - 1; y <= 1; y ++ ) for ( let x = - 1; x <= 1; x ++ ) {

				const o = vec3( x, y, z );
				const cell = mod( ip.add( o ), cells );
				const h = vec3( hash3( cell ), hash3( cell.add( 19.7 ) ), hash3( cell.add( 41.3 ) ) );
				d.assign( min( d, length( o.add( h ).sub( fp ) ) ) );

			}

			return d;

		};

		// tileable 3D gradient (perlin) noise, about -1..1
		const gnoise = ( p, cells ) => {

			const q = p.mul( cells );
			const i = floor( q );
			const f = fract( q );
			const u = f.mul( f ).mul( f.mul( f.mul( 6 ).sub( 15 ) ).add( 10 ) );
			const g = ( o ) => {

				const c = mod( i.add( o ), cells );
				const gv = vec3( hash3( c ), hash3( c.add( 13.1 ) ), hash3( c.add( 27.7 ) ) ).mul( 2 ).sub( 1 );
				return dot( gv, f.sub( o ) );

			};

			const x00 = mix( g( vec3( 0, 0, 0 ) ), g( vec3( 1, 0, 0 ) ), u.x );
			const x10 = mix( g( vec3( 0, 1, 0 ) ), g( vec3( 1, 1, 0 ) ), u.x );
			const x01 = mix( g( vec3( 0, 0, 1 ) ), g( vec3( 1, 0, 1 ) ), u.x );
			const x11 = mix( g( vec3( 0, 1, 1 ) ), g( vec3( 1, 1, 1 ) ), u.x );
			return mix( mix( x00, x10, u.y ), mix( x01, x11, u.y ), u.z );

		};

		// billows: inverted worley fbm (1 at the feature points)
		const billows = ( p, c ) => float( 1 ).sub( worley3( p, c ).mul( 0.625 ).add( worley3( p, c * 2 ).mul( 0.25 ) ).add( worley3( p, c * 4 ).mul( 0.125 ) ) );
		const perlinFbm = ( p, c ) => gnoise( p, c ).mul( 0.5 ).add( gnoise( p, c * 2 ).mul( 0.25 ) ).add( gnoise( p, c * 4 ).mul( 0.125 ) );

		// shape: r = billowy base shape (worley fbm dilated by perlin), gb = low frequency swirl for
		// the detail lookups
		const shapeKernel = Fn( () => {

			const p = vec3( globalId ).add( 0.5 ).div( SHAPE_RES );
			const b = billows( p, 4 );
			const pn = perlinFbm( p, 4 ).mul( 0.9 ).add( 0.5 );
			// perlin-worley: remap( perlin, 0, 1, worley, 1 ) keeps the billows, breaks their regularity
			const pw = b.add( saturate( pn ).mul( float( 1 ).sub( b ) ).mul( 0.5 ) );
			const base = saturate( pw.sub( 0.42 ).div( 0.5 ) );
			const cx = gnoise( p.add( 0.31 ), 4 ).mul( 0.7 ).add( 0.5 );
			const cz = gnoise( p.add( 0.67 ), 4 ).mul( 0.7 ).add( 0.5 );
			textureStore( this.shapeTex, uvec3( globalId ), vec4( base, saturate( cx ), saturate( cz ), 1 ) );

		} )().computeKernel( [ 4, 4, 4 ] ).setName( 'Cloud Shape Noise' );

		// detail: worley fbm distance (0 at the centre of a lump, high in the creases between lumps) at
		// three frequencies (sky-pro-webgpu's base noise profile: cells 4 / 8 / 16 per period, octaves x2, x4)
		const detailKernel = Fn( () => {

			const p = vec3( globalId ).add( 0.5 ).div( DETAIL_RES );
			const fbm = ( c ) => worley3( p, c ).mul( 0.625 ).add( worley3( p, c * 2 ).mul( 0.25 ) ).add( worley3( p, c * 4 ).mul( 0.125 ) );
			textureStore( this.detailTex, uvec3( globalId ), vec4( saturate( fbm( 4 ) ), saturate( fbm( 8 ) ), saturate( fbm( 16 ) ), 1 ) );

		} )().computeKernel( [ 4, 4, 4 ] ).setName( 'Cloud Detail Noise' );

		// ---- weather: r = small cell profile, g = its top (fraction of the layer), b = turrets,
		// a = big cell profile
		const vnoise2 = ( p, cells ) => {

			const q = p.mul( cells );
			const i = floor( q );
			const f = fract( q );
			const u = f.mul( f ).mul( f.mul( f.mul( 6 ).sub( 15 ) ).add( 10 ) );
			const h = ( o ) => hash2( mod( i.add( o ), cells ) );
			return mix( mix( h( vec2( 0, 0 ) ), h( vec2( 1, 0 ) ), u.x ), mix( h( vec2( 0, 1 ) ), h( vec2( 1, 1 ) ), u.x ), u.y );

		};

		// cells of varying size: max over cells of a blob that fades out at its random radius,
		// cells switch on where the mesoscale field allows. returns vec2( blob, its size )
		const blobs = ( p, cells, meso, seed ) => {

			const q = p.mul( cells );
			const ip = floor( q );
			const fp = fract( q );
			const b = vec2( 0 ).toVar();
			for ( let y = - 1; y <= 1; y ++ ) for ( let x = - 1; x <= 1; x ++ ) {

				const o = vec2( x, y );
				const cell = mod( ip.add( o ), cells ).add( seed );
				const hx = hash2( cell ), hy = hash2( cell.add( 19.7 ) ), hr = hash2( cell.add( 41.3 ) ), hp = hash2( cell.add( 7.1 ) );
				const d = length( o.add( vec2( hx, hy ).mul( 0.6 ).add( 0.2 ) ).sub( fp ) );
				const rad = hr.mul( hr ).mul( 0.5 ).add( 0.4 );
				const on = smoothstep( hp.sub( 0.15 ), hp.add( 0.15 ), meso.add( 0.12 ) );
				const v = saturate( float( 1 ).sub( d.div( rad ) ) ).mul( on );
				If( v.greaterThan( b.x ), () => {

					b.assign( vec2( v, hr ) );

				} );

			}

			return b;

		};

		const weatherKernel = Fn( () => {

			const p = vec2( globalId.xy ).add( 0.5 ).div( WEATHER_RES );
			const meso = vnoise2( p, vec2( 4 ) ).mul( 0.6 ).add( vnoise2( p, vec2( 8 ) ).mul( 0.3 ) ).add( vnoise2( p, vec2( 16 ) ).mul( 0.1 ) );
			// low frequency warp: irregular footprints, wavy streets
			const pw = p.add( vec2( vnoise2( p, vec2( 24 ) ), vnoise2( p.add( 0.37 ), vec2( 24 ) ) ).sub( 0.5 ).mul( 0.03 ) );
			// cloud streets along the wind (x), about 3 km apart
			const street = sin( pw.y.mul( 2 * PI * 11 ).add( vnoise2( p, vec2( 3 ) ).mul( 6 ) ) ).mul( 0.5 ).add( 0.5 );
			// fair weather cumulus: cells of 0.5 - 2 km, mostly along the streets
			const small = blobs( pw, vec2( 15, 19 ), meso.mul( 0.7 ).add( street.mul( 0.4 ) ).sub( 0.25 ), 0 );
			// big cells and towers (used far from the island only)
			const big = blobs( pw, vec2( 6, 8 ), meso.sub( 0.1 ), 3.7 );
			// turrets: several bumps per cell
			const lump = vnoise2( pw, vec2( 80 ) ).mul( 0.65 ).add( vnoise2( pw.add( 0.5 ), vec2( 160 ) ).mul( 0.35 ) );
			// top of the small cells (fraction of the layer): bigger cells grow taller
			const top = small.y.mul( 0.35 ).add( 0.35 ).mul( vnoise2( p, vec2( 12 ) ).mul( 0.5 ).add( 0.75 ) );
			textureStore( this.weatherTex, uvec2( globalId.xy ), vec4( small.x, saturate( top ), lump, big.x ) );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Cloud Weather' );

		// ---- 2D noise helpers
		const gnoise2 = ( p, cells ) => {

			const q = p.mul( cells );
			const i = floor( q );
			const f = fract( q );
			const u = f.mul( f ).mul( f.mul( f.mul( 6 ).sub( 15 ) ).add( 10 ) );
			const g = ( o ) => {

				const c = mod( i.add( o ), cells );
				const a = hash2( c ).mul( 2 * PI );
				return dot( vec2( cos( a ), sin( a ) ), f.sub( o ) );

			};

			return mix( mix( g( vec2( 0, 0 ) ), g( vec2( 1, 0 ) ), u.x ), mix( g( vec2( 0, 1 ) ), g( vec2( 1, 1 ) ), u.x ), u.y );

		};

		const worley2 = ( p, cells ) => {

			const q = p.mul( cells );
			const ip = floor( q );
			const fp = fract( q );
			const d = float( 1e3 ).toVar();
			for ( let y = - 1; y <= 1; y ++ ) for ( let x = - 1; x <= 1; x ++ ) {

				const o = vec2( x, y );
				const cell = mod( ip.add( o ), cells );
				d.assign( min( d, length( o.add( vec2( hash2( cell ), hash2( cell.add( 19.7 ) ) ).mul( 0.8 ).add( 0.1 ) ).sub( fp ) ) ) );

			}

			return d;

		};

		const fbm2 = ( p, cells, seed ) => vnoise2( p.add( seed ), vec2( cells ) ).mul( 0.5 )
			.add( vnoise2( p.add( seed * 1.7 ), vec2( cells * 2 ) ).mul( 0.3 ) ).add( vnoise2( p.add( seed * 2.3 ), vec2( cells * 4 ) ).mul( 0.2 ) );
		// gradient noise fbm remapped to about 0..1 (smoother than value noise, no grid artefacts)
		const gfbm = ( p, cells, seed ) => gnoise2( p.add( seed ), cells ).mul( 0.55 )
			.add( gnoise2( p.add( seed * 1.7 ), cells.mul( 2 ) ).mul( 0.3 ) ).add( gnoise2( p.add( seed * 2.3 ), cells.mul( 4 ) ).mul( 0.15 ) ).mul( 1.6 ).add( 0.5 );

		// ---- cirrus coverage: smooth fields of hundreds of km (u = x along the upper wind)
		const synKernel = Fn( () => {

			const p = vec2( globalId.xy ).add( 0.5 ).div( SYN_RES );
			const cov = gfbm( p, vec2( 2, 3 ), 0.31 );
			const streets = gfbm( p, vec2( 4, 14 ), 0.59 );
			const patches = gfbm( p, vec2( 10, 14 ), 0.83 ).mul( 0.7 ).add( gfbm( p, vec2( 28, 40 ), 0.21 ).mul( 0.3 ) );
			textureStore( this.synTex, uvec2( globalId.xy ), saturate( vec4( cov, streets, patches, 1 ) ) );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Cloud Synoptic Fields' );

		// ---- cirrus fibres: sparse seeds integrated along a smooth, gently meandering flow (line
		// integral convolution): long filaments of varied width, brightness and length
		const auxKernel = Fn( () => {

			const p = vec2( globalId.xy ).add( 0.5 ).div( FIB_RES );
			const psi = ( q ) => gnoise2( q, vec2( 3 ) ).mul( 0.5 ).add( gnoise2( q.add( 0.37 ), vec2( 7 ) ).mul( 0.25 ) ).add( gnoise2( q.add( 0.71 ), vec2( 15 ) ).mul( 0.1 ) );
			const e = 1 / FIB_RES;
			const curl = vec2( psi( p.add( vec2( 0, e ) ) ).sub( psi( p.sub( vec2( 0, e ) ) ) ), psi( p.sub( vec2( e, 0 ) ) ).sub( psi( p.add( vec2( e, 0 ) ) ) ) ).div( 2 * e );
			const dir = normalize( vec2( 1, 0 ).add( vec2( 0, gnoise2( p.add( 0.13 ), vec2( 3 ) ).mul( 0.5 ) ) ).add( curl.mul( 0.05 ) ) );
			const cluster = saturate( gfbm( p, vec2( 5 ), 0.9 ).mul( 1.6 ).sub( 0.3 ) );
			// sparse jittered points: n cells per texture, probability, radius (cells), seed
			const points = ( n, prob, sigma, seed ) => {

				const q = p.mul( n );
				const ip = floor( q );
				const fp = fract( q );
				const v = float( 0 ).toVar();
				for ( let y = - 1; y <= 1; y ++ ) for ( let x = - 1; x <= 1; x ++ ) {

					const o = vec2( x, y );
					const c = mod( ip.add( o ), vec2( n ) ).add( seed );
					const pos = o.add( vec2( hash2( c ), hash2( c.add( 19.7 ) ) ).mul( 0.8 ).add( 0.1 ) );
					const on = select( hash2( c.add( 41.3 ) ).lessThan( cluster.mul( prob ) ), float( 1 ), float( 0 ) );
					const d = pos.sub( fp );
					v.assign( max( v, exp( dot( d, d ).mul( - 0.5 / ( sigma * sigma ) ) ).mul( on ).mul( hash2( c.add( 7.1 ) ).mul( 0.7 ).add( 0.3 ) ) ) );

				}

				return v;

			};

			const seeds = max( points( 110, 0.3, 0.18, 0 ), points( 44, 0.3, 0.18, 3.3 ).mul( 0.8 ) );
			// tufts (heads of hooked filaments) and, just downwind of them, a sideways sag of the flow
			const tuft = points( 40, 0.4, 0.1, 6.1 );
			const droop = points( 40, 0.4, 0.3, 6.1 );
			textureStore( this.auxTex, uvec2( globalId.xy ), vec4( seeds, atan( dir.y, dir.x ), tuft, droop ) );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Cloud Fibre Flow' );

		// short strands: each wisp fades in and out along its length and is broken up by fine noise
		const LIC_STEPS = 44, LIC_STEP = 1 / FIB_RES;
		const fibKernel = Fn( () => {

			const p = vec2( globalId.xy ).add( 0.5 ).div( FIB_RES );
			const x1 = p.toVar(), x2 = p.toVar();
			const fib = float( 0 ).toVar();
			const head = float( 0 ).toVar();
			Loop( LIC_STEPS, ( { i } ) => {

				const k = float( i );
				const a = texture( this.auxTex, x1 ).level( 0 );
				fib.addAssign( a.x.mul( sin( k.add( 0.5 ).mul( PI / LIC_STEPS ) ) ) );
				x1.subAssign( vec2( cos( a.y ), sin( a.y ) ).mul( LIC_STEP ) );
				// hooked tails: from the tuft downwind, sagging sideways
				const b = texture( this.auxTex, x2 ).level( 0 );
				head.addAssign( b.z.mul( exp( k.mul( - 1 / 12 ) ) ) );
				x2.subAssign( normalize( vec2( cos( b.y ), sin( b.y ).add( b.w.mul( 1.2 ) ) ) ).mul( LIC_STEP ) );

			} );

			const breakup = saturate( gfbm( p, vec2( 24 ), 0.71 ).mul( 1.6 ).sub( 0.25 ) );
			const wisps = float( 1 ).sub( exp( fib.mul( - 0.8 ) ) ).mul( breakup );
			const hooks = float( 1 ).sub( exp( head.mul( - 0.5 ) ) );
			const veil = gfbm( p, vec2( 4 ), 0.33 );
			textureStore( this.fibTex, uvec2( globalId.xy ), vec4( saturate( max( wisps, hooks ) ), saturate( veil ), 0, 1 ) );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Cloud Fibres' );

		r.compute( synKernel, [ SYN_RES / 8, SYN_RES / 8, 1 ] );
		r.compute( auxKernel, [ FIB_RES / 8, FIB_RES / 8, 1 ] );
		r.compute( fibKernel, [ FIB_RES / 8, FIB_RES / 8, 1 ] );
		r.compute( shapeKernel, [ SHAPE_RES / 4, SHAPE_RES / 4, SHAPE_RES / 4 ] );
		r.compute( detailKernel, [ DETAIL_RES / 4, DETAIL_RES / 4, DETAIL_RES / 4 ] );
		r.compute( weatherKernel, [ WEATHER_RES / 8, WEATHER_RES / 8, 1 ] );

	}

	_makeTargets() {

		const make = ( w, h, name, filter = THREE.LinearFilter ) => {

			const t = new THREE.StorageTexture( w, h );
			t.type = THREE.HalfFloatType;
			t.format = THREE.RGBAFormat;
			t.magFilter = t.minFilter = filter;
			t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
			t.generateMipmaps = false;
			t.name = name;
			return t;

		};

		this.panorama = make( PANO_W, PANO_H, 'cloudPanorama' );
		this.panorama.wrapS = THREE.RepeatWrapping;
		// nearest filtered (unfilterable): materials read it without a sampler binding
		this.shadowMap = make( SHADOW_RES, SHADOW_RES, 'cloudShadow', THREE.NearestFilter );
		// 3D distance field (in cells) to anything that may hold cloud; ping-pong for the passes
		const make3 = ( name ) => {

			const t = new THREE.Storage3DTexture( SDF_RES, SDF_RES, SDF_H );
			t.format = THREE.RGBAFormat;
			t.type = THREE.UnsignedByteType;
			t.magFilter = t.minFilter = THREE.NearestFilter;
			t.wrapS = t.wrapT = THREE.RepeatWrapping;
			t.wrapR = THREE.ClampToEdgeWrapping;
			t.generateMipmaps = false;
			t.name = name;
			return t;

		};

		this.sdfA = make3( 'cloudSdfA' );
		this.sdfB = make3( 'cloudSdfB' );
		this.sdfTex = make3( 'cloudSdf' );

		// screen space view buffers (resized with the canvas)
		this.viewSize = uniform( new THREE.Vector2( 4, 4 ) ).setName( 'clViewSize' );
		this.displayH = uniform( 4 ).setName( 'clDisplayH' ); // render height (px): filters the noise
		this.subPixel = uniform( new THREE.Vector2() ).setName( 'clSubPx' ); // ray offset in the traced pixel
		this.traceSize = uniform( new THREE.Vector2( 1, 1 ) ).setName( 'clTraceSize' );
		this.traceTex = make( 1, 1, 'cloudTrace' );
		this.traceDepth = make( 1, 1, 'cloudTraceDepth', THREE.NearestFilter );
		this.highTrace = make( 1, 1, 'cloudHighTrace' );
		this.motionTex = make( 1, 1, 'cloudMotion', THREE.NearestFilter );
		// range of this frame's samples around each block (neighborhood clamp of the history)
		this.boxMin = make( 1, 1, 'cloudBoxMin', THREE.NearestFilter );
		this.boxMax = make( 1, 1, 'cloudBoxMax', THREE.NearestFilter );
		this.history = [ make( 4, 4, 'cloudViewA' ), make( 4, 4, 'cloudViewB' ) ];
		this.viewTexNode = texture( this.history[ 0 ] );
		this._w = 0;
		this._h = 0;

	}

	// ------------------------------------------------------------ density

	_buildFunctions() {

		const c = Math.cos( this._rot ), s = Math.sin( this._rot );

		// weather space (rotated so cloud streets follow the wind) uv of a camera relative position
		const weatherUV = ( pxz ) => vec2( pxz.x.mul( c ).add( pxz.y.mul( s ) ), pxz.y.mul( c ).sub( pxz.x.mul( s ) ) ).add( this.wOrigin ).div( TILE );

		// x: cell profile (after the coverage control), y: top of this column (fraction of the layer).
		// pxz: camera relative; big cells and towers are only allowed far from the island (origin)
		this._weather = Fn( ( [ pxz ] ) => {

			const w = texture( this.weatherTex, weatherUV( pxz ) ).level( 0 );
			const thr = float( 1 ).sub( this.coverage.mul( 1.3 ) );
			const far = smoothstep( ISLAND_NEAR, ISLAND_FAR, length( pxz.add( this.camXZ ) ) );
			// cells too weak to hold more than a scrap of cloud are dropped (no scattered specks)
			const cs = saturate( w.x.sub( thr ).div( max( float( 1 ).sub( thr ), 0.05 ) ).sub( 0.08 ).mul( 1.09 ) );
			const cb = saturate( w.w.mul( far ).sub( thr ).div( max( float( 1 ).sub( thr ), 0.05 ) ).sub( 0.08 ).mul( 1.09 ) );
			return vec2( max( cs, cb ), max( smallTop( cs, w.y, w.z ), bigTop( cb, w.z ) ) );

		} ).setLayout( { name: 'cloudWeather', type: 'vec2', inputs: [ { name: 'pxz', type: 'vec2' } ] } );

		// distance (m) that is certainly free of cloud around p (0 near clouds)
		this._skip = Fn( ( [ p ] ) => {

			const alt = p.y.add( p.x.mul( p.x ).add( p.z.mul( p.z ) ).div( 2 * EARTH_R ) );
			const H = this.top.sub( this.bottom );
			const d = texture3D( this.sdfTex, vec3( weatherUV( p.xz ), saturate( alt.sub( this.bottom ).div( H ) ) ) ).level( 0 ).x.mul( 255 );
			return max( d.sub( 1 ), 0 ).mul( min( H.div( SDF_H ), TILE / SDF_RES ) );

		} ).setLayout( { name: 'cloudSkip', type: 'float', inputs: [ { name: 'p', type: 'vec3' } ] } );

		// base shape. p: camera relative (sheared) position, w: weather
		// returns vec4( raw shape value, height inside the cloud 0..1, swirl xy ); density = ramp( x )
		this._base = Fn( ( [ p, w ] ) => {

			const alt = p.y.add( p.x.mul( p.x ).add( p.z.mul( p.z ) ).div( 2 * EARTH_R ) );
			const hL = alt.sub( this.bottom ).div( this.top.sub( this.bottom ) );
			const np = p.xz.add( this.nOrigin );
			const n = texture3D( this.shapeTex, vec3( np.x, alt.mul( 1.4 ), np.y ).div( SHAPE_SIZE ) ).level( 0 );
			// flat, slightly uneven base, the column's (domed) top, edges from the cell profile (steep,
			// so the shape noise can't break fragments off the rim)
			const hb = n.y.mul( 0.025 );
			const Ce = saturate( w.y.sub( hL ).mul( 2.2 ) ).mul( smoothstep( 0.0, 0.45, w.x ) ).mul( smoothstep( hb, hb.add( 0.012 ), hL ) );
			// the shape noise carves the boundary. x is the raw shape value (density before the ramp,
			// negative outside): the detail erodes it in the same units, and the coarse search uses it
			// to slow down near a cloud
			return vec4( n.x.add( Ce ).sub( 1 ), saturate( hL.div( max( w.y, 0.05 ) ) ), n.y, n.z );

		} ).setLayout( { name: 'cloudBase', type: 'vec4', inputs: [ { name: 'p', type: 'vec3' }, { name: 'w', type: 'vec2' } ] } );

		// full density with detail erosion. b: result of _base, foot: pixel footprint (m) that filters the
		// octaves. Returns vec2( density, lump ) (1 on a lump, 0 in a crease: shades the crevices)
		this._erode = Fn( ( [ p, b, foot ] ) => {

			const alt = p.y.add( p.x.mul( p.x ).add( p.z.mul( p.z ) ).div( 2 * EARTH_R ) );
			const np = p.xz.add( this.nOrigin );
			// low frequency swirl of the detail lookup
			const sw = b.zw.sub( 0.5 ).mul( float( 0.5 ).mul( float( 1 ).sub( b.y.mul( 0.6 ) ) ) );
			const dp = vec3( np.x, alt, np.y ).div( DETAIL_SIZE ).add( vec3( sw.x, 0, sw.y ) ).add( vec3( G.time.mul( 0.0015 ), 0, 0 ) ).toVar();
			// three octaves per fetch; s: size (m) of the coarsest features
			const octaves = ( uvw, s ) => {

				const d = texture3D( this.detailTex, uvw ).level( 0 ).xyz;
				const w = vec3( smoothstep( s * 0.24, s * 0.09, foot ), smoothstep( s * 0.12, s * 0.045, foot ), smoothstep( s * 0.06, s * 0.0225, foot ) );
				return dot( mix( vec3( D_MEAN ), d, w ), vec3( 0.6, 0.25, 0.15 ) );

			};

			const f1 = float( D_MEAN ).toVar(), f2 = float( D_MEAN ).toVar();
			If( foot.lessThan( D_S1 * 0.24 ), () => {

				f1.assign( octaves( dp, D_S1 ) );
				// close range: a finer fetch (lumps down to ~2 m) keeps near clouds crisp
				If( foot.lessThan( D_S2 * 0.24 ), () => {

					f2.assign( octaves( dp.mul( D_NEAR ).add( 0.37 ), D_S2 ) );

				} );

			} );
			const crease = creaseOf( erosionField( f1.mul( 0.65 ).add( f2.mul( 0.35 ) ), b ) );
			// the creases are eaten: round lumps (cauliflower) on the upper parts, wisps underneath
			return vec2( saturate( b.x.sub( crease.mul( erosionAmount( b ) ) ).sub( 0.012 ).mul( EDGE ) ), float( 1 ).sub( crease ) );

		} ).setLayout( { name: 'cloudErode', type: 'vec2', inputs: [ { name: 'p', type: 'vec3' }, { name: 'b', type: 'vec4' }, { name: 'foot', type: 'float' } ] } );

		// the cloud field leans downwind with height (wind shear)
		this._sheared = ( p ) => {

			const lean = max( p.y.sub( this.bottom ), 0 ).mul( SHEAR );
			return vec3( p.x.sub( this.windN.x.mul( lean ) ), p.y, p.z.sub( this.windN.y.mul( lean ) ) );

		};

	}

	// Cirrus veil seen along rd: vec4( radiance, transmittance ). A thin sheet on a curved-earth
	// shell, so its fibres converge toward the horizon in true perspective. Its coverage varies
	// only over hundreds of km (clearer and thicker parts of the sky, no outlines); long, gently
	// curved fibres follow the upper wind and give a low contrast texture. The fibre texture is
	// filtered anisotropically with the footprint of one pixel (pxAngle, radians), which turns
	// distant fibres into a smooth veil. Lighting: single scattering by ice crystals (strong
	// forward peak: bright near the sun, a faint 22 degree halo) plus multiple scattering and sky
	// light; lit by the sun from below its horizon for a while after sunset.
	_high( rd, pxAngle ) {

		const atmo = this.atmosphere;
		const H = this.cirrusAlt;
		const light = keyLight( atmo, float( 9 ) );
		const sunDir = light.dir;
		const cosT = dot( rd, sunDir );

		// geometry: hit point, local incidence, pixel footprint on the sheet (across the view and
		// along it, stretched by 1 / mu)
		const t = shell( this.camY, rd, H );
		const pxz = rd.xz.mul( t );
		const up = normalize( vec3( pxz.x.div( EARTH_R ), 1, pxz.y.div( EARTH_R ) ) );
		const mu = max( dot( rd, up ), 0.02 );
		const radial = normalize( rd.xz.add( vec2( 1e-6, 0 ) ) );
		const fA = vec2( radial.y.negate(), radial.x ).mul( t.mul( pxAngle ) );
		const fB = radial.mul( t.mul( pxAngle ).div( mu ) );
		const q = pxz.add( this.camXZ ).sub( this.hOffset );

		// frame of the upper wind (veered from the trades)
		const ang = this._rot + 0.6, c = Math.cos( ang ), s = Math.sin( ang );
		const toWind = ( v, k = 1 ) => vec2( v.x.mul( c ).add( v.y.mul( s ) ), v.y.mul( c ).sub( v.x.mul( s ) ) ).mul( k );

		// anisotropic filtering: n taps along the long axis of the footprint at the mip of its short
		// axis (three only emits textureSampleGrad in fragment shaders)
		const fibres = ( tile, rot, n ) => {

			const cr = Math.cos( rot ), sr = Math.sin( rot );
			const frame = ( v ) => { const w = toWind( v ); return vec2( w.x.mul( cr ).add( w.y.mul( sr ) ), w.y.mul( cr ).sub( w.x.mul( sr ) ) ).div( tile ); };
			const uv = frame( q ), a = frame( fA ), b = frame( fB );
			const la = length( a ).mul( FIB_RES ), lb = length( b ).mul( FIB_RES );
			const major = select( la.greaterThan( lb ), a, b );
			const lod = max( log2( max( min( la, lb ), max( la, lb ).div( n ) ) ), 0 );
			let sum = null;
			for ( let i = 0; i < n; i ++ ) {

				const v = bspline( this.fibTex, uv.add( major.mul( ( i + 0.5 ) / n - 0.5 ) ), lod, FIB_RES );
				sum = sum ? sum.add( v ) : v;

			}

			return sum.div( n );

		};

		// coverage: patches of 5 - 50 km with clear gaps, modulated over hundreds of km, broad bands
		const syn = texture( this.synTex, toWind( q, 1 / SYN_SIZE ) ).level( 0 );
		const cov = smoothstep( 0.15, 0.85, syn.x ).mul( smoothstep( 0.4, 0.72, syn.z ) ).mul( syn.y.mul( 0.4 ).add( 0.6 ) );
		// fibres at two scales (hides the tiling of the texture)
		const f1 = fibres( FIB_TILE, 0, 3 ), f2 = fibres( FIB_TILE * 2.3, 0.5, 3 );
		const fib = f1.x.mul( 0.6 ).add( f2.x.mul( 0.4 ) );
		const veil = f1.y.mul( 0.5 ).add( f2.y.mul( 0.5 ) );
		// vertical optical depth: faint wisps (opacity mostly 0.05 - 0.2 at the default amount) over
		// a thinner veil
		const amount = this.cirrus.mul( this.coverage.mul( 0.5 ).add( 0.78 ) );
		const tau = amount.mul( 0.4 ).mul( cov ).mul( fib.mul( 0.75 ).add( 0.25 ) ).mul( veil.mul( 0.4 ).add( 0.8 ) );
		// slant path, bounded so the veil doesn't turn into a white band at the horizon
		const alpha = float( 1 ).sub( exp( tau.div( max( mu, 0.25 ) ).negate() ) );

		// lighting: the key light at the sheet (sunset colours depend on the altitude)
		const muS = dot( sunDir, up );
		const E = select( light.isMoon, light.E, atmo.sampleTransmittance( float( 6360 ).add( H.div( 1000 ) ), muS ).mul( atmo.sunIlluminance ) )
			.mul( earthShadow( light, H, pxz ) );
		// ice: strong forward peak, a faint 22 degree halo, some backscatter; thin: mostly single
		// scattering, a little multiply scattered light
		const hd = acos( clamp( cosT, - 1, 1 ) ).sub( 0.384 ).div( 0.02 );
		const halo = exp( hd.mul( hd ).negate() ).mul( 0.04 );
		const phase = phaseHG( cosT, 0.85 ).mul( 0.4 ).add( phaseHG( cosT, 0.3 ).mul( 0.35 ) ).add( phaseHG( cosT, - 0.15 ).mul( 0.25 ) ).add( halo );
		const Ts = exp( tau.mul( - 0.5 ).div( max( muS, 0.1 ) ) );
		const L = E.mul( phase.mul( Ts ).add( float( 1 ).sub( Ts ).mul( 0.06 ) ) ).add( G.skyIrradiance.mul( 0.9 ) ).mul( alpha );
		// aerial perspective: haze in front of the sheet shows sky light where the sheet hides it
		const tr = exp( t.mul( - 0.25 / AP_DIST ) );
		const skyL = atmo.skyLuminance( rd );
		return vec4( L.mul( tr ).add( skyL.mul( alpha ).mul( float( 1 ).sub( tr ) ) ), float( 1 ).sub( alpha ) );

	}

	// Ray march the cumulus layer along rd (camera relative, camera on the planet axis).
	// Returns { L, T, depth }: sky * T + L is the composite, L includes aerial perspective.
	_march( rd, jitter, q ) {

		const camY = this.camY;
		const atmo = this.atmosphere;
		const light = keyLight( atmo, this.bottom.add( this.top ).mul( 0.0005 ) );
		const sunDir = light.dir;
		const t0 = max( shell( camY, rd, this.bottom ), 0 ).toVar();
		const t1 = min( shell( camY, rd, this.top ), q.maxDist ).toVar();
		const L = vec3( 0 ).toVar();
		const T = float( 1 ).toVar();
		const opac = float( 0 ).toVar();
		const tAcc = float( 0 ).toVar();
		const wAcc = float( 0 ).toVar();

		// Samples lie on a lattice along the ray, shared by all rays and jittered per frame: steps of
		// ds in distance bands ( [0, 5), [5, 10), [10, 20), [20, 40), 40+ km, ds doubling), fine points
		// every ds, coarse points every `coarse` fine points. Jumps over empty space snap onto it, so
		// neighbouring pixels always sample the clouds at consistent positions (arbitrary landing
		// points after a jump would average into contour lines)
		const B0 = 2500;
		const band = ( t ) => clamp( floor( log2( max( t, B0 ).div( B0 ) ) ), 0, 4 );
		const bandStart = ( b ) => select( b.equal( 0 ), float( 0 ), exp2( b ).mul( B0 ) );
		const bandStep = ( b ) => min( exp2( b ).mul( q.ds0 ), q.dsMax );
		const k0 = floor( fract( jitter.mul( 7.31 ).add( 0.37 ) ).mul( q.coarse ) ); // coarse phase
		const posMod = ( x, c ) => x.sub( floor( x.div( c ) ).mul( c ) );
		const snapCoarse = ( t, up ) => {

			const b = band( t ), s0 = bandStart( b ), ds = bandStep( b );
			const i = t.sub( s0 ).div( ds ).sub( jitter );
			const m = up ? ceil( i.sub( 1e-3 ) ) : floor( i.add( 1e-3 ) );
			const mc = up ? m.add( posMod( k0.sub( m ), q.coarse ) ) : m.sub( posMod( m.sub( k0 ), q.coarse ) );
			return s0.add( mc.add( jitter ).mul( ds ) );

		};

		const cosT = dot( rd, sunDir );
		// multiple scattering octaves (Hillaire 2016): scattering a^i, extinction b^i, eccentricity c^i
		// (sky-pro-webgpu) normalized droplet phase: 80% forward / 20% back; each order halves the asymmetry
		const phase = vec3( phaseHG( cosT, 0.8 ), phaseHG( cosT, 0.4 ), phaseHG( cosT, 0.2 ) ).mul( 0.8 )
			.add( vec3( phaseHG( cosT, - 0.2 ), phaseHG( cosT, - 0.1 ), phaseHG( cosT, - 0.05 ) ).mul( 0.2 ) ).toVar();
		const ph3 = float( 0.15 / ( 4 * PI ) ); // light diffused through the whole cloud (keeps thick bodies from going black)
		const sunE = light.E.toVar();
		const amb = G.skyIrradiance.mul( q.ambient );
		// in-scatter probability (Schneider 2015 'powder', Nubis 2017): light scattered toward the viewer
		// builds up inside the cloud, so thin edges and the underside look darker; faded out looking
		// toward the sun, where the forward peak makes the thin edges glow (silver lining)
		const powderK = saturate( cosT.mul( - 0.5 ).add( 0.6 ) );
		// warm light bounced by the sunlit sea onto the undersides (at low sun the sunlight is warm)
		const bounce = G.sunColor.mul( max( sunDir.y, 0 ).mul( 0.05 ).add( 0.012 ) );

		const t = snapCoarse( t0, true ).toVar();
		const bd = band( t ).toVar();
		// Nubis stepping: coarse steps with the cheap density until something is hit, then step
		// back and walk through it with fine, fully detailed samples
		const fineSteps = int( 0 ).toVar();

		If( rd.y.greaterThan( this.horizonY ).and( t1.greaterThan( t0 ) ), () => {

			Loop( q.maxSteps, () => {

				If( t.greaterThan( t1 ).or( T.lessThan( 0.015 ) ), () => {

					Break();

				} );

				// entering the next distance band: onto its lattice
				If( band( t ).notEqual( bd ), () => {

					bd.assign( band( t ) );
					If( fineSteps.equal( 0 ), () => {

						t.assign( snapCoarse( t, true ) );

					} ).Else( () => {

						const s0 = bandStart( bd ), d0 = bandStep( bd );
						t.assign( s0.add( ceil( t.sub( s0 ).div( d0 ).sub( jitter ).sub( 1e-3 ) ).add( jitter ).mul( d0 ) ) );

					} );

				} );

				const pr = vec3( rd.x.mul( t ), camY.add( rd.y.mul( t ) ), rd.z.mul( t ) ).toVar();
				const p = this._sheared( pr ).toVar();
				const ds = bandStep( bd ).toVar();

				If( fineSteps.equal( 0 ), () => {

					const sk = this._skip( p ).toVar();
					// the shear can move the sample up to 12% more than the ray
					const jump = sk.mul( 0.89 );
					If( jump.greaterThan( ds.mul( q.coarse ) ), () => {

						// far from any cloud: jump, landing on the coarse lattice (at least one stride on)
						t.assign( max( snapCoarse( t.add( jump ), false ), t.add( ds.mul( q.coarse ) ) ) );
						bd.assign( band( t ) );

					} ).Else( () => {

						const b = this._base( p, this._weather( p.xz ) );
						// switch to fine steps a little before the surface so thin wisps aren't skipped
						If( b.x.greaterThan( - 0.08 ), () => {

							t.subAssign( ds.mul( q.coarse - 1 ) );
							fineSteps.assign( 5 );

						} ).Else( () => {

							t.addAssign( ds.mul( q.coarse ) );

						} );

					} );

				} ).Else( () => {

					const w = this._weather( p.xz ).toVar();
					const b = this._base( p, w ).toVar();
					fineSteps.subAssign( 1 );

					If( b.x.greaterThan( 0.002 ), () => {

						fineSteps.assign( 3 );
						// detail level from the pixel footprint (m): the close range octave (lumps of ~7 - 28 m)
						// fades out beyond ~2 km, the erosion (30 - 120 m) beyond ~15 km, so nothing smaller
						// than about two pixels is ever sampled (that would only alias into grain)
						const foot = t.mul( q.pxAngle ).toVar();
						const er = vec2( meanDensity( b ), float( 1 ).sub( meanCrease( b ) ) ).toVar();
						// detail erosion (every octave under 2 px: the mean erosion above)
						if ( q.detail ) If( foot.lessThan( D_S1 * 0.24 ), () => {

							er.assign( this._erode( p, b, foot ) );

						} );
						const dens = er.x;

						If( dens.greaterThan( 0.002 ), () => {

							// light march toward the sun: near samples share the weather and keep the
							// detail, far ones only see the base shape
							const od = float( 0 ).toVar();
							const LS = q.lightSteps;
							// jittered along the light ray: its sampling pattern turns into noise the
							// temporal filter removes, instead of streaks across the cloud
							const lj = fract( jitter.add( 0.5 ) ).mul( 0.3 ).add( 0.85 );
							let prev = 0;
							for ( let k = 0; k < LS.length; k ++ ) {

								const dist = LS[ k ];
								const len = dist - prev;
								prev = dist;
								const lp = this._sheared( pr.add( sunDir.mul( lj.mul( dist - len * 0.5 ) ) ) );
								if ( k < q.lightDetail ) {

									// detailed self shadowing matters near the visible surface only
									const lb = this._base( lp, w ).toVar();
									If( T.greaterThan( 0.5 ), () => {

										// filtered at the size of the light segment
										od.addAssign( this._erode( lp, lb, max( foot, len * 0.35 ) ).x.mul( len ) );

									} ).Else( () => {

										od.addAssign( meanDensity( lb ).mul( len ) );

									} );

								} else {

									const lb = this._base( lp, this._weather( lp.xz ) );
									od.addAssign( meanDensity( lb ).mul( len ) );

								}

							}

							const sig = dens.mul( this.densityScale );
							// light optical depth (the local segment included); multiple scattering lowers the
							// effective extinction of the light
							const tau = od.add( dens.mul( 12 ) ).mul( this.densityScale ).mul( 0.55 );
							// three orders from one exponential: extinction and energy halve each order
							const quarter = exp( tau.mul( - 0.25 ) );
							const halfT = quarter.mul( quarter );
							const sun = dot( vec3( halfT.mul( halfT ), halfT.mul( 0.5 ), quarter.mul( 0.25 ) ), phase ).add( ph3.mul( exp( tau.mul( - 0.06 ) ) ) );
							// skylight occlusion: two broad upward probes (125 m, 600 m) of the filtered density
							const pu1 = this._sheared( pr.add( vec3( 0, 125, 0 ) ) ), pu2 = this._sheared( pr.add( vec3( 0, 600, 0 ) ) );
							const skyTau = meanDensity( this._base( pu1, w ) ).mul( 250 ).add( meanDensity( this._base( pu2, this._weather( pu2.xz ) ) ).mul( 700 ) )
								.add( dens.mul( 25 ) ).mul( this.densityScale );
							const skyVis = float( 0.2 ).add( float( 0.8 ).div( skyTau.mul( 0.35 ).add( 1 ) ) );
							// darker bases (their direct light is scattered away by the cloud above)
							const baseShadow = mix( float( 1 ), mix( float( 0.35 ), float( 1 ), smoothstep( - 0.1, 0.45, b.y ) ), 0.6 );
							const depthP = pow( dens, mix( 0.5, 1.6, b.y ) ).mul( 0.95 ).add( 0.05 );
							const vertP = pow( smoothstep( 0.02, 0.2, b.y ), 0.8 ).mul( 0.85 ).add( 0.15 );
							const powder = mix( float( 1 ), depthP.mul( vertP ), powderK );
							// ambient: the sky lights the tops; the bases only see the dark sea and the
							// horizon (darker, bluer), and crevices of the detail noise are occluded
							const up = saturate( b.y.mul( 1.4 ) );
							const ambH = mix( vec3( 0.38, 0.43, 0.52 ), vec3( 1 ), sqrt( up ) ).mul( skyVis ).mul( er.y.mul( 0.6 ).add( 0.55 ) );
							// after sunset the tops stay lit longest
							const alt = pr.y.add( dot( pr.xz, pr.xz ).div( 2 * EARTH_R ) );
							const S = sunE.mul( sun.mul( powder ).mul( baseShadow ).mul( earthShadow( light, alt, pr.xz ) ) ).add( amb.mul( ambH ) )
								.add( bounce.mul( float( 1 ).sub( up ) ) );
							const Tstep = exp( sig.mul( ds ).negate() );
							const tap = exp( t.mul( - 1 / AP_DIST ) );
							const dT = T.mul( float( 1 ).sub( Tstep ) );
							L.addAssign( S.mul( dT ).mul( tap ) );
							opac.addAssign( dT.mul( tap ) );
							tAcc.addAssign( t.mul( dT ) );
							wAcc.addAssign( dT );
							T.mulAssign( Tstep );

						} );

					} );

					t.addAssign( ds );
					If( fineSteps.equal( 0 ), () => {

						t.assign( snapCoarse( t, true ) );

					} );

				} );

			} );

		} );

		const depth = select( wAcc.greaterThan( 1e-4 ), tAcc.div( max( wAcc, 1e-4 ) ), t0.add( 4000 ) );
		// aerial perspective: the haze in front of distant clouds shows sky light where the cloud
		// hides the sky (sky luminance times the hidden fraction not reached by the cloud's light)
		// the march stops at 98.5% opacity: the rest counts as opaque (the sun disc must not shine through)
		const Tc = saturate( T.sub( 0.015 ).div( 0.985 ) );
		const haze = float( 1 ).sub( Tc ).sub( opac );
		return { L: L.add( atmo.skyLuminance( rd ).mul( max( haze, 0 ) ) ), T: Tc, depth };

	}

	// ------------------------------------------------------------ kernels

	_buildKernels() {

		// ---- coverage dependent 3D distance field (Chebyshev, separable passes) of the dome footprint.
		// Built for a coverage a little above the current one, so small changes need no rebuild
		this.sdfCoverage = uniform( 0 ).setName( 'clSdfCoverage' );
		// occupancy: a cell may hold cloud if the dome of any weather texel influencing it reaches
		// the cell's lowest altitude
		this.sdfOccKernel = Fn( () => {

			const ix = int( globalId.x ), iy = int( globalId.y );
			const h0 = float( globalId.z ).div( SDF_H );
			const thr = float( 1 ).sub( this.sdfCoverage.mul( 1.3 ) );
			// conservative: maxima of every weather quantity around the cell (they are interpolated
			// separately), big cells assumed allowed
			const mx = vec4( 0 ).toVar();
			for ( let y = - 1; y <= 2; y ++ ) for ( let x = - 1; x <= 2; x ++ ) {

				const wc = ivec2( mod( ix.mul( 2 ).add( x ).add( WEATHER_RES ), WEATHER_RES ), mod( iy.mul( 2 ).add( y ).add( WEATHER_RES ), WEATHER_RES ) );
				mx.assign( max( mx, textureLoad( this.weatherTex, wc ) ) );

			}

			const cs = saturate( mx.x.sub( thr ).div( max( float( 1 ).sub( thr ), 0.05 ) ) );
			const cb = saturate( mx.w.sub( thr ).div( max( float( 1 ).sub( thr ), 0.05 ) ) );
			const occ = max( select( cs.greaterThan( 0 ), smallTop( cs, mx.y, mx.z ), float( - 1 ) ), select( cb.greaterThan( 0 ), bigTop( cb, mx.z ), float( - 1 ) ) ).sub( h0 );

			textureStore( this.sdfA, uvec3( globalId ), vec4( select( occ.greaterThan( 0 ), float( 0 ), float( 1 ) ), 0, 0, 1 ) );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Cloud SDF Occupancy' );

		const load = ( tex, x, y, z ) => texture3DLoad( tex, ivec3( x, y, z ) ).x.mul( 255 );

		this.sdfXKernel = Fn( () => {

			const ix = int( globalId.x ), iy = int( globalId.y ), iz = int( globalId.z );
			const d = float( SDF_R + 1 ).toVar();
			for ( let k = - SDF_R; k <= SDF_R; k ++ ) {

				If( load( this.sdfA, mod( ix.add( k + SDF_RES ), SDF_RES ), iy, iz ).lessThan( 0.5 ), () => {

					d.assign( min( d, float( Math.abs( k ) ) ) );

				} );

			}

			textureStore( this.sdfB, uvec3( globalId ), vec4( d.div( 255 ), 0, 0, 1 ) );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Cloud SDF X' );

		this.sdfYKernel = Fn( () => {

			const ix = int( globalId.x ), iy = int( globalId.y ), iz = int( globalId.z );
			const d = float( SDF_R + 1 ).toVar();
			for ( let k = - SDF_R; k <= SDF_R; k ++ ) {

				d.assign( min( d, max( load( this.sdfB, ix, mod( iy.add( k + SDF_RES ), SDF_RES ), iz ), float( Math.abs( k ) ) ) ) );

			}

			textureStore( this.sdfA, uvec3( globalId ), vec4( d.div( 255 ), 0, 0, 1 ) );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Cloud SDF Y' );

		this.sdfZKernel = Fn( () => {

			const ix = int( globalId.x ), iy = int( globalId.y ), iz = int( globalId.z );
			const d = float( SDF_R + 1 ).toVar();
			for ( let k = - ( SDF_H - 1 ); k <= SDF_H - 1; k ++ ) {

				const z = iz.add( k );
				If( z.greaterThanEqual( 0 ).and( z.lessThan( SDF_H ) ), () => {

					d.assign( min( d, max( load( this.sdfA, ix, iy, z ), float( Math.abs( k ) ) ) ) );

				} );

			}

			textureStore( this.sdfTex, uvec3( globalId ), vec4( d.div( 255 ), 0, 0, 1 ) );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Cloud SDF Z' );

		// ---- view: one pixel of every 4x4 block per frame
		this.slot = uniform( new THREE.Vector2() ).setName( 'clSlot' );
		// camera basis (unit vectors) and frustum tangents, this frame and the previous one
		const cam = () => ( {
			right: uniform( new THREE.Vector3( 1, 0, 0 ) ), up: uniform( new THREE.Vector3( 0, 1, 0 ) ),
			fwd: uniform( new THREE.Vector3( 0, 0, - 1 ) ), tan: uniform( new THREE.Vector2( 1, 1 ) ),
		} );
		this.cam = cam();
		this.prev = cam();
		// camera the view texture was last traced with, and whether it holds data at all
		this.viewCam = cam();
		this.viewValid = uniform( 0 ).setName( 'clViewValid' );
		this.camDelta = uniform( new THREE.Vector3() ).setName( 'clCamDelta' ); // camera motion minus wind drift
		this.camDeltaHi = uniform( new THREE.Vector3() ).setName( 'clCamDeltaHi' ); // same for the high layers
		this.frameNoise = uniform( 0 ).setName( 'clFrameN' );
		this.rebuildK = uniform( - 1 ).setName( 'clRebuildK' ); // >= 0: rebuilding after a camera cut (slots traced so far)
		// weight of a new sample for static pixels: a running average right after a cut, then an
		// exponential one over about 16 samples
		this.minAlpha = uniform( 0.12 ).setName( 'clMinAlpha' );

		// march settings: steps of ds0 doubling in distance bands (up to dsMax), coarse search steps
		// `coarse` times longer; light samples at the given distances (the first `lightDetail`
		// with detail erosion)
		this.viewQuality = { maxSteps: 200, maxDist: 50000, ds0: 24, dsMax: 120, coarse: 4, lightSteps: [ 12, 50, 140, 350, 900, 1800 ], lightDetail: 2, detail: true, ambient: 1.2, pxAngle: this.cam.tan.y.mul( 2 ).div( this.displayH ) };
		// reflections / environment: no detail erosion (sub-texel there), short light march
		this.panoQuality = { maxSteps: 56, maxDist: 50000, ds0: 60, dsMax: 320, coarse: 2, lightSteps: [ 120, 500 ], lightDetail: 0, detail: false, ambient: 1.2, pxAngle: float( 2 * PI / PANO_W ) };

		const viewDir = ( uv ) => {

			const c = this.cam;
			const ndc = vec2( uv.x.mul( 2 ).sub( 1 ), float( 1 ).sub( uv.y.mul( 2 ) ) ).mul( c.tan );
			return normalize( c.fwd.add( c.right.mul( ndc.x ) ).add( c.up.mul( ndc.y ) ) );

		};

		// interleaved gradient noise (Jimenez 2014)
		const ign = ( px ) => fract( fract( dot( px, vec2( 0.06711056, 0.00583715 ) ) ).mul( 52.9829189 ) );

		// per pixel random offset (white noise: subsampled every 4 pixels, structured noise such as
		// interleaved gradient noise would alias into a visible grid) for the golden ratio sequence
		const pixelHash = ( px ) => {

			const p3 = fract( vec3( px.x, px.y, px.x ).mul( vec3( 0.1031, 0.1030, 0.0973 ) ) ).toVar();
			p3.addAssign( dot( p3, p3.yzx.add( 33.33 ) ) );
			return fract( p3.x.add( p3.y ).mul( p3.z ) );

		};

		this.traceKernel = Fn( () => {

			const tp = uvec2( globalId.xy );
			If( tp.x.greaterThanEqual( uint( this.traceSize.x ) ).or( tp.y.greaterThanEqual( uint( this.traceSize.y ) ) ), () => {

				Return();

			} );

			const px = vec2( tp.mul( 4 ).add( uvec2( this.slot ) ) ).add( 0.5 ).add( this.subPixel );
			const rd = viewDir( px.div( this.viewSize ) ).toVar();
			// well distributed per frame (interleaved gradient noise on the grid of traced pixels, the
			// offset changes every trace): the residual noise is high frequency, easy to average
			const jitter = ign( vec2( tp ).add( this.frameNoise ) );
			const m = this._march( rd, jitter, this.viewQuality );
			textureStore( this.traceTex, tp, vec4( m.L, m.T ) );
			// depth + opacity of the cumulus: the resolve reprojects either the cumulus or the high layers
			textureStore( this.traceDepth, tp, vec4( m.depth, float( 1 ).sub( m.T ), 0, 1 ) );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Clouds Trace' );

		// high layers for the same pixels (separate kernel: keeps the march kernel lean)
		this.highTraceKernel = Fn( () => {

			const tp = uvec2( globalId.xy );
			If( tp.x.greaterThanEqual( uint( this.traceSize.x ) ).or( tp.y.greaterThanEqual( uint( this.traceSize.y ) ) ), () => {

				Return();

			} );

			const px = vec2( tp.mul( 4 ).add( uvec2( this.slot ) ) ).add( 0.5 ).add( this.subPixel );
			const rd = viewDir( px.div( this.viewSize ) ).toVar();
			const hi = vec4( 0, 0, 0, 1 ).toVar();
			If( rd.y.greaterThan( - 0.01 ), () => {

				hi.assign( this._high( rd, this.cam.tan.y.mul( 2 ).div( this.viewSize.y ) ) );

			} );
			textureStore( this.highTrace, tp, hi );

			// motion for the resolve: the closest cumulus sample around this block (like the closest
			// depth dilation of TAA), so the edges of a cloud move with it and foreground wins
			const best = vec3( 1e9, 0, 0 ).toVar(); // depth key, depth, opacity
			const tmax = ivec2( this.traceSize ).sub( 1 );
			for ( let y = - 1; y <= 1; y ++ ) for ( let x = - 1; x <= 1; x ++ ) {

				const dz = textureLoad( this.traceDepth, clamp( ivec2( tp ).add( ivec2( x, y ) ), ivec2( 0 ), tmax ) );
				const key = select( dz.y.greaterThan( 0.15 ), dz.x, float( 1e9 ) );
				If( key.lessThan( best.x ), () => {

					best.assign( vec3( key, dz.xy ) );

				} );

			}

			If( best.x.greaterThan( 1e8 ), () => {

				best.assign( vec3( 0, textureLoad( this.traceDepth, ivec2( tp ) ).xy ) );

			} );

			textureStore( this.motionTex, tp, vec4( best.yz, 0, 1 ) );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Clouds High Trace' );

		// range (min, max) of this frame's composited samples in the 3x3 blocks around each block
		this.boxKernel = Fn( () => {

			const tp = ivec2( globalId.xy );
			If( uvec2( tp ).x.greaterThanEqual( uint( this.traceSize.x ) ).or( uvec2( tp ).y.greaterThanEqual( uint( this.traceSize.y ) ) ), () => {

				Return();

			} );

			const tmax = ivec2( this.traceSize ).sub( 1 );
			const lo = vec4( 1e4 ).toVar(), hi = vec4( - 1e4 ).toVar();
			for ( let y = - 1; y <= 1; y ++ ) for ( let x = - 1; x <= 1; x ++ ) {

				const q = clamp( tp.add( ivec2( x, y ) ), ivec2( 0 ), tmax );
				const c = over( textureLoad( this.traceTex, q ), textureLoad( this.highTrace, q ) );
				lo.assign( min( lo, c ) );
				hi.assign( max( hi, c ) );

			}

			textureStore( this.boxMin, uvec2( tp ), lo );
			textureStore( this.boxMax, uvec2( tp ), hi );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Clouds Box' );

		// ---- resolve: reproject the history, refresh the traced pixels
		const project = ( d, c ) => {

			const z = max( dot( d, c.fwd ), 1e-4 );
			const ndc = vec2( dot( d, c.right ), dot( d, c.up ) ).div( c.tan.mul( z ) );
			return vec2( ndc.x.mul( 0.5 ).add( 0.5 ), float( 0.5 ).sub( ndc.y.mul( 0.5 ) ) );

		};

		this._project = project;

		// bicubic Catmull-Rom in 5 bilinear taps (corners dropped)
		const catmullRom = ( tap, uv, size ) => {

			const sp = uv.mul( size );
			const tp1 = floor( sp.sub( 0.5 ) ).add( 0.5 );
			const f = sp.sub( tp1 );
			const w0 = f.mul( f.mul( f.mul( - 0.5 ).add( 1.0 ) ).sub( 0.5 ) );
			const w1 = f.mul( f ).mul( f.mul( 1.5 ).sub( 2.5 ) ).add( 1.0 );
			const w2 = f.mul( f.mul( f.mul( - 1.5 ).add( 2.0 ) ).add( 0.5 ) );
			const w3 = f.mul( f ).mul( f.mul( 0.5 ).sub( 0.5 ) );
			const w12 = w1.add( w2 );
			const tc0 = tp1.sub( 1 ).div( size ), tc3 = tp1.add( 2 ).div( size ), tc12 = tp1.add( w2.div( w12 ) ).div( size );
			const s = ( x, y ) => tap( vec2( x, y ) );
			const a = w12.x.mul( w0.y ), b = w0.x.mul( w12.y ), cc = w12.x.mul( w12.y ), d = w3.x.mul( w12.y ), e = w12.x.mul( w3.y );
			const sum = s( tc12.x, tc0.y ).mul( a ).add( s( tc0.x, tc12.y ).mul( b ) ).add( s( tc12.x, tc12.y ).mul( cc ) )
				.add( s( tc3.x, tc12.y ).mul( d ) ).add( s( tc12.x, tc3.y ).mul( e ) );
			return sum.div( a.add( b ).add( cc ).add( d ).add( e ) );

		};

		this._catmullRom = catmullRom;

		const buildResolve = ( src, dst ) => Fn( () => {

			const p = uvec2( globalId.xy );
			If( p.x.greaterThanEqual( uint( this.viewSize.x ) ).or( p.y.greaterThanEqual( uint( this.viewSize.y ) ) ), () => {

				Return();

			} );

			const uv = vec2( p ).add( 0.5 ).div( this.viewSize );
			const rd = viewDir( uv ).toVar();
			If( rd.y.lessThan( - 0.06 ), () => {

				Return();

			} );

			const tp = p.div( 4 );
			const fresh = p.x.mod( 4 ).equal( uint( this.slot.x ) ).and( p.y.mod( 4 ).equal( uint( this.slot.y ) ) );
			const dz = textureLoad( this.motionTex, ivec2( tp ) );
			// where was this cloud point last frame (camera motion and wind drift): cumulus, or the high
			// layers where there is no cumulus in front
			const pdC = normalize( rd.mul( dz.x ).add( this.camDelta ) );
			const pdH = normalize( rd.mul( shell( this.camY, rd, this.cirrusAlt ) ).add( this.camDeltaHi ) );
			const pd = select( dz.y.greaterThan( 0.3 ), pdC, pdH );
			const puv = mix( project( pdH, this.prev ), project( pdC, this.prev ), smoothstep( 0.1, 0.5, dz.y ) ).toVar();
			const valid = this.historyValid.greaterThan( 0.5 ).and( dot( pd, this.prev.fwd ).greaterThan( 0.01 ) )
				.and( puv.x.greaterThan( 0 ) ).and( puv.x.lessThan( 1 ) ).and( puv.y.greaterThan( 0 ) ).and( puv.y.lessThan( 1 ) )
				.and( pd.y.greaterThan( - 0.05 ) ).toVar();
			const out = vec4( 0, 0, 0, 1 ).toVar();
			If( rd.y.greaterThan( - 0.035 ), () => {

				// this frame's samples upsampled (they sit at the slot of each block)
				const upsampled = () => {

					const suv = vec2( p ).sub( this.slot ).div( 4 ).add( 0.5 ).div( this.traceSize );
					return over( texture( this.traceTex, suv ).level( 0 ), texture( this.highTrace, suv ).level( 0 ) );

				};

				If( valid, () => {

					// reprojected history, clamped to the range of this frame's samples around it (with a
					// margin): real changes (lighting, motion the reprojection missed) can't leave ghosts,
					// so the samples can be averaged over many frames, which removes the noise
					const h = max( catmullRom( ( c ) => texture( src, c ).level( 0 ), puv, this.viewSize ), vec4( 0 ) );
					const lo = textureLoad( this.boxMin, ivec2( tp ) ), hi = textureLoad( this.boxMax, ivec2( tp ) );
					const pad = hi.sub( lo ).mul( 0.25 ).add( 0.01 );
					out.assign( select( this.rebuildK.lessThan( 0 ), clamp( h, lo.sub( pad ), hi.add( pad ) ), h ) );

				} ).Else( () => {

					out.assign( upsampled() );

				} );

				If( fresh, () => {

					const cu = textureLoad( this.traceTex, ivec2( tp ) );
					const hl = textureLoad( this.highTrace, ivec2( tp ) );
					const cur = over( cu, hl );
					// average the jittered samples over time (removes the ray march noise); a little
					// shorter where the clouds move fast on screen (the history is resampled every frame)
					const hist = out;
					const motion = length( puv.sub( uv ).mul( this.viewSize ) );
					const a0 = this.minAlpha;
					const a = clamp( motion.mul( 0.1 ).add( a0 ), a0, 0.35 );
					// rebuilding after a camera cut: each pixel takes its own first sample as is
					out.assign( select( valid.and( this.rebuildK.lessThan( 0 ) ), mix( hist, cur, a ), cur ) );

				} ).ElseIf( this.rebuildK.greaterThanEqual( 0 ).and( valid ), () => {

					// rebuilding: pixels without a sample of their own since the cut average the upsampled
					// samples of every slot traced so far (their refresh rank is the 4x4 Bayer index)
					const x = float( p.x.mod( 4 ) ), y = float( p.y.mod( 4 ) );
					const b2 = ( a, b ) => abs( a.sub( b ) ).mul( 2 ).add( b );
					const rank = b2( x.mod( 2 ), y.mod( 2 ) ).mul( 4 ).add( b2( floor( x.div( 2 ) ), floor( y.div( 2 ) ) ) );
					If( rank.greaterThan( this.rebuildK ), () => {

						out.assign( mix( out, upsampled(), float( 1 ).div( this.rebuildK.add( 1 ) ) ) );

					} );

				} );

			} );

			textureStore( dst, p, vec4( out.rgb, clamp( out.a, 0, 1 ) ) );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Clouds Resolve' );

		this.resolveKernels = [ buildResolve( this.history[ 1 ], this.history[ 0 ] ), buildResolve( this.history[ 0 ], this.history[ 1 ] ) ];

		// ---- panorama (reflections / environment): interleaved progressive refresh
		this.panoSlot = uniform( new THREE.Vector2() ).setName( 'clPanoSlot' );
		this.panoKernel = Fn( () => {

			const px = uvec2( globalId.xy ).mul( uvec2( 8, 4 ) ).add( uvec2( this.panoSlot ) );
			const uv = vec2( px ).add( 0.5 ).div( vec2( PANO_W, PANO_H ) );
			const az = uv.x.mul( 2 * PI );
			const elev = uv.y.mul( uv.y ).mul( ( 94 / 180 ) * PI ).sub( ( 4 / 180 ) * PI );
			const rd = vec3( cos( elev ).mul( cos( az ) ), sin( elev ), cos( elev ).mul( sin( az ) ) ).toVar();
			const jitter = pixelHash( vec2( px ) );
			const m = this._march( rd, jitter, this.panoQuality );
			// panorama texel: 2 pi / PANO_W across; the elevation mapping is similar near the horizon
			const hi = this._high( rd, float( 2 * PI / PANO_W ) );
			textureStore( this.panorama, px, over( vec4( m.L, m.T ), hi ) );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Clouds Panorama' );
		this.panoKernel.dispatchSize = [ PANO_W / 64, PANO_H / 32, 1 ];

		// ---- cloud shadow: transmittance of the layer along the key light (sun, or the moon at
		// night), 1/4 of the rows per frame
		this.shadowPhase = uniform( 0 ).setName( 'clShadowPhase' );
		this.shadowKernel = Fn( () => {

			const px = uvec2( globalId.x, globalId.y.mul( 4 ).add( uint( this.shadowPhase ) ) );
			const uv = vec2( px ).add( 0.5 ).div( SHADOW_RES );
			// camera relative ground position
			const gxz = this.shadowCenter.add( uv.sub( 0.5 ).mul( this.shadowSize ) ).sub( this.camXZ );
			const sunDir = G.sunDir;
			const mu = max( sunDir.y, 0.08 );
			const tb = this.bottom.div( mu );
			const tt = this.top.div( mu );
			const od = float( 0 ).toVar();
			const steps = 8;
			for ( let k = 0; k < steps; k ++ ) {

				const t = mix( tb, tt, ( k + 0.5 ) / steps );
				const p = this._sheared( vec3( gxz.x, 0, gxz.y ).add( sunDir.mul( t ) ) );
				od.addAssign( meanDensity( this._base( p, this._weather( p.xz ) ) ) );

			}

			const T = exp( od.mul( this.densityScale ).mul( tt.sub( tb ).div( steps ) ).mul( - 0.5 ) );
			textureStore( this.shadowMap, px, vec4( T, 0, 0, 1 ) );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Cloud Shadow' );
		this.shadowKernel.dispatchSize = [ SHADOW_RES / 8, SHADOW_RES / 32, 1 ];

	}

	_resize( fw, fh ) {

		this._w = fw;
		this._h = fh;
		this._scale = this.resolutionScale;
		const w = Math.max( 4, Math.round( fw * this._scale * HISTORY_SCALE ) ), h = Math.max( 4, Math.round( fh * this._scale * HISTORY_SCALE ) );
		this.displayH.value = Math.max( 4, fh * this._scale );
		const tw = Math.ceil( w / 4 ), th = Math.ceil( h / 4 );
		this.viewSize.value.set( w, h );
		this.traceSize.value.set( tw, th );
		this.traceTex.setSize( tw, th );
		this.traceDepth.setSize( tw, th );
		this.highTrace.setSize( tw, th );
		this.boxMin.setSize( tw, th );
		this.boxMax.setSize( tw, th );
		this.motionTex.setSize( tw, th );
		this.history[ 0 ].setSize( w, h );
		this.history[ 1 ].setSize( w, h );
		this.traceKernel.dispatchSize = [ Math.ceil( tw / 8 ), Math.ceil( th / 8 ), 1 ];
		this.highTraceKernel.dispatchSize = this.traceKernel.dispatchSize;
		this.boxKernel.dispatchSize = this.traceKernel.dispatchSize;
		for ( const k of this.resolveKernels ) k.dispatchSize = [ Math.ceil( w / 8 ), Math.ceil( h / 8 ), 1 ];
		// the new textures hold nothing until traced: the sky uses the panorama meanwhile
		this.viewValid.value = 0;
		this.resetHistory();

	}

	// ------------------------------------------------------------ per frame

	update( dt, camera ) {

		const r = this.renderer;

		// wind drift, kept bounded (every noise tiles within TILE)
		const wind = this.wind.value;
		const c = Math.cos( this._rot ), s = Math.sin( this._rot );
		const o = this.offset.value;
		o.addScaledVector( wind, dt );
		o.set( ( ( o.x % TILE ) + TILE ) % TILE, ( ( o.y % TILE ) + TILE ) % TILE );
		const ow = this._offsetW;
		ow.x += ( wind.x * c + wind.y * s ) * dt;
		ow.y += ( wind.y * c - wind.x * s ) * dt;
		ow.set( ( ( ow.x % TILE ) + TILE ) % TILE, ( ( ow.y % TILE ) + TILE ) % TILE );

		const oh = this._offsetH;
		oh.addScaledVector( wind, dt * 1.6 );
		oh.set( ( ( oh.x % TILE ) + TILE ) % TILE, ( ( oh.y % TILE ) + TILE ) % TILE );

		camera.updateMatrixWorld();
		const cp = camera.position;
		// the march assumes a camera below the cloud base
		this.camY.value = THREE.MathUtils.clamp( cp.y, 1, this.bottom.value - 50 );
		if ( wind.lengthSq() > 1e-6 ) this.windN.value.copy( wind ).normalize();
		this.horizonY.value = - Math.sqrt( 2 * this.camY.value / EARTH_R );
		// the field moves along +wind: sample it at (position - offset)
		this.nOrigin.value.set( cp.x - o.x, cp.z - o.y );
		this.wOrigin.value.set( cp.x * c + cp.z * s - ow.x, cp.z * c - cp.x * s - ow.y );

		// ---- coverage changes: rebuild the distance field when it no longer bounds the clouds (or
		// has become too loose to skip well); the view and panorama converge on their own
		const cov = this.coverage.value;
		if ( cov > this.sdfCoverage.value || cov < this.sdfCoverage.value - 0.1 ) {

			this.sdfCoverage.value = Math.min( cov + 0.04, 1 );
			this._buildSDF();

		}

		// ---- view camera frame
		const size = r.getDrawingBufferSize( this._size || ( this._size = new THREE.Vector2() ) );
		if ( size.x !== this._w || size.y !== this._h || this.resolutionScale !== this._scale ) this._resize( size.x, size.y );
		const e = camera.matrixWorld.elements;
		const tanY = Math.tan( THREE.MathUtils.degToRad( camera.fov * 0.5 ) ) / camera.zoom;
		const C = this.cam, P = this.prev;
		P.right.value.copy( C.right.value );
		P.up.value.copy( C.up.value );
		P.fwd.value.copy( C.fwd.value );
		P.tan.value.copy( C.tan.value );
		C.right.value.set( e[ 0 ], e[ 1 ], e[ 2 ] ).normalize();
		C.up.value.set( e[ 4 ], e[ 5 ], e[ 6 ] ).normalize();
		C.fwd.value.set( - e[ 8 ], - e[ 9 ], - e[ 10 ] ).normalize();
		C.tan.value.set( tanY * camera.aspect, tanY );
		// camera cuts (teleports, big turns, time of day jumps, zoom): the history is useless, rebuild
		// it at full rate. Ordinary camera motion is handled by the reprojection
		const moved = this._hasPrev ? cp.distanceTo( this._prevCam ) : Infinity;
		const turned = C.fwd.value.angleTo( P.fwd.value );
		this._turned = turned;
		const sun = G.sunDir.value;
		if ( moved > 8 || turned > 0.5 || sun.angleTo( this._prevSun ) > 0.05 || Math.abs( C.tan.value.y - P.tan.value.y ) > 1e-3 * C.tan.value.y ) this.resetHistory();
		this._prevSun.copy( sun );
		// camera motion since last frame minus the wind drift of the clouds
		this.camDelta.value.set( cp.x - this._prevCam.x - wind.x * dt, cp.y - this._prevCam.y, cp.z - this._prevCam.z - wind.y * dt );
		this.camDeltaHi.value.set( cp.x - this._prevCam.x - wind.x * dt * 1.6, cp.y - this._prevCam.y, cp.z - this._prevCam.z - wind.y * dt * 1.6 );
		this._prevCam.copy( cp );
		this._hasPrev = true;

		this.shadowPhase.value = this.frame % 4;
		if ( this.frame % 4 === 0 ) this.shadowCenter.value.set( cp.x, cp.z );
		this.camXZ.value.set( cp.x, cp.z );
		this.hOffset.value.copy( oh );

		// ---- panorama: 1/32 of the texels per frame (8x4 blocks); all of them after a reset
		if ( this.panoWarm > 0 ) {

			for ( let k = 0; k < 32; k ++ ) {

				this._panoSlot( k );
				r.compute( this.panoKernel );

			}

			this.panoWarm = 0;

		}

		this._panoSlot( this.frame );
		if ( G.cameraUnderwater.value > 0.5 ) {

			// the sky is only seen through Snell's window (panorama): no view clouds
			r.compute( [ this.shadowKernel, this.panoKernel ] );
			this.resetHistory();

		} else if ( this._rebuild < 16 ) {

			// after a cut: several slots per frame, until every pixel has a sample of its own
			for ( let k = 0; k < REBUILD_SLOTS && this._rebuild < 16; k ++ ) {

				if ( k > 0 ) this._holdCamera();
				this.rebuildK.value = this._rebuild;
				this._trace( ORDER[ this._rebuild ++ ] );

			}

			r.compute( [ this.shadowKernel, this.panoKernel ] );

		} else {

			// samples every pixel holds since the cut
			const n = 1 + Math.floor( this._since ++ / 16 );
			this.minAlpha.value = Math.max( 0.12, 1 / ( n + 1 ) );
			this.rebuildK.value = - 1;
			this._trace( ORDER[ this.frame % 16 ], [ this.shadowKernel, this.panoKernel ] );
			// turning camera: a second slot per frame (every pixel refreshed in 8 frames instead of 16)
			// keeps the reprojected history from softening
			if ( this._turned > 0.003 ) {

				this._holdCamera();
				this._trace( ORDER[ ( this.frame + 8 ) % 16 ] );

			}

		}

		this.frame ++;

	}

	// trace one slot of every 4x4 block and resolve it into the history (plus extra kernels)
	_trace( order, extra = [] ) {

		this.slot.value.set( order % 4, Math.floor( order / 4 ) );
		// R2 sequence over the traces (one offset per 16 frame cycle, so each pixel sees them all)
		const n = Math.floor( this._traces / 16 ) + 1;
		this.subPixel.value.set( ( ( 0.5 + n * 0.7548776662 ) % 1 ) - 0.5, ( ( 0.5 + n * 0.5698402910 ) % 1 ) - 0.5 ).multiplyScalar( 0.25 );
		this.frameNoise.value = ( this._traces ++ % 64 ) * 5.588238;
		this.renderer.compute( [ this.traceKernel, this.highTraceKernel, this.boxKernel, this.resolveKernels[ this._pp ], ...extra ] );
		this.viewTexNode.value = this.history[ this._pp ];
		this._pp = 1 - this._pp;
		this.historyValid.value = 1;
		const C = this.cam, V = this.viewCam;
		V.right.value.copy( C.right.value );
		V.up.value.copy( C.up.value );
		V.fwd.value.copy( C.fwd.value );
		V.tan.value.copy( C.tan.value );
		this.viewValid.value = 1;

	}

	// no camera motion between the traces of one frame
	_holdCamera() {

		const C = this.cam, P = this.prev;
		P.right.value.copy( C.right.value );
		P.up.value.copy( C.up.value );
		P.fwd.value.copy( C.fwd.value );
		P.tan.value.copy( C.tan.value );
		this.camDelta.value.set( 0, 0, 0 );
		this.camDeltaHi.value.set( 0, 0, 0 );

	}

	// Restart the view accumulation (camera cuts are also detected on their own), rebuilding it at
	// full rate over the next frames so no seam shows.
	resetHistory() {

		this._rebuild = 0;
		this._since = 0;
		this.historyValid.value = 0;

	}

	_panoSlot( k ) {

		const o = ORDER[ k % 16 ];
		this.panoSlot.value.set( ( o % 4 ) * 2 + ( Math.floor( k / 16 ) % 2 ), Math.floor( o / 4 ) );

	}

	_buildSDF() {

		const d = [ SDF_RES / 8, SDF_RES / 8, SDF_H ];
		for ( const k of [ this.sdfOccKernel, this.sdfXKernel, this.sdfYKernel, this.sdfZKernel ] ) k.dispatchSize = d;
		this.renderer.compute( [ this.sdfOccKernel, this.sdfXKernel, this.sdfYKernel, this.sdfZKernel ] );

	}

	// restart all accumulation (view, panorama)
	invalidate() {

		this.panoWarm = 1;
		this.resetHistory();

	}

	// ------------------------------------------------------------ TSL sampling

	// vec4(rgb in-scattered radiance, a transmittance) for a view direction (panorama)
	sample( dir ) {

		const az = atan( dir.z, dir.x );
		const u = fract( az.div( 2 * PI ) );
		const elev = acos( clamp( dir.y, - 1, 1 ) ).negate().add( PI / 2 );
		const t = clamp( elev.add( ( 4 / 180 ) * PI ).div( ( 94 / 180 ) * PI ), 0, 1 );
		const v = sqrt( t );
		const s = texture( this.panorama, vec2( u, v ) );
		// below the panorama range: no clouds
		const below = smoothstep( - 0.07, - 0.03, dir.y );
		return vec4( s.rgb.mul( below ), mix( float( 1 ), s.a, below ) );

	}

	// Full resolution clouds for the main background. The view texture is looked up with the camera
	// it was traced with (it can lag behind: underwater frames skip the tracing); directions outside
	// it, or a texture without data yet (startup, resize), fall back to the panorama, so the sky can
	// never show empty texels
	sampleView( dir ) {

		const c = this.viewCam;
		const uv = this._project( dir, c );
		const inside = this.viewValid.greaterThan( 0.5 ).and( dot( dir, c.fwd ).greaterThan( 0.01 ) )
			.and( uv.x.greaterThanEqual( 0 ) ).and( uv.x.lessThanEqual( 1 ) ).and( uv.y.greaterThanEqual( 0 ) ).and( uv.y.lessThanEqual( 1 ) );
		const s = vec4( 0 ).toVar();
		If( inside, () => {

			// bicubic (Catmull-Rom) upsampling of the half resolution history keeps the edges crisp
			const v = max( this._catmullRom( ( c ) => this.viewTexNode.sample( c ).level( 0 ), uv, this.viewSize ), vec4( 0 ) );
			const above = smoothstep( - 0.05, - 0.03, dir.y );
			s.assign( vec4( v.rgb.mul( above ), mix( float( 1 ), min( v.a, 1 ), above ) ) );

		} ).Else( () => {

			s.assign( this.sample( dir ) );

		} );
		return s;

	}

	// cloud shadow transmittance (1 = clear) at a world position. Manual bilinear filtering of an
	// unfilterable texture: costs no sampler in the (sampler hungry) scene materials.
	shadow( worldXZ ) {

		const uv = worldXZ.sub( this.shadowCenter ).div( this.shadowSize ).add( 0.5 );
		const st = uv.mul( SHADOW_RES ).sub( 0.5 );
		const i0 = ivec2( floor( st ) );
		const f = fract( st );
		const lo = ivec2( 0 ), hi = ivec2( SHADOW_RES - 1 );
		const tap = ( x, y ) => textureLoad( this.shadowMap, clamp( i0.add( ivec2( x, y ) ), lo, hi ) ).x;
		const s = mix( mix( tap( 0, 0 ), tap( 1, 0 ), f.x ), mix( tap( 0, 1 ), tap( 1, 1 ), f.x ), f.y );
		// no data outside the map: fade to unshadowed at its border
		const e = abs( uv.sub( 0.5 ) );
		const inside = smoothstep( 0.5, 0.42, max( e.x, e.y ) );
		return mix( float( 1 ), s, this.shadowStrength.mul( inside ) );

	}

}
