import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, vec2, vec3, vec4, ivec2, texture, textureLoad, textureSize, screenUV, cameraPosition,
	cameraWorldMatrix, cameraProjectionMatrix, cameraNear, cameraFar, perspectiveDepthToViewZ, normalize, length,
	dot, exp, max, min, abs, clamp, mix, smoothstep, saturate, select, refract, pow, Loop, fract, sin,
	If, sqrt, int, uint, screenCoordinate, frameId,
} from 'three/tsl';
import { G } from '../core/Globals.js';
import { FLASH, spotProfile } from '../materials/LocalLights.js';

const IOR = 1.333;

// Everything that happens when the camera is (partly) underwater:
//  - the medium at the lens, per pixel. The lens is the near clip plane: the water between the eye
//    and the lens is clipped away and the view starts at the lens, so where the surface crosses the
//    near plane the view splits into an above-water and an underwater part (mediumNode)
//  - participating medium: Beer-Lambert absorption + single scattering of depth-attenuated
//    sun and sky light (analytic), with ray-marched caustic light shafts near the camera
//  - the meniscus band along the waterline on the lens: refraction through a rounded water edge,
//    a dark contact line and a bright rim
// m: closer than this to the surface the near plane can cross it (farther, the whole view is in the
// camera's medium)
export const LENS_REACH = 0.35;
const STRADDLE = LENS_REACH;
const BAND = 16; // px: widest meniscus band searched

export class Underwater {

	constructor( { depthTexture, maskTexture, query, caustics, fft } ) {

		this.depthTexture = depthTexture;
		this.maskTexture = maskTexture;
		this.query = query;
		this.caustics = caustics;
		this.fft = fft;
		this.lensDistance = uniform( 0.1 ).setName( 'uwLens' ); // = camera.near (the lens is the near clip plane)
		this.shafts = uniform( 1.0 ).setName( 'uwShafts' );
		this.band = uniform( 9 ).setName( 'uwBand' ); // meniscus half-width in pixels (<= BAND)
		this.mediumTexture = null; // render target of mediumNode(), set by the post chain
		this.enabled = uniform( 1 ).setName( 'uwOn' );
		this.torchBeam = uniform( 3 ).setName( 'uwTorchBeam' ); // flashlight beam in-scatter strength

		// explicit scene-camera uniforms: inside post passes the built-in camera nodes refer to
		// the full-screen quad camera
		this.camPos = uniform( new THREE.Vector3() ).setName( 'uwCamPos' );
		this.camWorld = uniform( new THREE.Matrix4() ).setName( 'uwCamWorld' );
		this.proj = uniform( new THREE.Vector4( 1, 1, 0.1, 1000 ) ).setName( 'uwProj' ); // p00, p11, near, far

	}

	updateCamera( camera ) {

		camera.updateMatrixWorld();
		this.camPos.value.setFromMatrixPosition( camera.matrixWorld );
		this.camWorld.value.copy( camera.matrixWorld );
		const e = camera.projectionMatrix.elements;
		this.proj.value.set( e[ 0 ], e[ 5 ], camera.near, camera.far );
		this.lensDistance.value = camera.near;

	}

	// reversed-Z depth -> view Z (negative)
	viewZ( d ) {

		const n = this.proj.z, f = this.proj.w;
		return n.mul( f ).div( n.sub( f ).mul( d ).sub( n ) );

	}

	// view-space ray for screen uv (unnormalized, z = -1)
	viewRay( uv ) {

		const ndc = vec2( uv.x.mul( 2 ).sub( 1 ), float( 1 ).sub( uv.y ).mul( 2 ).sub( 1 ) );
		return vec3( ndc.x.div( this.proj.x ), ndc.y.div( this.proj.y ), - 1 );

	}

	worldDir( uv ) {

		return normalize( this.camWorld.mul( vec4( this.viewRay( uv ), 0 ) ).xyz );

	}

	// Medium at the near clip plane for each pixel (r = 1: water, 0: air).
	//  - a water surface in view: the side it is seen from (written by the water material) is the
	//    medium between the lens and the surface
	//  - no surface in view: the ray never crosses the surface beyond the lens, so it is in the medium
	//    of its far end: an object hit below the surface is in water; the sky is air (a downward ray
	//    that reaches nothing is in water)
	// The waterline on the lens is then exactly where the rendered surface crosses the near plane:
	// the surface fragments on one side end right there, and the far ends on the other side agree.
	// Farther than STRADDLE from the surface the near plane can't reach it: the camera's medium.
	mediumNode() {

		return Fn( () => {

			const uv = screenUV;
			const size = textureSize( textureLoad( this.maskTexture ), 0 );
			const px = ivec2( clamp( uv, 0, 0.9999 ).mul( vec2( size ) ) ).toVar();
			const m = textureLoad( this.maskTexture, px );
			const camH = this.camPos.y.sub( this.query.cameraState().x );
			const water = select( camH.lessThan( 0 ), float( 1 ), float( 0 ) ).toVar();
			If( m.y.greaterThan( 0.5 ), () => {

				water.assign( m.x );

			} ).ElseIf( abs( camH ).lessThan( STRADDLE ), () => {

				const d = textureLoad( this.depthTexture, px ).x;
				const ray = this.viewRay( uv );
				const dirW = this.camWorld.mul( vec4( ray, 0 ) ).xyz;
				If( d.lessThan( 1e-7 ), () => {

					water.assign( select( dirW.y.lessThan( 0 ), float( 1 ), float( 0 ) ) );

				} ).Else( () => {

					const q = this.camPos.add( dirW.mul( this.viewZ( d ).negate() ) );
					water.assign( select( q.y.lessThan( this.query.heightAtNode( q.xz ) ), float( 1 ), float( 0 ) ) );

				} );

			} );
			return vec4( water, 0, 0, 1 );

		} )();

	}

	// ---------------------------------------------------------------- post node

	node( colorTexNode ) {

		return Fn( () => {

			const uv = screenUV;
			const med = this.mediumTexture;
			const mSize = ivec2( med.size() );
			const pc = ivec2( uv.mul( vec2( mSize ) ) ).clamp( ivec2( 0 ), mSize.sub( 1 ) ).toVar();
			const at = ( dy ) => med.load( ivec2( pc.x, pc.y.add( dy ).clamp( 0, mSize.y.sub( 1 ) ) ) ).r;
			const here = at( 0 ).toVar();
			const under = here.greaterThan( 0.5 );

			// distance to the waterline on the lens in pixels, searched vertically (the line runs roughly
			// across the screen: no camera roll, and over the few cm of the lens the surface is a plane)
			const lineDist = float( 1e4 ).toVar();
			const lineDir = float( 0 ).toVar(); // uv.y direction toward the other medium
			const camH = this.camPos.y.sub( this.query.cameraState().x );
			If( abs( camH ).lessThan( STRADDLE ), () => {

				const flips = abs( at( - 8 ).sub( here ) ).add( abs( at( 8 ).sub( here ) ) ).add( abs( at( - BAND ).sub( here ) ) ).add( abs( at( BAND ).sub( here ) ) );
				If( flips.greaterThan( 0.5 ), () => {

					Loop( { start: int( 1 ), end: int( BAND + 1 ), type: 'int', condition: '<' }, ( { i } ) => {

						const up = abs( at( i.negate() ).sub( here ) ).greaterThan( 0.5 );
						const down = abs( at( i ).sub( here ) ).greaterThan( 0.5 );
						If( lineDist.greaterThan( 1e3 ).and( up.or( down ) ), () => {

							lineDist.assign( float( i ).sub( 0.5 ) );
							lineDir.assign( select( up, float( - 1 ), float( 1 ) ) );

						} );

					} );

				} );

			} );

			// ---- meniscus band: rounded water edge acting like a cylindrical lens; samples are pushed
			// away from the line on both sides
			const bandW = this.band;
			const tBand = clamp( lineDist.div( bandW ), 0, 1 );
			const bendPx = tBand.mul( sqrt( max( float( 1 ).sub( tBand.mul( tBand ) ), 0 ) ) ).mul( bandW ).mul( 0.8 );
			const uvB = uv.sub( vec2( 0, lineDir.mul( bendPx ).div( float( mSize.y ) ) ) );
			const base = colorTexNode.sample( uvB ).rgb.toVar();

			const result = base.toVar();

			If( under.and( this.enabled.greaterThan( 0.5 ) ), () => {

				// scene distance along this pixel
				const size = textureSize( textureLoad( this.depthTexture ), 0 );
				const d = textureLoad( this.depthTexture, ivec2( clamp( uv, 0, 0.9999 ).mul( vec2( size ) ) ) ).x;
				const vz = this.viewZ( d );
				const ray = this.viewRay( uv );
				// the water between the eye and the lens is clipped away: the medium starts at the lens
				const dist = max( min( length( ray.mul( vz.negate() ) ), 600 ).sub( this.lensDistance.mul( length( ray ) ) ), 0 ).toVar();
				const dir = this.worldDir( uv );

				const st = this.query.cameraState();
				const zc = max( st.x.sub( this.camPos.y ), 0 ); // camera depth below the surface
				const sigA = G.waterAbsorption;
				const sigS = G.waterScattering;
				const sigT = sigA.add( sigS );

				const Ls = refract( G.sunDir.negate(), vec3( 0, 1, 0 ), 1 / IOR ).negate(); // toward the sun, underwater
				const mu = max( Ls.y, 0.15 );
				const cosPh = dot( dir, Ls );
				const g = 0.85;
				const phase = float( ( 1 - g * g ) / ( 4 * Math.PI ) ).div( pow( max( float( 1 + g * g ).sub( cosPh.mul( 2 * g ) ), 1e-4 ), 1.5 ) ).mul( 0.75 ).add( 0.25 / ( 4 * Math.PI ) );

				// light arriving at depth z: E0 exp(-sigT z / m); along the ray z(s) = zc - dir.y s, seen through
				// exp(-sigT s). The in-scatter integral is (exp(-a) - exp(-(a + k d))) / k with a = sigT zc / m
				// and k = sigT (1 - dir.y / m): both exponents stay <= 0 while the ray is in the water, so it
				// can't overflow looking up through deep water (exp(-a) (1 - exp(-k d)) / k does, and the
				// Inf / NaN would stick in the TAA history)
				const sunE = G.sunColor.mul( 0.96 );
				const ambE = G.skyIrradiance.mul( Math.PI ).mul( 0.9 );
				const lit = ( m ) => {

					const a = sigT.mul( zc ).div( m );
					const kk = sigT.mul( float( 1 ).sub( dir.y.div( m ) ) );
					const e0 = exp( a.negate() );
					const e1 = exp( max( a.add( kk.mul( dist ) ), 0 ).negate() );
					return select( abs( kk ).lessThan( 1e-4 ), e0.mul( dist ), e0.sub( e1 ).div( kk ) );

				};

				const bb = sigS.mul( 0.035 );
				const msAlb = bb.mul( 1.3 ).div( sigA.add( bb ) );
				const inSun = sunE.mul( sigS.mul( phase ).add( msAlb.mul( sigT ).mul( 1 / Math.PI ) ) ).mul( lit( mu ) );
				const inAmb = ambE.mul( sigS.mul( 1 / ( 4 * Math.PI ) ).add( msAlb.mul( sigT ).mul( 1 / Math.PI ) ) ).mul( lit( float( 0.8 ) ) );

				// light shafts: march the first metres and modulate the sun in-scatter by caustics
				const shafts = vec3( 0 ).toVar();
				if ( this.caustics ) {

					const steps = 20;
					const maxD = min( dist, 22 );
					const ds = maxD.div( steps );
					// interleaved gradient noise on the real pixel grid, moved every frame (Jimenez 2014) so the
					// temporal resolve integrates the steps instead of freezing a noise pattern on screen
					const fragPx = screenCoordinate.xy.add( float( frameId.mod( uint( 64 ) ) ).mul( 5.588238 ) );
					const jitter = fract( fract( dot( fragPx, vec2( 0.06711056, 0.00583715 ) ) ).mul( 52.9829189 ) );
					Loop( steps, ( { i } ) => {

						const s = float( i ).add( jitter ).mul( ds );
						const p = this.camPos.add( dir.mul( s ) );
						const z = max( st.x.sub( p.y ), 0.01 );
						const caus = this.caustics.sample( p, z, 1.5 );
						const Tl = exp( sigT.mul( s.add( z.div( mu ) ) ).negate() );
						shafts.addAssign( caus.sub( 1 ).mul( Tl ).mul( ds ) );

					} );

					shafts.assign( shafts.mul( sunE ).mul( sigS ).mul( phase ).mul( this.shafts ).mul( 2.5 ) );

				}

				// diver's torch: single scattering of the flashlight cone along the view ray (the torch sits
				// beside the eye, so the beam is a shaft slightly off the view axis), extinction on the light
				// and the view legs, same phase function, up to the scene depth
				const torch = vec3( 0 ).toVar();
				If( FLASH.on.greaterThan( 0.5 ), () => {

					const tSteps = 8;
					const tMax = min( dist, 25 );
					const tds = tMax.div( tSteps );
					const tPx = screenCoordinate.xy.add( float( frameId.mod( uint( 64 ) ) ).mul( 5.588238 ) ).add( 17.3 );
					const tJit = fract( fract( dot( tPx, vec2( 0.06711056, 0.00583715 ) ) ).mul( 52.9829189 ) );
					Loop( tSteps, ( { i } ) => {

						const s = float( i ).add( tJit ).mul( tds );
						const v = this.camPos.add( dir.mul( s ) ).sub( FLASH.pos );
						const r2 = max( dot( v, v ), 1e-4 );
						const r = sqrt( r2 );
						const Lr = v.div( r );
						const spot = spotProfile( dot( Lr, FLASH.dir ), FLASH.cone.x, FLASH.cone.y );
						// same lobe as the sun in-scatter (0.75 forward g = 0.85 + 0.25 isotropic), Schlick's form
						const sk = 1.55 * g - 0.55 * g * g * g;
						const sd = float( 1 ).sub( dot( dir, Lr.negate() ).mul( sk ) );
						const ph = float( ( 1 - sk * sk ) / ( 4 * Math.PI ) * 0.75 ).div( sd.mul( sd ) ).add( 0.25 / ( 4 * Math.PI ) );
						torch.addAssign( exp( sigT.mul( r.add( s ) ).negate() ).mul( spot.mul( ph ).div( r2.add( 0.15 ) ) ) );

					} );
					// x3: the suspended particles scatter more than the clear-water coefficient (the beam reads)
					torch.assign( torch.mul( FLASH.col ).mul( sigS ).mul( tds ).mul( this.torchBeam ) );

				} );

				const T = exp( sigT.mul( dist ).negate() );
				result.assign( base.mul( T ).add( inSun ).add( inAmb ).add( max( shafts, vec3( 0 ) ) ).add( torch ) );

			} );

			// meniscus contact line and bright rim
			const line = smoothstep( 2.0, 0.0, lineDist );
			const rim = smoothstep( bandW, bandW.mul( 0.4 ), lineDist ).mul( smoothstep( 0.5, 2.5, lineDist ) );
			const withLine = result.mul( float( 1 ).sub( line.mul( 0.55 ) ) ).add( rim.mul( 0.15 ).mul( G.skyIrradiance.mul( 2.5 ).add( G.sunColor.mul( 0.02 ) ) ) );

			return vec4( withLine, 1 );

		} )();

	}

}
