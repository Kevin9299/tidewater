import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, vec2, vec3, vec4, normalize, dot, reflect, refract, max, min, abs, clamp, saturate,
	mix, pow, exp, sqrt, length, smoothstep, select, If, positionWorld, cameraPosition, frontFacing,
	varyingProperty, screenUV, viewportSharedTexture, viewportDepthTexture, cameraNear, cameraFar,
	perspectiveDepthToViewZ, positionView, cameraViewMatrix, cameraProjectionMatrixInverse, cameraWorldMatrix,
	getViewPosition, fwidth, log2, texture, property, Loop, sin, cos, fract, floor, mx_noise_float,
	mx_worley_noise_float, transformDirection, luminance, cameraProjectionMatrix,
	textureLoad, textureSize, ivec2, Break, int, mrt, Discard,
} from 'three/tsl';
import { G } from '../core/Globals.js';
import { staticVelocity } from '../post/CameraVelocity.js';
import { whaleWater } from './WhaleWater.js';

const IOR = 1.333;

// exact unpolarized dielectric Fresnel, cosI > 0, eta = n2/n1
export const fresnelDielectric = ( cosI, eta ) => {

	const c = clamp( cosI, 0, 1 );
	const g2 = float( eta * eta - 1 ).add( c.mul( c ) );
	const tir = g2.lessThan( 0 );
	const g = sqrt( max( g2, 0 ) );
	const a = g.sub( c ).div( g.add( c ) );
	const b = c.mul( g.add( c ) ).sub( 1 ).div( c.mul( g.sub( c ) ).add( 1 ) );
	return select( tir, float( 1 ), float( 0.5 ).mul( a.mul( a ) ).mul( b.mul( b ).add( 1 ) ) );

};

// view-space position from screen uv and (negative) linear view Z; robust to reversed depth
export const viewPositionFromViewZ = ( uv, viewZ ) => {

	const ndc = vec2( uv.x.mul( 2 ).sub( 1 ), float( 1 ).sub( uv.y ).mul( 2 ).sub( 1 ) );
	const p00 = cameraProjectionMatrix.element( 0 ).element( 0 );
	const p11 = cameraProjectionMatrix.element( 1 ).element( 1 );
	return vec3( ndc.x.div( p00 ), ndc.y.div( p11 ), - 1 ).mul( viewZ.negate() );

};

const D_GGX = ( NdH, a2 ) => {

	const d = NdH.mul( NdH ).mul( a2.sub( 1 ) ).add( 1 );
	return a2.div( d.mul( d ).mul( Math.PI ) );

};

const V_SmithGGX = ( NdL, NdV, a2 ) => {

	const gv = NdL.mul( sqrt( NdV.mul( NdV ).mul( float( 1 ).sub( a2 ) ).add( a2 ) ) );
	const gl = NdV.mul( sqrt( NdL.mul( NdL ).mul( float( 1 ).sub( a2 ) ).add( a2 ) ) );
	return float( 0.5 ).div( max( gv.add( gl ), 1e-5 ) );

};

// Henyey-Greenstein
const phaseHG = ( cosT, g ) => {

	const g2 = g * g;
	return float( ( 1 - g2 ) / ( 4 * Math.PI ) ).div( pow( max( float( 1 + g2 ).sub( cosT.mul( 2 * g ) ), 1e-4 ), 1.5 ) );

};

class WaterLightingModel extends THREE.LightingModel {

	constructor( material ) {

		super();
		this.material = material;

	}

	direct( { lightColor } ) {

		this.material.sunLight.addAssign( lightColor );

	}

	indirect() {}

	finish( builder ) {

		builder.context.outgoingLight.assign( this.material.shade( builder ) );

	}

}

export class WaterMaterial extends THREE.NodeMaterial {

	constructor( { surface, sky, sceneCopy, reflection = null, hullMask = null, hullMaskActive = null } ) {

		super();
		this.isWaterMaterial = true;
		this.lights = true;
		this.side = THREE.DoubleSide;
		// Drawn in its own pass after the opaques (see SceneRenderer); sceneCopy holds the
		// opaque color + depth for refraction and absorption.
		this.transparent = false;
		this.forceSinglePass = true;
		this.blending = THREE.NoBlending;
		this.depthWrite = true;
		this.surface = surface;
		this.sky = sky;
		this.reflection = reflection;

		this.params = {
			absorption: G.waterAbsorption,
			scattering: G.waterScattering,
			backscatter: uniform( 0.035 ).setName( 'wBackscatter' ),
			sss: uniform( 1.0 ).setName( 'wSSS' ),
			refraction: uniform( 0.06 ).setName( 'wRefr' ),
			foamIntensity: uniform( 1.0 ).setName( 'wFoamI' ),
			roughness: uniform( 0.035 ).setName( 'wRough' ),
			reflectionStrength: uniform( 1.0 ).setName( 'wReflS' ),
			ssr: uniform( 1 ).setName( 'wSSR' ), // screen-space reflections on/off
		};

		this.sunLight = property( 'vec3', 'waterSunLight' );

		// camera velocity for TAA, plus the water mask (SceneRenderer) for the underwater pass: whether
		// the visible surface is seen from below
		this.seenFromBelow = property( 'float', 'waterSeenFromBelow' );
		this.mrtNode = mrt( {
			velocity: staticVelocity,
			waterMask: vec4( this.seenFromBelow, 1, 0, 1 ),
		} );

		// opaque scene color/depth copies (made by SceneRenderer right before the water pass)
		this.sceneDepthTexture = sceneCopy.depthTexture;
		this.sceneColorTexture = sceneCopy.texture;
		// depth via exact texel loads (float depth textures + filtering samplers are unreliable)
		this.sceneDepthAt = ( uv ) => {

			const size = textureSize( textureLoad( this.sceneDepthTexture ), 0 );
			const p = ivec2( clamp( uv, 0, 0.9999 ).mul( vec2( size ) ) );
			return textureLoad( this.sceneDepthTexture, p ).x;

		};

		this.sceneDepthNode = this.sceneDepthAt( screenUV );
		// camera distance to the nearest hull-volume surface per pixel (SceneRenderer, 0 = none)
		this.hullMaskTexture = hullMask;
		this.hullMaskActive = hullMaskActive;
		// one binding for every lookup of the opaque color copy (refraction, SSR, edge blend)
		this.sceneColorTex = texture( this.sceneColorTexture );
		this.sceneColorNode = this.sceneColorTex.sample( screenUV );
		this.debugMode = uniform( 0, 'int' ).setName( 'wDebug' );

		this.lagXZ = varyingProperty( 'vec2', 'vLagXZ' );
		this.vHeight = varyingProperty( 'float', 'vWaveH' );
		this.vDepth = varyingProperty( 'float', 'vSeaDepth' );
		this.vFoam = varyingProperty( 'float', 'vFoam' );
		this.vShoreN = varyingProperty( 'vec3', 'vShoreN' );
		this.vShoreFoam = varyingProperty( 'float', 'vShoreFoam' );
		this.vSwash = varyingProperty( 'float', 'vSwash' );
		this.vSurfMask = varyingProperty( 'vec2', 'vSurfMask' );

		this.positionNode = Fn( () => {

			const r = surface.vertex();
			this.lagXZ.assign( r.lagXZ );
			this.vHeight.assign( r.height );
			this.vDepth.assign( r.depth );
			this.vFoam.assign( r.foam );
			this.vShoreN.assign( r.shoreN );
			this.vShoreFoam.assign( r.shoreFoam );
			this.vSwash.assign( r.swash );
			if ( r.surfMask ) this.vSurfMask.assign( r.surfMask );
			return r.position;

		} )();

	}

	setupLightingModel() {

		return new WaterLightingModel( this );

	}

	setupVariants() {

		this.sunLight.assign( vec3( 0 ) );
		this.seenFromBelow.assign( 0 );

	}

	// --------------------------------------------------------------- screen-space reflection

	// March the reflected ray through the opaque depth copy (view space, geometric steps, then a
	// short bisection). Returns { color, weight }: weight fades at screen edges, for rays heading
	// back toward the camera and at the end of the search range.
	ssr( posV, Rv ) {

		const hit = float( 0 ).toVar();
		const hitT = float( 0 ).toVar();
		const t = float( 0.15 ).toVar();
		const dt = float( 0.25 ).toVar();
		const prevT = float( 0 ).toVar();
		const project = ( p ) => {

			const clip = cameraProjectionMatrix.mul( vec4( p, 1 ) );
			const ndc = clip.xy.div( max( clip.w, 1e-4 ) );
			return vec2( ndc.x.mul( 0.5 ).add( 0.5 ), ndc.y.mul( - 0.5 ).add( 0.5 ) );

		};

		const sceneZAt = ( uv ) => perspectiveDepthToViewZ( this.sceneDepthAt( uv ), cameraNear, cameraFar );

		Loop( { start: int( 0 ), end: int( 14 ), type: 'int', condition: '<' }, () => {

			prevT.assign( t );
			t.addAssign( dt );
			dt.mulAssign( 1.5 );
			const p = posV.add( Rv.mul( t ) );
			const uv = project( p );
			If( uv.x.lessThan( 0 ).or( uv.x.greaterThan( 1 ) ).or( uv.y.lessThan( 0 ) ).or( uv.y.greaterThan( 1 ) ).or( p.z.greaterThan( - 0.1 ) ), () => {

				Break();

			} );
			const sz = sceneZAt( uv );
			// behind the visible surface, within a thickness covering the last step
			If( p.z.lessThan( sz ).and( sz.sub( p.z ).lessThan( max( dt.mul( 1.3 ), max( t.mul( 0.08 ), 0.3 ) ) ) ), () => {

				hit.assign( 1 );
				Break();

			} );

		} );

		const color = vec3( 0 ).toVar();
		const weight = float( 0 ).toVar();
		If( hit.greaterThan( 0.5 ), () => {

			// refine between the last miss and the hit
			const a = prevT.toVar(), b = t.toVar();
			Loop( 4, () => {

				const m = a.add( b ).mul( 0.5 );
				const p = posV.add( Rv.mul( m ) );
				const behind = p.z.lessThan( sceneZAt( project( p ) ) );
				b.assign( select( behind, m, b ) );
				a.assign( select( behind, a, m ) );

			} );
			hitT.assign( b );
			const hitV = posV.add( Rv.mul( b ) ).toVar();
			const uv = project( hitV ).toVar();
			// after refining, the ray must really touch the surface there (a ray that only passed far
			// behind a thin or distant object is a false hit)
			const gap = abs( sceneZAt( uv ).sub( hitV.z ) );
			const touch = smoothstep( max( b.mul( 0.04 ), 0.4 ), max( b.mul( 0.02 ), 0.2 ), gap );
			// anything under the surface (seabed seen through the water, submerged hull) is not
			// visible to a reflected ray: those rays run into the next wave instead
			const hitY = cameraWorldMatrix.mul( vec4( hitV, 1 ) ).y;
			color.assign( this.sceneColorTex.sample( uv ).rgb );
			const edge = smoothstep( 0.0, 0.06, uv.x ).mul( smoothstep( 1.0, 0.94, uv.x ) ).mul( smoothstep( 0.0, 0.06, uv.y ) ).mul( smoothstep( 1.0, 0.94, uv.y ) );
			const facing = smoothstep( 0.5, 0.1, Rv.z ); // rays toward the camera leave the screen
			weight.assign( edge.mul( facing ).mul( touch ).mul( smoothstep( 260, 120, hitT ) ).mul( smoothstep( G.seaLevel.sub( 0.15 ), G.seaLevel.add( 0.35 ), hitY ) ) );

		} );

		return { color, weight };

	}

	// --------------------------------------------------------------- shading

	shade() {

		if ( this.cheap ) return vec3( 0.02, 0.05, 0.1 );
		const S = this.surface;
		const P = this.params;
		const sky = this.sky;

		const pos = positionWorld;
		// No sea inside a hull: the surface behind the nearest face of the hull volume is water the hull
		// keeps out (without this it shows through the cockpit sole when the stern squats or the boat heels)
		if ( this.hullMaskTexture && this.hullMaskActive ) {

			If( this.hullMaskActive.greaterThan( 0.5 ).and( frontFacing ), () => {

				const mSize = textureSize( textureLoad( this.hullMaskTexture ), 0 );
				const hullDist = textureLoad( this.hullMaskTexture, ivec2( clamp( screenUV, 0, 0.9999 ).mul( vec2( mSize ) ) ) ).x;
				Discard( hullDist.greaterThan( 0.01 ).and( length( pos.sub( cameraPosition ) ).greaterThan( hullDist.sub( 0.02 ) ) ) );

			} );

		}

		const toCam = cameraPosition.sub( pos );
		const dist = length( toCam );
		const V = toCam.div( dist ).toVar();
		const L = G.sunDir;
		if ( this.clouds ) this.sunLight.mulAssign( this.clouds.shadow( pos.xz ) );
		// the island's own shadow (heightfield horizon): the shadow map's range is too short to hold it
		if ( S.terrain && S.terrain.sunShadowAt ) this.sunLight.mulAssign( S.terrain.sunShadowAt( pos ) );

		// footprint of this pixel on the surface (m) — for filtering / roughness
		const footprint = max( length( fwidth( this.lagXZ ) ), 1e-4 );

		const simState = S.shoreSim ? S.shoreSim.sample( pos.xz ).toVar() : null;
		const surf = S.fragment( this.lagXZ, footprint, this.vDepth, this.vFoam, S.shore ? this.vShoreN : null, S.shore ? this.vShoreFoam : null,
			simState ? simState.x : null, { simState, surfMask: S.shore ? this.vSurfMask : null } );
		const foam = surf.foam.toVar();
		// the whale's churned white water and flat fluke-print slick (WhaleWater.js)
		const whaleW = whaleWater( pos.xz ).toVar();
		foam.assign( max( foam, whaleW.x ) );

		// Which medium is the view ray in before it reaches this fragment? The water surface is a closed
		// interface: a front face (its air side towards the camera) is seen from the air, a back face from
		// the water. For the visible (nearest) fragment this is the medium the ray starts in at the near
		// clip plane, which is exactly how the clip plane slices the water. The winding can't be trusted
		// in folds of the choppy / breaking surface: there, and well above or below the surface, the
		// camera's own medium decides.
		const camH = cameraPosition.y.sub( this.cameraWaterHeightNode || G.cameraWaterHeight );
		const shoreSteep = S.shore ? normalize( this.vShoreN ).y.lessThan( 0.35 ) : null;
		const folded = shoreSteep ? surf.jacobian.lessThan( 0.1 ).or( shoreSteep ) : surf.jacobian.lessThan( 0.1 );
		const nearSurface = abs( camH ).lessThan( 1.5 );
		const viewFromBelow = select( nearSurface.and( folded.not() ), frontFacing.not(), camH.lessThan( 0 ) ).toVar();
		this.seenFromBelow.assign( select( viewFromBelow, float( 1 ), float( 0 ) ) );
		// shading normal on the viewer's side of the interface. Triangle winding can't be trusted
		// (tiny self-intersections of the choppy FFT surface render as back faces seen from above),
		// so pick the side from the camera and bend facets that face away to grazing instead of
		// flipping them (a flipped normal turns a fold into a white sky-mirror patch).
		const Nup = normalize( mix( surf.normal, vec3( 0, 1, 0 ), whaleW.y.mul( 0.75 ) ) );
		const Nside = select( viewFromBelow, Nup.negate(), Nup );
		const Nview = normalize( Nside.add( V.mul( max( dot( Nside, V ).negate().add( 0.03 ), 0 ) ) ) ).toVar();

		// roughness from unresolved slope variance (Cox-Munk: mss = 0.003 + 0.00512 U)
		const mss = float( 0.003 ).add( G.windSpeed.mul( 0.00512 ) ).mul( S.slopeScale );
		const kpx = float( Math.PI ).div( footprint );
		const unresolved = saturate( log2( float( 110 ).div( kpx ) ).div( 9.0 ) );
		const roughVar = surf.rough.mul( surf.rough );
		const alpha2 = P.roughness.mul( P.roughness ).add( mss.mul( 2 ).mul( unresolved ).mul( roughVar ) ).add( foam.mul( 0.2 ) ).add( surf.aeration.mul( 0.03 ) ).mul( float( 1 ).sub( whaleW.y.mul( 0.6 ) ) ).toVar();
		// slope spread the mesh / normal maps can't show at this distance (for the reflection)
		const sigmaUnres = sqrt( mss.mul( unresolved ).mul( roughVar ) ).toVar();

		const out = vec3( 0 ).toVar();
		this._ssrW = float( 0 ).toVar();

		If( viewFromBelow.not(), () => {

			// ================= ABOVE WATER =================
			// water film thickness at this pixel; near the leading edge the surface bends down
			// to meet the sand like a rounded bead (meniscus), tilting the normal toward dry land
			let thickness = S.terrain ? pos.y.sub( S.terrain.heightAt( pos.xz ) ) : float( 10 );
			// the swash sheet ends exactly on its analytic leading edge (ShoreWaves.swashClip), not on the mesh triangles
			if ( S.terrain && S.shore && S.shore.swashClip ) thickness = S.shore.swashClip( pos.xz, thickness );
			const edgeW = float( 1 ).sub( smoothstep( 0.0, 0.006, thickness ) ).toVar();
			const nr = S.terrain ? S.terrain.normalRock( pos.xz ) : vec4( 0 );
			const uphill = normalize( vec2( nr.x, nr.y ).negate().add( vec2( 1e-5, 0 ) ) );
			const N = normalize( Nview.add( vec3( uphill.x, 0, uphill.y ).mul( edgeW.mul( edgeW ).mul( 0.7 ) ) ) ).toVar();
			this._edgeW = edgeW;
			this._thickness = thickness;
			const NdV = max( dot( N, V ), 1e-4 ).toVar();
			const F = fresnelDielectric( NdV, IOR ).toVar();

			// ---- reflection
			const Rraw = reflect( V.negate(), N ).toVar();
			// unresolved facets tilt the average reflection toward the higher, darker sky: rough
			// patches (gusts) darken toward the horizon, slicks stay bright and mirror-like
			const Rup = max( Rraw.y, 0.004 ).add( sigmaUnres.mul( 1.3 ).mul( float( 1 ).sub( max( Rraw.y, 0 ) ) ) );
			const R = normalize( vec3( Rraw.x, Rup, Rraw.z ) ).toVar();
			// reflections pointing below the horizon hit other waves: fade toward a dark sea color
			const skyRefl = sky.reflectionRadiance( R ).toVar();
			const horizonOcc = max( smoothstep( - 0.12, 0.08, Rraw.y ), smoothstep( 0.25, 0.06, thickness ) );
			let reflCol = mix( G.horizonColor.mul( 0.35 ), skyRefl, horizonOcc ).toVar();

			// objects (pier, boat, hills, village) reflected from the screen; only rays close to the
			// horizon can hit anything, so steep reflections skip the march entirely
			// (looking down, F is tiny: the reflection can't be seen, skip the march)
			If( Rraw.y.lessThan( 0.45 ).and( F.greaterThan( 0.05 ) ).and( P.ssr.greaterThan( 0.5 ) ), () => {

				const Rv = transformDirection( Rraw, cameraViewMatrix );
				const r = this.ssr( positionView, Rv );
				reflCol.assign( mix( reflCol, r.color, r.weight ) );
				this._ssrW.assign( r.weight );

			} );

			if ( this.reflection ) {

				// planar reflection of scene objects (alpha = coverage)
				const rOffset = N.xz.mul( 0.8 ).div( max( dist, 1 ) ).mul( 4 );
				const rs = this.reflection.sample( screenUV, rOffset );
				reflCol = mix( reflCol, rs.rgb, rs.a );

			}

			reflCol = reflCol.mul( P.reflectionStrength );

			// ---- sun specular (GGX), sun light already includes shadowing
			const H = normalize( L.add( V ) );
			const NdL = max( dot( N, L ), 0 );
			const NdH = max( dot( N, H ), 0 );
			const VdH = max( dot( V, H ), 0 );
			const Fs = fresnelDielectric( VdH, IOR );
			const spec = D_GGX( NdH, alpha2 ).mul( V_SmithGGX( NdL, NdV, alpha2 ) ).mul( Fs ).mul( NdL );
			// physically the glint is ~1e5x brighter than the sky; clamp to stay inside fp16 range
			const sunSpec = this.sunLight.mul( min( spec, 400 ) );

			// ---- refraction / water volume
			// Trace the refracted view ray (Snell) to the sea floor instead of using the straight
			// screen ray: at grazing angles the straight ray overestimates the water path ~10x.
			// view ray inside the water (unit, downward). Facets of a curling crest can refract it
			// upward on a coarse mesh; keep it heading down into the water body.
			const Tr = refract( V.negate(), N, 1 / IOR );
			const Tv = normalize( vec3( Tr.x, min( Tr.y, - 0.08 ), Tr.z ) ).toVar();
			const tDown = max( Tv.y.negate(), 0.04 );
			const surfViewZ = positionView.z;

			// water column below the surface along the refracted ray (terrain, 2 refinements)
			const L0 = S.terrain ? max( pos.y.sub( S.terrain.heightAt( pos.xz ) ), 0 ).div( tDown ) : float( 400 );
			const L1 = S.terrain ? max( pos.y.sub( S.terrain.heightAt( pos.xz.add( Tv.xz.mul( min( L0, 200 ) ) ) ) ), 0 ).div( tDown ) : L0;
			const Lt = S.terrain ? max( pos.y.sub( S.terrain.heightAt( pos.xz.add( Tv.xz.mul( min( L1.mul( 0.5 ).add( L0.mul( 0.5 ) ), 200 ) ) ) ) ), 0 ).div( tDown ) : L1;
			const Lter = clamp( Lt, 0, 400 ).toVar();
			// thin breaking crests: the refracted ray leaves through the back of the wave into the sky
			const crestT = S.shore && S.shore.crestPath ? S.shore.crestPath( this.lagXZ, this.vDepth, Tv ) : float( 1e4 );
			const thruCrest = crestT.lessThan( Lter );

			// project the refracted end point to the screen
			const pEnd = pos.add( Tv.mul( min( Lter, 80 ) ) );
			const clipEnd = cameraProjectionMatrix.mul( cameraViewMatrix.mul( vec4( pEnd, 1 ) ) );
			const ndcEnd = clipEnd.xy.div( max( clipEnd.w, 1e-4 ) );
			const uvR = vec2( ndcEnd.x.mul( 0.5 ).add( 0.5 ), ndcEnd.y.mul( - 0.5 ).add( 0.5 ) ).toVar();
			const dR = this.sceneDepthAt( uvR );
			const sceneZR = perspectiveDepthToViewZ( dR, cameraNear, cameraFar );
			// the refracted sample must lie behind the water surface (else something above water is in the way)
			const valid = surfViewZ.sub( sceneZR ).greaterThan( 0.05 ).and( uvR.x.greaterThan( 0 ) ).and( uvR.x.lessThan( 1 ) ).and( uvR.y.greaterThan( 0 ) ).and( uvR.y.lessThan( 1 ) );
			const uvF = select( valid, uvR, screenUV ).toVar();
			const sceneCol = select( thruCrest, sky.reflectionRadiance( normalize( vec3( Tv.x, max( Tv.y.abs(), 0.03 ), Tv.z ) ) ), this.sceneColorTex.sample( uvF ).rgb );

			// objects in front of the sea floor (pylons, rocks, reef) shorten the path
			const qView = viewPositionFromViewZ( uvF, perspectiveDepthToViewZ( select( valid, dR, this.sceneDepthNode ), cameraNear, cameraFar ) );
			const qDist = length( qView.sub( positionView ) );
			const pathLen = clamp( min( Lter, qDist ), 0, 400 ).toVar();
			pathLen.assign( min( pathLen, crestT ) );
			this._dbgPath = pathLen;
			this._dbgScene = sceneCol;

			// surf zone: sand and bubbles stirred up by the breakers (see ShoreWaves.surfMedium)
			const surfMed = S.shore && S.shore.surfMedium ? S.shore.surfMedium( pos.xz, this.vDepth ) : null;
			// bubbles mixed into the water (the surf behind breakers, wakes): a strong scatterer, the water
			// turns milky turquoise and the bottom disappears (WaterSurface.fragment aeration)
			const aer = surf.aeration.toVar();
			// sand stirred up where the bores have just passed (the foam they left marks that water):
			// clouds of sediment, not a uniform tint
			const sandK = simState ? saturate( simState.x.mul( 2.5 ) ).mul( 1.8 ).add( 0.45 ) : float( 1 );
			const sigA = surfMed ? P.absorption.add( surfMed.absorb.mul( sandK ) ) : P.absorption;
			// (bubble plumes are shallow and patchy: a moderate scatterer, milky turquoise rather than a glow)
			const sigS = ( surfMed ? P.scattering.add( surfMed.scatter.mul( sandK ) ) : P.scattering ).add( aer.mul( 1.6 ) );
			const sigT = sigA.add( sigS );

			// refracted sun direction
			const Ls = refract( L.negate(), vec3( 0, 1, 0 ), 1 / IOR ).negate(); // toward the sun from underwater
			const muS = max( Ls.y, 0.1 );
			const muV = max( Tv.y.negate(), 0.15 );

			const Tview = exp( sigT.mul( pathLen ).negate() );

			// in-scattered light along the view ray (single scattering sun + ambient), analytic
			// light at depth z: E0 * exp(-sigT * z / mu). Along the view ray z = s * muV.
			const sunIn = this.sunLight.mul( float( 1 ).sub( fresnelDielectric( max( L.y, 0.02 ), IOR ) ) );
			const kSun = sigT.mul( float( 1 ).add( muV.div( muS ) ) );
			const kAmb = sigT.mul( float( 1 ).add( muV.div( 0.75 ) ) );
			const cosPh = dot( Tv, Ls );
			const phase = phaseHG( cosPh, 0.86 ).mul( 0.7 ).add( 0.3 / ( 4 * Math.PI ) );
			const bb = sigS.mul( mix( P.backscatter, float( 0.06 ), saturate( aer.mul( 2 ) ) ) );
			// multiple-scattering boosted backscatter (Gordon R = 0.33 bb/(a+bb))
			const albedoMS = bb.mul( 0.33 * 4 ).div( sigA.add( bb ) );
			const inSun = sunIn.mul( sigS.mul( phase ).add( albedoMS.mul( sigT ).mul( 1 / Math.PI ) ) )
				.mul( float( 1 ).sub( exp( kSun.mul( pathLen ).negate() ) ) ).div( kSun );
			const inAmb = G.skyIrradiance.mul( sigS.mul( 0.25 ).add( albedoMS.mul( sigT ) ) )
				.mul( float( 1 ).sub( exp( kAmb.mul( pathLen ).negate() ) ) ).div( kAmb );

			// crest translucency (sun shining through thin wave tips)
			const vH = normalize( vec2( V.x, V.z ) );
			const lH = normalize( vec2( L.x, L.z ).add( 1e-5 ) );
			// (light entering the top and back of a thin crest scatters out of the face over a broad lobe:
			// side-lit waves glow green too, not only when looking straight into the sun)
			const back = pow( saturate( dot( vH, lH.negate() ).mul( 0.6 ).add( 0.4 ) ), 2.5 );
			const crest = saturate( this.vHeight.mul( 0.9 ).add( 0.1 ) ).mul( saturate( float( 1 ).sub( N.y ).mul( 4 ) ).add( 0.25 ) );
			const sssCol = vec3( 0.12, 0.55, 0.45 ).mul( 0.06 );
			const sss = this.sunLight.mul( sssCol ).mul( back ).mul( crest ).mul( P.sss ).mul( smoothstep( 0.0, 0.25, L.y ) );

			const transmitted = sceneCol.mul( Tview ).add( inSun ).add( inAmb ).add( sss );


			// ---- foam
			// foam: bright diffuse scatterer (albedo ~0.85), wrapped sun + sky irradiance (skyIrradiance = E/PI)
			const foamLit = surf.foamInfo ? surf.foamInfo.light( { N, L, V, sun: this.sunLight } ) : this.sunLight.mul( max( dot( N, L ), 0 ).mul( 0.75 ).add( 0.25 ) ).div( Math.PI ).add( G.skyIrradiance.mul( 0.95 ) ).mul( 0.85 );
			const foamCol = foamLit.mul( P.foamIntensity );

			const water = mix( transmitted, reflCol, F ).add( sunSpec );
			const shaded = mix( water, foamCol.add( sunSpec.mul( 0.05 ) ), saturate( foam ) );
			// fade into the sand right at the leading edge (anti-aliased by the film thickness)
			const edgeAA = smoothstep( 0.0, max( fwidth( thickness ).mul( 1.5 ), 0.004 ), thickness );
			out.assign( mix( this.sceneColorTex.sample( screenUV ).rgb, shaded, edgeAA ) );

		} ).Else( () => {

			// ================= BELOW WATER (looking up at the surface) =================
			const N = Nview;
			const NdV = max( dot( N, V ), 1e-4 );
			// from water (n=1.333) into air: eta = 1/1.333
			const F = fresnelDielectric( NdV, 1 / IOR );
			const T = refract( V.negate(), N, IOR );
			const tValid = dot( T, T ).greaterThan( 0.5 );
			const Td = normalize( select( tValid, T, vec3( 0, 1, 0 ) ) );
			// sky through Snell's window; the sun disk is bounded so grazing refractions of it far
			// away cannot bloom through the fog
			const skyT = min( sky.radianceWithClouds( Td, true ), vec3( 60 ) );

			// total internal reflection mirrors the lit water body below: the radiance of an
			// infinitely long view ray through the medium in the reflected direction
			const sigA = P.absorption, sigS = P.scattering, sigT = sigA.add( sigS );
			const bb = sigS.mul( P.backscatter );
			const albedoMS = bb.mul( 0.33 * 4 ).div( sigA.add( bb ) );
			const Rr = reflect( V.negate(), N );
			const LsU = refract( L.negate(), vec3( 0, 1, 0 ), 1 / IOR ).negate();
			const muU = max( LsU.y, 0.15 );
			const phR = phaseHG( dot( Rr, LsU ), 0.86 ).mul( 0.7 ).add( 0.3 / ( 4 * Math.PI ) );
			const kS = sigT.mul( float( 1 ).sub( min( Rr.y, 0 ).div( muU ) ) );
			const kA = sigT.mul( float( 1 ).sub( min( Rr.y, 0 ).div( 0.8 ) ) );
			const eSunU = this.sunLight.mul( float( 1 ).sub( fresnelDielectric( max( L.y, 0.02 ), IOR ) ) );
			const deepCol = eSunU.mul( sigS.mul( phR ).add( albedoMS.mul( sigT ).mul( 1 / Math.PI ) ) ).div( kS )
				.add( G.skyIrradiance.mul( Math.PI ).mul( sigS.mul( 1 / ( 4 * Math.PI ) ).add( albedoMS.mul( sigT ).mul( 1 / Math.PI ) ) ).div( kA ) );

			// objects above the water seen through Snell's window (from the viewport)
			const sceneZ = perspectiveDepthToViewZ( this.sceneDepthNode, cameraNear, cameraFar );
			const hasObj = positionView.z.sub( sceneZ ).greaterThan( 0 ).and( sceneZ.greaterThan( cameraFar.negate().mul( 0.9 ) ) );
			const objCol = this.sceneColorNode.rgb;
			const transmitted = select( hasObj, objCol, skyT );

			const foamUnder = G.skyIrradiance.add( this.sunLight.mul( 0.5 ) ).mul( 0.25 );
			const col = mix( transmitted.mul( float( 1 ).sub( F ) ).add( deepCol.mul( F ) ), foamUnder, saturate( foam ).mul( 0.7 ) );
			out.assign( col );

		} );

		// debug views: 1 = back faces red, 2 = normals, 3 = foam
		const dbg = this.debugMode;
		const res = min( out, vec3( 16000 ) ).toVar();
		If( dbg.equal( 1 ), () => {

			res.assign( select( frontFacing, res, vec3( 50, 0, 0 ) ) );

		} ).ElseIf( dbg.equal( 2 ), () => {

			res.assign( Nview.mul( 0.5 ).add( 0.5 ) );

		} ).ElseIf( dbg.equal( 3 ), () => {

			res.assign( vec3( foam ) );

		} ).ElseIf( dbg.equal( 4 ), () => {

			const nanN = Nup.x.notEqual( Nup.x ).or( Nup.y.notEqual( Nup.y ) ).or( Nup.z.notEqual( Nup.z ) );
			const nanL = this.lagXZ.x.notEqual( this.lagXZ.x ).or( this.lagXZ.y.notEqual( this.lagXZ.y ) );
			const big = length( surf.slopes ).greaterThan( 4 );
			res.assign( vec3( select( nanN, float( 1 ), float( 0 ) ), select( nanL, float( 1 ), float( 0 ) ), select( big, float( 1 ), float( 0 ) ) ).add( 0.05 ) );

		} ).ElseIf( dbg.equal( 10 ), () => {

			res.assign( vec3( this._ssrW ) );

		} ).ElseIf( dbg.equal( 6 ), () => {

			res.assign( vec3( this._dbgPath.mul( 0.02 ), 0, 0 ) );

		} ).ElseIf( dbg.equal( 9 ), () => {

			const dz = perspectiveDepthToViewZ( this.sceneDepthNode, cameraNear, cameraFar );
			res.assign( vec3( this.sceneDepthNode.mul( 100 ), dz.negate().mul( 0.02 ), positionView.z.negate().mul( 0.02 ) ) );

		} ).ElseIf( dbg.equal( 8 ), () => {

			res.assign( vec3( 0, this.vDepth.mul( 0.02 ), 0 ) );

		} ).ElseIf( dbg.equal( 7 ), () => {

			res.assign( this._dbgScene );

		} ).ElseIf( dbg.equal( 11 ), () => {

			// surf foam sources: whitewater of the breaking wave (r), foam carried by the shore sim (g), clear plunging face (b)
			res.assign( vec3( this.vShoreFoam, simState ? simState.x : float( 0 ), this.vSurfMask.x ) );

		} ).ElseIf( dbg.equal( 5 ), () => {

			res.assign( vec3( fract( this.lagXZ.x.mul( 0.1 ) ), fract( this.vHeight ), fract( this.lagXZ.y.mul( 0.1 ) ) ) );

		} );
		return res;

	}

}
