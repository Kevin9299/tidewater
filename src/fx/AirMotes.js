import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, vec2, vec3, vec4, ivec2, instanceIndex, positionGeometry, cameraPosition, cameraViewMatrix,
	cameraProjectionMatrix, screenSize, fract, sin, cos, dot, floor, exp, pow, abs, smoothstep, length, max, mix,
	normalize, cross, select, saturate, texture, varyingProperty, mrt, If, Discard,
} from 'three/tsl';
import { G } from '../core/Globals.js';
import { LAYERS } from '../core/SceneRenderer.js';
import { staticVelocity } from '../post/CameraVelocity.js';

const hash3 = ( n ) => fract( sin( vec3( n, n.add( 17.13 ), n.add( 43.71 ) ) ).mul( vec3( 43758.5453, 22578.1459, 19642.3490 ) ) );

// Henyey-Greenstein phase (1/sr)
const phaseHG = ( cosT, g ) => float( ( 1 - g * g ) / ( 4 * Math.PI ) ).div( pow( max( float( 1 + g * g ).sub( cosT.mul( 2 * g ) ), 1e-4 ), 1.5 ) );

const GNAT_SWARMS = 4, GNATS_PER_SWARM = 12, SEEDS = 48;
const VISIBILITY = 8;

// Life in the air around the camera: dust and pollen motes, fine salt aerosol near the surf, a few
// drifting seed tufts and, rarely, a loose swarm of gnats hovering over the vegetation. Like
// MarineSnow the positions are procedural (hash of the instance) in a box that wraps around the
// camera: no simulation, no CPU work beyond a few uniforms. Everything drifts with the wind plus a
// slow per-particle swirl; the gnats hold station in small swarms and dart around in loops.
// The specks are far smaller than a pixel: each is drawn at least ~1.2 px wide with its opacity
// scaled by the coverage of the real particle, so it neither flickers under the TAA jitter nor turns
// into a blob. They are lit per particle (vertex stage) by the sun through a forward-scattering
// phase function and the sun's visibility at the particle (cloud shadow, hill shadow and the near
// shadow cascade, so motes in a palm's shade stay dark): nearly invisible most of the time, they
// light up when backlit in the sun against a dark background. Drawn premultiplied in the late pass;
// the core of each speck writes depth and its own motion vector (wind + swirl), so the temporal
// resolve follows it instead of clipping it away. Cost: one instanced draw, ~0.02 ms.
export class AirMotes {

	constructor( { terrain, clouds = null, csm = null, reversedDepth = true, count = 3000, box = 12 } ) {

		this.terrain = terrain;
		this.csm = csm;
		this.box = box;
		this.camPos = uniform( new THREE.Vector3() ).setName( 'airCam' );
		this.drift = uniform( new THREE.Vector2() ).setName( 'airDrift' ); // wind drift, wrapped to the box (m)
		this.intensity = uniform( 1 ).setName( 'airIntensity' );
		this.windScale = 0.25; // near the ground, among the plants: a fraction of the 10 m wind
		// near shadow cascade (0-10 m): world -> shadow map, and its depth map (bound once it exists)
		// (read when the motes are drawn: the cascade is refitted when its map renders, earlier in the frame)
		this.csmMatrix = uniform( new THREE.Matrix4() ).setName( 'airCsmMatrix' ).onObjectUpdate( () => {

			const l = this.csm && this.csm.lights && this.csm.lights[ 0 ];
			if ( l && l.shadow ) this.csmMatrix.value.copy( l.shadow.matrix );

		} );
		this.csmActive = uniform( 0 ).setName( 'airCsmActive' );
		this.csmSize = uniform( new THREE.Vector2( 1, 1 ) ).setName( 'airCsmSize' );
		const placeholder = new THREE.DepthTexture( 1, 1 );
		this.csmDepth = texture( placeholder );

		const geo = new THREE.PlaneGeometry( 2, 2 );
		const inst = new THREE.InstancedBufferGeometry();
		inst.index = geo.index;
		inst.setAttribute( 'position', geo.getAttribute( 'position' ) );
		inst.instanceCount = count;
		inst.boundingSphere = new THREE.Sphere( new THREE.Vector3(), 1e7 );

		const mat = new THREE.NodeMaterial();
		mat.name = 'AirMotes';
		mat.transparent = true;
		// the core of each speck writes depth: the temporal resolve dilates its motion vectors by depth,
		// so the speck's own motion is used around it and its history follows it (instead of being
		// clipped away as it moves over a background with other motion)
		mat.depthWrite = true;
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

		const vCol = varyingProperty( 'vec4', 'vAirCol' ); // radiance, opacity
		const vUV = varyingProperty( 'vec2', 'vAirUV' );
		const vMotion = varyingProperty( 'vec2', 'vAirMotion' ); // own motion over the last frame (NDC)

		const B = float( box );
		const NG = GNAT_SWARMS * GNATS_PER_SWARM;
		const L = G.sunDir;

		// sun visibility in the near shadow cascade (one depth load; outside the cascade: lit)
		const csmVisibility = ( p ) => {

			const sc = this.csmMatrix.mul( vec4( p, 1 ) );
			const c = sc.xyz.div( sc.w );
			const suv = vec2( c.x, c.y.oneMinus() );
			const inside = suv.x.greaterThan( 0.001 ).and( suv.x.lessThan( 0.999 ) ).and( suv.y.greaterThan( 0.001 ) ).and( suv.y.lessThan( 0.999 ) );
			const d = this.csmDepth.load( ivec2( suv.clamp( 0, 0.999 ).mul( this.csmSize ) ) ).x;
			// reversed depth: closer to the light = larger
			const lit = reversedDepth ? select( c.z.add( 2e-5 ).greaterThanEqual( d ), float( 1 ), float( 0 ) ) : select( c.z.sub( 2e-5 ).lessThanEqual( d ), float( 1 ), float( 0 ) );
			return mix( float( 1 ), lit, select( inside, this.csmActive, float( 0 ) ) );

		};

		mat.positionNode = Fn( () => {

			const id = float( instanceIndex );
			const h = hash3( id.mul( 0.7131 ).add( 0.37 ) );
			const h2 = hash3( id.mul( 1.3917 ).add( 5.1 ) );
			const h3 = hash3( id.mul( 2.1733 ).add( 11.7 ) );
			const isGnat = id.lessThan( NG );
			const isSeed = id.greaterThanEqual( NG ).and( id.lessThan( NG + SEEDS ) );
			const t = G.time;

			// ---- where: wrapped around the camera. Gnats: a swarm home fixed in the world (no drift)
			const sid = floor( id.div( GNATS_PER_SWARM ) );
			const hs = hash3( sid.mul( 3.917 ).add( 1.3 ) );
			const home = select( isGnat, hs.xz, h.xz );
			const drift = select( isGnat, vec2( 0 ), this.drift );
			const local = fract( home.add( drift.sub( this.camPos.xz ).div( B ) ) ).sub( 0.5 ).mul( B ).toVar();
			const xz0 = this.camPos.xz.add( local ).toVar();
			const ht = terrain.heightAt( xz0 ).toVar();
			const hA = ht.sub( G.seaLevel ).toVar();

			// ---- motion: slow swirl (two octaves, random phases) + wind; gnats dart in loops
			const ph = h3.mul( 6.2832 );
			const w1 = h2.mul( 0.5 ).add( 0.35 ); // rad/s
			const w2 = h2.zxy.mul( 1.4 ).add( 1.5 );
			const A1 = vec3( 0.45, 0.22, 0.45 ).mul( select( isSeed, float( 1.6 ), float( 1 ) ) );
			const A2 = vec3( 0.07, 0.05, 0.07 );
			const a1 = w1.mul( t ).add( ph ), a2 = w2.mul( t ).add( ph.zxy );
			const swirl = sin( a1 ).mul( A1 ).add( sin( a2 ).mul( A2 ) );
			const swirlV = cos( a1 ).mul( A1.mul( w1 ) ).add( cos( a2 ).mul( A2.mul( w2 ) ) );
			const gw = h2.mul( 2.2 ).add( 1.8 );
			const gA = vec3( 0.3, 0.2, 0.3 ).mul( h3.x.mul( 0.6 ).add( 0.6 ) );
			const g1 = gw.mul( t ).add( ph ), g2 = gw.zxy.mul( t.mul( 1.7 ) ).add( ph.yzx );
			const gnatOff = h.sub( 0.5 ).mul( vec3( 0.9, 0.6, 0.9 ) ).add( sin( g1 ).mul( gA ) ).add( sin( g2 ).mul( gA.mul( 0.45 ) ) );
			const gnatV = cos( g1 ).mul( gA.mul( gw ) ).add( cos( g2 ).mul( gA.mul( 0.45 ).mul( gw.zxy ).mul( 1.7 ) ) );
			const off = select( isGnat, gnatOff, swirl );
			const windV = vec3( G.windDir.x, 0, G.windDir.y ).mul( G.windSpeed.mul( this.windScale ) );
			const vel = select( isGnat, gnatV, swirlV.add( windV ) );

			// height above the ground (or the sea): mostly low down, seeds and gnats in the first metres
			const ground = max( ht, G.seaLevel.add( 0.25 ) );
			const yMote = pow( h.y, 1.8 ).mul( 6.5 ).add( 0.2 );
			const ySeed = h.y.mul( 2.5 ).add( 0.3 );
			const yGnat = hs.y.mul( 1.4 ).add( 0.8 );
			const y = ground.add( select( isGnat, yGnat, select( isSeed, ySeed, yMote ) ) );
			const p = vec3( xz0.x, y, xz0.y ).add( off ).toVar();

			// ---- how many live here: most over the beach and the plants, very few far out over the sea,
			// fewer at night; gnats only inland over the vegetation, in a few places (per world tile)
			const land = smoothstep( 0.3, 1.5, hA );
			const shore = smoothstep( - 7.0, - 2.0, hA );
			const day = float( 1 ).sub( G.night.mul( 0.75 ) );
			const tile = floor( xz0.div( B ) );
			const tileHash = fract( sin( dot( tile, vec2( 12.9898, 78.233 ) ).add( sid.mul( 7.31 ) ) ).mul( 43758.5453 ) );
			const dMote = max( land.mul( 0.85 ), shore ).max( 0.06 ).mul( day );
			const dSeed = land.mul( day );
			const dGnat = smoothstep( 1.5, 4.0, hA ).mul( select( tileHash.lessThan( 0.3 ), float( 1 ), float( 0 ) ) ).mul( float( 1 ).sub( G.night ) );
			const density = select( isGnat, dGnat, select( isSeed, dSeed, dMote ) );
			const keep = saturate( density.sub( select( isGnat, float( 0.5 ), h3.y ) ).mul( 8 ) );

			// ---- what: salt aerosol near the surf / over the sea, dust and pollen over the land
			const salt = smoothstep( 0.8, - 0.5, hA );
			// (the size of what catches the eye in film: lint, pollen clumps, bits of plant, sea salt crystals)
			const rMote = mix( 0.00015, 0.0005, h2.x.mul( h2.x ) ).mul( mix( float( 1 ), float( 0.6 ), salt ) ).mul( select( h2.y.greaterThan( 0.96 ), float( 2.2 ), float( 1 ) ) );
			const r = select( isGnat, mix( 0.0008, 0.0011, h2.x ), select( isSeed, mix( 0.0025, 0.0045, h2.x ), rMote ) );

			// ---- lighting (per particle): forward-scattering phase, sun visibility at the particle
			const toCam = cameraPosition.sub( p );
			const dist = max( length( toCam ), 0.05 ).toVar();
			const Vd = toCam.div( dist );
			const cosT = dot( Vd, L ).negate(); // 1 = looking toward the sun through the particle
			const vis = ( clouds ? clouds.shadow( p.xz ) : float( 1 ) ).mul( terrain.sunShadowAt( p ) ).mul( csmVisibility( p ) );
			const sun = G.sunColor.mul( vis );
			// diffraction + refraction: a strong forward lobe with a narrow glint core, a weak broad part
			// (the broad part keeps sunlit motes faintly visible from the side against shade)
			const phDust = phaseHG( cosT, 0.93 ).mul( 0.22 ).add( phaseHG( cosT, 0.7 ).mul( 0.43 ) ).add( phaseHG( cosT, 0.15 ).mul( 0.35 ) );
			const phSalt = phaseHG( cosT, 0.93 ).mul( 0.35 ).add( phaseHG( cosT, 0.75 ).mul( 0.5 ) ).add( phaseHG( cosT, 0.2 ).mul( 0.15 ) );
			const phMote = mix( phDust, phSalt, salt );
			const albMote = mix( vec3( 0.95, 0.85, 0.66 ), vec3( 1.0 ), salt );
			const cMote = albMote.mul( sun.mul( phMote ).add( G.skyIrradiance.mul( 0.2 ) ) );
			// seed tufts: white fluff, diffuse from any side plus a forward glow
			const cSeed = sun.mul( phaseHG( cosT, 0.55 ).mul( 0.8 ).add( 0.05 ) ).add( G.skyIrradiance.mul( 0.35 ) );
			// gnats: dark bodies (specks against the sky), wings that catch the light when backlit
			const cGnat = sun.mul( phaseHG( cosT, 0.9 ).mul( 0.12 ) ).add( G.skyIrradiance.mul( 0.02 ) );
			const col = select( isGnat, cGnat, select( isSeed, cSeed, cMote ) );

			// ---- footprint: at least ~1.2 px wide, opacity = the real particle's coverage of it
			const p11 = cameraProjectionMatrix.element( 1 ).element( 1 );
			const pixel = dist.mul( 2 ).div( p11.mul( screenSize.y ) );
			const size = max( r, pixel.mul( 1.2 ) );
			// (x VISIBILITY: in film these read larger than they are, defocused and bloomed)
			const cover = r.div( size ).pow2().mul( 3.5 * VISIBILITY ).min( 1 );
			// fades: box edges (before the wrap), right in front of the lens, near the water surface
			const edge = smoothstep( B.mul( 0.5 ), B.mul( 0.36 ), max( abs( p.x.sub( this.camPos.x ) ), abs( p.z.sub( this.camPos.z ) ) ) );
			const fade = edge.mul( smoothstep( B.mul( 0.5 ), B.mul( 0.3 ), dist ) ).mul( smoothstep( 0.12, 0.5, dist ) )
				.mul( smoothstep( 0.25, 0.9, p.y.sub( G.seaLevel ) ) );
			const a = cover.mul( fade ).mul( keep ).mul( this.intensity ).toVar();

			// camera-facing quad
			const right = normalize( cross( vec3( 0, 1, 0 ), Vd ).add( vec3( 1e-5, 0, 0 ) ) );
			const up = cross( Vd, right );
			const corner = positionGeometry.xy; // plane is 2 x 2: half-width = size
			const world = p.add( right.mul( corner.x.mul( size ) ) ).add( up.mul( corner.y.mul( size ) ) );

			// own motion over the last frame, in NDC (camera motion comes from staticVelocity)
			const c0 = cameraProjectionMatrix.mul( cameraViewMatrix.mul( vec4( p, 1 ) ) );
			const c1 = cameraProjectionMatrix.mul( cameraViewMatrix.mul( vec4( p.sub( vel.mul( G.dt ) ), 1 ) ) );
			vMotion.assign( c0.xy.div( c0.w ).sub( c1.xy.div( c1.w ) ) );
			vCol.assign( vec4( col, a ) );
			vUV.assign( corner );

			// nothing to draw: collapse the quad off-screen
			return select( a.greaterThan( 2e-4 ), world, vec3( 0, - 1e5, 0 ) );

		} )();

		// gaussian speck cut at its core (the cut tails' energy is folded into the core: 1 / 0.65)
		const alpha = Fn( () => {

			const g = exp( dot( vUV, vUV ).mul( - 3.5 ) );
			const a = vCol.w.mul( g ).div( 0.65 ).min( 1 ).toVar();
			If( g.lessThan( 0.35 ).or( a.lessThan( 0.003 ) ), () => {

				Discard();

			} );
			return a;

		} )();
		mat.outputNode = vec4( vCol.rgb.mul( alpha ), alpha );
		// the speck owns the motion vector of the pixels it covers (see depthWrite)
		mat.mrtNode = mrt( { velocity: vec4( staticVelocity.add( vMotion ), 0, 1 ) } );
		mat.mrtNode.setBlendMode( 'velocity', new THREE.BlendMode( THREE.NoBlending ) );

		this.mesh = new THREE.Mesh( inst, mat );
		this.mesh.name = 'AirMotes';
		this.mesh.frustumCulled = false;
		this.mesh.castShadow = false;
		this.mesh.receiveShadow = false;
		this.mesh.renderOrder = 21;
		this.mesh.layers.set( LAYERS.TRANSPARENT );

	}

	// cameraWaterHeight: water level at the camera (none drawn with the camera under water)
	update( dt, camera, cameraWaterHeight = 0 ) {

		const visible = this.intensity.value > 0 && camera.position.y > cameraWaterHeight;
		this.mesh.visible = visible;
		if ( ! visible ) return;
		this.camPos.value.copy( camera.position );
		const s = G.windSpeed.value * this.windScale * dt;
		const d = this.drift.value;
		d.x = ( d.x + G.windDir.value.x * s ) % this.box;
		d.y = ( d.y + G.windDir.value.y * s ) % this.box;
		const l = this.csm && this.csm.lights && this.csm.lights[ 0 ];
		const map = l && l.shadow && l.shadow.map;
		if ( map && map.depthTexture ) {

			if ( this.csmDepth.value !== map.depthTexture ) {

				this.csmDepth.value = map.depthTexture;
				this.csmSize.value.set( map.width, map.height );

			}

			this.csmActive.value = 1;

		}

	}

}
