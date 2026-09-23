import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, vec2, vec3, vec4, instanceIndex, positionGeometry, cameraViewMatrix, cameraProjectionMatrix,
	fract, sin, dot, floor, exp, smoothstep, length, max, texture, mix, varyingProperty, uv, saturate, If, Discard,
} from 'three/tsl';
import { G } from '../core/Globals.js';
import { LAYERS } from '../core/SceneRenderer.js';
import { staticVelocityMRT } from '../post/CameraVelocity.js';
import { FLASH, spotProfile } from '../materials/LocalLights.js';

const hash3 = ( n ) => fract( sin( vec3( n, n.add( 17.13 ), n.add( 43.71 ) ) ).mul( vec3( 43758.5453, 22578.1459, 19642.3490 ) ) );

// Suspended particles in the water around the camera (marine snow, plankton, sand grains).
// Positions are procedural (hash of the instance) inside a box that wraps around the camera, so
// there is no simulation and no CPU work. The particles drift with a slow current and sway back
// and forth with the orbital motion of the passing swell (longest FFT cascades, decaying with
// depth). Only drawn while the view can be under water, and only below the surface (when the
// waterline crosses the lens, none float in the air half). The specks are tiny depth-writing discs
// in the opaque pass, so the underwater fog treats each one at its own distance (TAA smooths the
// edges).
export class MarineSnow {

	constructor( { fft, query = null, count = 12000, box = 8 } ) {

		this.fft = fft;
		this.box = box;
		this.camPos = uniform( new THREE.Vector3() ).setName( 'snowCam' );
		this.opacity = uniform( 1 ).setName( 'snowOpacity' );

		const geo = new THREE.PlaneGeometry( 1, 1 );
		const inst = new THREE.InstancedBufferGeometry();
		inst.index = geo.index;
		inst.setAttribute( 'position', geo.getAttribute( 'position' ) );
		inst.setAttribute( 'uv', geo.getAttribute( 'uv' ) );
		inst.instanceCount = count;

		const mat = new THREE.MeshBasicNodeMaterial();
		mat.mrtNode = staticVelocityMRT; // nearly still in the world: camera motion only

		const vFade = varyingProperty( 'float', 'vSnowFade' );
		const vDepth = varyingProperty( 'float', 'vSnowDepth' );
		const vTorch = varyingProperty( 'vec3', 'vSnowTorch' );

		mat.vertexNode = Fn( () => {

			const id = float( instanceIndex );
			const h = hash3( id.mul( 0.7131 ) );
			const h2 = hash3( id.mul( 1.3917 ).add( 5.1 ) );
			const B = float( box );
			// slow current + sinking, wrapped into the camera box
			const drift = vec3( 0.05, - 0.012, 0.03 ).mul( G.time ).add( h2.sub( 0.5 ).mul( G.time.mul( 0.02 ) ) );
			const local = fract( h.add( drift.div( B ) ).sub( this.camPos.div( B ) ) ).sub( 0.5 ).mul( B );
			const p = this.camPos.add( local ).toVar();
			// orbital sway from the passing swell (decays ~exp(-k z) with depth)
			const depth = max( G.seaLevel.sub( p.y ), 0 );
			let sway = vec3( 0 );
			for ( let c = 0; c < 2; c ++ ) {

				const L = fft.sizes[ c ];
				const d = texture( fft.displacementTexture, p.xz.div( L ) ).depth( c ).level( 3 ).xyz;
				sway = sway.add( d.mul( exp( depth.mul( - 2 * Math.PI / L ) ) ) );

			}

			p.addAssign( vec3( sway.x, sway.y.mul( 0.5 ), sway.z ).mul( 0.8 ) );

			// camera-facing quad, 3-8 mm (a few flecks larger)
			const size = mix( 0.003, 0.008, h2.x.mul( h2.x ) ).add( h2.y.greaterThan( 0.98 ).select( 0.008, 0 ) );
			const pv0 = cameraViewMatrix.mul( vec4( p, 1 ) );
			// fade in the box edges and right in front of the lens (by shrinking)
			const dist = length( local );
			let fade = smoothstep( B.mul( 0.5 ), B.mul( 0.3 ), dist ).mul( smoothstep( 0.08, 0.3, pv0.z.negate() ) );
			if ( query ) fade = fade.mul( smoothstep( 0.0, 0.04, query.heightAtNode( p.xz ).sub( p.y ) ) );
			const pv = vec4( pv0.xy.add( positionGeometry.xy.mul( size ).mul( fade ) ), pv0.z, pv0.w );
			vFade.assign( fade );
			vDepth.assign( depth );
			// flashlight on the speck (per vertex): the backscatter sparkle of a diver's torch; the
			// view leg is attenuated by the underwater composite
			const tv = p.sub( FLASH.pos );
			const tr2 = max( dot( tv, tv ), 1e-4 );
			const tr = tr2.sqrt();
			const tspot = spotProfile( dot( tv.div( tr ), FLASH.dir ), FLASH.cone.x, FLASH.cone.y );
			const sigT = G.waterAbsorption.add( G.waterScattering );
			vTorch.assign( FLASH.col.mul( exp( sigT.mul( tr ).negate() ) ).mul( tspot.div( tr2.add( 0.15 ) ).mul( FLASH.on ).mul( 0.35 ) ) );
			return cameraProjectionMatrix.mul( pv );

		} )();

		mat.colorNode = Fn( () => {

			const r = length( uv().sub( 0.5 ) ).mul( 2 );
			If( r.greaterThan( 1 ).or( vFade.lessThan( 0.02 ) ), () => {

				Discard();

			} );
			// lit by the light that reaches this depth, slightly greenish-white (organic matter)
			const sigT = G.waterAbsorption.add( G.waterScattering );
			const light = G.sunColor.mul( 0.1 ).add( G.skyIrradiance.mul( 1.2 ) ).mul( exp( sigT.mul( vDepth ).negate() ) );
			const col = light.add( vTorch ).mul( vec3( 0.9, 1.0, 0.95 ) ).mul( mix( 1.3, 0.8, r ) ).mul( this.opacity );
			return vec4( col, 1 );

		} )();

		this.mesh = new THREE.Mesh( inst, mat );
		this.mesh.frustumCulled = false;
		this.mesh.layers.set( LAYERS.OPAQUE );
		this.mesh.visible = false;

	}

	update( camera, underwater ) {

		this.mesh.visible = underwater;
		if ( underwater ) this.camPos.value.copy( camera.position );

	}

}
