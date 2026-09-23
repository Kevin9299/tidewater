import * as THREE from 'three/webgpu';
import {
	Fn, float, vec3, vec4, attribute, texture, uv, mix, smoothstep, max, abs, select, positionLocal, positionWorld,
	normalMap, luminance, saturate, Discard,
} from 'three/tsl';
import { bayer4, bandFade } from '../../materials/LODFade.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { standard } from '../../materials/Materials.js';
import { TerrainLightingModel } from '../Terrain.js';
import { srgb } from '../terrain/TerrainShading.js';
import { staticVelocityMRT } from '../../post/CameraVelocity.js';

// Photoscanned debris (CC0, Poly Haven; see public/models/debris/CREDITS.md): dead wood (logs,
// branches) and conch shells, instanced with three LODs.
//
// Each LOD level is ONE InstancedMesh holding the geometry of every asset (vertex attribute
// aAsset); an instance draws only its own asset (iData.x), the other assets' vertices collapse
// to a point. So the whole set costs 3 draw calls (+ 3 shadow draws for the near LOD) and one
// material. The asset textures (1K albedo / normal / AO-rough-metal) are packed into 2K atlases
// (2 x 2 tiles): 3 samplers.
// Instances are re-bucketed into the LODs (and distance culled) on the CPU when the camera moves.

export const SCAN_ASSETS = [ 'dead_quiver_trunk', 'dead_quiver_branch_02', 'dead_quiver_branch_01', 'lambis_shell' ];
export const SCAN = { TRUNK: 0, BRANCH_A: 1, BRANCH_B: 2, SHELL: 3 };
// size (m) of each asset after orientation (long axis x, up y), from the Blender export
export const SCAN_SIZE = [ [ 1.994, 0.306, 0.268 ], [ 0.526, 0.287, 0.201 ], [ 0.439, 0.255, 0.227 ], [ 0.139, 0.047, 0.075 ] ];
const LOD_DIST = [ 22, 70, 260 ]; // m (+ 6 x instance size): LOD0 | LOD1 | LOD2 | faded out
const BAND = 0.12; // cross-fade band (Bayer screen-door, see LODFade), share of the switch distance
const BASE = ( ( import.meta.env && import.meta.env.BASE_URL ) || '/' ) + 'models/debris/';

// RGBA8 pixels of an image (top row first). The headless test runner may provide a decoder.
async function loadPixels( url ) {

	if ( globalThis.__debrisImage ) return globalThis.__debrisImage( url );
	const blob = await ( await fetch( url ) ).blob();
	const bmp = await createImageBitmap( blob );
	const cv = new OffscreenCanvas( bmp.width, bmp.height );
	const ctx = cv.getContext( '2d', { willReadFrequently: true } );
	ctx.drawImage( bmp, 0, 0 );
	const d = ctx.getImageData( 0, 0, bmp.width, bmp.height ).data;
	return { data: new Uint8Array( d.buffer, d.byteOffset, d.byteLength ), width: bmp.width, height: bmp.height };

}

export class ScannedDebris {

	// instances: [ { asset, x, y, z, yaw, pitch, roll, sx, sy, sz, rnd } ]
	constructor( { gpu, instances, parent } ) {

		this.gpu = gpu;
		this.instances = instances;
		this.parent = parent;
		this.meshes = [];
		this.ready = false;
		this._lastCam = new THREE.Vector3( 1e9, 0, 0 );
		this._lastQuat = new THREE.Quaternion();
		this._frustum = new THREE.Frustum();
		this._m4 = new THREE.Matrix4();
		this._sphere = new THREE.Sphere();
		this._v = new THREE.Vector3();
		for ( const r of instances ) {

			const q = new THREE.Quaternion().setFromEuler( new THREE.Euler( r.roll || 0, r.yaw, r.pitch || 0, 'YXZ' ) );
			r.matrix = new THREE.Matrix4().compose( new THREE.Vector3( r.x, r.y, r.z ), q, new THREE.Vector3( r.sx, r.sy, r.sz ) );
			const s = SCAN_SIZE[ r.asset ];
			r.radius = Math.hypot( s[ 0 ] * r.sx, s[ 1 ] * r.sy, s[ 2 ] * r.sz ) / 2;

		}

		this.promise = this._load().catch( ( e ) => console.error( 'ScannedDebris: load failed', e ) );

	}

	async _load() {

		const t0 = performance.now();
		const loader = new GLTFLoader();
		const gltfs = await Promise.all( SCAN_ASSETS.map( ( id ) => loader.loadAsync( BASE + id + '.glb' ) ) );
		// 2 x 2 atlas per map
		const maps = [ 'albedo', 'normal', 'arm' ];
		const atlases = {};
		await Promise.all( maps.map( async ( m ) => {

			const tiles = await Promise.all( SCAN_ASSETS.map( ( id ) => loadPixels( BASE + id + '_' + m + '.jpg' ) ) );
			const T = tiles[ 0 ].width;
			const data = new Uint8Array( T * 2 * T * 2 * 4 );
			tiles.forEach( ( t, k ) => {

				const ox = ( k % 2 ) * T, oy = Math.floor( k / 2 ) * T;
				for ( let y = 0; y < T; y ++ ) data.set( t.data.subarray( y * T * 4, ( y + 1 ) * T * 4 ), ( ( oy + y ) * T * 2 + ox ) * 4 );

			} );
			const tex = new THREE.DataTexture( data, T * 2, T * 2, THREE.RGBAFormat, THREE.UnsignedByteType );
			tex.colorSpace = m === 'albedo' ? THREE.SRGBColorSpace : THREE.NoColorSpace;
			tex.magFilter = THREE.LinearFilter;
			tex.minFilter = THREE.LinearMipmapLinearFilter;
			tex.generateMipmaps = true;
			tex.anisotropy = 4;
			tex.name = 'debrisScan_' + m;
			tex.needsUpdate = true;
			atlases[ m ] = tex;

		} ) );
		this.textures = atlases;

		// merged geometry per LOD: all assets, uvs moved into their atlas tile
		const lodGeos = [];
		for ( let l = 0; l < 3; l ++ ) {

			const pos = [], nor = [], uvs = [], asset = [], idx = [];
			gltfs.forEach( ( g, k ) => {

				let mesh = null;
				g.scene.traverse( ( o ) => {

					if ( o.isMesh && o.name.startsWith( 'LOD' + l ) ) mesh = o;

				} );
				if ( ! mesh ) return;
				mesh.updateMatrixWorld( true );
				const geo = mesh.geometry.clone().applyMatrix4( mesh.matrixWorld );
				const P = geo.attributes.position, N = geo.attributes.normal, U = geo.attributes.uv;
				const base = pos.length / 3;
				const ox = ( k % 2 ) * 0.5, oy = Math.floor( k / 2 ) * 0.5;
				for ( let i = 0; i < P.count; i ++ ) {

					pos.push( P.getX( i ), P.getY( i ), P.getZ( i ) );
					nor.push( N.getX( i ), N.getY( i ), N.getZ( i ) );
					const u = Math.min( 0.998, Math.max( 0.002, U.getX( i ) ) ), v = Math.min( 0.998, Math.max( 0.002, U.getY( i ) ) );
					uvs.push( u * 0.5 + ox, v * 0.5 + oy );
					asset.push( k );

				}

				const I = geo.index;
				for ( let i = 0; i < I.count; i ++ ) idx.push( base + I.getX( i ) );

			} );
			const geo = new THREE.BufferGeometry();
			geo.setAttribute( 'position', new THREE.Float32BufferAttribute( pos, 3 ) );
			geo.setAttribute( 'normal', new THREE.Float32BufferAttribute( nor, 3 ) );
			geo.setAttribute( 'uv', new THREE.Float32BufferAttribute( uvs, 2 ) );
			geo.setAttribute( 'aAsset', new THREE.Float32BufferAttribute( asset, 1 ) );
			geo.setIndex( pos.length / 3 > 65535 ? new THREE.Uint32BufferAttribute( idx, 1 ) : new THREE.Uint16BufferAttribute( idx, 1 ) );
			lodGeos.push( geo );

		}

		this.material = this._createMaterial();
		const n = Math.max( 1, this.instances.length );
		this.levels = lodGeos.map( ( geo, l ) => {

			const iData = new THREE.InstancedBufferAttribute( new Float32Array( n * 4 ), 4 );
			iData.setUsage( THREE.DynamicDrawUsage );
			geo.setAttribute( 'iData', iData );
			const mesh = new THREE.InstancedMesh( geo, this.material, n );
			mesh.name = 'debris-scan-lod' + l;
			mesh.count = 0;
			mesh.castShadow = l === 0;
			mesh.receiveShadow = true;
			mesh.frustumCulled = false;
			mesh.instanceMatrix.setUsage( THREE.DynamicDrawUsage );
			this.parent.add( mesh );
			this.meshes.push( mesh );
			return { mesh, iData, tris: geo.index.count / 3 };

		} );
		this.loadMs = performance.now() - t0;
		this.ready = true;
		this._lastCam.set( 1e9, 0, 0 );

	}

	_createMaterial() {

		const gpu = this.gpu;
		const A = this.textures;
		const mat = standard( { roughness: 0.85, metalness: 0 } );
		mat.name = 'DebrisScanned';
		mat.mrtNode = staticVelocityMRT;
		mat.underwaterLighting = 'lite';
		mat.setupLightingModel = () => new TerrainLightingModel( mat, gpu.sunShadowAt( positionWorld ) );
		const iData = attribute( 'iData', 'vec4' ); // asset, random, LOD fade, outgoing
		// an instance keeps only its own asset's vertices (the rest collapse to a point)
		mat.positionNode = Fn( () => select( abs( attribute( 'aAsset', 'float' ).sub( iData.x ) ).lessThan( 0.5 ), positionLocal, vec3( 0 ) ) )();
		mat.castShadowPositionNode = mat.positionNode;

		const st = uv();
		const albedo = texture( A.albedo, st );
		const arm = texture( A.arm, st );
		const rnd = iData.y;
		const isWood = iData.x.lessThan( 2.5 );
		mat.colorNode = Fn( () => {

			const t = bayer4();
			Discard( select( iData.w.greaterThan( 0.5 ), t.lessThan( iData.z ), t.greaterThanEqual( iData.z ) ) );
			const p = positionWorld;
			let c = albedo.rgb;
			// driftwood: sun-bleached toward silver-grey, per instance
			const grey = vec3( luminance( c ) ).mul( vec3( 1.04, 1.02, 0.98 ) ).mul( 1.12 );
			c = mix( c, grey, select( isWood, rnd.mul( 0.4 ).add( 0.45 ), float( 0 ) ) );
			// ground contact: dusted with sand and darker where it touches / is buried
			const ground = gpu.heightAt( p.xz );
			const contact = float( 1 ).sub( smoothstep( 0.0, 0.1, p.y.sub( ground ) ) );
			const sandy = smoothstep( 0.6, 1.6, ground );
			c = mix( c, srgb( 0.8, 0.72, 0.58 ), contact.mul( sandy ).mul( 0.45 ) );
			// wet below the swash line: darker (and glossier, see roughness)
			const wet = float( 1 ).sub( smoothstep( 0.4, 1.1, p.y ) );
			c = c.mul( float( 1 ).sub( wet.mul( 0.45 ) ) );
			return vec4( c, 1 );

		} )();
		mat.roughnessNode = Fn( () => {

			const wet = float( 1 ).sub( smoothstep( 0.4, 1.1, positionWorld.y ) );
			return mix( max( arm.g, 0.35 ), float( 0.22 ), wet.mul( 0.85 ) );

		} )();
		mat.metalnessNode = float( 0 );
		mat.aoNode = Fn( () => {

			const p = positionWorld;
			const contact = float( 1 ).sub( smoothstep( 0.0, 0.12, p.y.sub( gpu.heightAt( p.xz ) ) ) );
			return saturate( arm.r.mul( float( 1 ).sub( contact.mul( 0.4 ) ) ) );

		} )();
		mat.normalNode = normalMap( texture( A.normal, st ) );
		return mat;

	}

	// LOD buckets + frustum / distance culling (only when the camera moved)
	update( camera ) {

		if ( ! this.ready ) return;
		const cam = camera.getWorldPosition( this._v );
		if ( cam.distanceToSquared( this._lastCam ) < 0.25 && camera.quaternion.equals( this._lastQuat ) ) return;
		this._lastCam.copy( cam );
		this._lastQuat.copy( camera.quaternion );
		camera.updateMatrixWorld();
		this._m4.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse );
		this._frustum.setFromProjectionMatrix( this._m4, camera.coordinateSystem, camera.reversedDepth );
		const counts = [ 0, 0, 0 ];
		for ( const r of this.instances ) {

			this._sphere.center.set( r.x, r.y, r.z );
			this._sphere.radius = r.radius + 6; // shadow casters just outside the view
			if ( ! this._frustum.intersectsSphere( this._sphere ) ) continue;
			const d = Math.hypot( r.x - cam.x, r.y - cam.y, r.z - cam.z ) - r.radius * 6;
			const [ a, b, c ] = LOD_DIST;
			const t01 = bandFade( d, a * ( 1 - BAND / 2 ), a * ( 1 + BAND / 2 ) );
			const t12 = bandFade( d, b * ( 1 - BAND / 2 ), b * ( 1 + BAND / 2 ) );
			const out = bandFade( d, c * ( 1 - BAND ), c );
			if ( out >= 1 ) continue;
			// ( level, fade, outgoing ): an instance is in two levels only inside a band
			if ( t01 < 1 ) this._put( 0, counts, r, t01, 1 );
			if ( t01 > 0 && t12 < 1 ) this._put( 1, counts, r, t12 > 0 ? t12 : t01, t12 > 0 ? 1 : 0 );
			if ( t12 > 0 ) this._put( 2, counts, r, t12 * ( 1 - out ), 0 );

		}

		this.levels.forEach( ( L, l ) => {

			L.mesh.count = counts[ l ];
			L.mesh.instanceMatrix.clearUpdateRanges();
			L.mesh.instanceMatrix.addUpdateRange( 0, Math.max( 1, counts[ l ] ) * 16 );
			L.mesh.instanceMatrix.needsUpdate = true;
			L.iData.clearUpdateRanges();
			L.iData.addUpdateRange( 0, Math.max( 1, counts[ l ] ) * 4 );
			L.iData.needsUpdate = true;

		} );

	}

	_put( l, counts, r, fade, outgoing ) {

		const L = this.levels[ l ];
		const i = counts[ l ] ++;
		L.mesh.setMatrixAt( i, r.matrix );
		L.iData.array.set( [ r.asset, r.rnd, fade, outgoing ], i * 4 );

	}

	get triangles() {

		if ( ! this.ready ) return 0;
		// each instance processes its LOD's merged geometry (other assets collapse)
		return this.levels.reduce( ( a, L ) => a + L.mesh.count * L.tris, 0 );

	}

	dispose() {

		for ( const m of this.meshes ) {

			m.geometry.dispose();
			this.parent.remove( m );

		}

		if ( this.material ) this.material.dispose();
		if ( this.textures ) for ( const k in this.textures ) this.textures[ k ].dispose();

	}

}
