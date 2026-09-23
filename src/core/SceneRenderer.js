import * as THREE from 'three/webgpu';
import { mrt, output, velocity, vec4, positionWorld, cameraPosition, uniform } from 'three/tsl';

const _local = new THREE.Vector3();
const _sphere = new THREE.Sphere();
const _frustum = new THREE.Frustum();
const _vp = new THREE.Matrix4();

export const LAYERS = {
	OPAQUE: 0,
	WATER: 1,
	TRANSPARENT: 2,
	REFLECT_ONLY: 3,
};

// Scene renderer (internal resolution = drawing buffer * scale, upscaled later by TAAU):
//   1. opaque layer -> sceneRT (HDR color + velocity MRT, float reversed-Z depth)
//   2. explicit copies of color/depth -> opaqueCopy (sampled by the water for refraction/absorption)
//   3. water + transparent layers -> sceneRT on top (depth-tested against the opaques). Here the
//      velocity target blends premultiplied (mrtLate): outputs with alpha 1 overwrite it, blended
//      effects write ( v * a, 0, a ) for an opacity-weighted motion vector, and glass writes 0 so the
//      temporal resolve follows what is seen through it.
// The post-processing chain then reads sceneRT.
export class SceneRenderer {

	constructor( renderer, scene, camera ) {

		this.renderer = renderer;
		this.scene = scene;
		this.camera = camera;
		this.scale = 1;

		const make = ( name, count ) => {

			const rt = new THREE.RenderTarget( 1, 1, { type: THREE.HalfFloatType, depthBuffer: true, generateMipmaps: false, count } );
			rt.texture.name = name;
			rt.depthTexture = new THREE.DepthTexture( 1, 1, THREE.FloatType );
			rt.depthTexture.name = name + 'Depth';
			return rt;

		};

		// MRT outputs bind to render target textures by name
		this.sceneRT = make( 'scene', 3 );
		this.sceneRT.textures[ 0 ].name = 'output';
		this.sceneRT.textures[ 1 ].name = 'velocity';
		this.velocityTexture = this.sceneRT.textures[ 1 ];
		// water surface mask, written by the water material only: r = 1 where the visible surface is
		// seen from below (the view starts in water), g = 1 where a water surface is the visible surface.
		// Everything else writes (0, 0, 0, 0), which leaves it untouched under the blending (alpha 0).
		this.sceneRT.textures[ 2 ].name = 'waterMask';
		this.sceneRT.textures[ 2 ].type = THREE.UnsignedByteType;
		this.sceneRT.textures[ 2 ].format = THREE.RGBAFormat;
		this.waterMaskTexture = this.sceneRT.textures[ 2 ];
		this.opaqueCopy = make( 'opaqueCopy', 1 );
		this.opaqueCopy.texture.minFilter = THREE.LinearFilter;

		// Hull interiors: camera distance to the nearest surface of each registered closed hull volume
		// (0 = no hull). The water drops its surface behind it, so the sea never shows through the
		// cockpit sole when the boat squats, heels or sits in a trough (WaterMaterial hullMask).
		this.hullMaskRT = new THREE.RenderTarget( 1, 1, { type: THREE.HalfFloatType, format: THREE.RedFormat, depthBuffer: true, generateMipmaps: false } );
		this.hullMaskRT.texture.name = 'hullMask';
		this.hullMaskScene = new THREE.Scene();
		this.hullMaskMaterial = new THREE.MeshBasicNodeMaterial( { side: THREE.DoubleSide } );
		this.hullMaskMaterial.colorNode = vec4( positionWorld.sub( cameraPosition ).length(), 0, 0, 1 );
		this.hullMasks = [];
		this.hullMaskActive = uniform( 0 ).setName( 'hullMaskOn' );

		this.mrt = mrt( { output, velocity, waterMask: vec4( 0 ) } );
		this.mrt.setBlendMode( 'waterMask', new THREE.BlendMode( THREE.NormalBlending ) );
		// water + late pass (MRT blend modes only apply at the renderer level, not per material)
		this.mrtLate = mrt( { output, velocity, waterMask: vec4( 0 ) } );
		this.mrtLate.setBlendMode( 'waterMask', new THREE.BlendMode( THREE.NormalBlending ) );
		const premultiplied = new THREE.BlendMode( THREE.CustomBlending );
		premultiplied.blendSrc = THREE.OneFactor;
		premultiplied.blendDst = THREE.OneMinusSrcAlphaFactor;
		this.mrtLate.setBlendMode( 'velocity', premultiplied );

		this._size = new THREE.Vector2();
		this.width = 1;
		this.height = 1;
		this.onBeforeWater = null;

	}

	internalSize( out ) {

		this.renderer.getDrawingBufferSize( out );
		out.set( Math.max( 1, Math.floor( out.x * this.scale ) ), Math.max( 1, Math.floor( out.y * this.scale ) ) );
		return out;

	}

	setSize( w, h ) {

		this.width = w;
		this.height = h;
		this.sceneRT.setSize( w, h );
		this.opaqueCopy.setSize( w, h );
		this.hullMaskRT.setSize( w, h );
		// the copy target is never rendered to, so force its GPU textures to (re)allocate
		const r = this.renderer;
		const prev = r.getRenderTarget();
		r.setRenderTarget( this.opaqueCopy );
		r.clear();
		r.setRenderTarget( prev );

	}

	// Register a closed volume that follows `object` (e.g. BoatModel.createHullVolumeGeometry()).
	addHullMask( geometry, object ) {

		geometry.computeBoundingBox();
		geometry.computeBoundingSphere();
		const mesh = new THREE.Mesh( geometry, this.hullMaskMaterial );
		mesh.matrixAutoUpdate = false;
		mesh.frustumCulled = false;
		this.hullMaskScene.add( mesh );
		this.hullMasks.push( { mesh, object, box: geometry.boundingBox.clone().expandByScalar( 0.05 ), sphere: geometry.boundingSphere } );
		return mesh;

	}

	// Hull masks that can affect this frame: in view, and not containing the camera (from inside a
	// hull every sea pixel would lie behind one of its walls).
	_renderHullMasks( camera ) {

		let active = 0;
		_frustum.setFromProjectionMatrix( _vp.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse ) );
		for ( const h of this.hullMasks ) {

			// the mask scene is flat: its matrix is the hull's world matrix
			h.object.updateWorldMatrix( true, false );
			h.mesh.matrix.copy( h.object.matrixWorld );
			h.mesh.matrixWorldNeedsUpdate = true;
			_local.setFromMatrixPosition( camera.matrixWorld ).applyMatrix4( _vp.copy( h.object.matrixWorld ).invert() );
			const on = ! h.box.containsPoint( _local ) && _frustum.intersectsSphere( _sphere.copy( h.sphere ).applyMatrix4( h.object.matrixWorld ) );
			h.mesh.visible = on;
			if ( on ) active ++;

		}

		this.hullMaskActive.value = active > 0 ? 1 : 0;
		if ( ! active ) return;
		const renderer = this.renderer;
		renderer.setMRT( null );
		renderer.setRenderTarget( this.hullMaskRT );
		renderer.render( this.hullMaskScene, camera );
		renderer.setRenderTarget( this.sceneRT );
		renderer.setMRT( this.mrtLate );

	}

	render() {

		const { renderer, scene, camera } = this;
		this.internalSize( this._size );
		if ( this.sceneRT.width !== this._size.x || this.sceneRT.height !== this._size.y ) this.setSize( this._size.x, this._size.y );

		const prevTarget = renderer.getRenderTarget();
		const prevAutoClear = renderer.autoClear;
		const prevMRT = renderer.getMRT();
		const layers = camera.layers.mask;

		// 1. opaques (+ background)
		camera.layers.set( LAYERS.OPAQUE );
		renderer.setRenderTarget( this.sceneRT );
		renderer.setMRT( this.mrt );
		renderer.autoClear = true;
		renderer.render( scene, camera );

		// 2. copies for refraction
		renderer.copyTextureToTexture( this.sceneRT.texture, this.opaqueCopy.texture );
		renderer.copyTextureToTexture( this.sceneRT.depthTexture, this.opaqueCopy.depthTexture );

		// 2b. hull interiors (same jittered camera as the water pass)
		if ( this.hullMasks.length > 0 ) this._renderHullMasks( camera );

		if ( this.onBeforeWater ) this.onBeforeWater();

		// 3. water + transparents, no background, no clear
		const bg = scene.backgroundNode;
		const bgt = scene.background;
		scene.backgroundNode = null;
		scene.background = null;
		renderer.autoClear = false;
		renderer.setMRT( this.mrtLate );
		camera.layers.set( LAYERS.WATER );
		camera.layers.enable( LAYERS.TRANSPARENT );
		renderer.render( scene, camera );

		scene.backgroundNode = bg;
		scene.background = bgt;
		camera.layers.mask = layers;
		renderer.setMRT( prevMRT );
		renderer.autoClear = prevAutoClear;
		renderer.setRenderTarget( prevTarget );

	}

}
