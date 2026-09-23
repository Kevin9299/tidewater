import * as THREE from 'three/webgpu';
import { Fn, vec4, normalize, normalWorldGeometry } from 'three/tsl';

// Renders the sky (atmosphere + clouds, no sun disk) into a cube map and prefilters it (PMREM) for
// image based lighting (scene.environment). Refreshed when the sun moves or periodically so
// drifting clouds stay in sync. The work is spread over frames so none pays for a whole refresh:
// one cube face per frame, then the PMREM filter one roughness level per frame into a back
// buffer, which is copied into the environment texture once complete.
export class Environment {

	constructor( renderer, scene, sky, size = 128 ) {

		this.renderer = renderer;
		this.scene = scene;
		this.sky = sky;
		this.envScene = new THREE.Scene();
		this.envScene.backgroundNode = Fn( () => {

			const dir = normalize( normalWorldGeometry );
			return vec4( sky.radianceWithClouds( dir, false ), 1 );

		} )();

		this.target = new THREE.CubeRenderTarget( size, { type: THREE.HalfFloatType, generateMipmaps: false } );
		this.camera = new THREE.CubeCamera( 0.5, 100, this.target );
		this.envScene.add( this.camera );
		this.lastSun = new THREE.Vector3( 0, - 2, 0 );
		this.timer = 0;
		this.interval = 3;
		this.step = - 1; // progress of the refresh under way (-1: idle)
		this.primed = false;
		scene.environmentIntensity = 1;

		// progressive PMREM through three's generator internals (r186); falls back to the stock path
		// (the whole PMREM in one frame) if they are not there
		const pm = new THREE.PMREMGenerator( renderer );
		this.progressive = [ '_setSize', '_allocateTarget', '_init', '_textureToCubeUV', '_applyGGXFilter' ].every( ( k ) => typeof pm[ k ] === 'function' );
		if ( this.progressive ) {

			pm._setSize( size );
			this.back = pm._allocateTarget( false );
			this.front = pm._allocateTarget( false );
			pm._init( this.back );
			this.pmrem = pm;
			// the front target is only ever copied into: allocate its GPU texture now
			const prev = renderer.getRenderTarget();
			renderer.setRenderTarget( this.front );
			renderer.clear();
			renderer.setRenderTarget( prev );
			scene.environment = this.front.texture;

		} else {

			pm.dispose();
			scene.environment = this.target.texture;

		}

		this.steps = this._buildSteps();

	}

	// the refresh as a list of small units of work
	_buildSteps() {

		const steps = [];
		for ( let i = 0; i < 6; i ++ ) steps.push( () => this._renderFace( i ) );
		if ( ! this.progressive ) {

			steps.push( () => {

				this.target.texture.needsPMREMUpdate = true;

			} );
			return steps;

		}

		const pm = this.pmrem;
		const n = pm._lodMeshes.length;
		steps.push( () => this._pmremStep( () => pm._textureToCubeUV( this.target.texture, this.back ) ) );
		for ( let i = 1; i < n; i ++ ) steps.push( () => this._pmremStep( () => pm._applyGGXFilter( this.back, i - 1, i ) ) );

		steps.push( () => this.renderer.copyTextureToTexture( this.back.texture, this.front.texture ) );
		return steps;

	}

	_pmremStep( fn ) {

		const r = this.renderer;
		const prevTarget = r.getRenderTarget();
		const prevFace = r.getActiveCubeFace();
		const prevMip = r.getActiveMipmapLevel();
		const autoClear = r.autoClear;
		r.autoClear = false;
		fn();
		r.autoClear = autoClear;
		r.setRenderTarget( prevTarget, prevFace, prevMip );

	}

	update( dt, force = false ) {

		this.timer -= dt;
		const sun = this.sky.atmosphere.sunDir.value;

		// first use (or forced): the whole refresh at once
		if ( force || ! this.primed ) {

			this.primed = true;
			this.timer = this.interval;
			this.lastSun.copy( sun );
			for ( const s of this.steps ) s();
			this.step = - 1;
			return;

		}

		if ( this.step < 0 ) {

			const moved = sun.angleTo( this.lastSun ) > 0.004;
			if ( ! moved && this.timer > 0 ) return;
			this.timer = this.interval;
			this.lastSun.copy( sun );
			this.step = 0;

		}

		this.steps[ this.step ++ ]();
		if ( this.step === this.steps.length ) this.step = - 1;

	}

	_renderFace( i ) {

		const r = this.renderer;
		const cube = this.camera;
		if ( cube.coordinateSystem !== r.coordinateSystem ) {

			cube.coordinateSystem = r.coordinateSystem;
			cube.updateCoordinateSystem();

		}

		const prevTarget = r.getRenderTarget();
		const prevFace = r.getActiveCubeFace();
		const prevMip = r.getActiveMipmapLevel();
		r.setRenderTarget( this.target, i );
		r.render( this.envScene, cube.children[ i ] );
		r.setRenderTarget( prevTarget, prevFace, prevMip );

	}

}
