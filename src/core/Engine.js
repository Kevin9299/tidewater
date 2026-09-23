import * as THREE from 'three/webgpu';

export class Engine {

	constructor( container ) {

		this.container = container;
		this.renderScale = 1;
		this.clock = new THREE.Timer();
		this.frame = 0;
		this.onUpdate = [];
		this.onResize = [];

	}

	async init() {

		if ( ! navigator.gpu ) throw new Error( 'WebGPU is not available in this browser.' );

		const adapter = await navigator.gpu.requestAdapter( { powerPreference: 'high-performance' } );
		if ( ! adapter ) throw new Error( 'No WebGPU adapter found.' );
		const L = adapter.limits;
		const want = ( name, v ) => Math.min( v, L[ name ] ?? v );
		const requiredLimits = {
			maxSampledTexturesPerShaderStage: want( 'maxSampledTexturesPerShaderStage', 32 ),
			maxStorageBuffersPerShaderStage: want( 'maxStorageBuffersPerShaderStage', 10 ),
			maxComputeWorkgroupStorageSize: want( 'maxComputeWorkgroupStorageSize', 32768 ),
			maxStorageTexturesPerShaderStage: want( 'maxStorageTexturesPerShaderStage', 8 ),
			maxColorAttachmentBytesPerSample: want( 'maxColorAttachmentBytesPerSample', 64 ),
			maxStorageBuffersInVertexStage: want( 'maxStorageBuffersInVertexStage', 4 ),
			maxStorageBuffersInFragmentStage: want( 'maxStorageBuffersInFragmentStage', 4 ),
		};
		for ( const k of Object.keys( requiredLimits ) ) if ( L[ k ] === undefined ) delete requiredLimits[ k ];

		this.hasFloat32Filterable = adapter.features.has( 'float32-filterable' );
		this.hasTimestamp = adapter.features.has( 'timestamp-query' );

		const renderer = new THREE.WebGPURenderer( {
			antialias: false,
			powerPreference: 'high-performance',
			reversedDepthBuffer: true,
			requiredLimits,
			trackTimestamp: this.hasTimestamp && new URLSearchParams( location.search ).has( 'profile' ),
		} );
		renderer.setPixelRatio( 1 );
		renderer.setSize( window.innerWidth, window.innerHeight );
		renderer.toneMapping = THREE.ACESFilmicToneMapping;
		renderer.toneMappingExposure = 1;
		renderer.shadowMap.enabled = true;
		renderer.shadowMap.type = THREE.PCFSoftShadowMap;
		this.container.appendChild( renderer.domElement );
		renderer.domElement.tabIndex = 0;

		await renderer.init();
		this.renderer = renderer;

		this.camera = new THREE.PerspectiveCamera( 62, window.innerWidth / window.innerHeight, 0.06, 60000 );
		this.scene = new THREE.Scene();

		window.addEventListener( 'resize', () => this.resize() );
		this.resize();

	}

	setRenderScale( s ) {

		this.renderScale = s;
		this.resize();

	}

	resize() {

		const w = window.innerWidth, h = window.innerHeight;
		this.renderer.setPixelRatio( this.renderScale );
		this.renderer.setSize( w, h );
		this.camera.aspect = w / h;
		this.camera.updateProjectionMatrix();
		for ( const f of this.onResize ) f( w, h );

	}

	start( update ) {

		this.renderer.setAnimationLoop( ( t ) => {

			this.clock.update( t );
			let dt = this.clock.getDelta();
			if ( dt > 0.1 ) dt = 0.1;
			this.frame ++;
			update( dt, this.clock.getElapsed() );

		} );

	}

}
