// Side-by-side fidelity check of the reef and the fish: the same scene and camera sequence rendered
// by the port (engine, test/life-harness.mjs) or by the three.js reference (three r186
// WebGPURenderer on Dawn, sources imported read-only from ../threejs-water-claude).
//
//   BACKEND=port node test/life-ref.mjs <outDir>     -> <outDir>/cmp-port-<view>.png
//   BACKEND=ref  node test/life-ref.mjs <outDir>     -> <outDir>/cmp-ref-<view>.png
//   MODE=reef | fish (default: both)
//
// Common minimal scene: TerrainData ground, a translucent stub sea sheet, sun (0.45, 0.62, 0.35)
// with colour (3.2, 3.0, 2.7), sky irradiance (0.22, 0.3, 0.42), horizon (0.55, 0.65, 0.78),
// ACES (fitted) at exposure 0.6. Environment lighting is reduced to what both sides can express
// identically: a hemisphere irradiance mix( horizon * 0.25, sky, N.y * 0.5 + 0.5 ) and no
// environment specular (the port installs matching SceneLighting hooks; the reference uses a
// HemisphereLight). Shadows differ (engine CSM + PCSS vs one three.js shadow map).
import './headless.mjs';
import { writePNG } from './headless.mjs';

const BACKEND = process.env.BACKEND || 'port';
const MODE = process.env.MODE || 'all';
const out = process.argv[ 2 ] || '.';
const W = +( process.env.W || 2560 ), H = +( process.env.H || 1267 );
const REF = new URL( '../../threejs-water-claude/', import.meta.url ).href;
const SUN = [ 0.45, 0.62, 0.35 ], SUNC = [ 3.2, 3.0, 2.7 ], SKY = [ 0.22, 0.3, 0.42 ], HOR = [ 0.55, 0.65, 0.78 ];
const BG = [ 0.45, 0.6, 0.85 ];

// ------------------------------------------------------------------------------ port backend

async function setupPort() {

	const { setupLife } = await import( './life-harness.mjs' );
	const { SceneLighting } = await import( '../src/engine/render/wgsl/lighting.js' );
	const { ShaderModule } = await import( '../src/engine/gpu/Shader.js' );
	const { surfaceModule } = await import( '../src/engine/render/wgsl/lighting.js' );
	if ( MODE === 'veg' ) ( await import( 'node:module' ) ).register( './life-veg-stubs.mjs', import.meta.url );
	const ground = MODE === 'veg' ? { center: [ 20, - 160 ], size: 900, res: 1 } : { center: [ - 78, 58 ], size: 360, res: 0.5 };
	const L = await setupLife( { W, H, ground, water: true, sun: SUN, shadowSplits: MODE === 'veg' ? [ 15, 80, 500 ] : undefined } );
	L.sky = [ ...BG, 1 ];
	const { G } = await import( '../src/engine/render/Frame.js' );
	G.horizonColor.value.setRGB( ...HOR );
	SceneLighting.set( 'envSpecular', new ShaderModule( { name: 'cmp-envSpec', deps: [ surfaceModule ], code: 'fn hookEnvSpecular( R: vec3f, roughness: f32 ) -> vec3f { return vec3f( 0.0 ); }' } ) );
	SceneLighting.set( 'envDiffuse', new ShaderModule( { name: 'cmp-envDiff', deps: [ surfaceModule ], code: 'fn hookEnvDiffuse( N: vec3f ) -> vec3f { return mix( frame.horizonColor * 0.25, frame.skyIrradiance, N.y * 0.5 + 0.5 ); }' } ) );
	const { Reef } = await import( '../src/world/Reef.js' );
	const { FishSchools } = await import( '../src/world/Fish.js' );
	const { WORLD } = await import( '../src/world/WorldLayout.js' );
	const E = await import( '../src/engine/index.js' );
	const { Material } = await import( '../src/engine/render/Material.js' );
	const spheres = ( list ) => list.map( ( o, i ) => {

		const m = new E.Mesh( new E.SphereGeometry( 0.5, 64, 32 ), new Material( { color: o.color, roughness: o.roughness, metalness: o.metalness, surface: o.spec !== undefined ? `s.specularIntensity = ${ o.spec.toFixed( 3 ) };` : '' } ) );
		m.position.set( ...o.pos );
		return m;

	} );
	const debugChannel = ( fish, ch ) => {

		for ( const m of [ fish.material, fish.fadeMaterial ] ) {

			m.surface += `\n\ts.emissive = vec3f( s.${ ch } ) * 0.5; s.albedo = vec3f( 0.0 ); s.specularIntensity = 0.0; s.translucency = vec3f( 0.0 );`;
			m.needsUpdate = true;

		}

	};
	const { Vegetation } = MODE === 'veg' ? await import( '../src/world/Vegetation.js' ) : {};
	return { L, Reef, FishSchools, WORLD, Vegetation, spheres, debugChannel, save: ( name ) => L.save( `${ out }/cmp-port-${ name }.png` ) };

}

// ------------------------------------------------------------------------------ reference backend

async function setupRef() {

	// canvas shim: three's WebGPU backend configures a context on its canvas even when we only
	// render into render targets
	let device = null;
	const ctx = {
		configure( d ) { device = d.device; this.format = d.format; },
		unconfigure() {},
		getCurrentTexture() {

			if ( ! this._tex || this._tex.width !== canvas.width || this._tex.height !== canvas.height ) this._tex = device.createTexture( { size: [ canvas.width, canvas.height ], format: this.format || 'bgra8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING } );
			return this._tex;

		},
	};
	const canvas = { width: W, height: H, style: {}, clientWidth: W, clientHeight: H, getContext: () => ctx, addEventListener() {}, removeEventListener() {}, setAttribute() {}, getBoundingClientRect: () => ( { left: 0, top: 0, width: W, height: H } ) };
	globalThis.document = globalThis.document || { createElementNS: () => canvas, createElement: () => canvas };
	globalThis.window = globalThis.window || globalThis;
	globalThis.self = globalThis.self || globalThis;
	globalThis.requestAnimationFrame = globalThis.requestAnimationFrame || ( ( f ) => setTimeout( () => f( performance.now() ), 0 ) );

	await import( REF + 'src/core/TSLPatches.js' );
	const THREE = await import( REF + 'node_modules/three/build/three.webgpu.js' );
	const TSL = await import( REF + 'node_modules/three/build/three.tsl.js' );
	const { G } = await import( REF + 'src/core/Globals.js' );
	const { TerrainData } = await import( REF + 'src/world/TerrainData.js' );
	const { Reef } = await import( REF + 'src/world/Reef.js' );
	const { FishSchools } = await import( REF + 'src/world/Fish.js' );
	const { WORLD } = await import( REF + 'src/world/WorldLayout.js' );
	const { Vegetation } = MODE === 'veg' ? await import( REF + 'src/world/Vegetation.js' ) : {};

	const renderer = new THREE.WebGPURenderer( { canvas, antialias: false } );
	await renderer.init();
	renderer.setPixelRatio( 1 );
	renderer.setSize( W, H, false );
	renderer.toneMapping = THREE.NoToneMapping;
	renderer.shadowMap.enabled = true;
	renderer.shadowMap.type = THREE.PCFSoftShadowMap;

	G.sunDir.value.set( ...SUN ).normalize();
	G.sunColor.value.setRGB( ...SUNC );
	G.skyIrradiance.value.setRGB( ...SKY );
	G.horizonColor.value.setRGB( ...HOR );

	const scene = new THREE.Scene();
	scene.background = new THREE.Color().setRGB( ...BG, THREE.LinearSRGBColorSpace );
	const camera = new THREE.PerspectiveCamera( 55, W / H, 0.1, 5000 );
	const sun = new THREE.DirectionalLight( new THREE.Color().setRGB( ...SUNC, THREE.LinearSRGBColorSpace ), 1 );
	sun.castShadow = true;
	sun.shadow.mapSize.set( 4096, 4096 );
	const sc = sun.shadow.camera;
	sc.left = sc.bottom = - 40; sc.right = sc.top = 40; sc.near = 1; sc.far = 400;
	sun.shadow.bias = - 0.0002;
	scene.add( sun, sun.target );
	const PI = Math.PI;
	const hemi = new THREE.HemisphereLight( new THREE.Color().setRGB( SKY[ 0 ] * PI, SKY[ 1 ] * PI, SKY[ 2 ] * PI, THREE.LinearSRGBColorSpace ), new THREE.Color().setRGB( HOR[ 0 ] * 0.25 * PI, HOR[ 1 ] * 0.25 * PI, HOR[ 2 ] * 0.25 * PI, THREE.LinearSRGBColorSpace ), 1 );
	scene.add( hemi );

	const terrain = new TerrainData();
	// ground: same heightfield and colours as test/life-harness.mjs
	{

		const [ cx, cz, size, res ] = MODE === 'veg' ? [ 20, - 160, 900, 1 ] : [ - 78, 58, 360, 0.5 ];
		const n = Math.round( size / res ) + 1;
		const pos = new Float32Array( n * n * 3 );
		for ( let j = 0; j < n; j ++ ) for ( let i = 0; i < n; i ++ ) {

			const x = cx - size / 2 + i * res, z = cz - size / 2 + j * res, k = ( j * n + i ) * 3;
			pos[ k ] = x; pos[ k + 1 ] = terrain.heightAt( x, z ); pos[ k + 2 ] = z;

		}

		const idx = new Uint32Array( ( n - 1 ) * ( n - 1 ) * 6 );
		let o = 0;
		for ( let j = 0; j < n - 1; j ++ ) for ( let i = 0; i < n - 1; i ++ ) {

			const a = j * n + i, b = a + 1, c = a + n, d = c + 1;
			idx[ o ++ ] = a; idx[ o ++ ] = c; idx[ o ++ ] = b; idx[ o ++ ] = b; idx[ o ++ ] = c; idx[ o ++ ] = d;

		}

		const g = new THREE.BufferGeometry();
		g.setAttribute( 'position', new THREE.BufferAttribute( pos, 3 ) );
		g.setIndex( new THREE.BufferAttribute( idx, 1 ) );
		g.computeVertexNormals();
		const { vec3, positionWorld, normalWorld, smoothstep, mix, float } = TSL;
		const m = new THREE.MeshStandardNodeMaterial( { roughness: 0.95, metalness: 0 } );
		const h = positionWorld.y, slope = float( 1 ).sub( normalWorld.y );
		let c = mix( vec3( 0.62, 0.55, 0.42 ), vec3( 0.16, 0.22, 0.08 ), smoothstep( 1.5, 4.0, h ) );
		c = mix( c, vec3( 0.35, 0.32, 0.29 ), smoothstep( 0.25, 0.5, slope ) );
		c = mix( c, vec3( 0.55, 0.5, 0.38 ), smoothstep( 0.0, - 3.0, h ) );
		m.colorNode = c;
		const mesh = new THREE.Mesh( g, m );
		mesh.castShadow = mesh.receiveShadow = true;
		scene.add( mesh );

	}

	const wm = new THREE.MeshStandardNodeMaterial( { transparent: true, roughness: 0.08, color: 0x0a3a4a, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false } );
	const water = new THREE.Mesh( new THREE.PlaneGeometry( 4000, 4000 ).rotateX( - Math.PI / 2 ), wm );
	water.frustumCulled = false;
	scene.add( water );

	const rt = new THREE.RenderTarget( W, H, { type: THREE.HalfFloatType, depthBuffer: true } );
	let time = 0;
	const L = {
		scene, camera, terrain, water,
		async run( n = 3, onFrame = null, dt = 1 / 60 ) {

			for ( let i = 0; i < n; i ++ ) {

				time += dt;
				G.time.value = time;
				G.dt.value = dt;
				camera.updateMatrixWorld();
				if ( onFrame ) onFrame( dt, time );
				// shadow map centred on what the camera looks at
				const f = new THREE.Vector3( 0, 0, - 1 ).applyQuaternion( camera.quaternion ).multiplyScalar( 20 ).add( camera.position );
				sun.target.position.copy( f );
				sun.position.copy( f ).addScaledVector( new THREE.Vector3( ...SUN ).normalize(), 200 );
				// the reference's velocity MRT outputs need the app's MRT pass: drop them (colour only)
				scene.traverse( ( o ) => {

					for ( const m of [ o.material ].flat() ) if ( m && m.mrtNode ) { m.mrtNode = null; m.needsUpdate = true; }

				} );
				renderer.setRenderTarget( rt );
				renderer.render( scene, camera );
				renderer.setRenderTarget( null );
				await renderer.backend.device.queue.onSubmittedWorkDone();

			}

		},
		async save( name ) {

			const px = await renderer.readRenderTargetPixelsAsync( rt, 0, 0, W, H );
			const rgba = new Uint8Array( W * H * 4 );
			const half = THREE.DataUtils.fromHalfFloat;
			const f = px instanceof Uint16Array ? ( i ) => half( px[ i ] ) : ( i ) => px[ i ];
			const aces = ( x ) => {

				const a = x * 0.6;
				const t = ( a * ( 2.51 * a + 0.03 ) ) / ( a * ( 2.43 * a + 0.59 ) + 0.14 );
				const c = Math.min( 1, Math.max( 0, t ) );
				return Math.round( 255 * ( c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow( c, 1 / 2.4 ) - 0.055 ) );

			};

			// readback rows may come bottom-up depending on the backend: WebGPU is top-down
			for ( let i = 0; i < W * H; i ++ ) {

				rgba[ i * 4 ] = aces( f( i * 4 ) ); rgba[ i * 4 + 1 ] = aces( f( i * 4 + 1 ) ); rgba[ i * 4 + 2 ] = aces( f( i * 4 + 2 ) ); rgba[ i * 4 + 3 ] = 255;

			}

			writePNG( `${ out }/cmp-ref-${ name }.png`, W, H, rgba );
			console.log( 'wrote', `${ out }/cmp-ref-${ name }.png` );

		},
		async exit() {

			await new Promise( ( r ) => setTimeout( r, 200 ) );
			process.exit( 0 );

		},
	};
	const { physical } = await import( REF + 'src/materials/Materials.js' );
	const spheres = ( list ) => list.map( ( o ) => {

		const mat = physical( { color: o.color, roughness: o.roughness, metalness: o.metalness } );
		if ( o.spec !== undefined ) mat.specularIntensityNode = TSL.float( o.spec );
		const m = new THREE.Mesh( new THREE.SphereGeometry( 0.5, 64, 32 ), mat );
		m.position.set( ...o.pos );
		m.castShadow = m.receiveShadow = true;
		return m;

	} );
	const debugChannel = ( fish, ch ) => {

		for ( const m of [ fish.material, fish.fadeMaterial ] ) {

			const node = m[ ch + 'Node' ];
			m.colorNode = m.colorNode.mul( 0 );
			m.emissiveNode = TSL.vec3( TSL.float( node ) ).mul( 0.5 );
			m.specularIntensityNode = null; m.specularIntensity = 0;
			m.translucencyNode = null;
			m.needsUpdate = true;

		}

	};
	return { L, Reef, FishSchools, WORLD, Vegetation, spheres, debugChannel, TSL, save: ( name ) => L.save( name ) };

}

// ------------------------------------------------------------------------------ shared sequence

const B = BACKEND === 'ref' ? await setupRef() : await setupPort();
const { L } = B;
const look = ( p, t ) => {

	L.camera.position.set( ...p );
	L.camera.lookAt( ...t );

};

if ( MODE === 'sphere' ) {

	// lighting model check: dielectric spheres (roughness 0.2 / 0.35 / 0.6, specular 1 and 0.5) and a metal
	const y = 3;
	const list = [];
	[ 0.2, 0.35, 0.6 ].forEach( ( r, i ) => {

		list.push( { pos: [ - 78 + i * 1.2, y, 58 ], color: 0x8a5a20, roughness: r, metalness: 0 } );
		list.push( { pos: [ - 78 + i * 1.2, y + 1.2, 58 ], color: 0x8a5a20, roughness: r, metalness: 0, spec: 0.5 } );

	} );
	list.push( { pos: [ - 78 + 3.6, y, 58 ], color: 0xcccccc, roughness: 0.3, metalness: 1 } );
	for ( const m of B.spheres( list ) ) L.scene.add( m );
	L.water.visible = false;
	look( [ - 76, y + 0.6, 63 ], [ - 76.2, y + 0.6, 58 ] );
	await L.run( 3 );
	await B.save( 'sphere' );

}

if ( MODE === 'veg' ) {

	// the reference vegetation at the views of test/life-veg.mjs (ref backend only)
	const veg = new B.Vegetation( { scene: L.scene, terrain: L.terrain } );
	const upd = ( dt ) => veg.update( dt, L.camera );
	const gy = ( x, z, dy ) => L.terrain.heightAt( x, z ) + dy;
	const views = {
		beach: [ [ 10, gy( 10, - 52, 1.7 ), - 52 ], [ 30, gy( 30, - 80, 4 ), - 80 ] ],
		palms: [ [ - 40, gy( - 40, - 55, 1.7 ), - 55 ], [ - 60, gy( - 60, - 75, 5 ), - 75 ] ],
		forest: [ [ 20, 60, 20 ], [ 20, 60, - 300 ] ],
		under: [ [ 60, gy( 60, - 200, 1.7 ), - 200 ], [ 60, gy( 60, - 230, 2 ), - 230 ] ],
		aerial: [ [ 150, 160, 150 ], [ 0, 20, - 250 ] ],
	};
	L.water.visible = true;
	for ( const [ name, [ p, t ] ] of Object.entries( views ) ) {

		if ( process.env.VIEW && process.env.VIEW !== name ) continue;
		look( p, t );
		await L.run( 4, upd );
		await B.save( 'veg-' + name );

	}

}

if ( MODE === 'all' || MODE === 'reef' ) {

	const reef = new B.Reef( { scene: L.scene, terrain: L.terrain } );
	const upd = ( dt ) => reef.update( dt, L.camera.position );
	const fy = ( x, z, dy ) => reef.floorHeightAt( x, z ) + dy;
	const views = {
		above: [ [ - 60, 9, 95 ], [ - 78, - 4, 58 ] ],
		under: [ [ - 70, fy( - 70, 70, 2.2 ), 70 ], [ - 82, fy( - 82, 52, 0.3 ), 52 ] ],
		close: [ [ - 76, fy( - 76, 62, 1.3 ), 62 ], [ - 79, fy( - 79, 58, 0.2 ), 58 ] ],
		fans: [ [ - 60, fy( - 60, 40, 1.8 ), 40 ], [ - 70, fy( - 70, 34, 0.8 ), 34 ] ],
	};
	for ( const [ name, [ p, t ] ] of Object.entries( views ) ) {

		if ( process.env.VIEW && process.env.VIEW !== name ) continue;
		look( p, t );
		L.water.visible = name === 'above';
		await L.run( 4, upd );
		await B.save( 'reef-' + name );

	}

	L.scene.remove( reef.group );

}

if ( MODE === 'all' || MODE === 'fish' ) {

	const rc = B.WORLD.reef.center;
	L.water.visible = false;
	const fish = new B.FishSchools( { parent: L.scene, terrain: L.terrain, center: rc.clone(), radius: B.WORLD.reef.radius + 10 } );
	// DEBUG=roughness | metalness | specularIntensity: show that material channel as emissive
	if ( process.env.DEBUG ) B.debugChannel( fish, process.env.DEBUG );
	if ( process.env.REFSPEC && B.TSL ) for ( const m of [ fish.material, fish.fadeMaterial ] ) { m.specularIntensityNode = B.TSL.float( + process.env.REFSPEC ); m.needsUpdate = true; }
	const step = ( dt ) => fish.update( dt, L.camera.position );
	const at = ( g ) => [ fish.pos[ g.offset * 3 ], fish.pos[ g.offset * 3 + 1 ], fish.pos[ g.offset * 3 + 2 ] ];
	const y0 = L.terrain.heightAt( rc.x, rc.z );
	look( [ rc.x + 14, Math.max( y0 + 2, - 6 ), rc.z + 10 ], [ rc.x, y0 + 0.5, rc.z ] );
	await L.run( 20, step );
	await B.save( 'fish-school' );
	for ( const name of [ 'grunt', 'yellowtail', 'parrot', 'turtle', 'stingray', 'eagleRay', 'barracuda', 'bait', 'chromis', 'tang', 'wrasse', 'sergeant' ] ) {

		const g = fish.groups.find( ( x ) => x.sp.name === name );
		if ( ! g ) continue;
		const Ls = fish.size[ g.offset ];
		const d = Math.max( 0.3, Ls * 1.4 );
		const place = () => {

			const c = at( g );
			look( [ c[ 0 ] + d * 0.8, c[ 1 ] + d * 0.3, c[ 2 ] + d * 0.5 ], c );

		};

		place();
		await L.run( 3, ( dt ) => {

			step( dt );
			place();

		} );
		await B.save( 'fish-' + name );

	}

}

await L.exit();
