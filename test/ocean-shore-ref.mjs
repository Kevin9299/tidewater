// Evaluates the REFERENCE (three.js / TSL) ShoreWaves on the analytic stub beach of
// ocean-shore-stubs.mjs and writes the results to a binary file for ocean-shore-compare.mjs.
// Imports (read-only) from ../threejs-water-claude. node test/ocean-shore-ref.mjs out.bin
const REF = '../../threejs-water-claude';
import { create, globals } from 'webgpu';
Object.assign( globalThis, globals );
Object.defineProperty( globalThis.navigator, 'gpu', { value: create( [] ), configurable: true } );
globalThis.requestAnimationFrame = ( f ) => setTimeout( () => f( performance.now() ), 16 );
globalThis.cancelAnimationFrame = ( id ) => clearTimeout( id );
globalThis.self = globalThis;
function makeCanvas( width, height ) {

	const ctx = { device: null, format: null, texture: null, configure( c ) { this.device = c.device; this.format = c.format; }, unconfigure() {},
		getCurrentTexture() { return this.texture || ( this.texture = this.device.createTexture( { size: [ width, height ], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT } ) ); } };
	return { width, height, style: {}, addEventListener() {}, removeEventListener() {}, getContext: () => ctx, getBoundingClientRect: () => ( { left: 0, top: 0, width, height } ), clientWidth: width, clientHeight: height };

}
await import( REF + '/src/core/TSLPatches.js' );
const THREE = await import( REF + '/node_modules/three/build/three.webgpu.js' );
const { Fn, float, vec2, vec3, vec4, sin, sqrt, max, select, instancedArray, instanceIndex, normalize } = await import( REF + '/node_modules/three/build/three.tsl.js' );
const { G } = await import( REF + '/src/core/Globals.js' );
const { ShoreWaves } = await import( REF + '/src/ocean/ShoreWaves.js' );
const { ShoreSim } = await import( REF + '/src/ocean/ShoreSim.js' );
const { SeaDetail } = await import( REF + '/src/ocean/SeaDetail.js' );
const { textureLoad, ivec2, uvec2 } = await import( REF + '/node_modules/three/build/three.tsl.js' );
import { writeFileSync } from 'node:fs';

const SLOPE = 0.04, K = 2 / Math.sqrt( 9.81 * SLOPE );
const terrain = {
	heightAt: ( xz ) => xz.y.mul( - SLOPE ).add( sin( xz.x.mul( 0.05 ) ).mul( 0.25 ) ).sub( select( xz.y.lessThan( - 3 ), xz.y.add( 3 ).mul( 0.06 ), float( 0 ) ) ),
	shoreSample: ( xz ) => vec4( float( K ).mul( float( 20 ).sub( sqrt( max( xz.y, 0.25 ) ) ) ), 0, - 1, K * ( 20 - 0.5 ) ),
	normalRock: () => vec4( 0, SLOPE, 0, 1 ),
	origin: - 400, size: 800,
};
{

	const res = 256, data = new Float32Array( res * res * 4 );
	const Tof = ( z ) => K * ( 20 - Math.sqrt( Math.max( z, 0.25 ) ) );
	for ( let j = 0; j < res; j ++ ) for ( let i = 0; i < res; i ++ ) {

		const z = - 400 + ( j + 0.5 ) / res * 800, k = ( j * res + i ) * 4;
		data[ k ] = Tof( z ); data[ k + 1 ] = 0; data[ k + 2 ] = - 1; data[ k + 3 ] = Tof( 0 );

	}

	terrain.shoreTexture = new THREE.DataTexture( data, res, res, THREE.RGBAFormat, THREE.FloatType );

}

const renderer = new THREE.WebGPURenderer( { canvas: makeCanvas( 64, 64 ) } );
await renderer.init();
const shore = new ShoreWaves( terrain );
G.time.value = Number( process.env.T || 47.3 );
const N = 128;
const out = instancedArray( N * N * 4, 'vec4' );
const kernel = Fn( () => {

	const i = instanceIndex;
	const ix = float( i.mod( N ) ), iz = float( i.div( N ) );
	const xz = vec2( ix.add( 0.5 ).div( N ).mul( 120 ).sub( 60 ), iz.add( 0.5 ).div( N ).mul( 120 ).sub( 20 ) ).toVar();
	const ground = terrain.heightAt( xz ).toVar();
	const depth = G.seaLevel.sub( ground ).toVar();
	const sw = shore.evaluate( xz, depth, ground );
	const base = i.mul( 4 );
	out.element( base ).assign( vec4( sw.disp, sw.foam ) );
	out.element( base.add( 1 ) ).assign( vec4( sw.nShore, sw.swashLevel ) );
	out.element( base.add( 2 ) ).assign( vec4( sw.face, sw.roller, select( sw.swashCovered, float( 1 ), float( 0 ) ), sw.flowSpeed ) );
	const med = shore.surfMedium( xz, depth );
	out.element( base.add( 3 ) ).assign( vec4( shore.swashClip( xz, float( 0.1 ) ), shore.crestPath( xz, depth, normalize( vec3( 0.2, - 0.5, 0.8 ) ) ), med.scatter.y, sw.breaking ) );

} )().compute( N * N );
renderer.compute( kernel );
const buf = await renderer.getArrayBufferAsync( out.value );
writeFileSync( process.argv[ 2 ] || '/tmp/shore-ref.bin', Buffer.from( buf ) );

// ---- ShoreSim: 300 frames at 30 fps, then its state (res 256 over 160 m)
if ( process.env.SIM ) {

	const R = 256;
	const sim = new ShoreSim( renderer, { terrainGPU: terrain, shore, center: new THREE.Vector2( 0, 20 ), size: 160, res: R } );
	G.dt.value = 1 / 30;
	const t0 = G.time.value;
	for ( let f = 0; f < 300; f ++ ) {

		G.time.value = t0 + f / 30;
		sim.update();

	}

	const st = instancedArray( R * R, 'vec4' );
	renderer.compute( Fn( () => {

		const i = instanceIndex;
		st.element( i ).assign( textureLoad( sim.stateA, ivec2( i.mod( R ), i.div( R ) ) ) );

	} )().compute( R * R ) );
	writeFileSync( process.argv[ 2 ] + '.sim', Buffer.from( await renderer.getArrayBufferAsync( st.value ) ) );

	// SurfFoam on a y = 0 plane seen straight down (ortho, 512^2 over x -40..40, z -10..70)
	{

		const { SurfFoam } = await import( REF + '/src/ocean/SurfFoam.js' );
		const { positionWorld, saturate } = await import( REF + '/node_modules/three/build/three.tsl.js' );
		const sf = new SurfFoam( { shoreSim: sim } );
		const W = 512;
		const rt = new THREE.RenderTarget( W, W, { type: THREE.FloatType } );
		const cam = new THREE.OrthographicCamera( - 40, 40, 40, - 40, 1, 200 );
		cam.position.set( 0, 50, 30 ); cam.up.set( 0, 0, - 1 ); cam.lookAt( 0, 0, 30 ); cam.updateMatrixWorld();
		const mat = new THREE.MeshBasicNodeMaterial();
		mat.fragmentNode = Fn( () => {

			const xz = positionWorld.xz;
			const ground = terrain.heightAt( xz ).toVar();
			const depth = G.seaLevel.sub( ground ).toVar();
			const s = sim.sample( xz ).toVar();
			const sw = shore.evaluate( xz, depth, ground );
			const cov = saturate( s.x.add( sw.foam ) ).toVar();
			const n = normalize( sw.nShore ).toVar();
			const info = sf.shading( { coverage: cov, foam: cov.mul( 0.8 ), footprint: float( 0.02 ), depth, bubbles: float( 0.5 ), normal: n, baseNormal: n, fresh: sw.foam, sim: s.x, simState: s, roller: sw.roller } );
			const lit = info.light( { N: n, L: normalize( vec3( 0.3, 0.6, 0.5 ) ), V: vec3( 0, 1, 0 ), sun: vec3( 3 ) } );
			return vec4( info.foam, info.density, lit.x, lit.y );

		} )();
		const scene = new THREE.Scene();
		const plane = new THREE.Mesh( new THREE.PlaneGeometry( 80, 80 ).rotateX( - Math.PI / 2 ), mat );
		plane.position.set( 0, 0, 30 );
		scene.add( plane );
		renderer.toneMapping = THREE.NoToneMapping;
		renderer.setRenderTarget( rt );
		renderer.render( scene, cam );
		renderer.setRenderTarget( null );
		const px = await renderer.readRenderTargetPixelsAsync( rt, 0, 0, W, W );
		writeFileSync( process.argv[ 2 ] + '.foam', Buffer.from( px.buffer, px.byteOffset, px.byteLength ) );
		console.log( 'surf foam', px.constructor.name, px.length );

	}
	// SeaDetail on a 128^2 grid over 4 km
	const det = new SeaDetail();
	det.offset.value.set( 13, - 7 );
	const dv = instancedArray( N * N, 'vec4' );
	renderer.compute( Fn( () => {

		const i = instanceIndex;
		const xz = vec2( float( i.mod( N ) ).mul( 31.3 ).sub( 2000 ), float( i.div( N ) ).mul( 31.3 ).sub( 2000 ) );
		const d = det.sample( xz );
		dv.element( i ).assign( vec4( d.rough, d.gust, d.slick, d.streak ) );

	} )().compute( N * N ) );
	writeFileSync( process.argv[ 2 ] + '.det', Buffer.from( await renderer.getArrayBufferAsync( dv.value ) ) );

}
console.log( 'wrote', buf.byteLength );
// ---- Breakers crest finder (no spray): the crest buffer after one update
if ( process.env.SIM ) {

	const { Breakers } = await import( REF + '/src/ocean/Breakers.js' );
	const { SurfFoam } = await import( REF + '/src/ocean/SurfFoam.js' );
	const zeros = new THREE.DataArrayTexture( new Uint16Array( 4 * 4 * 4 * 4 ), 4, 4, 4 );
	zeros.type = THREE.HalfFloatType; zeros.format = THREE.RGBAFormat; zeros.needsUpdate = true;
	const surface = {
		fft: { cascades: 4, sizes: [ 733, 157, 33.3, 7.1 ], displacementTexture: zeros },
		cascadeAttenuation: () => float( 1 ), amplitude: float( 1 ), terrain,
	};
	const cpuH = ( x, z ) => - SLOPE * z + 0.25 * Math.sin( x * 0.05 ) - ( z < - 3 ? ( z + 3 ) * 0.06 : 0 );
	const { mix, pow, max: tmax } = await import( REF + '/node_modules/three/build/three.tsl.js' );
	const sky = { reflectionRadiance: ( d ) => mix( vec3( 0.75, 0.82, 0.9 ), vec3( 0.18, 0.35, 0.75 ), pow( d.y.clamp( 0, 1 ), 0.5 ) ).mul( 1.2 ) };
	const brk = new Breakers( renderer, { surface, shore, terrainData: { heightAt: cpuH }, sky, spray: null } );
	G.time.value = Number( process.env.T || 47.3 );
	brk.update( { position: new THREE.Vector3() } );
	writeFileSync( process.argv[ 2 ] + '.brk', Buffer.from( await renderer.getArrayBufferAsync( brk.crestAttr ) ) );
	console.log( 'breaker stations', brk.NS );
	// the lip sheet from the beach (premultiplied colour over black)
	{

		const c = new Float32Array( await renderer.getArrayBufferAsync( brk.crestAttr ) );
		let best = - 1, bd = 9;
		for ( let i = 0; i < brk.NS * 2; i ++ ) if ( c[ i * 12 + 11 ] > 0.5 && Math.abs( c[ i * 12 + 3 ] - 0.65 ) < bd ) { bd = Math.abs( c[ i * 12 + 3 ] - 0.65 ); best = i; }
		const x = c[ best * 12 ], y = c[ best * 12 + 1 ], z = c[ best * 12 + 2 ];
		const cam = new THREE.PerspectiveCamera( 50, 16 / 9, 0.1, 500 );
		cam.position.set( x - 6, y + 0.4, z - 9 ); cam.lookAt( x, y - 0.3, z ); cam.updateMatrixWorld();
		G.sunDir.value.set( 0.2, 0.35, 0.9 ).normalize();
		G.sunColor.value.setRGB( 3, 2.8, 2.5 );
		G.skyIrradiance.value.setRGB( 0.3, 0.38, 0.5 );
		const rt = new THREE.RenderTarget( 640, 360, { type: THREE.FloatType } );
		const scene = new THREE.Scene();
		scene.add( brk.mesh );
		brk.mesh.layers.set( 0 );
		brk.mesh.material.mrtNode = null; // (velocity MRT: not rendered here)
		renderer.setRenderTarget( rt );
		renderer.setClearColor( 0x000000, 0 );
		renderer.clear();
		renderer.render( scene, cam );
		renderer.setRenderTarget( null );
		const px = await renderer.readRenderTargetPixelsAsync( rt, 0, 0, 640, 360 );
		writeFileSync( process.argv[ 2 ] + '.lip', Buffer.from( px.buffer, px.byteOffset, px.byteLength ) );
		writeFileSync( process.argv[ 2 ] + '.lipcam', JSON.stringify( { p: cam.position.toArray(), t: [ x, y - 0.3, z ] } ) );

	}

}

process.exit( 0 );
