// Compares the WGSL ShoreWaves with the reference TSL one (ocean-shore-ref.mjs output) on the stub beach.
// node test/ocean-shore-ref.mjs ref.bin && node test/ocean-shore-compare.mjs ref.bin
import './headless.mjs';
import { readFileSync } from 'node:fs';
import { GPU, StorageBuffer, ComputeKernel, readBuffer, G, commonModule } from '../src/engine/webgpu.js';
import { ShoreWaves } from '../src/ocean/ShoreWaves.js';
import { ShoreSim } from '../src/ocean/ShoreSim.js';
import { SeaDetail } from '../src/ocean/SeaDetail.js';
import { SurfFoam } from '../src/ocean/SurfFoam.js';
import { FullscreenPass, RenderTarget, readTexture } from '../src/engine/webgpu.js';
import { Vector2 } from '../src/engine/index.js';
import { existsSync } from 'node:fs';
import { makeTerrain, makeFFT, makeSky, heightAt } from './ocean-shore-stubs.mjs';
import { MeshRenderer, SunShadows, setFrameCamera } from '../src/engine/webgpu.js';
import { PerspectiveCamera, Scene } from '../src/engine/index.js';
import { Breakers } from '../src/ocean/Breakers.js';

await GPU.init( { headless: true } );
const terrain = makeTerrain();
const shore = new ShoreWaves( terrain );
G.time.value = Number( process.env.T || 47.3 );
const N = 128;
const out = new StorageBuffer( { count: N * N * 4, type: 'vec4f' } );
const k = new ComputeKernel( {
	modules: [ commonModule, shore.module, terrain.module ],
	bindings: { outB: { storage: out, access: 'read_write' } },
	workgroupSize: [ 64, 1, 1 ],
	code: /* wgsl */`
@compute @workgroup_size( WG_X ) fn main( @builtin( global_invocation_id ) gid: vec3u ) {
	let i = gid.x;
	if ( i >= ${ N * N }u ) { return; }
	let xz = vec2f( ( f32( i % ${ N }u ) + 0.5 ) / ${ N }.0 * 120.0 - 60.0, ( f32( i / ${ N }u ) + 0.5 ) / ${ N }.0 * 120.0 - 20.0 );
	let ground = terrainHeightAt( xz );
	let depth = frame.seaLevel - ground;
	let sw = shoreEvaluate( xz, depth, ground );
	let b = i * 4u;
	outB[ b ] = vec4f( sw.disp, sw.foam );
	outB[ b + 1u ] = vec4f( sw.nShore, sw.swashLevel );
	outB[ b + 2u ] = vec4f( sw.face, sw.roller, sw.swashCovered, sw.flowSpeed );
	let med = shoreSurfMedium( xz, depth );
	outB[ b + 3u ] = vec4f( shoreSwashClip( xz, 0.1 ), shoreCrestPath( xz, depth, normalize( vec3f( 0.2, -0.5, 0.8 ) ) ), med.scatter.y, sw.breaking );
}`,
} );
GPU.beginFrame();
k.dispatch( k.groups( N * N ) );
GPU.submit();
const mine = new Float32Array( await readBuffer( out.getGPU(), N * N * 64 ) );
const ref = new Float32Array( readFileSync( process.argv[ 2 ] ).buffer.slice( 0 ) );
const names = [ 'disp.x', 'disp.y', 'disp.z', 'foam', 'n.x', 'n.y', 'n.z', 'swashLevel', 'face', 'roller', 'covered', 'flowSpeed', 'swashClip', 'crestPath', 'medium', 'breaking' ];
let bad = 0;
for ( let c = 0; c < 16; c ++ ) {

	let maxErr = 0, at = - 1, n = 0, sumErr = 0, range = 0;
	for ( let i = 0; i < N * N; i ++ ) {

		const a = mine[ i * 16 + c ], b = ref[ i * 16 + c ];
		range = Math.max( range, Math.abs( b ) );
		const e = Math.abs( a - b );
		if ( ! ( e <= 1e-3 + 1e-3 * Math.abs( b ) ) ) n ++;
		sumErr += e;
		if ( e > maxErr || Number.isNaN( e ) ) { maxErr = e; at = i; }

	}

	if ( n ) bad ++;
	console.log( names[ c ].padEnd( 11 ), 'max |ref|', range.toFixed( 3 ).padStart( 9 ), 'max err', maxErr.toExponential( 2 ), 'mean', ( sumErr / N / N ).toExponential( 2 ), 'mismatches', n, at >= 0 && n ? `(at ${ at % N },${ Math.floor( at / N ) }: ${ mine[ at * 16 + c ] } vs ${ ref[ at * 16 + c ] })` : '' );

}

console.log( bad ? 'DIFFERENCES in ' + bad + ' channels' : 'MATCH' );

function stats( label, a, b, chNames ) {

	const C = chNames.length, n = a.length / C;
	for ( let c = 0; c < C; c ++ ) {

		let maxErr = 0, sum = 0, sumRef = 0, big = 0;
		for ( let i = 0; i < n; i ++ ) {

			const e = Math.abs( a[ i * C + c ] - b[ i * C + c ] );
			if ( ! ( e <= maxErr ) ) maxErr = e;
			sum += e; sumRef += Math.abs( b[ i * C + c ] );
			if ( e > 0.05 ) big ++;

		}

		console.log( label, chNames[ c ].padEnd( 8 ), 'mean |ref|', ( sumRef / n ).toFixed( 4 ), 'mean err', ( sum / n ).toExponential( 2 ), 'max err', maxErr.toFixed( 4 ), 'texels > 0.05:', big );

	}

}

if ( existsSync( process.argv[ 2 ] + '.sim' ) ) {

	const R = 256;
	const sim = new ShoreSim( null, { terrainGPU: terrain, shore, center: new Vector2( 0, 20 ), size: 160, res: R } );
	G.dt.value = 1 / 30;
	const t0 = G.time.value;
	for ( let f = 0; f < 300; f ++ ) {

		G.time.value = t0 + f / 30;
		GPU.beginFrame(); sim.update(); GPU.submit();

	}

	const st = new StorageBuffer( { count: R * R, type: 'vec4f' } );
	const cp = new ComputeKernel( { modules: [ sim.module ], bindings: { stB: { storage: st, access: 'read_write' } }, workgroupSize: [ 64, 1, 1 ], code: `
@compute @workgroup_size( WG_X ) fn main( @builtin( global_invocation_id ) gid: vec3u ) {
	if ( gid.x >= ${ R * R }u ) { return; }
	stB[ gid.x ] = textureLoad( shoreSimStateTex, vec2i( i32( gid.x % ${ R }u ), i32( gid.x / ${ R }u ) ), 0 );
}` } );
	GPU.beginFrame(); cp.dispatch( cp.groups( R * R ) ); GPU.submit();
	const a = new Float32Array( await readBuffer( st.getGPU(), R * R * 16 ) );
	const b = new Float32Array( readFileSync( process.argv[ 2 ] + '.sim' ).buffer.slice( 0 ) );
	stats( 'ShoreSim', a, b, [ 'foam', 'wet', 'residue', 'flow' ] );

	if ( existsSync( process.argv[ 2 ] + '.foam' ) ) {

		const sf = new SurfFoam( { shoreSim: sim } );
		const W = 512;
		const rt = new RenderTarget( W, W, { colors: [ 'rgba32float' ], label: 'foamCmp' } );
		const pass = new FullscreenPass( { label: 'foamCmp', colorFormats: [ 'rgba32float' ], modules: [ sf.module, terrain.module ], code: /* wgsl */`
fn fragment( in: FSIn ) -> vec4f {
	let xz = vec2f( -40.0 + in.uv.x * 80.0, -10.0 + in.uv.y * 80.0 );
	let P = vec3f( xz.x, 0.0, xz.y );
	let ground = terrainHeightAt( xz );
	let depth = frame.seaLevel - ground;
	let s = shoreSimSample( xz );
	let sw = shoreEvaluate( xz, depth, ground );
	let cov = sat( s.x + sw.foam );
	let n = normalize( sw.nShore );
	var a: SurfFoamArgs;
	a.coverage = cov; a.foam = cov * 0.8; a.footprint = 0.02; a.depth = depth; a.bubbles = 0.5; a.lagXZ = xz;
	a.normal = n; a.baseNormal = n; a.fresh = sw.foam; a.sim = s.x; a.simState = s; a.roller = sw.roller; a.P = P;
	let info = surfFoamShading( a );
	let lit = surfFoamLight( info, n, normalize( vec3f( 0.3, 0.6, 0.5 ) ), vec3f( 0.0, 1.0, 0.0 ), vec3f( 3.0 ), P );
	return vec4f( info.foam, info.density, lit.x, lit.y );
}` } );
		GPU.beginFrame(); pass.render( { colorViews: [ rt.texture ], clear: [ 0, 0, 0, 0 ] } ); GPU.submit();
		const img = await readTexture( rt.texture );
		const fa = new Float32Array( img.data.buffer || img.data, img.data.byteOffset || 0, W * W * 4 );
		const fb = new Float32Array( readFileSync( process.argv[ 2 ] + '.foam' ).buffer.slice( 0 ) );
		stats( 'SurfFoam', fa, fb, [ 'foam', 'density', 'lit.r', 'lit.g' ] );
		{ const pts = []; for ( let i = 0; i < W * W; i ++ ) if ( Math.abs( fa[ i * 4 + 2 ] - fb[ i * 4 + 2 ] ) > 0.05 ) pts.push( [ i % W, Math.floor( i / W ) ] ); console.log( "lit mismatch px", JSON.stringify( pts.filter( ( _, k ) => k % 25 === 0 ) ) ); }

	}

	const det = new SeaDetail();
	det.offset.value.set( 13, - 7 );
	const dv = new StorageBuffer( { count: N * N, type: 'vec4f' } );
	const dk = new ComputeKernel( { modules: [ det.module ], bindings: { dvB: { storage: dv, access: 'read_write' } }, workgroupSize: [ 64, 1, 1 ], code: `
@compute @workgroup_size( WG_X ) fn main( @builtin( global_invocation_id ) gid: vec3u ) {
	if ( gid.x >= ${ N * N }u ) { return; }
	let xz = vec2f( f32( gid.x % ${ N }u ) * 31.3 - 2000.0, f32( gid.x / ${ N }u ) * 31.3 - 2000.0 );
	let d = seaDetailSample( xz );
	dvB[ gid.x ] = vec4f( d.rough, d.gust, d.slick, d.streak );
}` } );
	GPU.beginFrame(); dk.dispatch( dk.groups( N * N ) ); GPU.submit();
	{

		const brk = new Breakers( null, { surface: { fft: makeFFT(), terrain, amplitude: { value: 1 } }, shore, terrainData: { heightAt }, sky: makeSky(), spray: null } );
		G.time.value = Number( process.env.T || 47.3 );
		GPU.beginFrame(); brk.update( { position: new Vector2() } ); GPU.submit();
		const a = new Float32Array( await readBuffer( brk.crest.getGPU(), brk.NS * 6 * 16 ) );
		const b = new Float32Array( readFileSync( process.argv[ 2 ] + '.brk' ).buffer.slice( 0 ) );
		console.log( 'breaker stations', brk.NS, 'valid crests', a.filter( ( v, i ) => i % 12 === 11 && v > 0.5 ).length, 'ref', b.filter( ( v, i ) => i % 12 === 11 && v > 0.5 ).length );
		if ( existsSync( process.argv[ 2 ] + '.lip' ) ) {

			const cam = JSON.parse( readFileSync( process.argv[ 2 ] + '.lipcam', 'utf8' ) );
			const camera = new PerspectiveCamera( 50, 16 / 9, 0.1, 500 );
			camera.position.fromArray( cam.p ); camera.lookAt( cam.t[ 0 ], cam.t[ 1 ], cam.t[ 2 ] );
			G.sunDir.value.set( 0.2, 0.35, 0.9 ).normalize();
			G.sunColor.value.setRGB( 3, 2.8, 2.5 );
			G.skyIrradiance.value.setRGB( 0.3, 0.38, 0.5 );
			const scene = new Scene(); scene.add( brk.mesh );
			const mr = new MeshRenderer(); new SunShadows();
			const rt = new RenderTarget( 640, 360, { colors: [ 'rgba16float' ], depth: 'depth32float', label: 'lipCmp' } );
			GPU.beginFrame();
			setFrameCamera( camera, 640, 360 );
			mr.render( scene, { camera, kind: 'color', layerMask: 0xffffffff, colorViews: [ rt.texture.view() ], colorFormats: [ 'rgba16float' ], clearColors: [ [ 0, 0, 0, 0 ] ], depthView: rt.depthTexture.view(), depthFormat: 'depth32float', clearDepth: 0 } );
			GPU.submit();
			const ob = new StorageBuffer( { count: 640 * 360, type: 'vec4f' } );
			const cp = new ComputeKernel( { bindings: { src: { texture: rt.texture }, ob: { storage: ob, access: 'read_write' } }, workgroupSize: [ 64, 1, 1 ], code: `
@compute @workgroup_size( WG_X ) fn main( @builtin( global_invocation_id ) gid: vec3u ) {
	if ( gid.x >= ${ 640 * 360 }u ) { return; }
	ob[ gid.x ] = textureLoad( src, vec2i( i32( gid.x % 640u ), i32( gid.x / 640u ) ), 0 );
}` } );
			GPU.beginFrame(); cp.dispatch( cp.groups( 640 * 360 ) ); GPU.submit();
			const la = new Float32Array( await readBuffer( ob.getGPU(), 640 * 360 * 16 ) );
			const lb = new Float32Array( readFileSync( process.argv[ 2 ] + '.lip' ).buffer.slice( 0 ) );
			let cov = 0; for ( let i = 0; i < 640 * 360; i ++ ) if ( lb[ i * 4 + 3 ] > 0.01 ) cov ++;
			console.log( 'lip coverage (ref px)', cov );
			stats( 'Lip', la, lb, [ 'r', 'g', 'b', 'a' ] );
			const toPng = ( f ) => { const o = new Uint8Array( 640 * 360 * 4 ); for ( let i = 0; i < 640 * 360 * 4; i ++ ) o[ i ] = i % 4 === 3 ? 255 : Math.min( 255, Math.pow( Math.max( f[ i ], 0 ) / ( 1 + f[ i ] ), 1 / 2.2 ) * 255 ); return o; };
			const { writePNG } = await import( './headless.mjs' );
			const dir = process.argv[ 2 ].replace( /[^/]*$/, '' );
			writePNG( dir + 'lip-native.png', 640, 360, toPng( la ) );
			writePNG( dir + 'lip-ref.png', 640, 360, toPng( lb ) );

		}

		stats( 'Crests', a, b, [ 'root.x', 'root.y', 'root.z', 'b', 'back.x', 'back.y', 'back.z', 'H', 'dir.x', 'dir.z', 'trough', 'id' ].slice( 0, 4 ).concat( [ 'back.x', 'back.y', 'back.z', 'H' ] ).concat( [ 'dir.x', 'dir.z', 'trough', 'id' ] ) );

	}

	stats( 'SeaDetail', new Float32Array( await readBuffer( dv.getGPU(), N * N * 16 ) ), new Float32Array( readFileSync( process.argv[ 2 ] + '.det' ).buffer.slice( 0 ) ), [ 'rough', 'gust', 'slick', 'streak' ] );

}
process.exit( 0 );
