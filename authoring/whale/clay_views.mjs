// Clay review renders of the finest level (rest pose): side, top, bottom, front, three-quarter.
//   node clay_views.mjs <model dir> <out dir>
import fs from 'node:fs';
// headless WebGPU canvas + PNG helpers (npm 'webgpu')
const { makeCanvas, readCanvas, writePNG } = await import( './headless.mjs' );
const THREE = await import( 'three/webgpu' );
const dir = process.argv[ 2 ], out = process.argv[ 3 ];
fs.mkdirSync( out, { recursive: true } );
const man = JSON.parse( fs.readFileSync( dir + '/humpback.json' ) );
const bin = fs.readFileSync( dir + '/humpback.bin' );
const buf = bin.buffer.slice( bin.byteOffset, bin.byteOffset + bin.byteLength );
const lv = man.levels[ 0 ];
const geo = new THREE.BufferGeometry();
geo.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array( buf, lv.position, lv.vertices * 3 ), 3 ) );
geo.setAttribute( 'normal', new THREE.BufferAttribute( new Int16Array( buf, lv.normal, lv.vertices * 3 ), 3, true ) );
geo.setIndex( new THREE.BufferAttribute( new Uint32Array( buf, lv.index, lv.indices ), 1 ) );
const W = 1600, H = 700;
const canvas = makeCanvas( W, H );
const renderer = new THREE.WebGPURenderer( { canvas, antialias: true } );
await renderer.init();
renderer.setSize( W, H, false );
const scene = new THREE.Scene();
scene.background = new THREE.Color( 0x6f7880 );
const mat = new THREE.MeshStandardNodeMaterial( { color: 0xb8b0a4, roughness: 0.75 } );
scene.add( new THREE.Mesh( geo, mat ) );
const key = new THREE.DirectionalLight( 0xffffff, 2.6 ); key.position.set( 4, 10, 6 ); scene.add( key );
const rim = new THREE.DirectionalLight( 0xffffff, 1.0 ); rim.position.set( - 6, 3, - 8 ); scene.add( rim );
scene.add( new THREE.HemisphereLight( 0xdde6ff, 0x404040, 1.1 ) );
const views = {
	side: [ [ 18, 0, - 1.5 ], [ 0, 0, - 1.5 ], 8.6 ], // looking at the left flank (+x)
	top: [ [ 0, 18, - 1.5 ], [ 0, 0, - 1.5 ], 8.6 ],
	bottom: [ [ 0, - 18, - 1.5 ], [ 0, 0, - 1.5 ], 8.6 ],
	front: [ [ 0, - 0.4, 22 ], [ 0, - 0.4, 0 ], 3.2 ],
	head: [ [ 5.5, 1.2, 6.5 ], [ 0.3, - 0.2, 3.2 ], 0 ],
	threequarter: [ [ 13, 5, 9 ], [ 0, - 0.5, - 1.5 ], 0 ],
};
for ( const [ name, [ p, t, half ] ] of Object.entries( views ) ) {
	let cam;
	if ( half > 0 ) { const a = W / H; cam = new THREE.OrthographicCamera( - half * a, half * a, half, - half, 0.1, 100 ); }
	else cam = new THREE.PerspectiveCamera( 35, W / H, 0.1, 200 );
	cam.position.set( ...p );
	if ( name === 'top' ) cam.up.set( 1, 0, 0 );
	if ( name === 'bottom' ) cam.up.set( - 1, 0, 0 );
	cam.lookAt( ...t );
	renderer.render( scene, cam );
	await renderer.backend.device.queue.onSubmittedWorkDone();
	writePNG( `${ out }/clay_${ name }.png`, await readCanvas( canvas ) );
}
console.log( 'clay views written to', out );
process.exit( 0 );
