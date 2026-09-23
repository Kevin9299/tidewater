// Bake the procedural skin (skin.mjs) into the texture atlas of the finest level of detail.
//   node bake_textures.mjs <model dir>   (headless WebGPU via the npm 'webgpu' package)
// Writes <model dir>/humpback_albedo.png (sRGB albedo + roughness in alpha) and
// <model dir>/humpback_height.png (relief height, 16 bit in R/G, cavity in B).
import fs from 'node:fs';
import zlib from 'node:zlib';
// headless WebGPU canvas helper (npm 'webgpu')
const { makeCanvas } = await import( './headless.mjs' );
const THREE = await import( 'three/webgpu' );
const TSL = await import( 'three/tsl' );
const { makeSkin } = await import( './skin.mjs' );
const { Fn, vec4, attribute, varyingProperty, positionLocal, mrt, output, float } = TSL;

const dir = process.argv[ 2 ];
const W = Number( process.argv[ 3 ] || 4096 ), H = W / 2;
const man = JSON.parse( fs.readFileSync( dir + '/humpback.json' ) );
const bin = fs.readFileSync( dir + '/humpback.bin' );
const buf = bin.buffer.slice( bin.byteOffset, bin.byteOffset + bin.byteLength );
const lv = man.levels[ 0 ];
const geo = new THREE.BufferGeometry();
geo.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array( buf, lv.position, lv.vertices * 3 ), 3 ) );
geo.setAttribute( 'uv', new THREE.BufferAttribute( new Float32Array( buf, lv.uv, lv.vertices * 2 ), 2 ) );
geo.setAttribute( 'uv1', new THREE.BufferAttribute( new Float32Array( buf, lv.uv1, lv.vertices * 2 ), 2 ) );
geo.setAttribute( 'rig', new THREE.BufferAttribute( new Float32Array( buf, lv.rig, lv.vertices * 4 ), 4 ) );
geo.setIndex( new THREE.BufferAttribute( new Uint32Array( buf, lv.index, lv.indices ), 1 ) );

const canvas = makeCanvas( 64, 64 );
const renderer = new THREE.WebGPURenderer( { canvas, antialias: false } );
await renderer.init();
const skin = makeSkin( man );
const vP = varyingProperty( 'vec3', 'bP' ), vPart = varyingProperty( 'float', 'bPart' ), vUV = varyingProperty( 'vec2', 'bUV' );
const mat = new THREE.NodeMaterial();
mat.side = THREE.DoubleSide;
mat.vertexNode = Fn( () => {

	vP.assign( positionLocal );
	vPart.assign( attribute( 'rig', 'vec4' ).y );
	vUV.assign( attribute( 'uv1', 'vec2' ) );
	const uv = attribute( 'uv', 'vec2' );
	return vec4( uv.x.mul( 2 ).sub( 1 ), float( 1 ).sub( uv.y.mul( 2 ) ), 0.5, 1 );

} )();
const rt = new THREE.RenderTarget( W, H, { type: THREE.FloatType, count: 2, depthBuffer: false, generateMipmaps: false } );
rt.textures[ 0 ].name = 'albedo';
rt.textures[ 1 ].name = 'height';
// one skin evaluation, two outputs: albedo + roughness, height + coverage
const both = Fn( () => {

	const s = skin( vP, vPart, vUV );
	return { c: s.color, h: vec4( s.height, 1, 0, 1 ) };

} );
let _s = null;
const colorOut = Fn( () => { const s = skin( vP, vPart, vUV ); return s.color; } )();
const heightOut = Fn( () => { const s = skin( vP, vPart, vUV ); return vec4( s.height, 1, 0, 1 ); } )();
mat.colorNode = colorOut;
mat.mrtNode = mrt( { albedo: colorOut, height: heightOut } );
const scene = new THREE.Scene();
const mesh = new THREE.Mesh( geo, mat );
mesh.frustumCulled = false;
scene.add( mesh );
const cam = new THREE.OrthographicCamera( - 1, 1, 1, - 1, 0, 1 );
renderer.setRenderTarget( rt );
renderer.setClearColor( 0x000000, 0 );
const t0 = performance.now();
renderer.render( scene, cam );
const col = await renderer.readRenderTargetPixelsAsync( rt, 0, 0, W, H, 0 );
const hgt = await renderer.readRenderTargetPixelsAsync( rt, 0, 0, W, H, 1 );
console.log( 'baked', W, H, ( ( performance.now() - t0 ) / 1000 ).toFixed( 1 ), 's' );

// ---- padding: grow every island by 12 texels (no seams under filtering / mips)
const cov = new Uint8Array( W * H );
for ( let i = 0; i < W * H; i ++ ) cov[ i ] = hgt[ i * 4 + 1 ] > 0.5 ? 1 : 0;
for ( let it = 0; it < 12; it ++ ) {

	const next = cov.slice();
	for ( let y = 0; y < H; y ++ ) for ( let x = 0; x < W; x ++ ) {

		const i = y * W + x;
		if ( cov[ i ] ) continue;
		let n = 0; const acc = [ 0, 0, 0, 0, 0 ];
		for ( const [ dx, dy ] of [ [ 1, 0 ], [ - 1, 0 ], [ 0, 1 ], [ 0, - 1 ] ] ) {

			const xx = x + dx, yy = y + dy;
			if ( xx < 0 || yy < 0 || xx >= W || yy >= H ) continue;
			const j = yy * W + xx;
			if ( ! cov[ j ] ) continue;
			for ( let k = 0; k < 4; k ++ ) acc[ k ] += col[ j * 4 + k ];
			acc[ 4 ] += hgt[ j * 4 ];
			n ++;

		}

		if ( n ) {

			for ( let k = 0; k < 4; k ++ ) col[ i * 4 + k ] = acc[ k ] / n;
			hgt[ i * 4 ] = acc[ 4 ] / n;
			next[ i ] = 1;

		}

	}

	cov.set( next );

}

// ---- encode (readback rows are bottom-up in WebGPU? detect by checking the body band position)
let hmin = Infinity, hmax = - Infinity;
for ( let i = 0; i < W * H; i ++ ) { const h = hgt[ i * 4 ]; if ( h < hmin ) hmin = h; if ( h > hmax ) hmax = h; }
console.log( 'height range', hmin.toFixed( 4 ), hmax.toFixed( 4 ) );
const srgb = ( v ) => { v = Math.min( 1, Math.max( 0, v ) ); return Math.round( 255 * ( v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow( v, 1 / 2.4 ) - 0.055 ) ); };
const A = new Uint8Array( W * H * 4 ), B = new Uint8Array( W * H * 4 );
// small-scale cavity from the height (darkens grooves, pits, barnacle gaps)
const blur = new Float32Array( W * H );
const R = 6;
for ( let y = 0; y < H; y ++ ) { let s = 0; for ( let x = - R; x < W + R; x ++ ) { const xa = Math.min( W - 1, Math.max( 0, x + R ) ), xb = Math.min( W - 1, Math.max( 0, x - R - 1 ) ); s += hgt[ ( y * W + xa ) * 4 ] - ( x - R - 1 >= - R ? hgt[ ( y * W + xb ) * 4 ] : 0 ); if ( x >= 0 && x < W ) blur[ y * W + x ] = s / ( 2 * R + 1 ); } }
const blur2 = new Float32Array( W * H );
for ( let x = 0; x < W; x ++ ) { let s = 0; for ( let y = - R; y < H + R; y ++ ) { const ya = Math.min( H - 1, Math.max( 0, y + R ) ), yb = Math.min( H - 1, Math.max( 0, y - R - 1 ) ); s += blur[ ya * W + x ] - ( y - R - 1 >= - R ? blur[ yb * W + x ] : 0 ); if ( y >= 0 && y < H ) blur2[ y * W + x ] = s / ( 2 * R + 1 ); } }
for ( let i = 0; i < W * H; i ++ ) {

	const cav = Math.max( 0.25, Math.min( 1, 1 + ( hgt[ i * 4 ] - blur2[ i ] ) * 70 ) );
	A[ i * 4 ] = srgb( col[ i * 4 ] * cav ); A[ i * 4 + 1 ] = srgb( col[ i * 4 + 1 ] * cav ); A[ i * 4 + 2 ] = srgb( col[ i * 4 + 2 ] * cav );
	A[ i * 4 + 3 ] = Math.round( 255 * Math.min( 1, Math.max( 0, col[ i * 4 + 3 ] ) ) );
	const hn = Math.round( ( hgt[ i * 4 ] - hmin ) / ( hmax - hmin ) * 65535 );
	B[ i * 4 ] = hn >> 8; B[ i * 4 + 1 ] = hn & 255; B[ i * 4 + 2 ] = Math.round( cav * 255 ); B[ i * 4 + 3 ] = 255;

}

function crc32( b ) { let c, crc = 0xffffffff; for ( let n = 0; n < b.length; n ++ ) { c = ( crc ^ b[ n ] ) & 0xff; for ( let k = 0; k < 8; k ++ ) c = c & 1 ? 0xedb88320 ^ ( c >>> 1 ) : c >>> 1; crc = ( crc >>> 8 ) ^ c; } return ( crc ^ 0xffffffff ) >>> 0; }
function png( path, data, flipRows ) {

	// filter type 2 (Up) on every row: compresses well and decodes with a single add per byte
	const stride = W * 4, raw = Buffer.alloc( ( stride + 1 ) * H );
	for ( let y = 0; y < H; y ++ ) {

		const sy = flipRows ? H - 1 - y : y, py = flipRows ? H - y : y - 1;
		raw[ y * ( stride + 1 ) ] = 2;
		for ( let x = 0; x < stride; x ++ ) {

			const cur = data[ sy * stride + x ], prev = y > 0 ? data[ py * stride + x ] : 0;
			raw[ y * ( stride + 1 ) + 1 + x ] = ( cur - prev ) & 255;

		}

	}

	const chunk = ( type, body ) => { const len = Buffer.alloc( 4 ); len.writeUInt32BE( body.length ); const tb = Buffer.concat( [ Buffer.from( type ), body ] ); const c = Buffer.alloc( 4 ); c.writeUInt32BE( crc32( tb ) ); return Buffer.concat( [ len, tb, c ] ); };
	const ihdr = Buffer.alloc( 13 ); ihdr.writeUInt32BE( W, 0 ); ihdr.writeUInt32BE( H, 4 ); ihdr[ 8 ] = 8; ihdr[ 9 ] = 6;
	fs.writeFileSync( path, Buffer.concat( [ Buffer.from( [ 137, 80, 78, 71, 13, 10, 26, 10 ] ), chunk( 'IHDR', ihdr ), chunk( 'IDAT', zlib.deflateSync( raw, { level: 9 } ) ), chunk( 'IEND', Buffer.alloc( 0 ) ) ] ) );

}

const flip = process.env.FLIP === '1';
png( dir + '/humpback_albedo.png', A, flip );
png( dir + '/humpback_height.png', B, flip );
man.textures = { albedo: 'humpback_albedo.png', height: 'humpback_height.png', size: [ W, H ], heightRange: [ hmin, hmax ] };
fs.writeFileSync( dir + '/humpback.json', JSON.stringify( man ) );
console.log( 'wrote', fs.statSync( dir + '/humpback_albedo.png' ).size, fs.statSync( dir + '/humpback_height.png' ).size );
process.exit( 0 );
