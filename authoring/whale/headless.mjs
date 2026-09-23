// Headless three.js WebGPURenderer on Dawn (node 'webgpu' package) for offline visual checks.
import { create, globals } from 'webgpu';
import zlib from 'node:zlib';
import fs from 'node:fs';

Object.assign( globalThis, globals );
const gpu = create( [] );
Object.defineProperty( globalThis.navigator, 'gpu', { value: gpu, configurable: true } );
globalThis.requestAnimationFrame = ( f ) => setTimeout( () => f( performance.now() ), 16 );
globalThis.cancelAnimationFrame = ( id ) => clearTimeout( id );
globalThis.self = globalThis;

export function makeCanvas( width, height ) {

	const ctx = {
		device: null, format: null, texture: null,
		configure( cfg ) {

			this.device = cfg.device; this.format = cfg.format;
			this.texture = null;

		},
		unconfigure() {},
		getCurrentTexture() {

			if ( ! this.texture || this.texture.width !== canvas.width || this.texture.height !== canvas.height ) {

				this.texture = this.device.createTexture( {
					size: [ canvas.width, canvas.height ],
					format: this.format,
					usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
				} );

			}

			return this.texture;

		},
	};
	const canvas = {
		width, height, style: {},
		addEventListener() {}, removeEventListener() {},
		getContext() { return ctx; },
		getBoundingClientRect() { return { left: 0, top: 0, width: canvas.width, height: canvas.height }; },
		clientWidth: width, clientHeight: height,
	};
	canvas._ctx = ctx;
	return canvas;

}

export async function readCanvas( canvas ) {

	const ctx = canvas._ctx;
	const device = ctx.device;
	const tex = ctx.texture;
	const w = tex.width, h = tex.height;
	const bpr = Math.ceil( w * 4 / 256 ) * 256;
	const buf = device.createBuffer( { size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ } );
	const enc = device.createCommandEncoder();
	enc.copyTextureToBuffer( { texture: tex }, { buffer: buf, bytesPerRow: bpr }, [ w, h ] );
	device.queue.submit( [ enc.finish() ] );
	await buf.mapAsync( GPUMapMode.READ );
	const src = new Uint8Array( buf.getMappedRange() );
	const out = new Uint8Array( w * h * 4 );
	const bgra = ctx.format.startsWith( 'bgra' );
	for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

		const s = y * bpr + x * 4, d = ( y * w + x ) * 4;
		out[ d ] = src[ s + ( bgra ? 2 : 0 ) ]; out[ d + 1 ] = src[ s + 1 ]; out[ d + 2 ] = src[ s + ( bgra ? 0 : 2 ) ]; out[ d + 3 ] = 255;

	}

	buf.unmap();
	return { w, h, data: out };

}

function crc32( buf ) {

	let c, crc = 0xffffffff;
	for ( let n = 0; n < buf.length; n ++ ) {

		c = ( crc ^ buf[ n ] ) & 0xff;
		for ( let k = 0; k < 8; k ++ ) c = c & 1 ? 0xedb88320 ^ ( c >>> 1 ) : c >>> 1;
		crc = ( crc >>> 8 ) ^ c;

	}

	return ( crc ^ 0xffffffff ) >>> 0;

}

export function writePNG( path, { w, h, data } ) {

	const raw = Buffer.alloc( ( w * 4 + 1 ) * h );
	for ( let y = 0; y < h; y ++ ) {

		raw[ y * ( w * 4 + 1 ) ] = 0;
		Buffer.from( data.buffer, y * w * 4, w * 4 ).copy( raw, y * ( w * 4 + 1 ) + 1 );

	}

	const chunk = ( type, body ) => {

		const len = Buffer.alloc( 4 ); len.writeUInt32BE( body.length );
		const tb = Buffer.concat( [ Buffer.from( type ), body ] );
		const crc = Buffer.alloc( 4 ); crc.writeUInt32BE( crc32( tb ) );
		return Buffer.concat( [ len, tb, crc ] );

	};

	const ihdr = Buffer.alloc( 13 );
	ihdr.writeUInt32BE( w, 0 ); ihdr.writeUInt32BE( h, 4 );
	ihdr[ 8 ] = 8; ihdr[ 9 ] = 6; ihdr[ 10 ] = 0; ihdr[ 11 ] = 0; ihdr[ 12 ] = 0;
	const png = Buffer.concat( [
		Buffer.from( [ 137, 80, 78, 71, 13, 10, 26, 10 ] ),
		chunk( 'IHDR', ihdr ),
		chunk( 'IDAT', zlib.deflateSync( raw ) ),
		chunk( 'IEND', Buffer.alloc( 0 ) ),
	] );
	fs.writeFileSync( path, png );

}
