import * as THREE from 'three/webgpu';
import { Fn, float, vec2, vec3, vec4, instanceIndex, instancedArray, storage, If, texture, max, min, normalize, length, smoothstep } from 'three/tsl';
import { G } from '../core/Globals.js';

export const MAX_QUERIES = 64;

// Water surface queries on the GPU.
//   - slot 0 is always the camera (consumed the same frame by the waterline/underwater passes)
//   - other slots are gameplay points (boat hull samples, swimmer, particles...)
// Results are read back asynchronously for CPU physics (1-3 frames latency).
//
// Height is Eulerian: the FFT displacement is Lagrangian (x0 -> x0 + D(x0)), so we solve
// x0 + D(x0) = xz with a few fixed-point iterations.
export class WaterQuery {

	constructor( renderer, surface ) {

		this.renderer = renderer;
		this.surface = surface;
		this.inputs = new Float32Array( MAX_QUERIES * 4 );
		this.inputAttr = new THREE.StorageInstancedBufferAttribute( this.inputs, 4 );
		this.inputNode = storage( this.inputAttr, 'vec4', MAX_QUERIES ).toReadOnly();
		this.results = instancedArray( MAX_QUERIES, 'vec4' ).setName( 'waterQueryResults' );
		this.count = 1;
		this.cpu = new Float32Array( MAX_QUERIES * 4 );
		this.cpuValid = false;
		this._pending = false;
		// read-back bookkeeping: `version` increments with every new result, `resultTime` is the
		// simulation time the result was computed at (it arrives 1-3 frames later)
		this.version = 0;
		this.resultTime = 0;
		this.resultInputs = new Float32Array( MAX_QUERIES * 4 ); // query points of the latest result
		this._issueInputs = new Float32Array( MAX_QUERIES * 4 );
		this.latency = 0.05; // s, smoothed age of the results when they arrive
		this.frameLatency = 0;
		this.slots = new Map();

		this._build();

	}

	// Returns the displacement (vec3) of the full water surface at Lagrangian point x0.
	_disp( x0, depth ) {

		const S = this.surface;
		const fft = S.fft;
		const d = vec3( 0 ).toVar();
		for ( let c = 0; c < fft.cascades; c ++ ) {

			const s = texture( fft.displacementTexture, x0.div( fft.sizes[ c ] ) ).depth( c ).level( c === fft.cascades - 1 ? 2 : 0 ).xyz;
			d.addAssign( s.mul( S.cascadeAttenuation( c, depth ) ) );

		}

		d.mulAssign( S.amplitude );
		if ( S.shore ) {

			const ground = S.terrain.heightAt( x0 );
			d.addAssign( S.shore.evaluate( x0, depth, ground, { withNormal: false } ).disp );

		}

		if ( S.wake ) d.addAssign( S.wake.displacement( x0, depth ) );
		return d;

	}

	heightAtNode( xz ) {

		if ( ! this._heightFn ) {

			const S = this.surface;
			const dispFn = Fn( ( [ x0, depth ] ) => this._disp( x0, depth ) ).setLayout( {
				name: 'waterDispAt', type: 'vec3',
				inputs: [ { name: 'x0', type: 'vec2' }, { name: 'depth', type: 'float' } ],
			} );

			this._heightFn = Fn( ( [ p ] ) => {

				const depth = S.seaDepth( p ).toVar();
				const x0 = vec2( p ).toVar();
				for ( let i = 0; i < 2; i ++ ) {

					x0.assign( p.sub( dispFn( x0, depth ).xz ) );

				}

				return G.seaLevel.add( dispFn( x0, depth ).y );

			} ).setLayout( { name: 'waterHeightAt', type: 'float', inputs: [ { name: 'p', type: 'vec2' } ] } );

		}

		return this._heightFn( xz );

	}

	_build() {

		this.kernel = Fn( () => {

			const q = this.inputNode.element( instanceIndex );
			const xz = q.xy;
			const h = this.heightAtNode( xz ).toVar();
			const e = 0.35;
			const hx = this.heightAtNode( xz.add( vec2( e, 0 ) ) );
			const hz = this.heightAtNode( xz.add( vec2( 0, e ) ) );
			const n = normalize( vec3( h.sub( hx ).div( e ), 1, h.sub( hz ).div( e ) ) );
			const seaFloor = this.surface.terrain ? this.surface.terrain.heightAt( xz ) : float( - 500 );
			this.results.element( instanceIndex ).assign( vec4( h, n.x, n.z, seaFloor ) );

		} )().compute( MAX_QUERIES ).setName( 'Water Queries' );

	}

	// ---------------------------------------------------------------- CPU API

	setCamera( x, z ) {

		this.inputs[ 0 ] = x;
		this.inputs[ 1 ] = z;

	}

	// allocate named slots (e.g. 'boat' -> 24 points). Returns the first index.
	allocate( name, n ) {

		if ( this.slots.has( name ) ) return this.slots.get( name ).start;
		const start = this.count;
		if ( start + n > MAX_QUERIES ) throw new Error( 'WaterQuery: out of slots' );
		this.slots.set( name, { start, n } );
		this.count += n;
		return start;

	}

	setPoint( i, x, z ) {

		this.inputs[ i * 4 ] = x;
		this.inputs[ i * 4 + 1 ] = z;

	}

	// last read-back results: { height, nx, nz, floor }
	get( i, out = {} ) {

		const c = this.cpu;
		out.height = c[ i * 4 ];
		out.nx = c[ i * 4 + 1 ];
		out.nz = c[ i * 4 + 2 ];
		out.floor = c[ i * 4 + 3 ];
		return out;

	}

	update() {

		this.inputAttr.needsUpdate = true;
		this.renderer.compute( this.kernel );

		if ( ! this._pending ) {

			this._pending = true;
			const issued = G.time.value;
			this._issueInputs.set( this.inputs );
			this.renderer.getArrayBufferAsync( this.results.value ).then( ( buf ) => {

				this.cpu.set( new Float32Array( buf ) );
				this.cpuValid = true;
				this._pending = false;
				this.latency += ( Math.min( G.time.value - issued, 0.25 ) - this.latency ) * 0.2;
				this.resultTime = issued;
				this.resultInputs.set( this._issueInputs );
				this.version ++;

			} ).catch( () => {

				this._pending = false;

			} );

		}

	}

	// TSL: camera water state (same frame)
	cameraState() {

		return this.results.element( 0 );

	}

}
