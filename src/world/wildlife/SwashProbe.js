import * as THREE from 'three/webgpu';
import { Fn, If, vec4, storage, instancedArray, instanceIndex } from 'three/tsl';
import { G } from '../../core/Globals.js';

// Where is the edge of the swash? The run-up of the waves on the sand is analytic on the GPU
// (ShoreWaves); this evaluates it at a few points (the shorebirds) with the very same code the
// water uses, and reads the result back for the CPU (1-3 frames old, like WaterQuery).
//
// Per point: run-up of the current wave Rt and the point's own "inland" distance (both in metres
// up the beach, measured with the nominal beach slope), the speed of the leading edge (dRt/dt,
// > 0 while the water runs up) and the phase of the swash cycle.

export class SwashProbe {

	constructor( renderer, { shore, terrainGPU, count = 32, interval = 6 } ) {

		this.renderer = renderer;
		this.shore = shore;
		this.n = count;
		this.interval = interval; // frames between dispatches (the owner extrapolates in between)
		this._frame = 0;
		this.inputs = new Float32Array( count * 4 );
		this.inputAttr = new THREE.StorageBufferAttribute( this.inputs, 4 );
		this.inputAttr.setUsage( THREE.DynamicDrawUsage );
		const input = storage( this.inputAttr, 'vec4', count ).toReadOnly();
		this.results = instancedArray( count, 'vec4' ).setName( 'swashProbe' );
		this.cpu = new Float32Array( count * 4 );
		this.valid = false;
		this.issued = new Float32Array( count * 4 ); // inputs of the latest result
		this.resultTime = 0;
		this._pending = false;
		this._inputsCopy = new Float32Array( count * 4 );
		this.active = false;

		const out = this.results;
		this.kernel = Fn( () => {

			const q = input.element( instanceIndex );
			If( q.w.greaterThan( 0.5 ), () => {

				const p = q.xy;
				const ground = terrainGPU.heightAt( p ).toVar();
				const sw = shore.evaluate( p, G.seaLevel.sub( ground ), ground, { withNormal: false } );
				out.element( instanceIndex ).assign( vec4( sw.runup, sw.inland, sw.dRdt, sw.tau ) );

			} );

		} )().compute( count ).setName( 'Swash Probe' );

	}

	set( i, x, z ) {

		const o = i * 4;
		this.inputs[ o ] = x;
		this.inputs[ o + 1 ] = z;
		this.inputs[ o + 3 ] = 1;
		this.active = true;

	}

	clear( i ) {

		this.inputs[ i * 4 + 3 ] = 0;

	}

	update() {

		if ( ! this.active ) return;
		this.active = false;
		if ( this._pending || ( this._frame ++ ) % this.interval !== 0 ) return;
		this.inputAttr.needsUpdate = true;
		this.renderer.compute( this.kernel );
		this._pending = true;
		const issued = G.time.value;
		this._inputsCopy.set( this.inputs );
		this.renderer.getArrayBufferAsync( this.results.value ).then( ( buf ) => {

			this.cpu.set( new Float32Array( buf ) );
			this.issued.set( this._inputsCopy );
			this.resultTime = issued;
			this.valid = true;
			this._pending = false;

		} ).catch( () => {

			this._pending = false;

		} );

	}

	// latest result for point i: { runup, inland, speed, tau, age } (null until one arrives)
	get( i, out ) {

		if ( ! this.valid || this.issued[ i * 4 + 3 ] < 0.5 ) return null;
		const c = this.cpu, o = i * 4;
		const age = G.time.value - this.resultTime;
		const tau = c[ o + 3 ] + age / Math.max( this.shore.period.value, 1 );
		out.inland = c[ o + 1 ];
		out.x = this.issued[ o ];
		out.z = this.issued[ o + 1 ];
		out.age = age;
		if ( tau < 1 ) {

			out.runup = c[ o ] + c[ o + 2 ] * age;
			out.speed = c[ o + 2 ];

		} else {

			// the next wave has reached the shoreline since: its edge races up the beach
			const t = ( tau - 1 ) * this.shore.period.value;
			out.speed = 3.2 * Math.max( 0.2, 1 - t / 3 );
			out.runup = 0.35 + 3.2 * t * ( 1 - t / 6 );

		}

		out.tau = tau % 1;
		return out;

	}

}
