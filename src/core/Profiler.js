// GPU timestamp profiler (enable with ?profile in the URL).
export class Profiler {

	constructor( renderer ) {

		this.renderer = renderer;
		this.enabled = renderer.backend.trackTimestamp === true;
		this.nodes = new Map(); // name -> compute node
		this.result = { compute: 0, render: 0, items: [] };
		this._busy = false;
		this._t = 0;

	}

	track( name, node ) {

		this.nodes.set( name, node );

	}

	async _resolve() {

		const r = this.renderer;
		const backend = r.backend;
		try {

			const c = await r.resolveTimestampsAsync( 'compute' );
			const g = await r.resolveTimestampsAsync( 'render' );
			const items = [];
			const pool = backend.timestampQueryPool.compute;
			for ( const [ name, node ] of this.nodes ) {

				const data = backend.get( node );
				if ( data && data.timestampUID && pool && pool.timestamps.has( data.timestampUID ) ) {

					items.push( { name, ms: pool.timestamps.get( data.timestampUID ) } );

				}

			}

			const rp = backend.timestampQueryPool.render;
			if ( rp ) for ( const [ uid, ms ] of rp.timestamps ) items.push( { name: 'render ' + uid, ms } );
			items.sort( ( a, b ) => b.ms - a.ms );
			this.result = { compute: c || 0, render: g || 0, items };

		} catch ( e ) { /* ignore */ }

		this._busy = false;

	}

	update( dt ) {

		if ( ! this.enabled ) return;
		this._t += dt;
		if ( this._t > 0.5 && ! this._busy ) {

			this._t = 0;
			this._busy = true;
			this._resolve();

		}

	}

}
