import { Fn, uniform, float, vec2, vec3, fract, sin, dot, floor, length, sqrt, max, min, smoothstep, mix, clamp, exp, If, screenSize, atan } from 'three/tsl';

const hash2 = ( p ) => fract( sin( vec2( dot( p, vec2( 127.1, 311.7 ) ), dot( p, vec2( 269.5, 183.3 ) ) ) ).mul( 43758.5453 ) );

// Water left on the camera lens after surfacing: droplets of many sizes (no full-screen warp). Small ones cling and evaporate, large ones slide down after a random
// delay and leave a thin wet trail. Each droplet is a tiny lens: it shows a blurred, inverted
// view of the scene, with a bright sky highlight and a dark edge. Applied at output resolution
// after the temporal resolve, so drops stay glued to the lens while the scene moves.
export class LensDroplets {

	constructor() {

		this.wet = uniform( 0 ).setName( 'lensWet' ); // 0..1 water left on the lens
		this.age = uniform( 100 ).setName( 'lensAge' ); // seconds since surfacing
		this.seed = uniform( 0 ).setName( 'lensSeed' );
		this.duration = 9; // seconds until the lens is dry
		this._wasUnder = false;

	}

	update( dt, underwater ) {

		if ( underwater ) {

			this.wet.value = 0;

		} else {

			if ( this._wasUnder ) {

				this.wet.value = 1;
				this.age.value = 0;
				this.seed.value = Math.random() * 97;

			}

			this.age.value += dt;
			this.wet.value = Math.max( 0, this.wet.value - dt / this.duration );

		}

		this._wasUnder = underwater;

	}

	// returns a function( sharp(uv) -> vec3, blurred(uv) -> vec3, uv ) -> vec3 color
	build() {

		const wet = this.wet, age = this.age, seed = this.seed;

		return ( sharp, blurred, uv ) => Fn( () => {

			const col = sharp( uv ).toVar();
			If( wet.greaterThan( 0.001 ), () => {

				const aspect = screenSize.x.div( screenSize.y );
				const p = vec2( uv.x.mul( aspect ), uv.y ); // y grows downward on screen

				// ---- droplets (two layers: clinging small drops, sliding large drops)
				const n2 = vec2( 0 ).toVar(); // droplet normal (xy) of the drop covering this pixel
				const cover = float( 0 ).toVar(); // soft coverage (drops are out of focus)
				const trail = float( 0 ).toVar();

				const layer = ( cell, rMin, rMax, density, slide ) => {

					const c0 = floor( p.div( cell ) );
					for ( let j = - 1; j <= 1; j ++ ) for ( let i = - 1; i <= 1; i ++ ) {

						const c = c0.add( vec2( i, j ) );
						const h = hash2( c.add( seed ) );
						const h2 = hash2( c.add( seed ).add( 17.3 ) );
						const present = h.x.lessThan( density );
						const center = c.add( vec2( 0.2 ).add( h2.mul( 0.6 ) ) ).mul( cell ).toVar();
						// evaporation shrinks drops; big ones last longer
						const life = clamp( wet.mul( 1.6 ).sub( h2.y.mul( 0.6 ) ), 0, 1 );
						const r = mix( float( rMin ), float( rMax ), h.y.mul( h.y ) ).mul( sqrt( life ) ).toVar();
						if ( slide ) {

							// heavy drops start sliding after a delay and accelerate
							const t0 = h2.x.mul( 3 ).add( 0.4 );
							const s = max( age.sub( t0 ), 0 );
							const dy = s.mul( s ).mul( r.mul( 3.5 ) );
							center.y.addAssign( dy );
							// wet streak above a sliding drop
							const dxT = p.x.sub( center.x ).abs();
							const above = center.y.sub( p.y );
							const tr = smoothstep( r.mul( 0.45 ), 0.0, dxT ).mul( smoothstep( 0.0, 0.01, above ) ).mul( smoothstep( dy.add( 0.01 ), 0.0, above ) );
							trail.assign( max( trail, tr.mul( present.select( 1, 0 ) ).mul( life ) ) );

						}

						// irregular outline: a few lobes, sliding drops stretched vertically
						const d = p.sub( center ).mul( vec2( 1, slide ? 0.8 : 1 ) );
						const ang = atan( d.y, d.x );
						const wobble = float( 1 ).add( sin( ang.mul( 3 ).add( h.x.mul( 40 ) ) ).mul( 0.12 ) ).add( sin( ang.mul( 5 ).add( h2.y.mul( 30 ) ) ).mul( 0.06 ) );
						const q = d.div( max( r.mul( wobble ), 1e-4 ) );
						const rq = dot( q, q );
						const a = smoothstep( 1.0, 0.7, rq ).mul( present.select( 1, 0 ) );
						If( a.greaterThan( cover ), () => {

							n2.assign( q );
							cover.assign( a );

						} );

					}

				};

				layer( 0.05, 0.003, 0.011, wet.mul( 0.5 ), false );
				layer( 0.13, 0.01, 0.026, wet.mul( 0.22 ), true );

				If( cover.greaterThan( 0.001 ), () => {

					const r2 = min( dot( n2, n2 ), 1 );
					const nz = sqrt( max( float( 1 ).sub( r2 ), 0 ) );
					// a drop is a strong fisheye lens: the image inside is inverted and blurred
					const off = n2.mul( - 0.05 ).mul( float( 1 ).sub( nz.mul( 0.5 ) ) );
					const inside = blurred( uv.add( vec2( off.x.div( aspect ), off.y ) ) );
					const edge = smoothstep( 0.45, 1.0, r2 );
					const highlight = smoothstep( 0.3, 0.0, length( n2.sub( vec2( - 0.3, - 0.4 ) ) ) ).mul( 0.35 );
					const dropCol = inside.mul( mix( float( 1.04 ), float( 0.7 ), edge ) ).add( inside.mul( highlight ) );
					col.assign( mix( col, dropCol, cover ) );

				} );

				// wet trails: slight blur and darkening
				col.assign( mix( col, blurred( uv ).mul( 0.9 ), trail.mul( 0.6 ) ) );

			} );

			return col;

		} )();

	}

}
