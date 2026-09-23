import * as THREE from 'three/webgpu';
import {
	float, vec2, vec3, vec4, smoothstep, saturate, mix, max, dot, normalize, positionWorld, dFdx, dFdy, cross, abs, sign,
	pow, If, clamp, fract, floor, sin, texture, sqrt, length,
} from 'three/tsl';
import { G } from '../core/Globals.js';

// Surf-zone foam: the lace texture shared by the shore simulation, and the foam look used by the
// water shader (hooks called from WaterSurface.fragment and WaterMaterial.shade).
//
// Whitewater is a volume of bubbles: a dense mat that is bright from every side, with a lumpy,
// bubbly surface (darker in its own dips), tearing into patches and then into lace (thin bubble
// strands around clear holes) as it decays. Thin foam is translucent over the water colour.

export const LACE_TILE = 3.5; // metres per tile of the lace texture

// Tileable lace texture (generated once on the CPU, with CPU mipmaps: no GPU pipeline needed)
//   R: distance to the nearest bubble strand (warped cell edges + noise contours), 0 on a strand,
//      1 in the middle of a hole: thresholding it by the foam amount gives a dense mat, foam with
//      holes, lace and thin strands as the foam decays
//   G: small bubbles clustered along the strands
//   B: soft mottling (foam density variation)
//   A: per-cell random value (staggers when stranded foam pops)
let cachedLace = null;

export function makeLaceTexture( size = 512 ) {

	if ( cachedLace ) return cachedLace;
	const t0 = performance.now();
	const data = laceData( size );
	const mips = [ { data, width: size, height: size } ];
	let src = data, s = size;
	while ( s > 1 ) {

		const h = s >> 1;
		const dst = new Uint8Array( h * h * 4 );
		for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < h; x ++ ) for ( let c = 0; c < 4; c ++ ) {

			const a = src[ ( ( 2 * y ) * s + 2 * x ) * 4 + c ], b = src[ ( ( 2 * y ) * s + 2 * x + 1 ) * 4 + c ];
			const d = src[ ( ( 2 * y + 1 ) * s + 2 * x ) * 4 + c ], e = src[ ( ( 2 * y + 1 ) * s + 2 * x + 1 ) * 4 + c ];
			dst[ ( y * h + x ) * 4 + c ] = ( a + b + d + e + 2 ) >> 2;

		}

		mips.push( { data: dst, width: h, height: h } );
		src = dst;
		s = h;

	}

	const tex = new THREE.DataTexture( data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType );
	tex.mipmaps = mips;
	tex.generateMipmaps = false;
	tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
	tex.magFilter = THREE.LinearFilter;
	tex.minFilter = THREE.LinearMipmapLinearFilter;
	tex.anisotropy = 4;
	tex.colorSpace = THREE.NoColorSpace;
	tex.name = 'surfLace';
	tex.needsUpdate = true;
	tex.userData.ms = performance.now() - t0;
	// the same pattern for shaders with no sampler to spare: nearest filtering means no sampler
	// binding (read with loads, filtered by hand, see ShoreSim.laceLoad)
	const near = new THREE.DataTexture( data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType );
	near.magFilter = near.minFilter = THREE.NearestFilter;
	near.generateMipmaps = false;
	near.colorSpace = THREE.NoColorSpace;
	near.name = 'surfLaceNearest';
	near.needsUpdate = true;
	tex.userData.nearest = near;
	tex.userData.size = size;
	cachedLace = tex;
	return tex;

}

function laceData( size ) {

	const data = new Uint8Array( size * size * 4 );
	const hash = ( x, y, s ) => {

		let h = ( x * 374761393 + y * 668265263 + s * 2246822519 ) | 0;
		h = Math.imul( h ^ ( h >>> 13 ), 1274126177 );
		h ^= h >>> 16;
		return ( h >>> 0 ) / 4294967296;

	};

	const vnoise = ( x, y, n, s ) => {

		const i = Math.floor( x ), j = Math.floor( y );
		const fx = x - i, fy = y - j;
		const ux = fx * fx * ( 3 - 2 * fx ), uy = fy * fy * ( 3 - 2 * fy );
		const w = ( a ) => ( ( a % n ) + n ) % n;
		const a = hash( w( i ), w( j ), s ), b = hash( w( i + 1 ), w( j ), s );
		const c = hash( w( i ), w( j + 1 ), s ), d = hash( w( i + 1 ), w( j + 1 ), s );
		return ( a * ( 1 - ux ) + b * ux ) * ( 1 - uy ) + ( c * ( 1 - ux ) + d * ux ) * uy;

	};

	const fbm = ( x, y, n, s, oct = 3 ) => {

		let v = 0, a = 0.5, t = 0;
		for ( let o = 0; o < oct; o ++ ) {

			v += vnoise( x * ( 1 << o ), y * ( 1 << o ), n * ( 1 << o ), s + o * 17 ) * a;
			t += a;
			a *= 0.5;

		}

		return v / t;

	};

	// Voronoi F1, F2 and nearest cell id on an n x n periodic jittered grid (coordinates in cells)
	const voronoi = ( x, y, n, s ) => {

		const i = Math.floor( x ), j = Math.floor( y );
		let f1 = 9, f2 = 9, id = 0;
		for ( let dj = - 1; dj <= 1; dj ++ ) for ( let di = - 1; di <= 1; di ++ ) {

			const ci = i + di, cj = j + dj;
			const wi = ( ( ci % n ) + n ) % n, wj = ( ( cj % n ) + n ) % n;
			const px = ci + 0.15 + 0.7 * hash( wi, wj, s ), py = cj + 0.15 + 0.7 * hash( wi, wj, s + 1 );
			const d = Math.hypot( px - x, py - y );
			if ( d < f1 ) {

				f2 = f1; f1 = d; id = hash( wi, wj, s + 2 );

			} else if ( d < f2 ) f2 = d;

		}

		return [ f1, f2, id ];

	};

	const sstep = ( a, b, x ) => {

		const t = Math.min( 1, Math.max( 0, ( x - a ) / ( b - a ) ) );
		return t * t * ( 3 - 2 * t );

	};

	// pass 1: warped coordinates and the contour noise at every texel
	const N = size * size;
	const UU = new Float32Array( N ), VV = new Float32Array( N ), NC = new Float32Array( N );
	for ( let py = 0; py < size; py ++ ) for ( let px = 0; px < size; px ++ ) {

		const u = ( px + 0.5 ) / size, v = ( py + 0.5 ) / size;
		// two-level domain warp: organic, curvy strands
		const w1x = ( fbm( u * 3, v * 3, 3, 3 ) - 0.5 ) * 0.16, w1y = ( fbm( u * 3 + 5.2, v * 3 + 1.3, 3, 7 ) - 0.5 ) * 0.16;
		const w2x = ( fbm( ( u + w1x ) * 9, ( v + w1y ) * 9, 9, 13 ) - 0.5 ) * 0.07, w2y = ( fbm( ( u + w1x ) * 9 + 2.7, ( v + w1y ) * 9, 9, 17 ) - 0.5 ) * 0.07;
		const k = py * size + px;
		UU[ k ] = u + w1x + w2x;
		VV[ k ] = v + w1y + w2y;
		NC[ k ] = fbm( UU[ k ] * 6, VV[ k ] * 6, 6, 91, 3 );

	}

	// pass 2: distance to the nearest strand
	const N1 = 16, half = 0.5 / N1;
	const D = new Float32Array( N );
	for ( let py = 0; py < size; py ++ ) for ( let px = 0; px < size; px ++ ) {

		const u = ( px + 0.5 ) / size, v = ( py + 0.5 ) / size;
		const k = py * size + px;
		const uu = UU[ k ], vv = VV[ k ];
		// strands 1: edges of a warped cell network (distance to the edge ~ (F2 - F1) / 2, in cells)
		const [ a1, b1, id1 ] = voronoi( uu * N1, vv * N1, N1, 11 );
		const dCells = ( b1 - a1 ) * 0.5 / N1;
		// strands 2: contour loops of the warped noise; distance ~ |n - c| / |grad n| (texture units)
		const n1 = NC[ k ];
		const xl = ( px + size - 1 ) % size, xr = ( px + 1 ) % size, yd = ( py + size - 1 ) % size, yu = ( py + 1 ) % size;
		const gx = ( NC[ py * size + xr ] - NC[ py * size + xl ] ) * size * 0.5;
		const gy = ( NC[ yu * size + px ] - NC[ yd * size + px ] ) * size * 0.5;
		const g = Math.max( Math.hypot( gx, gy ), 0.5 );
		const dLoops = Math.min( Math.abs( n1 - 0.5 ), Math.abs( n1 - 0.36 ), Math.abs( n1 - 0.64 ) ) / g;
		// normalised by half a cell: 0 on a strand, ~1 in the middle of a hole. Irregular hole edges
		// (fine noise) and places where the strands break up (gaps).
		const hi = fbm( u * 24, v * 24, 24, 5, 2 );
		const gap = sstep( 0.52, 0.36, fbm( u * 10 + 3.1, v * 10, 10, 31 ) );
		const d = Math.min( dCells * 1.35 + 0.004, dLoops ) / half * ( 0.75 + 0.5 * hi ) + gap * 0.22;
		const mott = fbm( u * 3, v * 3, 3, 71, 4 );
		// bubbles: small dots, clustered near the strands
		const N3 = 120;
		const [ a3, , id3 ] = voronoi( u * N3, v * N3, N3, 61 );
		const rad = 0.1 + 0.22 * id3;
		const dot = sstep( rad, rad * 0.35, a3 ) * ( id3 > 0.45 ? 1 : 0 );
		const bub = dot * ( 0.25 + 0.75 * sstep( 0.5, 0.1, d ) );
		D[ k ] = Math.min( 1, d );
		data[ k * 4 + 1 ] = Math.round( Math.min( 1, bub ) * 255 );
		data[ k * 4 + 2 ] = Math.round( mott * 255 );
		data[ k * 4 + 3 ] = Math.round( id1 * 255 );

	}

	// pass 3: rounded holes: blurred distance inside the holes, the exact one near the strands
	const D0 = new Float32Array( D );
	const T = new Float32Array( N );
	for ( let it = 0; it < 2; it ++ ) {

		for ( let py = 0; py < size; py ++ ) for ( let px = 0; px < size; px ++ ) {

			let s = 0;
			for ( let o = - 2; o <= 2; o ++ ) s += D[ py * size + ( ( px + o + size ) % size ) ];
			T[ py * size + px ] = s / 5;

		}

		for ( let py = 0; py < size; py ++ ) for ( let px = 0; px < size; px ++ ) {

			let s = 0;
			for ( let o = - 2; o <= 2; o ++ ) s += T[ ( ( py + o + size ) % size ) * size + px ];
			D[ py * size + px ] = s / 5;

		}

	}

	for ( let k = 0; k < N; k ++ ) data[ k * 4 ] = Math.round( ( D0[ k ] + ( D[ k ] - D0[ k ] ) * sstep( 0.12, 0.45, D0[ k ] ) ) * 255 );
	return data;

}

// Henyey-Greenstein phase (1/sr)
const phaseHG = ( cosT, g ) => {

	const g2 = g * g;
	return float( ( 1 - g2 ) / ( 4 * Math.PI ) ).div( max( float( 1 + g2 ).sub( cosT.mul( 2 * g ) ), 1e-4 ).pow( 1.5 ) );

};

const hash11 = ( x ) => fract( sin( x.mul( 91.7 ).add( 17.3 ) ).mul( 43758.5453 ) );

// Foam look hooks for the water shader.
//   surface.foamShading = ( args ) => surfFoam.shading( args )   (WaterSurface.fragment)
//   foamInfo.light( { N, L, V, sun } )                           (WaterMaterial.shade)
//
// The pattern lives in world space, never in the coordinates of the displaced surface (those are
// squeezed and stretched by the big horizontal motion of the shore waves and would draw the foam
// into streaks). It is carried by the water with a dual-phase flow map: two copies of the pattern,
// half a cycle apart, each advected by the local flow (from ShoreSim) for one cycle and then
// re-placed at random, cross-faded so neither is ever stretched for long and the resets never show.
// Steep whitewater faces (bore / roller fronts) use a vertical projection that rolls down the face.
export class SurfFoam {

	constructor( { shoreSim } ) {

		this.sim = shoreSim;
		this.bump = 0.7; // relief of the whitewater and of thick foam (normal perturbation strength)
		this.period = 1.2; // s, flow map cycle
		this.maxFlow = 2.5; // m/s, the pattern lags behind faster flow (bounds the distortion per cycle)

	}

	// Lace distance (0 on a bubble strand .. 1 in a hole) at pattern coordinates q (m), carried by
	// `flow` (m/s in the same coordinates) with the dual-phase flow map. `salt` decorrelates layers.
	flowLace( q, flow, salt = 0 ) {

		const lace = this.sim.lace;
		const t = G.time.div( this.period );
		const v = clamp( flow, vec2( - this.maxFlow ), vec2( this.maxFlow ) ).mul( this.period );
		const out = vec4( 0 ).toVar();
		for ( let k = 0; k < 2; k ++ ) {

			const tk = t.add( k * 0.5 + salt * 0.37 );
			const ph = fract( tk );
			const cycle = floor( tk ).add( k * 13.1 + salt * 5.7 );
			const jitter = vec2( hash11( cycle ), hash11( cycle.add( 0.5 ) ) ).mul( 7.0 );
			const p = q.sub( v.mul( ph.sub( 0.5 ) ) ).div( LACE_TILE ).add( jitter );
			const w = float( 1 ).sub( ph.mul( 2 ).sub( 1 ).abs() );
			out.addAssign( texture( lace, p ).mul( w ) );

		}

		return out; // the two weights always add up to 1

	}

	// coverage: total foam amount (0..1, all sources); foam: the default (offshore whitecap) foam;
	// footprint: pixel size on the surface (m); depth: sea depth; bubbles: fine bubble detail;
	// normal: surface normal; fresh: whitewater made by the breaking wave right here (roller,
	// plunge point, swash front); sim: foam carried by the water (ShoreSim, already kept off the
	// clear face of plunging waves); simState: ShoreSim.sample() at this pixel; roller: relief of
	// the whitewater roller (m)
	shading( { coverage, foam, footprint, depth, bubbles, normal, baseNormal = null, fresh = float( 0 ), sim = float( 0 ), simState = null, roller = float( 0 ) } ) {

		const simS = this.sim;
		const xz = positionWorld.xz;
		const opacity = float( foam ).toVar();
		const density = saturate( coverage ).toVar();
		const height = float( 0 ).toVar();
		const wwOut = float( 0 ).toVar(); // how much of it is whitewater (deeper crevices)
		const wwRelief = float( 0 ).toVar(); // relief of the whitewater (m)
		const wwShadow = float( 1 ).toVar(); // sun visibility inside the churn
		const wwCav = float( 1 ).toVar(); // sky visibility in its crevices
		// surf look near the beach, the default whitecap look offshore
		const surf = smoothstep( 7.0, 3.0, depth ).mul( simS.inside( simS.uvOf( xz ) ) ).toVar();
		If( surf.greaterThan( 0 ).and( coverage.greaterThan( 0.04 ) ), () => {

			// flow of the water here, from the shore simulation (along the local wave direction)
			const dir = simS.shore.dirAt( xz ).xy.toVar();
			const speed = simState ? simState.w : float( 0 );
			const flow = dir.mul( speed );

			// --- pattern: world-space lace carried by the flow; on steep faces a vertical projection
			// (along the crest x height) rolling down the face with the roller
			// (from the normal of the wave itself: steep ripple facets must not switch the projection,
			// that drew combs of vertical streaks into the foam on flat water and on the swash)
			const steep = smoothstep( 0.82, 0.5, ( baseNormal || normal ).y ).toVar();
			const ww = saturate( fresh.mul( 1.4 ) ).toVar();
			const flat = this.flowLace( xz, flow, 0 ).toVar();
			const lace = flat.toVar();
			// lumps of tumbling whitewater (~0.6 m), only where there is whitewater
			const lumps = float( 0.5 ).toVar();
			const tangent = vec2( dir.y.negate(), dir.x );
			// (compressed vertically: a front only a metre or two high must not show single lumps as columns)
			// (the along-crest coordinate warped by low-frequency noise: the 3.5 m lace tile must not repeat as a
			// row of identical lumps and spikes along the break)
			const al = dot( xz, tangent );
			const qv = vec2( al.add( sin( al.mul( 0.19 ).add( 0.8 ) ).mul( 2.1 ) ).add( sin( al.mul( 0.47 ).add( 2.9 ) ).mul( 0.6 ) ), positionWorld.y.mul( 2.4 ) );
			If( steep.greaterThan( 0.01 ), () => {

				const vert = this.flowLace( qv, vec2( 0, - 1.8 ), 1 );
				lace.assign( mix( flat, vert, steep ) );

			} );
			// Churning whitewater is a pile of foam lumps at several scales (tumbling masses ~1.3 m,
			// clumps ~0.5 m, bubble clusters ~0.2 m): a relief (m) for the normals, sunlit caps and
			// self-shadowed crevices (a short march toward the sun through the lump field)
			If( ww.greaterThan( 0.02 ), () => {

				const L = G.sunDir;
				const pq = mix( xz, qv, steep ).toVar();
				const pflow = mix( flow, vec2( 0, - 1.8 ), steep ).toVar();
				const bigAt = ( qq ) => sqrt( this.flowLace( qq.div( 6.0 ), pflow.div( 6.0 ), 2 ).x );
				const big = bigAt( pq ).toVar();
				const mid = sqrt( this.flowLace( pq.div( 2.2 ), pflow.div( 2.2 ), 3 ).x );
				lumps.assign( big.mul( 0.6 ).add( mid.mul( 0.4 ) ) );
				const A = 0.22; // relief of the lumps (m)
				wwRelief.assign( big.mul( A ).add( mid.mul( A * 0.45 ) ).add( float( 1 ).sub( lace.x ).mul( A * 0.12 ) ).mul( ww ) );
				// the sun direction in the pattern's coordinates, and its elevation above the local surface
				const tangent2 = vec2( dir.y.negate(), dir.x );
				const Lp = mix( L.xz, vec2( dot( L.xz, tangent2 ), L.y.mul( 2.4 ) ), steep );
				const Ld = Lp.div( max( length( Lp ), 1e-3 ) );
				const NdL = dot( normal, L );
				const tanE = NdL.div( max( length( L.sub( normal.mul( NdL ) ) ), 0.05 ) );
				const occ = float( 0 ).toVar();
				for ( const d of [ 0.14, 0.34, 0.7 ] ) {

					const hk = bigAt( pq.add( Ld.mul( d ) ) );
					occ.assign( max( occ, smoothstep( 0.0, 0.05, hk.sub( big ).mul( A ).sub( tanE.mul( d ) ) ) ) );

				}

				wwShadow.assign( float( 1 ).sub( occ.mul( mix( float( 0.85 ), float( 0.7 ), steep ) ) ) );
				// crevices between the lumps: occluded from the sky too
				wwCav.assign( mix( float( 0.5 ), float( 1.0 ), smoothstep( 0.05, 0.75, lumps ) ) );

			} );

			// --- whitewater: the aerated mass of a roller / plunge / swash front. Dense and opaque, its
			// surface boiling: lumps with shaded crevices between them, bubble clusters on each; it only
			// tears (and shows water through) at its edges.
			const boil = lace.x.mul( 0.7 ).add( lace.z.mul( 0.3 ) );
			// (the edge of the churn is torn by its lumps: ragged fingers, not a clean boundary)
			const wwEdge = smoothstep( 0.25, 0.75, ww.mul( 1.35 ).sub( boil.mul( 0.3 ) ).sub( lumps.oneMinus().mul( 0.5 ) ) );
			const whitewater = ww.mul( 0.25 ).add( wwEdge.mul( 0.75 ) ).mul( boil.oneMinus().mul( 0.12 ).add( 0.88 ) );

			// --- foam carried by the water: a lacy web of bubble strands and clusters around holes. With
			// more foam the strands widen into a mat; as it spreads and thins it tears into filaments.
			// (w: strand half-width; the pattern covers 6% of the area at w = 0.1, 34% at 0.3, 82% at 0.6)
			const c = saturate( sim.add( coverage.sub( sim ).sub( fresh ).max( 0 ).mul( 0.5 ) ).sub( 0.05 ).div( 0.95 ) ).toVar();
			// hole edges are ragged (bubble clusters), hole sizes vary between patches
			const w = pow( c, 1.4 ).mul( 0.9 ).mul( lace.z.mul( 0.5 ).add( 0.75 ) ).add( lace.y.sub( 0.5 ).mul( 0.06 ) ).max( 0 ).toVar();
			const soft = float( 0.05 ).add( footprint.mul( 4.5 ) );
			const mat = float( 1 ).sub( smoothstep( w.sub( soft ), w.add( soft ), lace.x ) ).mul( smoothstep( 0.0, 0.05, c ) );
			// scattered bubbles in the holes next to the strands
			const bub = lace.y.mul( smoothstep( w.add( 0.25 ), w, lace.x ) ).mul( smoothstep( 0.02, 0.2, c ) ).mul( 0.5 );
			// thin foam is translucent and uneven, thick foam is opaque
			const inner = saturate( w.sub( lace.x ).div( 0.25 ) ).toVar();
			const laceFoam = max( mat.mul( saturate( inner.mul( 0.35 ).add( 0.45 ).add( lace.z.mul( 0.3 ) ) ) ), bub );

			// once a lace cell (~0.2 m) covers a few pixels, use the average of the pattern
			const far = smoothstep( 0.03, 0.12, footprint );
			const average = max( saturate( pow( w, 1.45 ).mul( 1.9 ) ).mul( 0.8 ), ww.mul( 0.95 ) );
			const near = max( laceFoam, whitewater );
			opacity.assign( mix( foam, mix( near, average, far ), surf ) );

			// optical thickness (thin foam is translucent, thick foam scatters like snow) and a relief
			// height for the lighting: lumpy boiling whitewater, thick foam higher than its thin edges
			density.assign( max( saturate( c.mul( 1.3 ) ).mul( inner.mul( 0.5 ).add( 0.5 ) ), ww ) );
			const relief = inner.mul( saturate( c.mul( 1.5 ) ) ).mul( float( 1 ).sub( ww ) );
			wwOut.assign( ww.mul( float( 1 ).sub( far.mul( 0.6 ) ) ) );
			// far away the lumps average out: a mean shadowing of the churn instead
			wwShadow.assign( mix( wwShadow, float( 0.82 ), far ) );
			wwCav.assign( mix( wwCav, float( 0.8 ), far ) );
			wwRelief.mulAssign( float( 1 ).sub( far ) );
			height.assign( relief.add( bubbles.mul( 0.15 ) ).mul( surf ).mul( float( 1 ).sub( far ) ) );

		} );

		return {
			foam: opacity,
			density,
			light: ( args ) => this.light( { ...args, density, height, surf, ww: wwOut, relief: wwRelief, selfShadow: wwShadow, cavity: wwCav } ),
		};

	}

	// Foam radiance: a dense scatterer, wrapped diffuse sun (light diffuses through the bubbles),
	// sky ambient, darker in the dips of the bubbly relief, glowing at thin edges when backlit.
	light( { N, L, V, sun, density, height, surf, ww = float( 0 ), relief = float( 0 ), selfShadow = float( 1 ), cavity = float( 1 ) } ) {

		// relief normal from the screen-space gradient of the height (Mikkelsen surface gradient): the
		// thin-foam relief (in units of ~3 cm) plus the whitewater lumps (m)
		const p = positionWorld;
		const hN = height.mul( this.bump * 0.04 ).add( relief );
		const dpdx = dFdx( p ), dpdy = dFdy( p );
		const dhdx = dFdx( hN ), dhdy = dFdy( hN );
		const r1 = cross( dpdy, N ), r2 = cross( N, dpdx );
		const det = dot( dpdx, r1 );
		const grad = r1.mul( dhdx ).add( r2.mul( dhdy ) ).mul( sign( det ) );
		const Nf = normalize( N.mul( abs( det ) ).sub( grad ).add( N.mul( 1e-6 ) ) );
		const NdL = dot( Nf, L );
		// foam lets light diffuse into it (wrapped lighting); churning whitewater much less so: its sides
		// facing away from the sun are shaded grey-blue by the sky, its caps sunlit, its crevices in the
		// shadow of the lumps around them
		const wrap = mix( float( 0.45 ), float( 0.12 ), ww );
		const diff = saturate( NdL.mul( float( 1 ).sub( wrap ) ).add( wrap ) ).mul( mix( float( 1 ), selfShadow, ww ) );
		// dips between the lumps are shaded by the lumps around them (sky occlusion)
		const ao = mix( mix( float( 1 ), height.mul( 0.3 ).add( 0.76 ), surf ), cavity, ww );
		// light through thin aerated water (torn edges, thin foam, spray-soaked lips): green-white,
		// strongly forward scattered
		const thin = float( 1 ).sub( density ).mul( 0.8 ).add( ww.mul( float( 1 ).sub( cavity ) ).mul( 0.3 ) );
		const trans = phaseHG( dot( V.negate(), L ), 0.55 ).mul( thin ).mul( 1.3 );
		const transCol = mix( vec3( 1 ), vec3( 0.6, 0.92, 0.82 ), ww.mul( 0.7 ).add( 0.3 ) );
		// light bounced around inside the churn (from its sunlit lumps) keeps the shaded foam from going
		// as dark and as blue as the open sky alone would make it
		const sky = G.skyIrradiance;
		const skyGrey = vec3( dot( sky, vec3( 0.2126, 0.7152, 0.0722 ) ) );
		const amb = mix( sky, skyGrey, ww.mul( 0.35 ) ).mul( ao ).add( sun.mul( ww.mul( 0.05 ) ).mul( ao ) );
		return sun.mul( diff.mul( mix( float( 1 ), cavity.sqrt(), ww ) ).div( Math.PI ).add( transCol.mul( trans ) ) ).add( amb ).mul( 0.86 );

	}
}
