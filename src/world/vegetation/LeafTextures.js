import * as THREE from 'three/webgpu';
import { Fn, float, vec2, vec4, uv, floor, fract, abs, cos, sin, atan, pow, max, clamp, mix, length, smoothstep, select, texture } from 'three/tsl';
import { hash12 } from './VegNodes.js';

// Leaf-cluster cards baked once on the GPU (instead of evaluating the leaf shapes per fragment):
// a 2 x 2 atlas of tiles, each tile a card of many small twig-end leaf whorls (tropical almond /
// sea hibiscus style: 8-25 cm leaves clustered at the twig tips) on a jittered grid with empty
// cells, in two offset layers: clumps with sky holes, no regular pattern. 4x supersampled,
// mipmapped.
//   R coverage, G brightness structure (leaf tip lightening, midrib) / 1.4, B per-leaf random
// Tiles: 0 tree, broad leaves   1 tree, narrow leaves   2 shrub, round leaves   3 shrub, narrow leaves
export const LEAF_TILES = [
	{ grid: 4, leaves: 6, width: 0.34 },
	{ grid: 4, leaves: 8, width: 0.2 },
	{ grid: 3, leaves: 6, width: 0.45 },
	{ grid: 3, leaves: 8, width: 0.22 },
];
const SIZE = 1024;

// leaf rosettes tiled G x G over a card: each cell holds one rosette with its own rotation /
// size; leaves have a short petiole gap at the centre. Returns { d: signed distance to the leaf
// edge (in rosette units, > 0 inside), bright, cell }.
const rosette = ( st, G, nLeaves, width, rot ) => {

	const cuv = st.mul( G );
	const cell = floor( cuv );
	const h1 = hash12( cell.add( rot.mul( 17.3 ) ) );
	const h2 = hash12( cell.mul( 1.7 ).add( rot.mul( 31.1 ) ).add( 5.2 ) );
	const h3 = hash12( cell.mul( 2.3 ).add( rot.mul( 7.7 ) ).add( 1.9 ) );
	// whorls of different sizes, off the cell centres; some cells stay empty (sky holes)
	const sc = mix( 0.62, 1.05, h2 ).mul( select( h3.lessThan( 0.2 ), float( 0.001 ), float( 1 ) ) );
	const off = vec2( h1.sub( 0.5 ), h3.sub( 0.5 ) ).mul( 0.34 );
	const p = fract( cuv ).sub( 0.5 ).sub( off ).mul( 2 ).div( sc );
	const r = length( p );
	const ang = atan( p.y, p.x ).add( rot ).add( h1.mul( 6.2832 ) );
	const sector = ang.mul( nLeaves / 6.2832 );
	const k = floor( sector.add( 0.5 ) );
	// irregular leaf spacing and lengths so clusters don't read as flowers
	const lr = hash12( vec2( k, h1.mul( 13.7 ) ) );
	const jit = lr.sub( 0.5 ).mul( 0.8 );
	const L = mix( 0.5, 1.0, fract( lr.mul( 7.13 ) ) );
	// each blade curves a little to one side (leaves, not a star)
	const curve = fract( lr.mul( 3.31 ) ).sub( 0.5 ).mul( 0.9 );
	const da = sector.sub( k ).sub( jit ).mul( 6.2832 / nLeaves ).sub( curve.mul( r.div( L ) ).mul( r.div( L ) ).mul( 0.35 ) );
	const along = r.mul( cos( da ) );
	const across = abs( r.mul( sin( da ) ) );
	const x = along.div( L );
	const bx = clamp( x.sub( 0.18 ).div( 0.82 ), 0, 1 ); // blade after a short petiole
	// obovate (widest beyond the middle) with a short pointed tip
	const shape = pow( max( sin( pow( bx, 0.62 ).mul( 3.14159 ) ), 0 ), 0.8 ).mul( smoothstep( 1.0, 0.86, bx ).mul( 0.25 ).add( 0.75 ) );
	const hw = shape.mul( width ).mul( L );
	const inBlade = x.greaterThan( 0.16 ).and( x.lessThan( 1 ) );
	const dBlade = select( inBlade, hw.sub( across ), float( - 1 ) );
	const dPet = select( x.lessThan( 0.2 ), float( 0.012 ).sub( across ), float( - 1 ) );
	const d = max( dBlade, dPet );
	const vein = smoothstep( 0.03, 0.0, across ).mul( smoothstep( 0.05, 0.25, bx ) ).mul( smoothstep( 1.0, 0.7, bx ) );
	const bright = h1.mul( 0.7 ).add( lr.mul( 0.3 ) ).mul( 0.22 ).add( 0.88 ).mul( mix( 0.82, 1.08, bx ) ).mul( vein.mul( 0.09 ).add( 1 ) );
	return { d, bright, cell: h1.mul( 0.7 ).add( lr.mul( 0.3 ) ) };

};

export class LeafAtlas {

	constructor() {

		this.rt = new THREE.RenderTarget( SIZE, SIZE, {
			type: THREE.UnsignedByteType, depthBuffer: false, generateMipmaps: true,
			minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter,
		} );
		this.rt.texture.name = 'vegLeafClusters';
		this.rt.texture.colorSpace = THREE.NoColorSpace;
		this.rt.texture.anisotropy = 4;
		this.texture = this.rt.texture;
		this.baked = false;

	}

	bake( renderer ) {

		const mat = new THREE.MeshBasicNodeMaterial();
		mat.colorNode = Fn( () => {

			const st = uv();
			const tile = floor( st.mul( 2 ) );
			const tid = tile.x.add( tile.y.mul( 2 ) );
			const local = fract( st.mul( 2 ) );
			const cov = float( 0 ).toVar(), bright = float( 0 ).toVar(), cellR = float( 0 ).toVar();
			const px = 1 / ( SIZE / 2 ); // texel size in tile uv
			for ( let t = 0; t < 4; t ++ ) {

				const T = LEAF_TILES[ t ];
				const isT = tid.equal( t );
				// 2 x 2 supersampling
				for ( let s = 0; s < 4; s ++ ) {

					const o = vec2( ( s % 2 ) - 0.5, Math.floor( s / 2 ) - 0.5 ).mul( px * 0.5 );
					const q = local.add( o );
					const A = rosette( q, T.grid, T.leaves, T.width, float( 0 ) );
					const B = rosette( q.add( 0.5 / T.grid ), T.grid, T.leaves, T.width, float( 2.1 ) );
					const inA = A.d.greaterThan( 0 ), inB = B.d.greaterThan( 0 );
					const hit = inA.or( inB ).and( isT );
					cov.addAssign( select( hit, float( 0.25 ), float( 0 ) ) );
					bright.addAssign( select( hit, select( inA, A.bright, B.bright ), float( 0 ) ).mul( 0.25 ) );
					cellR.addAssign( select( hit, select( inA, A.cell, B.cell ), float( 0 ) ).mul( 0.25 ) );

				}

			}

			// colour channels are stored un-premultiplied (valid where covered)
			const inv = float( 1 ).div( max( cov, 1e-3 ) );
			return vec4( cov, bright.mul( inv ).div( 1.4 ), cellR.mul( inv ), 1 );

		} )();

		const quad = new THREE.Mesh( new THREE.PlaneGeometry( 2, 2 ), mat );
		quad.frustumCulled = false;
		const scene = new THREE.Scene();
		scene.add( quad );
		const cam = new THREE.OrthographicCamera( - 1, 1, 1, - 1, 0, 2 );
		cam.position.z = 1;
		const prev = renderer.getRenderTarget();
		renderer.setRenderTarget( this.rt );
		renderer.render( scene, cam );
		renderer.setRenderTarget( prev );
		quad.geometry.dispose();
		mat.dispose();
		this.baked = true;

	}

	// TSL: sample tile `tile` (0..3, node) at card uv `st` -> vec4 (coverage, bright, cell, 1)
	sample( st, tile ) {

		const tuv = vec2( fract( tile.mul( 0.5 ) ).mul( 2 ), floor( tile.mul( 0.5 ) ) ).add( st.clamp( 0.004, 0.996 ) ).mul( 0.5 );
		// render targets are stored top row first
		return texture( this.texture, vec2( tuv.x, float( 1 ).sub( tuv.y ) ) );

	}

}
