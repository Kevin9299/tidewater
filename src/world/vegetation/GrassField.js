import * as THREE from 'three/webgpu';
import {
	Fn, float, vec2, vec3, vec4, ivec2, attribute, texture, varyingProperty, normalLocal, normalViewGeometry,
	positionWorld, positionViewDirection, cameraPosition,
	sin, cos, floor, mix, smoothstep, clamp, min, max, length, normalize, dot, select, saturate, pow, abs,
} from 'three/tsl';
import { physical } from '../../materials/Materials.js';
import { G } from '../../core/Globals.js';
import { mulberry32 } from '../../util/Noise.js';
import { hash12, vnoise, windStrength, windDir3, windPerp3, gustAt, uCamPos, UP } from './VegNodes.js';
import { getDetailTexture } from '../terrain/DetailTextures.js';
import { meadowTone, MEADOW } from '../terrain/TerrainShading.js';

// 2D rotation of an xz node by a constant angle (matches TerrainShading.rot2)
const rotXZ = ( v, a ) => {

	const c = Math.cos( a ), s = Math.sin( a );
	return vec2( v.x.mul( c ).sub( v.y.mul( s ) ), v.x.mul( s ).add( v.y.mul( c ) ) );

};

// Camera-following ground flora: tall meadow grass (knee to waist high), dune grass, sea oats and
// beach creeper.
//
// The world is divided into CELL x CELL metre cells. Every visible cell near the camera draws one
// instance of a "patch": a fixed blue-noise set of clump slots. In the vertex shader each slot is
// placed at cellOrigin + slot offset, its height comes from the terrain heightmap (manual bilinear
// textureLoad) and its presence / size from an RGBA density mask (R dune grass, G meadow grass,
// B sea oats, A creeper). Empty cells are skipped on the CPU and cells are frustum culled.
//
// Three levels share the same clumps and blades (identical parameters), so switching a cell
// between them never moves a blade:
//   near (< R_NEAR): every blade, 3 segments        mid (< R_MID): tiers 0-1, 2 segments
//   far (< R_FAR): tier 0, one triangle per blade
// Every blade has a tier; tier 2 blades fade out (narrow to nothing) before R_NEAR, tier 1 before
// R_MID, and tier 0 thins out (random per-blade cut-off) towards R_FAR where the terrain's meadow
// shading takes over. The remaining blades widen as the others fade, so the grass covers the
// ground equally at every distance (constant blade density x width).

export const CELL = 8;
export const R_NEAR = 18;
export const R_MID = 46;
export const R_FAR = 88;
export const FADE_T2 = [ 11, 17 ];
export const FADE_T1 = [ 36, 45 ];
export const FADE_T0 = [ 62, 87 ]; // tier 0 blades drop out at random distances in this range
const MAX_NEAR = 48;
const MAX_MID = 160;
const MAX_FAR = 420;
const CLUMPS = 384; // meadow / dune clump slots per cell (6 per m^2)
const BLADES = 7; // per clump
const BLADE_TIER = [ 0, 0, 1, 1, 2, 2, 2 ];
// blades per clump in each tier (coverage compensation)
const TIER_N = [ 2, 2, 3 ];
const OATS = 12; // sea oat slots per cell
const VINES = 16; // creeper slots per cell

const KIND = { GRASS: 0, OAT_STALK: 1, OAT_HEAD: 2, CREEPER: 3, FLOWER: 4 };

// Mitchell's best-candidate blue noise on a torus; any prefix is also well distributed.
function blueNoise( rand, n, size, k = 10 ) {

	const pts = [];
	for ( let i = 0; i < n; i ++ ) {

		let best = null, bestD = - 1;
		for ( let c = 0; c < ( i === 0 ? 1 : k ); c ++ ) {

			const x = rand() * size, z = rand() * size;
			let dmin = Infinity;
			for ( const p of pts ) {

				let dx = Math.abs( p[ 0 ] - x ), dz = Math.abs( p[ 1 ] - z );
				dx = Math.min( dx, size - dx );
				dz = Math.min( dz, size - dz );
				dmin = Math.min( dmin, dx * dx + dz * dz );

			}

			if ( dmin > bestD ) {

				bestD = dmin;
				best = [ x, z ];

			}

		}

		pts.push( best );

	}

	return pts;

}

class PatchBuilder {

	constructor() {

		this.pos = [];
		this.nor = [];
		this.side = [];
		this.slot = [];
		this.bladeData = [];
		this.idx = [];

	}

	get count() {

		return this.pos.length / 3;

	}

	// side: xyz offset direction * half width, w: across coordinate (-1 left edge, 1 right, 0 tip)
	v( p, n, side, slot, blade ) {

		this.pos.push( p[ 0 ], p[ 1 ], p[ 2 ] );
		this.nor.push( n[ 0 ], n[ 1 ], n[ 2 ] );
		this.side.push( side[ 0 ], side[ 1 ], side[ 2 ], side[ 3 ] ?? 0 );
		this.slot.push( slot[ 0 ], slot[ 1 ], slot[ 2 ], slot[ 3 ] );
		this.bladeData.push( blade[ 0 ], blade[ 1 ], blade[ 2 ], blade[ 3 ] );
		return this.count - 1;

	}

	// blade: centre-line points with half widths; tip row has zero width (single vertex).
	// w4: aBlade.w (grass: tier)
	blade( slot, kind, rows, width, dirX, dirZ, lean, height, curve, rnd, base = [ 0, 0 ], w4 = 0 ) {

		const sx = - dirZ, sz = dirX; // horizontal side direction
		const ids = [];
		for ( let r = 0; r < rows.length; r ++ ) {

			const h = rows[ r ];
			const out = ( Math.sin( lean ) * h + curve * h * h ) * height;
			const up = ( Math.cos( lean ) * h - curve * 0.35 * h * h ) * height;
			const p = [ base[ 0 ] + dirX * out, up, base[ 1 ] + dirZ * out ];
			// tangent for the normal
			const dOut = Math.sin( lean ) + 2 * curve * h, dUp = Math.cos( lean ) - 0.7 * curve * h;
			const tl = Math.hypot( dOut, dUp );
			const tx = dirX * dOut / tl, ty = dUp / tl, tz = dirZ * dOut / tl;
			// normal = side x tangent
			const nx = - sz * ty, ny = sz * tx - sx * tz, nz = sx * ty;
			const nl = Math.hypot( nx, ny, nz ) || 1;
			const n = [ nx / nl, ny / nl, nz / nl ];
			const hw = width * ( 1 - Math.pow( h, 1.6 ) ) * ( 0.75 + 0.25 * ( 1 - h ) );
			const blade = [ h, kind, rnd, w4 ];
			if ( r === rows.length - 1 ) {

				ids.push( [ this.v( p, n, [ 0, 0, 0, 0 ], slot, blade ) ] );

			} else {

				ids.push( [
					this.v( p, n, [ - sx * hw, 0, - sz * hw, - 1 ], slot, blade ),
					this.v( p, n, [ sx * hw, 0, sz * hw, 1 ], slot, blade ),
				] );

			}

		}

		for ( let r = 0; r < ids.length - 1; r ++ ) {

			const a = ids[ r ], b = ids[ r + 1 ];
			if ( b.length === 1 ) this.idx.push( a[ 0 ], a[ 1 ], b[ 0 ] );
			else this.idx.push( a[ 0 ], a[ 1 ], b[ 1 ], a[ 0 ], b[ 1 ], b[ 0 ] );

		}

	}

	// small leaf / flower lying on the ground: triangle fan around a centre point.
	// outline(a) gives the radius factor for angle a (0 = pointing along dir).
	fan( slot, kind, center, radius, dirX, dirZ, tilt, sides, outline, rnd, lift, hf = 0.1 ) {

		const c = this.v( [ center[ 0 ], lift, center[ 1 ] ], [ 0, 1, 0 ], [ 0, 0, 0, 0 ], slot, [ hf, kind, rnd, 1 ] );
		const ring = [];
		for ( let i = 0; i < sides; i ++ ) {

			const a = ( i / sides ) * Math.PI * 2;
			const r = radius * outline( a );
			const lx = Math.cos( a ) * r, lz = Math.sin( a ) * r * 0.85;
			const x = center[ 0 ] + lx * dirX - lz * dirZ;
			const z = center[ 1 ] + lx * dirZ + lz * dirX;
			const along = lx / radius;
			const y = lift + ( along + 0.6 ) * radius * tilt;
			ring.push( this.v( [ x, y, z ], [ 0, 1, 0 ], [ 0, 0, 0, 0 ], slot, [ hf + 0.05 * Math.max( 0, along ), kind, rnd, 0 ] ) );

		}

		for ( let i = 0; i < sides; i ++ ) this.idx.push( c, ring[ ( i + 1 ) % sides ], ring[ i ] );

	}

	// thin ribbon lying on the ground (creeper runner)
	ribbon( slot, kind, pts, width, rnd ) {

		const ids = [];
		for ( let i = 0; i < pts.length; i ++ ) {

			const p = pts[ i ], q = pts[ Math.min( pts.length - 1, i + 1 ) ], o = pts[ Math.max( 0, i - 1 ) ];
			let dx = q[ 0 ] - o[ 0 ], dz = q[ 2 ] - o[ 2 ];
			const l = Math.hypot( dx, dz ) || 1;
			dx /= l; dz /= l;
			const sx = - dz * width, sz = dx * width;
			ids.push( [
				this.v( [ p[ 0 ] - sx, p[ 1 ], p[ 2 ] - sz ], [ 0, 1, 0 ], [ 0, 0, 0, 0 ], slot, [ 0.05, kind, rnd, 0 ] ),
				this.v( [ p[ 0 ] + sx, p[ 1 ], p[ 2 ] + sz ], [ 0, 1, 0 ], [ 0, 0, 0, 0 ], slot, [ 0.05, kind, rnd, 0 ] ),
			] );

		}

		for ( let i = 0; i < ids.length - 1; i ++ ) {

			const a = ids[ i ], b = ids[ i + 1 ];
			this.idx.push( a[ 0 ], b[ 1 ], a[ 1 ], a[ 0 ], b[ 0 ], b[ 1 ] );

		}

	}

	build() {

		const g = new THREE.InstancedBufferGeometry();
		g.setAttribute( 'position', new THREE.Float32BufferAttribute( this.pos, 3 ) );
		g.setAttribute( 'normal', new THREE.Float32BufferAttribute( this.nor, 3 ) );
		g.setAttribute( 'aSide', new THREE.Float32BufferAttribute( this.side, 4 ) );
		g.setAttribute( 'aSlot', new THREE.Float32BufferAttribute( this.slot, 4 ) );
		g.setAttribute( 'aBlade', new THREE.Float32BufferAttribute( this.bladeData, 4 ) );
		g.setIndex( this.count > 65535 ? new THREE.Uint32BufferAttribute( this.idx, 1 ) : new THREE.Uint16BufferAttribute( this.idx, 1 ) );
		g.boundingSphere = new THREE.Sphere( new THREE.Vector3(), 1e7 );
		g.boundingBox = new THREE.Box3( new THREE.Vector3( - 1e7, - 1e7, - 1e7 ), new THREE.Vector3( 1e7, 1e7, 1e7 ) );
		return g;

	}

	get triangles() {

		return this.idx.length / 3;

	}

}

// Clump / blade parameters shared by all levels (same seeds -> identical blades).
function clumpParams( seed ) {

	const rand = mulberry32( seed );
	const slots = blueNoise( mulberry32( seed + 1 ), CLUMPS, CELL );
	return slots.map( ( [ x, z ] ) => {

		const slotRand = rand();
		const az0 = rand() * Math.PI * 2;
		const blades = [];
		for ( let k = 0; k < BLADES; k ++ ) {

			// golden-angle azimuths: any prefix of the blades spreads around the clump; the first
			// two (kept at every distance) lean away from each other
			const az = az0 + k * 2.39996 + ( rand() - 0.5 ) * 0.5;
			const outer = k === 0 ? 0.1 + 0.3 * rand() : k === 1 ? 0.35 + 0.4 * rand() : rand();
			blades.push( {
				dx: Math.cos( az ), dz: Math.sin( az ),
				lean: 0.06 + 0.55 * outer * ( 0.6 + 0.8 * rand() ),
				h: ( 0.7 + 0.5 * rand() ) * ( 1.06 - 0.22 * outer ),
				curve: 0.12 + 0.5 * outer * ( 0.5 + rand() ),
				rnd: rand(),
				base: [ Math.cos( az ) * 0.09 * rand(), Math.sin( az ) * 0.09 * rand() ],
				tier: BLADE_TIER[ k ],
			} );

		}

		return { x, z, slotRand, blades };

	} );

}

// level: 0 near, 1 mid, 2 far
function buildPatch( level, clumps, seed = 7 ) {

	const b = new PatchBuilder();
	const rows = [ [ 0, 0.3, 0.62, 1 ], [ 0, 0.55, 1 ], [ 0, 1 ] ][ level ];
	const maxTier = 2 - level;

	for ( const c of clumps ) {

		const slot = [ c.x, c.z, c.slotRand, 0 ];
		for ( const bl of c.blades ) {

			if ( bl.tier > maxTier ) continue;
			b.blade( slot, KIND.GRASS, rows, 0.012, bl.dx, bl.dz, bl.lean, bl.h, bl.curve, bl.rnd, bl.base, bl.tier );

		}

	}

	if ( level === 2 ) return b;

	// sea oats (tier 1: fade out before the far level) and, near only, beach creeper. Every slot
	// has its own random stream, so the near and mid versions of a plant are identical.
	const oatSlots = blueNoise( mulberry32( seed + 2 ), OATS, CELL );
	const near = level === 0;
	oatSlots.forEach( ( [ x, z ], si ) => {

		const rand = mulberry32( seed * 7919 + si * 131 + 17 );
		const slot = [ x, z, rand(), 1 ];
		const stalks = near ? 3 : 1;
		for ( let q = 0; q < 3; q ++ ) {

			const az = rand() * Math.PI * 2;
			const dx = Math.cos( az ), dz = Math.sin( az );
			const lean = 0.06 + 0.14 * rand();
			const hq = 0.8 + 0.3 * rand();
			const rnd = rand();
			const bx = q === 0 ? 0 : ( rand() - 0.5 ) * 0.22, bz = q === 0 ? 0 : ( rand() - 0.5 ) * 0.22;
			const hs = [ rand(), rand(), rand(), rand(), rand(), rand(), rand(), rand() ];
			if ( q >= stalks ) continue;
			const qTier = q === 0 ? 0 : 2;
			// stalk (unit height, scaled in the shader)
			b.blade( slot, KIND.OAT_STALK, near ? [ 0, 0.3, 0.6, 0.85, 1 ] : [ 0, 0.6, 1 ], near ? 0.0075 : 0.014, dx, dz, lean, hq, 0.2, rnd, [ bx, bz ], qTier );
			// drooping panicle: flat spikelets hanging off the upper stalk
			const heads = near ? 8 : 3;
			for ( let k = 0; k < heads; k ++ ) {

				const hf = 0.7 + 0.3 * ( k / Math.max( 1, heads - 1 ) );
				const out = ( Math.sin( lean ) * hf + 0.2 * hf * hf ) * hq;
				const up = ( Math.cos( lean ) * hf - 0.07 * hf * hf ) * hq;
				const sa = az + ( k % 2 ? 1 : - 1 ) * ( 0.45 + 0.6 * hs[ k ] );
				const sdx = Math.cos( sa ), sdz = Math.sin( sa );
				const cx = bx + dx * out + sdx * 0.02, cz = bz + dz * out + sdz * 0.02;
				const L = near ? 0.085 : 0.13, W = near ? 0.022 : 0.04;
				const n = [ 0, 1, 0 ];
				const slotB = [ hf, KIND.OAT_HEAD, rnd, qTier ];
				const a = b.v( [ cx, up + 0.01, cz ], n, [ 0, 0, 0, 0 ], slot, slotB );
				const l = b.v( [ cx + sdx * L * 0.45 - sdz * W, up - L * 0.4, cz + sdz * L * 0.45 + sdx * W ], n, [ 0, 0, 0, 0 ], slot, slotB );
				const r = b.v( [ cx + sdx * L * 0.45 + sdz * W, up - L * 0.4, cz + sdz * L * 0.45 - sdx * W ], n, [ 0, 0, 0, 0 ], slot, slotB );
				const t = b.v( [ cx + sdx * L * 0.8, up - L * 0.95, cz + sdz * L * 0.8 ], n, [ 0, 0, 0, 0 ], slot, slotB );
				b.idx.push( a, l, t, a, t, r );

			}

		}

		if ( near ) {

			for ( let k = 0; k < 4; k ++ ) {

				const la = ( k / 4 ) * Math.PI * 2 + rand();
				b.blade( slot, KIND.GRASS, [ 0, 0.45, 0.8, 1 ], 0.01, Math.cos( la ), Math.sin( la ), 0.55 + 0.4 * rand(), 0.45, 0.35, rand(), [ 0, 0 ], 2 );

			}

		}

	} );

	if ( ! near ) return b;

	// beach creeper (railroad vine): a runner crawling over the sand with notched leaves
	const notched = ( a ) => {

		const d = Math.min( Math.abs( a ), Math.abs( a - Math.PI * 2 ) );
		return 1 - 0.38 * Math.exp( - ( d * d ) / 0.12 ) - 0.15 * Math.pow( Math.sin( a * 0.5 ), 8 );

	};
	const petals = ( a ) => 0.82 + 0.18 * Math.cos( a * 5 );
	const vineSlots = blueNoise( mulberry32( seed + 3 ), VINES, CELL );
	vineSlots.forEach( ( [ x, z ], si ) => {

		const rand = mulberry32( seed * 3571 + si * 197 + 5 );
		const slot = [ x, z, rand(), 2 ];
		let az = rand() * Math.PI * 2;
		const pts = [];
		const n = 9;
		let px = - Math.cos( az ) * 0.65, pz = - Math.sin( az ) * 0.65;
		for ( let i = 0; i < n; i ++ ) {

			pts.push( [ px, 0.01, pz ] );
			az += ( rand() - 0.5 ) * 0.5;
			px += Math.cos( az ) * 0.16;
			pz += Math.sin( az ) * 0.16;

		}

		b.ribbon( slot, KIND.CREEPER, pts, 0.011, rand() );
		for ( let i = 1; i < pts.length; i ++ ) {

			const side = i % 2 ? 1 : - 1;
			const p0 = pts[ i - 1 ], p1 = pts[ i ];
			let dx = p1[ 0 ] - p0[ 0 ], dz = p1[ 2 ] - p0[ 2 ];
			const l = Math.hypot( dx, dz ) || 1;
			dx /= l; dz /= l;
			// leaves stand on short petioles, angled forward off the runner, tilted up
			const la = Math.atan2( dz, dx ) + side * ( 0.7 + 0.5 * rand() );
			const ldx = Math.cos( la ), ldz = Math.sin( la );
			const r = ( 0.05 + 0.03 * rand() ) * ( 0.6 + 0.4 * Math.sin( Math.PI * i / n ) );
			b.fan( slot, KIND.CREEPER, [ p1[ 0 ] + ldx * r * 0.9, p1[ 2 ] + ldz * r * 0.9 ], r, ldx, ldz, 0.3 + 0.3 * rand(), 7, notched, rand(), 0.015 );

		}

		const fp = pts[ 3 + Math.floor( rand() * 4 ) ];
		b.fan( slot, KIND.FLOWER, [ fp[ 0 ] + 0.03, fp[ 2 ] + 0.03 ], 0.036, 1, 0, 1.4, 10, petals, rand(), 0.07, 0.2 );

	} );

	return b;

}

// per-tier visibility at camera distance d (1 = full width, 0 = gone); tier 0 thins out per blade
const tierFade = ( tier, d, cut ) => {

	const f2 = float( 1 ).sub( smoothstep( FADE_T2[ 0 ], FADE_T2[ 1 ], d ) );
	const f1 = float( 1 ).sub( smoothstep( FADE_T1[ 0 ], FADE_T1[ 1 ], d ) );
	const f0 = float( 1 ).sub( smoothstep( cut.sub( 5 ), cut, d ) );
	return select( tier.greaterThan( 1.5 ), f2, select( tier.greaterThan( 0.5 ), f1, f0 ) );

};

export class GrassField {

	constructor( { terrain, mask } ) {

		this.terrain = terrain;
		const res = terrain.res;
		const maskData = mask.data, mres = mask.res;

		// terrain heights (float, loaded + bilinearly filtered manually) and density mask
		this.heightTex = new THREE.DataTexture( terrain.heights, res, res, THREE.RedFormat, THREE.FloatType );
		this.heightTex.minFilter = this.heightTex.magFilter = THREE.NearestFilter;
		this.heightTex.generateMipmaps = false;
		this.heightTex.needsUpdate = true;
		this.maskTex = new THREE.DataTexture( maskData, mres, mres, THREE.RGBAFormat, THREE.UnsignedByteType );
		this.maskTex.minFilter = this.maskTex.magFilter = THREE.LinearFilter;
		this.maskTex.generateMipmaps = false;
		this.maskTex.needsUpdate = true;

		// per-cell occupancy + height range (for skipping empty cells and frustum tests)
		const cellsPerSide = Math.ceil( terrain.size / CELL );
		this.cellsPerSide = cellsPerSide;
		this.cellFlags = new Uint8Array( cellsPerSide * cellsPerSide );
		this.cellMinY = new Float32Array( cellsPerSide * cellsPerSide );
		this.cellMaxY = new Float32Array( cellsPerSide * cellsPerSide );
		const mpc = CELL / ( terrain.size / mres ); // mask texels per cell
		const hpc = CELL / terrain.texel; // height texels per cell
		for ( let cj = 0; cj < cellsPerSide; cj ++ ) {

			for ( let ci = 0; ci < cellsPerSide; ci ++ ) {

				let any = 0, mn = Infinity, mx = - Infinity;
				const i0 = Math.max( 0, Math.floor( ci * mpc ) - 1 ), i1 = Math.min( mres - 1, Math.ceil( ( ci + 1 ) * mpc ) + 1 );
				const j0 = Math.max( 0, Math.floor( cj * mpc ) - 1 ), j1 = Math.min( mres - 1, Math.ceil( ( cj + 1 ) * mpc ) + 1 );
				for ( let j = j0; j <= j1 && ! any; j ++ ) for ( let i = i0; i <= i1; i ++ ) {

					const o = ( j * mres + i ) * 4;
					if ( maskData[ o ] > 8 || maskData[ o + 1 ] > 8 || maskData[ o + 2 ] > 8 || maskData[ o + 3 ] > 8 ) {

						any = 1;
						break;

					}

				}

				const c = cj * cellsPerSide + ci;
				this.cellFlags[ c ] = any;
				if ( ! any ) continue;
				const hi0 = Math.max( 0, Math.floor( ci * hpc ) - 1 ), hi1 = Math.min( res - 1, Math.ceil( ( ci + 1 ) * hpc ) + 1 );
				const hj0 = Math.max( 0, Math.floor( cj * hpc ) - 1 ), hj1 = Math.min( res - 1, Math.ceil( ( cj + 1 ) * hpc ) + 1 );
				for ( let j = hj0; j <= hj1; j ++ ) for ( let i = hi0; i <= hi1; i ++ ) {

					const h = terrain.heights[ j * res + i ];
					if ( h < mn ) mn = h;
					if ( h > mx ) mx = h;

				}

				this.cellMinY[ c ] = mn;
				this.cellMaxY[ c ] = mx;

			}

		}

		this.material = this._createMaterial();

		const clumps = clumpParams( 7 );
		const patches = [ buildPatch( 0, clumps ), buildPatch( 1, clumps ), buildPatch( 2, clumps ) ];
		this.patchTris = patches.map( ( p ) => p.triangles );
		this.levels = [
			this._createMesh( patches[ 0 ].build(), MAX_NEAR, 'veg-grass-near' ),
			this._createMesh( patches[ 1 ].build(), MAX_MID, 'veg-grass-mid' ),
			this._createMesh( patches[ 2 ].build(), MAX_FAR, 'veg-grass-far' ),
		];
		this.meshes = this.levels.map( ( l ) => l.mesh );

		this._frustum = new THREE.Frustum();
		this._mat = new THREE.Matrix4();
		this._box = new THREE.Box3();
		this._last = new Float64Array( 8 ).fill( NaN );

	}

	// back-compat accessors (stats)
	get near() {

		return this.levels[ 0 ];

	}

	get far() {

		return this.levels[ 2 ];

	}

	get nearTris() {

		return this.patchTris[ 0 ];

	}

	get farTris() {

		return this.patchTris[ 2 ];

	}

	_createMesh( geometry, max, name ) {

		const arr = new Float32Array( max * 4 );
		const attr = new THREE.InstancedBufferAttribute( arr, 4 );
		attr.setUsage( THREE.DynamicDrawUsage );
		geometry.setAttribute( 'iCell', attr );
		geometry.instanceCount = 0;
		const mesh = new THREE.Mesh( geometry, this.material );
		mesh.name = name;
		mesh.frustumCulled = false;
		mesh.castShadow = false;
		mesh.receiveShadow = true;
		mesh.matrixAutoUpdate = false;
		return { mesh, geometry, attr, arr, max, count: 0 };

	}

	_createMaterial() {

		const t = this.terrain;
		const res = t.res;
		const hTex = texture( this.heightTex );
		const mTex = texture( this.maskTex );

		const heightAt = Fn( ( [ xz ] ) => {

			const f = xz.sub( t.origin ).div( t.texel ).sub( 0.5 );
			const fi = floor( f );
			const fr = f.sub( fi );
			const ij = ivec2( clamp( fi, vec2( 0 ), vec2( res - 2 ) ) );
			const a = hTex.load( ij ).x;
			const b = hTex.load( ij.add( ivec2( 1, 0 ) ) ).x;
			const c = hTex.load( ij.add( ivec2( 0, 1 ) ) ).x;
			const d = hTex.load( ij.add( ivec2( 1, 1 ) ) ).x;
			return mix( mix( a, b, fr.x ), mix( c, d, fr.x ), fr.y );

		} ).setLayout( { name: 'vegTerrainHeight', type: 'float', inputs: [ { name: 'xz', type: 'vec2' } ] } );

		const vG = varyingProperty( 'vec4', 'vVegGrass' ); // hf, kind, dune fraction, rand
		const vTone = varyingProperty( 'vec4', 'vVegGrassTone' ); // meadow tone (shared with the terrain), dryness
		const det = texture( getDetailTexture() );
		const vG2 = varyingProperty( 'vec4', 'vVegGrass2' ); // density, leaf / flower centre flag, across, dry blade
		const vGust = varyingProperty( 'float', 'vVegGust' ); // current gust bend (wind sheen)

		const mat = physical( { side: THREE.DoubleSide, specularIntensity: 0.4 } );
		mat.name = 'veg-grass';

		mat.positionNode = Fn( () => {

			const cell = attribute( 'iCell', 'vec4' ).xy;
			const slot = attribute( 'aSlot', 'vec4' );
			const blade = attribute( 'aBlade', 'vec4' );
			const side4 = attribute( 'aSide', 'vec4' );
			const side = side4.xyz;
			const P = attribute( 'position', 'vec3' );
			const N = attribute( 'normal', 'vec3' );

			const xz = cell.add( slot.xy );
			const hf = blade.x;
			const kind = blade.y;
			const bRnd = blade.z;

			const m = mTex.sample( xz.sub( t.origin ).div( t.size ) ).level( 0 );
			const isGrass = kind.lessThan( 0.5 );
			const isOat = kind.greaterThan( 0.5 ).and( kind.lessThan( 2.5 ) );
			const lush = m.g;
			const dune = m.r;
			// dune tufts are sparse, the meadow is a closed sward
			const duneF = dune.div( dune.add( lush ).add( 1e-3 ) );
			const grassP = mix( lush, dune.mul( 0.4 ), duneF );
			const density = select( isGrass, grassP, select( isOat, m.b, m.a ) );
			const r = hash12( xz.mul( 1.37 ).add( 0.51 ) );
			const r2 = hash12( xz.mul( 2.11 ).add( 7.3 ) );
			const flowerOk = select( kind.greaterThan( 3.5 ), select( r2.lessThan( 0.35 ), 1, 0 ), 1 );
			const present = select( r.lessThan( density ), 1, 0 ).mul( flowerOk );

			// meadow tone at the clump: the same function and inputs as the terrain's meadow shading
			const mA = det.sample( rotXZ( xz, 0.7 ).div( 173 ) ).level( 0 ).w;
			const mB = det.sample( rotXZ( xz, 2.1 ).div( 47 ) ).level( 0 ).w;
			const mt = meadowTone( mA, mB, float( 0 ), float( 0 ) );
			// size: tall meadow grass, knee to waist high: swathes of taller grass in the lush patches,
			// lower where it is dry; wiry dune tufts
			const patch = vnoise( xz.mul( 1 / 6.5 ).add( 17.3 ) ).mul( 0.7 ).add( vnoise( xz.mul( 1 / 2.3 ) ).mul( 0.3 ) );
			const lushH = mix( 0.7, 1.3, patch ).mul( mt.lush.mul( 0.25 ).add( 1 ) ).mul( float( 1 ).sub( mt.dry.mul( 0.25 ) ) );
			const grassH = mix( lushH, float( 0.55 ), duneF ).mul( r2.mul( 0.35 ).add( 0.83 ) ).mul( density.mul( 0.35 ).add( 0.65 ) );
			const oatH = r2.mul( 0.55 ).add( 1.0 );
			const vineS = r2.mul( 0.4 ).add( 0.8 );
			const hScale = select( isGrass, grassH, select( isOat, oatH, vineS ) );
			// meadow clumps fan out wider than the wiry dune tufts
			const spread = select( isGrass, mix( 1.35, 1.0, duneF ), float( 1 ) );

			// distance LOD: tiers fade out (narrow to nothing), the remaining blades widen so the
			// coverage (blade density x width) stays constant; tier 0 thins out per blade near R_FAR
			const dist = length( xz.sub( uCamPos.xz ) );
			// tier: grass blades and oats carry their own (aBlade.w), otherwise the slot's
			const tier = max( select( kind.lessThan( 2.5 ), blade.w, float( 0 ) ), slot.w );
			const cut = mix( float( FADE_T0[ 0 ] + 5 ), float( FADE_T0[ 1 ] ), fract01( bRnd.mul( 7.13 ).add( r2 ) ) );
			const own = tierFade( tier, dist, cut );
			const f2 = float( 1 ).sub( smoothstep( FADE_T2[ 0 ], FADE_T2[ 1 ], dist ) );
			const f1 = float( 1 ).sub( smoothstep( FADE_T1[ 0 ], FADE_T1[ 1 ], dist ) );
			const comp = float( TIER_N[ 0 ] + TIER_N[ 1 ] + TIER_N[ 2 ] ).div( f2.mul( TIER_N[ 2 ] ).add( f1.mul( TIER_N[ 1 ] ) ).add( TIER_N[ 0 ] ) );
			// oats and creeper simply shrink out
			const widthK = select( isGrass, comp.mul( own ), float( 1 ) );
			const sizeK = select( isGrass, float( 1 ), own );
			const scale = hScale.mul( present ).mul( sizeK );

			// per-slot random yaw
			const yaw = hash12( xz.mul( 0.73 ).add( 3.3 ) ).mul( 6.2832 );
			const cy = cos( yaw ), sy = sin( yaw );
			const rot = ( v ) => vec3( v.x.mul( cy ).sub( v.z.mul( sy ) ), v.y, v.x.mul( sy ).add( v.z.mul( cy ) ) );

			const ground = heightAt( xz );
			const base = vec3( xz.x, ground.sub( 0.03 ), xz.y );
			const o = rot( vec3( P.x.mul( spread ), P.y, P.z.mul( spread ) ) ).mul( scale );

			// wind: travelling gusts bend blades downwind (length preserving), plus flutter
			const w = windStrength;
			const g = gustAt( xz );
			const tm = G.time;
			const ph = r.mul( 6.2832 );
			const bendAmt = w.mul( g.mul( 0.55 ).add( 0.22 ) ).add( sin( tm.mul( 1.9 ).add( ph ).add( xz.x.mul( 0.2 ) ) ).mul( w ).mul( 0.1 ) );
			const flut = sin( tm.mul( 7.3 ).add( ph.mul( 3 ) ).add( bRnd.mul( 20 ) ) ).mul( w.mul( 0.06 ).add( 0.015 ) );
			const stiff = select( isGrass, float( 1 ), select( isOat, float( 0.8 ), float( 0.08 ) ) );
			const hf2 = hf.mul( hf );
			const disp = windDir3.mul( bendAmt.mul( hf2 ).mul( stiff ) ).add( windPerp3.mul( flut.mul( hf2 ).mul( stiff ) ) ).mul( length( o ).add( 1e-4 ) );
			const oL = length( o );
			const ob = normalize( o.add( disp ).add( vec3( 0, 1e-5, 0 ) ) ).mul( oL );

			const wScale = select( isGrass, mix( 1.5, 1.2, duneF ), float( 1 ) );
			const wide = wScale.mul( widthK ).mul( min( scale.mul( 2 ), max( scale, 0.5 ) ) );
			const pos = base.add( ob ).add( rot( side ).mul( wide ) );

			// lighting normal: blade normal bent towards up (soft, grass-like shading), rounded across
			// the blade (a folded leaf is lit differently on its two halves)
			const across = side4.w;
			const sideDir = normalize( rot( side ).add( vec3( 1e-5, 0, 0 ) ) );
			normalLocal.assign( normalize( rot( N ).mul( 0.5 ).add( UP ).add( sideDir.mul( across.mul( 0.35 ) ) ) ) );

			// a share of the blades is dead / straw coloured (more in the dry patches)
			const dryBlade = select( fract01( bRnd.mul( 13.7 ).add( r.mul( 3.1 ) ) ).lessThan( mt.dry.mul( 0.3 ).add( 0.07 ) ), float( 1 ), float( 0 ) );
			vG.assign( vec4( hf, kind, duneF, bRnd ) );
			vTone.assign( vec4( mt.tone, mt.dry ) );
			vG2.assign( vec4( density, select( kind.lessThan( 2.5 ), float( 0 ), blade.w ), across, dryBlade ) );
			vGust.assign( saturate( g.mul( w ).mul( 0.6 ) ).mul( stiff ) );

			return pos;

		} )();

		const albedo = Fn( () => {

			const hf = vG.x;
			const kind = vG.y;
			const duneF = vG.z;
			const rnd = vG.w;
			const across = vG2.z;
			const dryBlade = vG2.w;
			// self-shadowing of the sward: dark at the base of the clump
			const ao = mix( 0.32, 1.0, smoothstep( 0.0, 0.75, hf ) );
			// dune grass: olive base, straw tips; meadow: the terrain's meadow tone, dark and brownish
			// (dead leaf sheaths) at the base, lighter / sun-bleached towards the tips
			const duneBase = mix( C( 0x5d6232 ), C( 0x7f8a40 ), rnd );
			const duneTip = mix( C( 0xb8ab6c ), C( 0x9aa452 ), rnd );
			const tone = vTone.xyz.mul( rnd.mul( 0.34 ).add( 0.83 ) );
			const lushBase = mix( tone.mul( 0.5 ), MEADOW.soil, 0.3 );
			const tipDry = vTone.w.mul( 0.4 ).add( smoothstep( 0.8, 1.0, rnd ).mul( 0.35 ) );
			const lushTip = mix( tone.mul( vec3( 1.25, 1.25, 1.05 ) ), MEADOW.straw, tipDry.mul( smoothstep( 0.55, 1.0, hf ) ) );
			const base = mix( lushBase, duneBase, duneF );
			const tip = mix( lushTip, duneTip, duneF );
			let grass0 = mix( base, tip, smoothstep( 0.05, 0.95, hf ) );
			// dead blades: straw to brown
			grass0 = mix( grass0, mix( MEADOW.straw, MEADOW.soil.mul( 1.8 ), rnd.mul( 0.6 ) ).mul( smoothstep( 0.0, 0.5, hf ).mul( 0.4 ).add( 0.6 ) ), dryBlade.mul( float( 1 ).sub( duneF ) ) );
			// pale midrib
			grass0 = grass0.mul( pow( float( 1 ).sub( abs( across ) ), 6 ).mul( smoothstep( 0.05, 0.4, hf ) ).mul( 0.18 ).add( 1 ) );
			// wind sheen: blades flattened by a gust show their paler undersides, so gusts read
			// as bright waves rolling across the grass
			const grass = mix( grass0, grass0.mul( vec3( 1.3, 1.28, 1.1 ) ).add( vec3( 0.03, 0.03, 0.015 ) ), vGust.mul( smoothstep( 0.15, 0.9, hf ) ).mul( 0.6 ) );
			const oatStalk = mix( C( 0x9c9a62 ), C( 0xc2b27a ), hf );
			const oatHead = mix( C( 0xc9b27a ), C( 0xa8905a ), rnd );
			const centre = vG2.y; // 1 at leaf / flower centre
			const vine = mix( mix( C( 0x2e5219 ), C( 0x4a7328 ), rnd ), C( 0x7a8f4a ), centre.mul( 0.3 ) ).mul( select( hf.lessThan( 0.075 ), vec3( 1.25, 1.05, 0.8 ), vec3( 1 ) ) );
			const flower = mix( C( 0xb8479c ), C( 0xf2e8ee ), smoothstep( 0.35, 0.9, centre ) );
			const c = select( kind.lessThan( 0.5 ), grass,
				select( kind.lessThan( 1.5 ), oatStalk,
					select( kind.lessThan( 2.5 ), oatHead,
						select( kind.lessThan( 3.5 ), vine, flower ) ) ) );
			return c.mul( select( kind.lessThan( 2.5 ), ao, float( 1 ) ) );

		} )();

		mat.colorNode = albedo;
		mat.roughnessNode = select( vG.y.greaterThan( 2.5 ).and( vG.y.lessThan( 3.5 ) ), float( 0.45 ), float( 0.8 ) );
		mat.metalnessNode = float( 0 );
		mat.normalNode = normalize( normalViewGeometry.add( positionViewDirection.mul( 0.4 ) ) );

		// back-lit thin blades glow (evaluated in the lighting model with the shadowed light, so
		// grass in shadow does not)
		mat.translucencyNode = ( lightColor ) => {

			const V = normalize( cameraPosition.sub( positionWorld ) );
			const back = pow( saturate( dot( V.negate(), G.sunDir ) ), 4 );
			return albedo.mul( vec3( 1.1, 1.3, 0.6 ) ).mul( lightColor ).mul( back.mul( 0.3 ).mul( smoothstep( 0.2, 1.0, vG.x ) ) ).mul( float( 1 ).sub( G.night ) );

		};

		return mat;

	}

	// Recompute visible cells (cheap; skipped when the camera did not change).
	update( camera ) {

		// skip when the camera has not moved / turned noticeably
		const e = camera.matrixWorld.elements;
		const L = this._last;
		const pe = camera.projectionMatrix.elements;
		if ( Math.abs( e[ 12 ] - L[ 0 ] ) < 0.05 && Math.abs( e[ 13 ] - L[ 1 ] ) < 0.05 && Math.abs( e[ 14 ] - L[ 2 ] ) < 0.05 &&
			Math.abs( e[ 8 ] - L[ 3 ] ) < 1e-3 && Math.abs( e[ 9 ] - L[ 4 ] ) < 1e-3 && Math.abs( e[ 10 ] - L[ 5 ] ) < 1e-3 &&
			pe[ 0 ] === L[ 6 ] && pe[ 5 ] === L[ 7 ] ) return false;
		L[ 0 ] = e[ 12 ]; L[ 1 ] = e[ 13 ]; L[ 2 ] = e[ 14 ];
		L[ 3 ] = e[ 8 ]; L[ 4 ] = e[ 9 ]; L[ 5 ] = e[ 10 ];
		L[ 6 ] = pe[ 0 ]; L[ 7 ] = pe[ 5 ];

		this._mat.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse );
		this._frustum.setFromProjectionMatrix( this._mat, camera.coordinateSystem, camera.reversedDepth );

		const cx = e[ 12 ], cy = e[ 13 ], cz = e[ 14 ];
		const t = this.terrain;
		const n = this.cellsPerSide;
		const i0 = Math.max( 0, Math.floor( ( cx - R_FAR - t.origin ) / CELL ) ), i1 = Math.min( n - 1, Math.floor( ( cx + R_FAR - t.origin ) / CELL ) );
		const j0 = Math.max( 0, Math.floor( ( cz - R_FAR - t.origin ) / CELL ) ), j1 = Math.min( n - 1, Math.floor( ( cz + R_FAR - t.origin ) / CELL ) );
		const [ near, mid, far ] = this.levels;
		let nc = 0, mc = 0, fc = 0;

		for ( let j = j0; j <= j1; j ++ ) {

			for ( let i = i0; i <= i1; i ++ ) {

				const c = j * n + i;
				if ( ! this.cellFlags[ c ] ) continue;
				const x0 = t.origin + i * CELL, z0 = t.origin + j * CELL;
				// nearest point of the cell (horizontal, like the shader's LOD distance)
				const dx = Math.max( x0 - cx, 0, cx - x0 - CELL );
				const dz = Math.max( z0 - cz, 0, cz - z0 - CELL );
				const d = Math.hypot( dx, dz );
				if ( d > R_FAR ) continue;
				this._box.min.set( x0, this.cellMinY[ c ] - 0.5, z0 );
				this._box.max.set( x0 + CELL, this.cellMaxY[ c ] + 1.8, z0 + CELL );
				if ( ! this._frustum.intersectsBox( this._box ) ) continue;
				if ( d < R_NEAR && nc < near.max ) {

					near.arr[ nc * 4 ] = x0;
					near.arr[ nc * 4 + 1 ] = z0;
					nc ++;

				} else if ( d < R_MID && mc < mid.max ) {

					mid.arr[ mc * 4 ] = x0;
					mid.arr[ mc * 4 + 1 ] = z0;
					mc ++;

				} else if ( d >= R_MID && fc < far.max ) {

					far.arr[ fc * 4 ] = x0;
					far.arr[ fc * 4 + 1 ] = z0;
					fc ++;

				}

			}

		}

		for ( const [ lvl, count ] of [ [ near, nc ], [ mid, mc ], [ far, fc ] ] ) {

			lvl.count = count;
			lvl.geometry.instanceCount = count;
			lvl.attr.clearUpdateRanges();
			lvl.attr.addUpdateRange( 0, Math.max( 1, count ) * 4 );
			lvl.attr.needsUpdate = true;

		}

		return true;

	}

	get triangles() {

		return this.levels.reduce( ( a, l, k ) => a + l.count * this.patchTris[ k ], 0 );

	}

	dispose() {

		this.heightTex.dispose();
		this.maskTex.dispose();
		this.material.dispose();
		for ( const m of this.meshes ) m.geometry.dispose();

	}

}

// fract() for a node (TSL's fract is imported lazily to keep the import list short)
const fract01 = ( x ) => x.sub( floor( x ) );

const C = ( hex ) => {

	const c = new THREE.Color( hex );
	return vec3( c.r, c.g, c.b );

};
