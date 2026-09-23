import * as THREE from 'three/webgpu';
import { GeoBuilder } from './GeoBuilder.js';
import { mulberry32 } from '../../util/Noise.js';

// Procedural plant geometry. All plants are built in a local frame with the base at the
// origin and +Y up. See VegNodes.plantDeform for the attribute conventions.
//
// Part ids (aMat.x):
//   plant-leaf material: 0 stem / trunk, 1 coconut frond, 2 fern frond, 3 banana leaf, 5 coconut
//   canopy material:     0 bark, 1 tree leaf-cluster card, 4 shrub leaf-cluster card, 5 shrub stem
// (far trees / shrubs are octahedral impostors baked from these meshes: Impostors.js)

export const PART = { STEM: 0, FROND: 1, FERN: 2, BANANA: 3, COCONUT: 5 };

const _up = new THREE.Vector3( 0, 1, 0 );
const _v0 = new THREE.Vector3();
const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _v3 = new THREE.Vector3();

const smooth = ( a, b, x ) => {

	const t = Math.min( 1, Math.max( 0, ( x - a ) / ( b - a ) ) );
	return t * t * ( 3 - 2 * t );

};

// Stem / trunk: a tube whose vertices carry the height fraction u (aVeg.x) and a radial
// offset. The real height is applied in the shader (u * H), so one mesh fits any height.
function addStem( b, { radial, rows, radius, Hgeo, mat = [ 0, 0, 0, 0 ], cap = true, texU = 1 } ) {

	const start = b.count;
	for ( let j = 0; j < rows.length; j ++ ) {

		const u = rows[ j ];
		const r = radius( u );
		const du = 0.01;
		const dr = ( radius( u + du ) - radius( u - du ) ) / ( 2 * du * Hgeo );
		for ( let i = 0; i <= radial; i ++ ) {

			const a = ( i / radial ) * Math.PI * 2;
			const c = Math.cos( a ), s = Math.sin( a );
			_v0.set( c * r, u * Hgeo, s * r );
			_v1.set( c, - dr, s ).normalize();
			b.vertex( _v0, _v1, ( i / radial ) * texU, u, [ u, 0, 0, 0 ], mat );

		}

	}

	for ( let j = 0; j < rows.length - 1; j ++ ) {

		for ( let i = 0; i < radial; i ++ ) {

			const a = start + j * ( radial + 1 ) + i;
			const c = a + radial + 1;
			b.quad( a, c, c + 1, a + 1 );

		}

	}

	if ( cap ) {

		const uTop = rows[ rows.length - 1 ];
		const top = b.vertex( _v0.set( 0, uTop * Hgeo, 0 ), _up, 0.5, uTop, [ uTop + 0.015, 0, 0, 0 ], mat );
		const ring = start + ( rows.length - 1 ) * ( radial + 1 );
		for ( let i = 0; i < radial; i ++ ) b.tri( ring + i, top, ring + i + 1 );

	}

}

// Pinnate frond / leaf: two wings (leaflet rows) hanging off a curved rachis. The leaflets
// themselves are cut out in the fragment shader with a procedural mask on (s, t).
function addFrond( b, o ) {

	const {
		origin, azimuth, elevation, bend, twist = 0, length,
		segs = 8, cross = 2,
		leafLen, leafAngle, droop, curl = 0.2,
		part = PART.FROND, age = 0, seed = 0, phase = 0, flutter = 1, minWidth = 0.035,
		bendPow = 1.4, stemU = 1,
	} = o;

	// rachis centre line
	const pts = [];
	const p = origin.clone();
	for ( let k = 0; k <= segs; k ++ ) {

		pts.push( p.clone() );
		const sm = ( k + 0.5 ) / segs;
		const el = elevation - bend * Math.pow( sm, bendPow );
		const az = azimuth + twist * sm;
		_v0.set( Math.cos( el ) * Math.cos( az ), Math.sin( el ), Math.cos( el ) * Math.sin( az ) );
		p.addScaledVector( _v0, length / segs );

	}

	const azPerp = new THREE.Vector3( - Math.sin( azimuth ), 0, Math.cos( azimuth ) );
	const T = new THREE.Vector3(), S = new THREE.Vector3(), Nup = new THREE.Vector3(), side = new THREE.Vector3(), ld = new THREE.Vector3();

	for ( const sigma of [ - 1, 1 ] ) {

		const grid = [];
		for ( let k = 0; k <= segs; k ++ ) {

			const s = k / segs;
			T.subVectors( pts[ Math.min( segs, k + 1 ) ], pts[ Math.max( 0, k - 1 ) ] ).normalize();
			S.crossVectors( T, _up );
			if ( S.lengthSq() < 0.04 ) S.copy( azPerp );
			S.normalize();
			Nup.crossVectors( S, T ).normalize();
			const Ll = Math.max( leafLen( s ), minWidth );
			const al = leafAngle( s );
			const be = droop( s );
			side.copy( S ).multiplyScalar( sigma * Math.cos( be ) ).addScaledVector( Nup, - Math.sin( be ) );
			ld.copy( T ).multiplyScalar( Math.cos( al ) ).addScaledVector( side, Math.sin( al ) ).normalize();
			const row = [];
			for ( let j = 0; j <= cross; j ++ ) {

				const t = j / cross;
				const q = pts[ k ].clone().addScaledVector( ld, Ll * t ).addScaledVector( Nup, - Ll * curl * t * t );
				row.push( { q, s, t, Ll, nup: Nup.clone() } );

			}

			grid.push( row );

		}

		// normals from the grid, oriented to the frond's upper side
		const idx = [];
		for ( let k = 0; k <= segs; k ++ ) {

			const r = [];
			for ( let j = 0; j <= cross; j ++ ) {

				const g = grid[ k ][ j ];
				const ka = Math.max( 0, k - 1 ), kb = Math.min( segs, k + 1 );
				const ja = Math.max( 0, j - 1 ), jb = Math.min( cross, j + 1 );
				_v1.subVectors( grid[ kb ][ j ].q, grid[ ka ][ j ].q );
				_v2.subVectors( grid[ k ][ jb ].q, grid[ k ][ ja ].q );
				_v3.crossVectors( _v1, _v2 );
				if ( _v3.lengthSq() < 1e-12 ) _v3.copy( g.nup );
				_v3.normalize();
				if ( _v3.dot( g.nup ) < 0 ) _v3.negate();
				const fl = g.t * smooth( 0.05, 0.3, g.s ) * flutter;
				r.push( b.vertex( g.q, _v3, g.s, g.t, [ stemU, g.s, fl, phase ], [ part, age, g.Ll, seed ] ) );

			}

			idx.push( r );

		}

		for ( let k = 0; k < segs; k ++ ) {

			for ( let j = 0; j < cross; j ++ ) {

				const a = idx[ k ][ j ], bb = idx[ k + 1 ][ j ], c = idx[ k + 1 ][ j + 1 ], d = idx[ k ][ j + 1 ];
				if ( sigma > 0 ) b.quad( a, d, c, bb );
				else b.quad( a, bb, c, d );

			}

		}

	}

}

function addIcosphere( b, center, radii, mat, veg ) {

	const g = new THREE.IcosahedronGeometry( 1, 0 );
	const p = g.attributes.position;
	const map = new Map();
	const idx = [];
	for ( let i = 0; i < p.count; i ++ ) {

		_v0.fromBufferAttribute( p, i );
		const key = _v0.x.toFixed( 3 ) + ',' + _v0.y.toFixed( 3 ) + ',' + _v0.z.toFixed( 3 );
		let v = map.get( key );
		if ( v === undefined ) {

			_v1.copy( _v0 ).multiply( radii ).add( center );
			_v2.copy( _v0 ).divide( radii ).normalize();
			v = b.vertex( _v1, _v2, 0.5, 0.5, veg, mat );
			map.set( key, v );

		}

		idx.push( v );

	}

	for ( let i = 0; i < idx.length; i += 3 ) b.tri( idx[ i ], idx[ i + 1 ], idx[ i + 2 ] );
	g.dispose();

}

// Coconut palm -----------------------------------------------------------------------

const PALM_H = 10; // geometry trunk height (the shader uses the per-instance height)

const palmRadius = ( u ) => {

	const y = u * PALM_H;
	let r = 0.155 + 0.045 * ( 1 - u ) + 0.19 * Math.exp( - Math.max( y, 0 ) / 0.42 );
	r *= 1 + 0.05 * Math.sin( y * 1.7 ) * ( 1 - u ); // slight irregularity
	r += 0.07 * smooth( 0.955, 0.99, u ) - 0.1 * smooth( 0.995, 1.02, u ); // leaf-base boot
	return r;

};

// Frond parameters shared by both LODs so their silhouettes agree.
function palmFrondParams( rand, count ) {

	const list = [];
	for ( let i = 0; i < count; i ++ ) {

		const a = count > 1 ? i / ( count - 1 ) : 0.5; // 0 youngest .. 1 oldest
		const dead = i === count - 1 && count > 10;
		const len = ( 3.9 + 1.3 * smooth( 0, 0.45, a ) ) * ( 0.9 + 0.2 * rand() );
		list.push( {
			a, dead,
			azimuth: i * 2.39996 + ( rand() - 0.5 ) * 0.35,
			elevation: dead ? - 1.3 : 1.02 - 1.45 * Math.pow( a, 0.8 ) + ( rand() - 0.5 ) * 0.25,
			bend: dead ? 0.15 : 0.62 + 0.8 * a + rand() * 0.3,
			twist: ( rand() - 0.5 ) * 0.3,
			length: dead ? len * 0.85 : len,
			attachY: 0.3 - 0.45 * a,
			seed: rand(),
			phase: rand(),
		} );

	}

	return list;

}

function addPalmCrown( b, fronds, { segs, cross, leafScale = 1 } ) {

	for ( const f of fronds ) {

		const origin = new THREE.Vector3( Math.cos( f.azimuth ) * 0.14, f.attachY, Math.sin( f.azimuth ) * 0.14 );
		const Lf = f.length;
		addFrond( b, {
			origin, azimuth: f.azimuth, elevation: f.elevation, bend: f.bend, twist: f.twist, length: Lf,
			segs, cross,
			leafLen: ( s ) => leafScale * 0.19 * Lf * ( smooth( 0.05, 0.27, s ) * ( 1 - 0.62 * smooth( 0.32, 1.0, s ) ) ),
			leafAngle: ( s ) => 1.05 - 0.45 * s,
			droop: ( s ) => ( f.dead ? 1.25 : 0.48 + 0.5 * f.a ) + 0.32 * s,
			curl: f.dead ? 0.1 : 0.22,
			part: PART.FROND,
			age: f.dead ? 1 : f.a * 0.55,
			seed: f.seed, phase: f.phase,
			flutter: f.dead ? 0.3 : 1,
			minWidth: 0.045,
		} );

	}

}

// Full detail coconut palm: ringed trunk, coconut cluster and 15 fronds in one geometry
// (one draw call per pass for the plant-leaf material).
export function buildPalmNear( seed = 11 ) {

	const rand = mulberry32( seed );
	const b = new GeoBuilder();

	const rows = [ - 0.03, 0, 0.015, 0.04, 0.08, 0.14, 0.22, 0.32, 0.43, 0.54, 0.65, 0.76, 0.86, 0.94, 0.975, 0.992, 1.005 ];
	addStem( b, { radial: 10, rows, radius: palmRadius, Hgeo: PALM_H, mat: [ PART.STEM, 0, 0, 0 ] } );

	// coconut cluster below the crown (crown-local coordinates)
	const nuts = 6;
	for ( let i = 0; i < nuts; i ++ ) {

		const az = i * 2.39996 + rand() * 0.5;
		const rr = 0.2 + rand() * 0.1;
		const c = new THREE.Vector3( Math.cos( az ) * rr, - 0.3 - rand() * 0.35, Math.sin( az ) * rr );
		const s = 0.12 + rand() * 0.035;
		addIcosphere( b, c, new THREE.Vector3( s, s * 1.12, s ), [ PART.COCONUT, rand(), 0, i / nuts ], [ 1, 0, 0, 0 ] );

	}

	addPalmCrown( b, palmFrondParams( rand, 15 ), { segs: 8, cross: 2 } );

	return { geometry: b.build( 16, new THREE.Vector3( 0, 6, 0 ) ), triangles: b.triangles };

}

// Low detail palm: one geometry (stem + crown) for the plant-leaf material.
export function buildPalmFar( seed = 11 ) {

	const rand = mulberry32( seed );
	const b = new GeoBuilder();
	addStem( b, { radial: 5, rows: [ - 0.03, 0.03, 0.2, 0.6, 1.0 ], radius: palmRadius, Hgeo: PALM_H, mat: [ PART.STEM, 0, 0, 0 ], cap: false } );
	const fronds = palmFrondParams( rand, 15 ).filter( ( f, i ) => i !== 1 && i !== 5 && i !== 9 );
	addPalmCrown( b, fronds, { segs: 3, cross: 1, leafScale: 1.15 } );
	return { geometry: b.build( 16, new THREE.Vector3( 0, 6, 0 ) ), triangles: b.triangles };

}

// Young palm (no trunk yet) / clumping understory palm.
export function buildYoungPalm( seed = 5, b = new GeoBuilder() ) {

	const rand = mulberry32( seed );
	addStem( b, { radial: 5, rows: [ - 0.2, 0.3, 1.0 ], radius: ( u ) => 0.16 - 0.07 * u, Hgeo: 0.6, mat: [ PART.STEM, 0.35, 0, 0 ], cap: false } );
	const n = 8;
	for ( let i = 0; i < n; i ++ ) {

		const a = i / ( n - 1 );
		const az = i * 2.39996 + rand() * 0.4;
		const Lf = ( 1.7 + 0.9 * a ) * ( 0.85 + 0.3 * rand() );
		addFrond( b, {
			origin: new THREE.Vector3( 0, 0.1 - 0.15 * a, 0 ), azimuth: az,
			elevation: 1.35 - 0.8 * a + ( rand() - 0.5 ) * 0.2, bend: 0.6 + 0.7 * a, twist: ( rand() - 0.5 ) * 0.3, length: Lf,
			segs: 5, cross: 1,
			leafLen: ( s ) => 0.21 * Lf * ( smooth( 0.04, 0.25, s ) * ( 1 - 0.6 * smooth( 0.3, 1, s ) ) ),
			leafAngle: ( s ) => 1.0 - 0.4 * s,
			droop: ( s ) => 0.25 + 0.4 * a + 0.2 * s,
			curl: 0.15, part: PART.FROND, age: a * 0.4, seed: rand(), phase: rand(), minWidth: 0.03,
		} );

	}

	return { geometry: b.build( 4, new THREE.Vector3( 0, 1, 0 ) ), triangles: b.triangles };

}

// Fern: fountain of arching pinnate fronds.
export function buildFern( seed = 3, b = new GeoBuilder() ) {

	const rand = mulberry32( seed );
	const n = 9;
	for ( let i = 0; i < n; i ++ ) {

		const a = i / ( n - 1 );
		const az = i * 2.39996 + rand() * 0.5;
		const Lf = ( 0.75 + 0.5 * a ) * ( 0.85 + 0.3 * rand() );
		addFrond( b, {
			origin: new THREE.Vector3( 0, 0.05, 0 ), azimuth: az,
			elevation: 1.3 - 0.55 * a + ( rand() - 0.5 ) * 0.2, bend: 1.3 + 0.8 * a, twist: ( rand() - 0.5 ) * 0.5, length: Lf,
			segs: 4, cross: 1, bendPow: 1.2,
			leafLen: ( s ) => 0.2 * Lf * ( smooth( 0.02, 0.2, s ) * ( 1 - 0.8 * smooth( 0.35, 1, s ) ) ),
			leafAngle: () => 1.35,
			droop: ( s ) => 0.12 + 0.2 * s,
			curl: 0.08, part: PART.FERN, age: 0, seed: rand(), phase: rand(), minWidth: 0.012, flutter: 0.6,
		} );

	}

	return { geometry: b.build( 1.6, new THREE.Vector3( 0, 0.4, 0 ) ), triangles: b.triangles };

}

// Banana plant: pseudostem with large paddle leaves (tears cut in the shader).
export function buildBanana( seed = 9, b = new GeoBuilder() ) {

	const rand = mulberry32( seed );
	addStem( b, { radial: 6, rows: [ - 0.05, 0.1, 0.5, 0.9, 1.0 ], radius: ( u ) => 0.13 - 0.05 * u + 0.05 * Math.exp( - u * 8 ), Hgeo: 1.8, mat: [ PART.STEM, 1, 0, 0 ], cap: true } );
	const n = 8;
	for ( let i = 0; i < n; i ++ ) {

		const a = i / ( n - 1 );
		const az = i * 2.39996 + rand() * 0.4;
		const Lf = ( 1.9 + 0.7 * rand() ) * ( a > 0.85 ? 0.9 : 1 );
		const W = 0.3 + 0.08 * rand();
		addFrond( b, {
			origin: new THREE.Vector3( Math.cos( az ) * 0.06, 0.15 - 0.25 * a, Math.sin( az ) * 0.06 ), azimuth: az,
			elevation: 1.25 - 0.75 * a + ( rand() - 0.5 ) * 0.2, bend: 0.5 + 0.9 * a, twist: ( rand() - 0.5 ) * 0.25, length: Lf,
			segs: 6, cross: 1,
			leafLen: ( s ) => W * Math.pow( Math.max( 0, Math.sin( Math.PI * Math.min( 1, Math.max( 0, ( s - 0.12 ) / 0.88 ) ) ) ), 0.55 ),
			leafAngle: () => 1.45,
			droop: ( s ) => 0.18 + 0.35 * s,
			curl: 0.12, part: PART.BANANA, age: a > 0.85 ? 0.9 : a * 0.3, seed: rand(), phase: rand(), minWidth: 0.03, flutter: 0.8,
		} );

	}

	return { geometry: b.build( 4, new THREE.Vector3( 0, 2, 0 ) ), triangles: b.triangles };

}

// Broadleaf trees and shrubs: lobed crowns ------------------------------------------------
//
// A crown is a set of lobes (leaf clusters at the branch ends) [x, y, z, radius] in plant-local
// space. Every leaf card stores the offset to its lobe centre and the lobe id (aLobe), so each
// instance can drop or resize lobes in the vertex shader: one mesh, irregular crowns with gaps.
// The far impostors are baked from the same variants (LOBE_TABLE rows), so the LOD switch keeps
// the crown's shape. Lobes 0-1 are never dropped.

export const TREE_H = 12.5; // nominal height (top of the highest lobe)
// deep, irregular crown: lobes from ~4 m up to the top, so from a distance the canopy is a
// continuous lumpy carpet rather than crowns on bare poles
export const TREE_LOBES = [
	[ 0.4, 9.9, - 0.3, 2.9 ],
	[ - 2.0, 8.3, 1.6, 2.6 ],
	[ 3.0, 7.7, 1.2, 2.4 ],
	[ 1.7, 6.8, - 3.0, 2.5 ],
	[ - 3.2, 6.3, - 1.8, 2.3 ],
	[ 0.4, 5.6, 3.4, 2.2 ],
	[ - 1.1, 11.2, - 1.7, 1.8 ],
	[ 3.7, 5.2, - 0.9, 1.9 ],
];
const TREE_FORK = [ 0.15, 3.7, - 0.05 ];

export const SHRUB_H = 1.6;
export const SHRUB_LOBES = [
	[ 0.0, 0.85, 0.0, 0.8 ],
	[ 0.75, 0.6, 0.35, 0.6 ],
	[ - 0.6, 0.55, 0.55, 0.6 ],
	[ - 0.25, 0.65, - 0.75, 0.62 ],
	[ 0.35, 1.25, - 0.25, 0.48 ],
];

function addBranch( b, p0, p1, r0, r1, radial, rows, mat, hScale, flexFn, radiusFn = null ) {

	const dir = new THREE.Vector3().subVectors( p1, p0 );
	const len = dir.length();
	dir.normalize();
	const tmp = Math.abs( dir.y ) < 0.9 ? _up : new THREE.Vector3( 1, 0, 0 );
	const X = new THREE.Vector3().crossVectors( dir, tmp ).normalize();
	const Z = new THREE.Vector3().crossVectors( X, dir ).normalize();
	const start = b.count;
	for ( let j = 0; j <= rows; j ++ ) {

		const f = j / rows;
		const c = new THREE.Vector3().copy( p0 ).addScaledVector( dir, len * f );
		for ( let i = 0; i <= radial; i ++ ) {

			const a = ( i / radial ) * Math.PI * 2;
			const r = radiusFn ? radiusFn( f, a, c ) : r0 + ( r1 - r0 ) * f;
			const n = new THREE.Vector3().copy( X ).multiplyScalar( Math.cos( a ) ).addScaledVector( Z, Math.sin( a ) );
			const p = c.clone().addScaledVector( n, r );
			b.vertex( p, n, i / radial, f * len, [ Math.max( 0, p.y / hScale ), flexFn( p ), 0, 0 ], mat );

		}

	}

	for ( let j = 0; j < rows; j ++ ) {

		for ( let i = 0; i < radial; i ++ ) {

			const a = start + j * ( radial + 1 ) + i;
			const c = a + radial + 1;
			b.quad( a, c, c + 1, a + 1 );

		}

	}

}

// Curved limb: a tapered tube along a quadratic Bezier p0 -> ctrl -> p1 (rings follow the curve,
// frames rotation-minimised from ring to ring).
function addLimb( b, p0, ctrl, p1, r0, r1, radial, rows, mat, hScale, flexFn ) {

	const start = b.count;
	const pt = ( t ) => new THREE.Vector3()
		.copy( p0 ).multiplyScalar( ( 1 - t ) * ( 1 - t ) )
		.addScaledVector( ctrl, 2 * ( 1 - t ) * t )
		.addScaledVector( p1, t * t );
	const tan = ( t ) => new THREE.Vector3().subVectors( ctrl, p0 ).multiplyScalar( 2 * ( 1 - t ) ).addScaledVector( new THREE.Vector3().subVectors( p1, ctrl ), 2 * t ).normalize();
	let T = tan( 0 );
	let X = new THREE.Vector3().crossVectors( T, Math.abs( T.y ) < 0.9 ? _up : new THREE.Vector3( 1, 0, 0 ) ).normalize();
	let len = 0;
	let prev = pt( 0 );
	for ( let j = 0; j <= rows; j ++ ) {

		const f = j / rows;
		const c = pt( f );
		len += c.distanceTo( prev );
		prev = c;
		const Tn = tan( f );
		// rotation-minimising frame: project the previous X onto the new normal plane
		X.addScaledVector( Tn, - X.dot( Tn ) ).normalize();
		T = Tn;
		const Z = new THREE.Vector3().crossVectors( X, T ).normalize();
		const r = r0 + ( r1 - r0 ) * Math.pow( f, 0.8 );
		for ( let i = 0; i <= radial; i ++ ) {

			const a = ( i / radial ) * Math.PI * 2;
			const n = new THREE.Vector3().copy( X ).multiplyScalar( Math.cos( a ) ).addScaledVector( Z, Math.sin( a ) );
			const p = c.clone().addScaledVector( n, r );
			b.vertex( p, n, i / radial, len, [ Math.max( 0, p.y / hScale ), flexFn( p ), 0, 0 ], mat );

		}

	}

	for ( let j = 0; j < rows; j ++ ) {

		for ( let i = 0; i < radial; i ++ ) {

			const a = start + j * ( radial + 1 ) + i;
			const c = a + radial + 1;
			b.quad( a, c, c + 1, a + 1 );

		}

	}

}

// Leaf-cluster card: a quad facing `normal`, rotated by `yaw` around it. Vertex normals blend the
// lobe sphere and the whole crown (volumetric shading), aMat.y = exposure (0 inner .. 1 outer).
function addCard( b, { center, size, normal, yaw, lobeC, lobeR, lobeId, crownC, crownR, rand, flexFn, hScale, part, clumpC = null, clumpR = 1 } ) {

	const n = normal.clone().normalize();
	const tmp = Math.abs( n.y ) < 0.95 ? _up : new THREE.Vector3( 1, 0, 0 );
	const X = new THREE.Vector3().crossVectors( tmp, n ).normalize();
	const Y = new THREE.Vector3().crossVectors( n, X ).normalize();
	const cy = Math.cos( yaw ), sy = Math.sin( yaw );
	const Xr = X.clone().multiplyScalar( cy ).addScaledVector( Y, sy );
	const Yr = Y.clone().multiplyScalar( cy ).addScaledVector( X, - sy );
	const cr = rand();
	const card = rand();
	const ph = rand();
	const ids = [];
	const corners = [ [ - 0.5, - 0.5, 0, 0 ], [ 0.5, - 0.5, 1, 0 ], [ 0.5, 0.5, 1, 1 ], [ - 0.5, 0.5, 0, 1 ] ];
	for ( const [ x, y, u, v ] of corners ) {

		const p = center.clone().addScaledVector( Xr, x * size ).addScaledVector( Yr, y * size );
		const dl = clumpC ? p.clone().sub( clumpC ).divideScalar( clumpR ) : p.clone().sub( lobeC ).divideScalar( lobeR );
		const dc = p.clone().sub( crownC ).divide( crownR );
		// normals bent outward from the clump (or lobe) centre and from the crown centre: soft,
		// volumetric shading of every clump
		const nn = dl.clone().normalize().multiplyScalar( 0.5 ).addScaledVector( dc.clone().normalize(), 0.4 ).addScaledVector( n, 0.2 ).normalize();
		// exposure: outer / upper leaves lit, the inside of each clump and of the crown dark
		const ext = Math.min( 1, 0.5 * Math.min( 1, dl.length() ) + 0.4 * Math.min( 1, dc.length() ) + 0.25 * Math.max( 0, dc.y ) );
		const hf = Math.max( 0, p.y / hScale );
		ids.push( b.vertex( p, nn, u, v, [ hf, flexFn( p ), 1, ph ], [ part, 0.2 + 0.8 * ext * ext, cr, card ], [ lobeC.x - p.x, lobeC.y - p.y, lobeC.z - p.z, lobeId ] ) );

	}

	b.quad( ids[ 0 ], ids[ 1 ], ids[ 2 ], ids[ 3 ] );

}

// fills a lobe with leaf cards, biased towards its shell (the lobe reads as a volume, with gaps);
// returns card specs (emitted later, outermost first, so near cards draw first: early depth test)
function lobeCards( lobe, id, { count, size, crownC, crownR, rand, flexFn, hScale, part, flatten = 0.6 } ) {

	const lobeC = new THREE.Vector3( lobe[ 0 ], lobe[ 1 ], lobe[ 2 ] );
	const R = lobe[ 3 ];
	const cards = [];
	for ( let k = 0; k < count; k ++ ) {

		let x, y, z;
		do {

			x = rand() * 2 - 1; y = rand() * 2 - 1; z = rand() * 2 - 1;

		} while ( x * x + y * y + z * z > 1 || y < - 0.75 );

		const l = Math.hypot( x, y, z ) || 1;
		const shell = 0.45 + 0.55 * Math.sqrt( l );
		const c = new THREE.Vector3( x / l * shell * R * 0.85, y / l * shell * R * 0.85 * flatten, z / l * shell * R * 0.85 ).add( lobeC );
		const out = c.clone().sub( lobeC ).normalize();
		const normal = out.lerp( _up, 0.35 + rand() * 0.35 ).normalize();
		cards.push( {
			center: c, size: size * ( 0.8 + rand() * 0.45 ), normal, yaw: rand() * 6.283,
			lobeC, lobeR: R, lobeId: id, crownC, crownR, rand, flexFn, hScale, part,
		} );

	}

	return cards;

}

// Leaf clumps at the branch tips of a lobe: each clump a few crossing cards around its centre
// (normals outward from the clump with random tilt), clumps of different sizes and depths with
// real gaps between them (sky and branches show through). Returns { cards, tips }.
function clumpCards( lobe, id, { clumps, clumpR, cardsPer, size, crownC, crownR, rand, flexFn, hScale, part, flatten = 0.7 } ) {

	const lobeC = new THREE.Vector3( lobe[ 0 ], lobe[ 1 ], lobe[ 2 ] );
	const R = lobe[ 3 ];
	const cards = [], tips = [];
	for ( let k = 0; k < clumps; k ++ ) {

		let x, y, z;
		do {

			x = rand() * 2 - 1; y = rand() * 2 - 1; z = rand() * 2 - 1;

		} while ( x * x + y * y + z * z > 1 || y < - 0.55 );

		const l = Math.hypot( x, y, z ) || 1;
		const reach = 0.3 + 0.7 * Math.sqrt( rand() ); // mostly toward the lobe surface, some deeper
		const c = new THREE.Vector3( x / l, y / l * flatten, z / l ).multiplyScalar( reach * R * 0.85 ).add( lobeC );
		const rc = clumpR * ( 0.65 + rand() * 0.7 );
		tips.push( { c, rc } );
		const out = c.clone().sub( lobeC ).normalize();
		for ( let j = 0; j < cardsPer; j ++ ) {

			// crossing cards: each faces a random direction around the clump, biased outward / up
			const dir = new THREE.Vector3( rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1 ).normalize()
				.addScaledVector( out, 0.9 ).addScaledVector( _up, 0.55 ).normalize();
			const center = c.clone().addScaledVector( dir, rc * ( 0.1 + 0.3 * rand() ) )
				.add( new THREE.Vector3( rand() - 0.5, ( rand() - 0.5 ) * 0.6, rand() - 0.5 ).multiplyScalar( rc * 0.5 ) );
			cards.push( {
				center, size: size * rc / clumpR * ( 0.8 + rand() * 0.4 ), normal: dir, yaw: rand() * 6.283,
				lobeC, lobeR: R, lobeId: id, crownC, crownR, rand, flexFn, hScale, part, clumpC: c, clumpR: rc,
			} );

		}

	}

	return { cards, tips };

}

function emitCards( b, cards, crownC ) {

	cards.sort( ( a, c ) => c.center.distanceToSquared( crownC ) - a.center.distanceToSquared( crownC ) );
	for ( const card of cards ) addCard( b, card );

}

// Broadleaf rainforest tree: buttressed trunk forking into limbs that end in leafy lobes.
export function buildTreeNear( seed = 21, b = new GeoBuilder() ) {

	const rand = mulberry32( seed );
	const H = TREE_H;
	const crownC = new THREE.Vector3( 0, 7.8, 0 );
	const crownR = new THREE.Vector3( 5.2, 3.7, 5.2 );
	const flex = ( p ) => Math.min( 1, Math.hypot( p.x, p.z ) / 5 ) * smooth( 3.2, 6.5, p.y );
	const fork = new THREE.Vector3( ...TREE_FORK );

	// trunk with buttress fins at the foot
	const trunkR = ( f, a, c ) => {

		const y = c.y;
		const r = 0.34 - 0.1 * f;
		const fin = Math.pow( Math.max( 0, Math.cos( 4 * a + 0.4 ) ), 4 ) * Math.exp( - Math.max( y, 0 ) / 0.8 );
		return r * ( 1 + 0.9 * fin + 0.25 * Math.exp( - Math.max( y + 0.3, 0 ) / 0.5 ) );

	};

	addBranch( b, new THREE.Vector3( 0, - 0.5, 0 ), fork, 0.34, 0.24, 10, 7, [ 0, 0, 0, 0 ], H, flex, trunkR );

	// limbs to the lobes (the first five from the fork, the rest from the middle of a limb)
	const mids = [];
	TREE_LOBES.forEach( ( L, i ) => {

		const lc = new THREE.Vector3( L[ 0 ], L[ 1 ], L[ 2 ] );
		const end = lc.clone().addScaledVector( lc.clone().sub( crownC ).setY( 0 ).normalize(), - L[ 3 ] * 0.2 ).setY( L[ 1 ] - L[ 3 ] * 0.25 );
		// limbs rise steeply from the fork and spread out (vase-shaped crown), with a random kink
		const bow = ( a, e, k ) => {

			const mid = a.clone().lerp( e, 0.5 );
			const d = e.clone().sub( a );
			const horiz = new THREE.Vector3( d.x, 0, d.z );
			return mid.addScaledVector( horiz, - 0.28 * k ).add( new THREE.Vector3( 0, d.length() * 0.16 * k, 0 ) )
				.add( new THREE.Vector3( ( rand() - 0.5 ) * 0.5, ( rand() - 0.5 ) * 0.3, ( rand() - 0.5 ) * 0.5 ).multiplyScalar( d.length() * 0.25 ) );

		};

		if ( i < 5 ) {

			const start = fork.clone().add( new THREE.Vector3( ( rand() - 0.5 ) * 0.2, - 0.2 - rand() * 0.3, ( rand() - 0.5 ) * 0.2 ) );
			const ctrl = bow( start, end, 1 );
			addLimb( b, start, ctrl, end, 0.18, 0.05, 6, 5, [ 0, 0, 0, 0 ], H, flex );
			// branching point for the secondary limbs: along the curve
			mids.push( start.clone().multiplyScalar( 0.25 ).addScaledVector( ctrl, 0.5 ).addScaledVector( end, 0.25 ) );

		} else {

			let best = mids[ 0 ];
			for ( const m of mids ) if ( m.distanceTo( lc ) < best.distanceTo( lc ) ) best = m;
			addLimb( b, best, bow( best, end, 0.6 ), end, 0.085, 0.03, 4, 3, [ 0, 0, 0, 0 ], H, flex );

		}

	} );

	const cards = [];
	TREE_LOBES.forEach( ( L, i ) => {

		const { cards: cs, tips } = clumpCards( L, i, {
			clumps: Math.round( 3.9 * L[ 3 ] ), clumpR: 0.82, cardsPer: 3, size: 1.45, crownC, crownR, rand, flexFn: flex, hScale: H, part: 1, flatten: 0.72,
		} );
		cards.push( ...cs );
		// short twigs carrying each clump (from part way along the lobe's limb, not one point);
		// only on the two lobes no crown variant drops (twigs don't follow the lobe scaling)
		const base = new THREE.Vector3( L[ 0 ], L[ 1 ] - L[ 3 ] * 0.3, L[ 2 ] );
		if ( i < 2 ) for ( const t of tips ) {

			const from = base.clone().lerp( t.c, 0.4 + rand() * 0.25 ).add( new THREE.Vector3( rand() - 0.5, ( rand() - 0.5 ) * 0.4, rand() - 0.5 ).multiplyScalar( 0.5 ) );
			// ends inside the clump even when the variant shrinks the lobe (no floating twigs)
			addBranch( b, from, from.clone().lerp( t.c, 0.7 ), 0.035, 0.014, 3, 1, [ 0, 0, 0, 0 ], H, flex );

		}

	} );
	emitCards( b, cards, crownC );

	return { geometry: b.build( 14, new THREE.Vector3( 0, 6.5, 0 ) ), triangles: b.triangles };

}

// Shrub: a few leafy lobes on short stems (stems are part 5: bark of a shrub).
export function buildShrubNear( seed = 31, b = new GeoBuilder() ) {

	const rand = mulberry32( seed );
	const crownC = new THREE.Vector3( 0, 0.7, 0 );
	const crownR = new THREE.Vector3( 1.1, 0.75, 1.1 );
	const flex = ( p ) => Math.min( 1, p.y / 1.3 );
	// a few bare stems at the foot
	for ( let i = 0; i < 3; i ++ ) {

		const L = SHRUB_LOBES[ i + 1 ];
		addBranch( b, new THREE.Vector3( 0, - 0.1, 0 ), new THREE.Vector3( L[ 0 ] * 0.6, L[ 1 ] * 0.7, L[ 2 ] * 0.6 ), 0.035, 0.015, 3, 1, [ 5, 0, 0, 0 ], SHRUB_H, flex );

	}

	const cards = [];
	SHRUB_LOBES.forEach( ( L, i ) => cards.push( ...lobeCards( L, i, {
		count: Math.round( 15 * L[ 3 ] ), size: 0.85, crownC, crownR, rand, flexFn: flex, hScale: SHRUB_H, part: 4, flatten: 0.8,
	} ) ) );
	emitCards( b, cards, crownC );

	return { geometry: b.build( 2.2, new THREE.Vector3( 0, 0.7, 0 ) ), triangles: b.triangles };

}

// Per-variant lobe scales (0 = dropped), generated once: trees 3 variants x 8 lobes, then shrubs
// 2 variants x 8 lobes. The near canopy (vertex shader) and the impostor bake read the same table.
export const TREE_VARIANTS = 3;
export const SHRUB_VARIANTS = 2;
export const LOBE_TABLE = ( () => {

	const rand = mulberry32( 4711 );
	const t = new Float32Array( ( TREE_VARIANTS + SHRUB_VARIANTS ) * 8 );
	for ( let v = 0; v < TREE_VARIANTS + SHRUB_VARIANTS; v ++ ) {

		const shrub = v >= TREE_VARIANTS;
		const n = shrub ? SHRUB_LOBES.length : TREE_LOBES.length;
		let dropped = 0;
		for ( let k = 0; k < 8; k ++ ) {

			if ( k >= n ) continue;
			const drop = k >= 2 && rand() < 0.3 && dropped < ( shrub ? 1 : 3 );
			if ( drop ) dropped ++;
			t[ v * 8 + k ] = drop ? 0 : 0.74 + 0.44 * rand();

		}

	}

	return t;

} )();

// Copy of a lobed geometry with the lobe scales of one variant applied on the CPU (for the bake).
export function lobeVariantGeometry( geometry, table, offset ) {

	const pos = geometry.attributes.position;
	const lobe = geometry.attributes.aLobe;
	const src = pos.data.array;
	const stride = pos.data.stride;
	const data = new Float32Array( src );
	for ( let i = 0; i < pos.count; i ++ ) {

		const o = i * stride;
		const id = src[ o + lobe.offset + 3 ];
		if ( id < 0 ) continue;
		const k = 1 - table[ offset + id ];
		data[ o + pos.offset ] += src[ o + lobe.offset ] * k;
		data[ o + pos.offset + 1 ] += src[ o + lobe.offset + 1 ] * k;
		data[ o + pos.offset + 2 ] += src[ o + lobe.offset + 2 ] * k;

	}

	const ib = new THREE.InterleavedBuffer( data, stride );
	const g = new THREE.BufferGeometry();
	for ( const name of Object.keys( geometry.attributes ) ) {

		const a = geometry.attributes[ name ];
		g.setAttribute( name, new THREE.InterleavedBufferAttribute( ib, a.itemSize, a.offset ) );

	}

	g.setIndex( geometry.index );
	g.boundingSphere = geometry.boundingSphere.clone();
	return g;

}

// Merged near geometry of trees + shrubs (one draw call): the vertex shader keeps the tree parts
// (0, 1) for tree instances and the shrub parts (4, 5) for shrub instances.
export function buildCanopyNear() {

	const b = new GeoBuilder();
	const tree = buildTreeNear( 21, b );
	const trees = b.triangles;
	buildShrubNear( 31, b );
	return { geometry: b.build( 14, new THREE.Vector3( 0, 6.5, 0 ) ), triangles: b.triangles, treeTriangles: trees, shrubTriangles: b.triangles - trees };

}

// Merged understory geometry (one draw call): young palm (kind 1), banana (kind 2), fern (kind 3);
// the vertex shader keeps the plant whose kind matches the instance (floor of its seed).
export const UNDERSTORY = { YOUNG: 1, BANANA: 2, FERN: 3 };
export function buildUnderstory() {

	const b = new GeoBuilder();
	const tris = {};
	let t0 = 0;
	b.kind = UNDERSTORY.YOUNG; buildYoungPalm( 5, b ); tris.young = b.triangles - t0; t0 = b.triangles;
	b.kind = UNDERSTORY.BANANA; buildBanana( 9, b ); tris.banana = b.triangles - t0; t0 = b.triangles;
	b.kind = UNDERSTORY.FERN; buildFern( 3, b ); tris.fern = b.triangles - t0;
	return { geometry: b.build( 4, new THREE.Vector3( 0, 1.5, 0 ) ), triangles: b.triangles, perKind: tris };

}
