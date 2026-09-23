import * as THREE from 'three/webgpu';
import {
	Fn, float, vec2, vec3, vec4, attribute, texture, varyingProperty, normalLocal, normalWorldGeometry, positionWorld,
	cameraPosition, mix, smoothstep, clamp, max, sin, cos, fract, step, select, length, normalize, dot,
	transformNormalToView, saturate,
} from 'three/tsl';
import { standard } from '../../materials/Materials.js';
import { TerrainLightingModel } from '../Terrain.js';
import { srgb, perturbNormal, triWeights } from '../terrain/TerrainShading.js';
import { staticVelocityMRT } from '../../post/CameraVelocity.js';
import { stoneSurface } from './NatureMaterial.js';
import { stonePart } from './DebrisShapes.js';

// Camera-following ground clutter: pebbles, cobbles and shell / coral grit on the beaches,
// rocky shores and paths (one draw call, no shadow casting).
//
// The ground is divided into PCELL x PCELL cells; every visible cell near the camera draws one
// instance of a patch of blue-noise slots (small pebbles, cobbles, flat shell / coral chips).
// In the vertex shader each slot looks up its density in the debris mask (DebrisPlacement: R
// pebbles, G cobbles, B grit, A stone palette), picks its size, shape, yaw from hashes of its
// world position (stable while the camera moves), sits on the terrain (TerrainGPU.heightAt) and
// shrinks to nothing with distance (small grit before cobbles): no popping. Empty cells are
// skipped on the CPU and cells are frustum culled when the camera moves.

export const PCELL = 4;
export const R_FAR = 25;
const FADE_SMALL = [ 8, 15 ];
const FADE_COBBLE = [ 15, 24 ];
const MAX_CELLS = 240;
const N_COBBLE = 18, N_CHIP = 44, N_SMALL = 150;

function blueNoise( rand, n, size, k = 12 ) {

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

// flat, slightly domed fragment with a ragged outline (shell / coral grit), unit radius
function chipPart( v ) {

	const p = [ 0, 0.3, 0 ], idx = [];
	const m = 6;
	for ( let k = 0; k < m; k ++ ) {

		const a = ( k + ( ( v * 7 + k * 3 ) % 5 ) * 0.12 ) / m * Math.PI * 2;
		const r = 0.65 + ( ( v * 13 + k * 7 ) % 9 ) / 9 * 0.45;
		p.push( Math.cos( a ) * r, 0, Math.sin( a ) * r );

	}

	for ( let k = 0; k < m; k ++ ) idx.push( 0, 1 + ( k + 1 ) % m, 1 + k );
	const n = new Array( p.length ).fill( 0 );
	for ( let t = 0; t < idx.length; t += 3 ) {

		const a = idx[ t ] * 3, b = idx[ t + 1 ] * 3, c = idx[ t + 2 ] * 3;
		const ux = p[ b ] - p[ a ], uy = p[ b + 1 ] - p[ a + 1 ], uz = p[ b + 2 ] - p[ a + 2 ];
		const vx = p[ c ] - p[ a ], vy = p[ c + 1 ] - p[ a + 1 ], vz = p[ c + 2 ] - p[ a + 2 ];
		const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
		for ( const o of [ a, b, c ] ) {

			n[ o ] += nx; n[ o + 1 ] += ny; n[ o + 2 ] += nz;

		}

	}

	for ( let i = 0; i < n.length; i += 3 ) {

		if ( n[ i + 1 ] < 0 ) {

			n[ i ] = - n[ i ]; n[ i + 1 ] = - n[ i + 1 ]; n[ i + 2 ] = - n[ i + 2 ];

		}

		n[ i + 1 ] += 0.6; // rounded edges read softer
		const l = Math.hypot( n[ i ], n[ i + 1 ], n[ i + 2 ] );
		n[ i ] /= l; n[ i + 1 ] /= l; n[ i + 2 ] /= l;

	}

	// faces up
	for ( let t = 0; t < idx.length; t += 3 ) {

		const a = idx[ t ] * 3, b = idx[ t + 1 ] * 3, c = idx[ t + 2 ] * 3;
		const ux = p[ b ] - p[ a ], uz = p[ b + 2 ] - p[ a + 2 ], vx = p[ c ] - p[ a ], vz = p[ c + 2 ] - p[ a + 2 ];
		if ( uz * vx - ux * vz < 0 ) {

			const s = idx[ t + 1 ];
			idx[ t + 1 ] = idx[ t + 2 ];
			idx[ t + 2 ] = s;

		}

	}

	return { p, n, idx };

}

function buildPatch() {

	let s = 1234567;
	const rand = () => {

		s = ( s * 16807 ) % 2147483647;
		return ( s - 1 ) / 2147483646;

	};

	const pts = blueNoise( rand, N_COBBLE + N_CHIP + N_SMALL, PCELL );
	const pos = [], nor = [], slot = [], idx = [];
	for ( let k = 0; k < pts.length; k ++ ) {

		const type = k < N_COBBLE ? 1 : k < N_COBBLE + N_CHIP ? 2 : 0;
		const part = type === 2 ? chipPart( k % 7 ) : type === 1 ? stonePart( 30 + ( k % 6 ), 1, k % 2 ? 0.5 : 0.05 ) : stonePart( 20 + ( k % 7 ), 0, k % 3 === 0 ? 0.8 : 0.1 );
		const base = pos.length / 3;
		for ( let i = 0; i < part.p.length; i ++ ) pos.push( part.p[ i ] );
		for ( let i = 0; i < part.n.length; i ++ ) nor.push( part.n[ i ] );
		for ( let i = 0; i < part.p.length / 3; i ++ ) slot.push( pts[ k ][ 0 ], pts[ k ][ 1 ], rand(), type );
		for ( let i = 0; i < part.idx.length; i ++ ) idx.push( base + part.idx[ i ] );

	}

	const g = new THREE.BufferGeometry();
	g.setAttribute( 'position', new THREE.Float32BufferAttribute( pos, 3 ) );
	g.setAttribute( 'normal', new THREE.Float32BufferAttribute( nor, 3 ) );
	g.setAttribute( 'aSlot', new THREE.Float32BufferAttribute( slot, 4 ) );
	g.setIndex( new THREE.Uint16BufferAttribute( idx, 1 ) );
	return g;

}

const hash12 = ( p ) => fract( sin( dot( p, vec2( 127.1, 311.7 ) ) ).mul( 43758.5453 ) );

export class PebbleField {

	// terrain: TerrainData, gpu: TerrainGPU, mask: { data (RGBA8), res, texel } over the terrain domain
	constructor( { terrain, gpu, mask } ) {

		this.terrain = terrain;
		this.gpu = gpu;
		const T = terrain;
		this.maskTex = new THREE.DataTexture( mask.data, mask.res, mask.res, THREE.RGBAFormat, THREE.UnsignedByteType );
		this.maskTex.magFilter = this.maskTex.minFilter = THREE.LinearFilter;
		this.maskTex.generateMipmaps = false;
		this.maskTex.colorSpace = THREE.NoColorSpace;
		this.maskTex.name = 'debrisPebbleMask';
		this.maskTex.needsUpdate = true;

		// occupied cells + height range
		const n = Math.ceil( T.size / PCELL );
		this.cellsPerSide = n;
		this.cellFlags = new Uint8Array( n * n );
		this.cellMinY = new Float32Array( n * n );
		this.cellMaxY = new Float32Array( n * n );
		const mpc = PCELL / mask.texel;
		const hpc = PCELL / T.texel;
		let occupied = 0;
		for ( let cj = 0; cj < n; cj ++ ) for ( let ci = 0; ci < n; ci ++ ) {

			let any = 0;
			const i0 = Math.max( 0, Math.floor( ci * mpc ) - 1 ), i1 = Math.min( mask.res - 1, Math.ceil( ( ci + 1 ) * mpc ) );
			const j0 = Math.max( 0, Math.floor( cj * mpc ) - 1 ), j1 = Math.min( mask.res - 1, Math.ceil( ( cj + 1 ) * mpc ) );
			for ( let j = j0; j <= j1 && ! any; j ++ ) for ( let i = i0; i <= i1; i ++ ) {

				const o = ( j * mask.res + i ) * 4;
				if ( mask.data[ o ] > 6 || mask.data[ o + 1 ] > 6 || mask.data[ o + 2 ] > 6 ) {

					any = 1;
					break;

				}

			}

			const c = cj * n + ci;
			this.cellFlags[ c ] = any;
			if ( ! any ) continue;
			occupied ++;
			let mn = Infinity, mx = - Infinity;
			const hi0 = Math.max( 0, Math.floor( ci * hpc ) - 1 ), hi1 = Math.min( T.res - 1, Math.ceil( ( ci + 1 ) * hpc ) );
			const hj0 = Math.max( 0, Math.floor( cj * hpc ) - 1 ), hj1 = Math.min( T.res - 1, Math.ceil( ( cj + 1 ) * hpc ) );
			for ( let j = hj0; j <= hj1; j ++ ) for ( let i = hi0; i <= hi1; i ++ ) {

				const h = T.heights[ j * T.res + i ];
				if ( h < mn ) mn = h;
				if ( h > mx ) mx = h;

			}

			this.cellMinY[ c ] = mn;
			this.cellMaxY[ c ] = mx;

		}

		this.occupiedCells = occupied;
		const geometry = buildPatch();
		this.patchTris = geometry.index.count / 3;
		this.arr = new Float32Array( MAX_CELLS * 4 );
		this.attr = new THREE.InstancedBufferAttribute( this.arr, 4 );
		this.attr.setUsage( THREE.DynamicDrawUsage );
		geometry.setAttribute( 'iCell', this.attr );
		geometry.instanceCount = 0;
		this.geometry = geometry;
		this.material = this._createMaterial();
		this.mesh = new THREE.Mesh( geometry, this.material );
		this.mesh.name = 'debris-pebbles';
		this.mesh.frustumCulled = false;
		this.mesh.castShadow = false;
		this.mesh.receiveShadow = true;
		this.mesh.matrixAutoUpdate = false;
		this.count = 0;

		this._frustum = new THREE.Frustum();
		this._mat = new THREE.Matrix4();
		this._box = new THREE.Box3();
		this._last = new Float64Array( 8 ).fill( NaN );

	}

	_createMaterial() {

		const T = this.terrain, gpu = this.gpu;
		const mTex = texture( this.maskTex );
		const det = gpu.detailTexture;
		const vPeb = varyingProperty( 'vec4', 'vPebble' ); // type, palette, seed, radius

		const mat = standard( { roughness: 0.8, metalness: 0 } );
		mat.name = 'DebrisPebbles';
		mat.mrtNode = staticVelocityMRT;
		mat.underwaterLighting = 'lite';
		mat.setupLightingModel = () => new TerrainLightingModel( mat, gpu.sunShadowAt( positionWorld ) );

		mat.positionNode = Fn( () => {

			const cell = attribute( 'iCell', 'vec4' ).xy;
			const slot = attribute( 'aSlot', 'vec4' );
			const P = attribute( 'position', 'vec3' );
			const N = attribute( 'normal', 'vec3' );
			const xz = cell.add( slot.xy );
			const m = mTex.sample( xz.sub( T.origin ).div( T.size ) ).level( 0 );
			const type = slot.w;
			const isSmall = type.lessThan( 0.5 );
			const isCob = type.greaterThan( 0.5 ).and( type.lessThan( 1.5 ) );
			const isChip = type.greaterThan( 1.5 );
			const dens = select( isSmall, m.x, select( isCob, m.y, m.z ) );
			const h1 = hash12( xz.mul( 1.37 ).add( 0.51 ) );
			const h2 = hash12( xz.mul( 2.11 ).add( 7.3 ) );
			const h3 = hash12( xz.mul( 3.7 ).add( 1.1 ) );
			const h4 = hash12( xz.mul( 5.3 ).add( 2.9 ) );
			const h5 = hash12( xz.mul( 8.9 ).add( 4.7 ) );
			const present = step( h1, dens );
			const d = length( xz.sub( cameraPosition.xz ) );
			const fade = float( 1 ).sub( smoothstep( select( isCob, float( FADE_COBBLE[ 0 ] ), float( FADE_SMALL[ 0 ] ) ), select( isCob, float( FADE_COBBLE[ 1 ] ), float( FADE_SMALL[ 1 ] ) ), d.add( h4.mul( 2.5 ) ) ) );
			// size: denser patches have more but slightly smaller stones
			const r = select( isSmall, mix( 0.007, 0.024, h2.mul( h2 ) ), select( isCob, mix( 0.03, 0.09, h2.mul( h2 ) ), mix( 0.008, 0.022, h2 ) ) )
				.mul( float( 1.15 ).sub( dens.mul( 0.3 ) ) );
			const sc = vec3( r.mul( mix( 1.0, 1.55, h3 ) ), r.mul( select( isChip, float( 0.28 ), mix( 0.42, 0.8, h4 ) ) ), r );
			const k = present.mul( fade );
			// tilt about the local x axis (the stones don't all lie flat), then yaw
			const tilt = h5.sub( 0.5 ).mul( select( isChip, float( 0.5 ), float( 0.9 ) ) );
			const ct = cos( tilt ), st = sin( tilt );
			const q0 = P.mul( sc ).mul( k );
			const q = vec3( q0.x, q0.y.mul( ct ).sub( q0.z.mul( st ) ), q0.y.mul( st ).add( q0.z.mul( ct ) ) );
			const yaw = h3.mul( 6.2832 );
			const c = cos( yaw ), s = sin( yaw );
			const gy = gpu.heightAt( xz );
			const lift = sc.y.mul( select( isChip, float( 0.35 ), float( 0.15 ) ) ).mul( k );
			const n0 = N.div( sc );
			const nl = vec3( n0.x, n0.y.mul( ct ).sub( n0.z.mul( st ) ), n0.y.mul( st ).add( n0.z.mul( ct ) ) );
			normalLocal.assign( normalize( vec3( nl.x.mul( c ).add( nl.z.mul( s ) ), nl.y, nl.z.mul( c ).sub( nl.x.mul( s ) ) ) ) );
			// palette: dark basalt / grey / pale coral limestone in proportions set by the region
			// (mask alpha: 0 rocky shore .. 1 white coral beach)
			const dark = mix( 0.72, 0.24, m.w ), grey = mix( 0.26, 0.3, m.w );
			const pal = select( h4.lessThan( dark ), mix( 0.02, 0.3, h2 ), select( h4.lessThan( dark.add( grey ) ), mix( 0.32, 0.56, h2 ), mix( 0.6, 0.97, h2 ) ) );
			vPeb.assign( vec4( type, pal, h2.add( h3 ), r ) );
			return vec3( xz.x.add( q.x.mul( c ) ).add( q.z.mul( s ) ), gy.add( lift ).add( q.y ), xz.y.add( q.z.mul( c ) ).sub( q.x.mul( s ) ) );

		} )();

		const outRough = float( 0.8 ).toVar( 'pebRough' );
		const outN = vec3( 0, 1, 0 ).toVar( 'pebN' );
		const outAO = float( 1 ).toVar( 'pebAO' );
		mat.colorNode = Fn( () => {

			const p = positionWorld;
			const N = normalWorldGeometry.toVar();
			const type = vPeb.x, pal = vPeb.y, seed = vPeb.z, r = vPeb.w;
			const isChip = step( 1.5, type );
			const tile = clamp( r.mul( 4 ), 0.03, 0.3 );
			const w = triWeights( N );
			const A = texture( det, p.zy.div( tile ) ), B = texture( det, p.xz.div( tile ).add( 0.37 ) ), Cc = texture( det, p.xy.div( tile ).add( 0.71 ) );
			const T3 = A.mul( w.x ).add( B.mul( w.y ) ).add( Cc.mul( w.z ) );
			const S = stoneSurface( { T: T3, N, pal, style: float( 0 ), seed, tile } );
			// shell and coral grit: white / cream / pink chips with growth bands
			const hue = fract( seed.mul( 7.13 ) );
			let chip = mix( srgb( 0.94, 0.92, 0.88 ), srgb( 0.86, 0.74, 0.56 ), smoothstep( 0.3, 0.45, hue ) );
			chip = mix( chip, srgb( 0.9, 0.6, 0.56 ), smoothstep( 0.58, 0.66, hue ) );
			chip = mix( chip, srgb( 0.58, 0.4, 0.28 ), smoothstep( 0.76, 0.82, hue ) );
			chip = mix( chip, srgb( 0.36, 0.33, 0.4 ), smoothstep( 0.9, 0.95, hue ) );
			chip = chip.mul( T3.a.sub( 0.5 ).mul( 0.5 ).add( 1 ) ).mul( sin( p.x.add( p.z ).mul( 900 ) ).mul( 0.06 ).add( 0.97 ) );
			// sea glass among the grit (frosted green / brown / white, glinting) and pumice among the
			// pebbles (pale, porous)
			const glass = step( 0.955, fract( seed.mul( 3.71 ) ) ).mul( isChip );
			const glassC = select( fract( seed.mul( 13.3 ) ).lessThan( 0.45 ), srgb( 0.42, 0.62, 0.45 ), select( fract( seed.mul( 13.3 ) ).lessThan( 0.75 ), srgb( 0.52, 0.34, 0.18 ), srgb( 0.8, 0.84, 0.82 ) ) );
			chip = mix( chip, glassC, glass );
			const pumice = step( 0.93, fract( seed.mul( 5.17 ) ) ).mul( float( 1 ).sub( isChip ) );
			const pumiceC = srgb( 0.66, 0.64, 0.6 ).mul( float( 1 ).sub( smoothstep( 0.55, 0.8, T3.b ).mul( 0.45 ) ) );
			let col = mix( mix( S.albedo, pumiceC, pumice ), chip, isChip );
			let rough = mix( mix( S.rough, float( 0.95 ), pumice ), mix( float( 0.5 ), float( 0.12 ), glass ), isChip );
			// wet near the sea, dusted with sand where they touch the ground
			const ground = gpu.heightAt( p.xz );
			const contact = float( 1 ).sub( smoothstep( 0.0, max( r.mul( 0.7 ), 0.006 ), p.y.sub( ground ) ) );
			const wet = float( 1 ).sub( smoothstep( 0.4, 1.0, p.y ) );
			col = mix( col, srgb( 0.78, 0.7, 0.56 ), contact.mul( smoothstep( 0.8, 1.6, ground ) ).mul( 0.35 ) );
			col = col.mul( float( 1 ).sub( wet.mul( 0.45 ) ) );
			rough = mix( rough, float( 0.22 ), wet.mul( 0.85 ) );
			outRough.assign( clamp( rough, 0.05, 1 ) );
			outAO.assign( float( 1 ).sub( contact.mul( 0.5 ) ) );
			outN.assign( perturbNormal( N, S.hd.mul( float( 1 ).sub( isChip ) ), 1.0 ) );
			return vec4( col, 1 );

		} )();
		mat.roughnessNode = outRough;
		mat.normalNode = transformNormalToView( outN );
		mat.aoNode = saturate( outAO );
		return mat;

	}

	// Recompute the visible cells (skipped when the camera did not move / turn).
	update( camera ) {

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
		const T = this.terrain;
		const n = this.cellsPerSide;
		let count = 0;
		// nothing to see from high above
		const hCam = cy - T.heightAt( cx, cz );
		if ( hCam < R_FAR ) {

			const i0 = Math.max( 0, Math.floor( ( cx - R_FAR - T.origin ) / PCELL ) ), i1 = Math.min( n - 1, Math.floor( ( cx + R_FAR - T.origin ) / PCELL ) );
			const j0 = Math.max( 0, Math.floor( ( cz - R_FAR - T.origin ) / PCELL ) ), j1 = Math.min( n - 1, Math.floor( ( cz + R_FAR - T.origin ) / PCELL ) );
			for ( let j = j0; j <= j1; j ++ ) for ( let i = i0; i <= i1; i ++ ) {

				const c = j * n + i;
				if ( ! this.cellFlags[ c ] ) continue;
				const x0 = T.origin + i * PCELL, z0 = T.origin + j * PCELL;
				const dx = Math.max( x0 - cx, 0, cx - x0 - PCELL ), dz = Math.max( z0 - cz, 0, cz - z0 - PCELL );
				if ( Math.hypot( dx, dz ) > R_FAR ) continue;
				this._box.min.set( x0, this.cellMinY[ c ] - 0.2, z0 );
				this._box.max.set( x0 + PCELL, this.cellMaxY[ c ] + 0.3, z0 + PCELL );
				if ( ! this._frustum.intersectsBox( this._box ) ) continue;
				if ( count >= MAX_CELLS ) break;
				this.arr[ count * 4 ] = x0;
				this.arr[ count * 4 + 1 ] = z0;
				count ++;

			}

		}

		this.count = count;
		this.geometry.instanceCount = count;
		this.attr.clearUpdateRanges();
		this.attr.addUpdateRange( 0, Math.max( 1, count ) * 4 );
		this.attr.needsUpdate = true;
		return true;

	}

	get triangles() {

		return this.count * this.patchTris;

	}

	dispose() {

		this.maskTex.dispose();
		this.material.dispose();
		this.geometry.dispose();

	}

}
