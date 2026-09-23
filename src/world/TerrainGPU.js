import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, float, vec2, vec4, ivec2, uvec2, clamp, floor, fract, mix, max, log2, length, smoothstep, textureLoad, texture,
	textureStore, uniform, select, globalId,
} from 'three/tsl';
import { G } from '../core/Globals.js';
import { bakeTerrainMaps, buildShadowHeights } from './terrain/TerrainBake.js';
import { getDetailTexture } from './terrain/DetailTextures.js';

// GPU-side terrain data shared by every shader that needs ground height:
// ocean (depth attenuation / hiding under land), shore waves, terrain mesh,
// particles, water queries.
//
// Textures (all over the terrain domain unless noted):
//   heightTexture  R32F   exact heights, bilinear filtering done manually (heightAt)
//   normalTexture  RGBA8  macro normal xz (encoded), rock mask, baked ambient occlusion (mipmapped)
//   splatTexture   RGBA8  loose sand, worn ground / paths, gullies (land) or seagrass (seabed),
//                         seabed rubble (mipmapped)
//   detailTexture  RGBA8  512^2 tileable detail heights (rock, soil, sand, fbm), see DetailTextures
//   shadowHeightTexture R16F 512^2 (4 m) heights with max-mips (input of the sun shadow bake)
//   sunShadowTexture    RGBA16F storage 512^2: 'shadow top' height and occluder distance toward the
//                       key light, re-baked on the GPU when G.sunDir moves (updateSunShadow)
const SUN_N = 512;
const SUN_T0 = 24; // m: nearer occluders are left to the shadow map / N.L
const SUN_DT0 = 6;
const SUN_GROWTH = 1.22;
const SUN_STEPS = 24; // reaches ~3 km

export class TerrainGPU {

	constructor( terrain, shoreField ) {

		this.terrain = terrain;
		const res = terrain.res;
		const t0 = performance.now();

		// heights: R32F, loaded with manual bilinear filtering (no sampler / float filtering needed)
		this.heightTexture = new THREE.DataTexture( terrain.heights, res, res, THREE.RedFormat, THREE.FloatType );
		this.heightTexture.magFilter = this.heightTexture.minFilter = THREE.NearestFilter;
		this.heightTexture.generateMipmaps = false;
		this.heightTexture.needsUpdate = true;

		const maps = bakeTerrainMaps( terrain );
		const make = ( data, name ) => {

			const tex = new THREE.DataTexture( data, res, res, THREE.RGBAFormat, THREE.UnsignedByteType );
			tex.magFilter = THREE.LinearFilter;
			tex.minFilter = THREE.LinearMipmapLinearFilter;
			tex.generateMipmaps = true;
			tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
			tex.colorSpace = THREE.NoColorSpace;
			tex.name = name;
			tex.needsUpdate = true;
			return tex;

		};

		this.normalTexture = make( maps.normal, 'terrainNormalRockAO' );
		this.splatTexture = make( maps.splat, 'terrainSplat' );
		this.detailTexture = getDetailTexture();

		// coarse heights for the sun shadow march (half float is filterable everywhere)
		const sh = buildShadowHeights( terrain, 4 );
		const st = new THREE.DataTexture( sh.levels[ 0 ].data, sh.levels[ 0 ].width, sh.levels[ 0 ].height, THREE.RedFormat, THREE.HalfFloatType );
		st.mipmaps = sh.levels;
		st.magFilter = THREE.LinearFilter;
		st.minFilter = THREE.LinearMipmapLinearFilter;
		st.generateMipmaps = false;
		st.wrapS = st.wrapT = THREE.ClampToEdgeWrapping;
		st.name = 'terrainShadowHeights';
		st.needsUpdate = true;
		this.shadowHeightTexture = st;
		this.shadowTexel = sh.texel;
		this.maxHeight = sh.maxHeight;

		this.size = terrain.size;
		this.origin = terrain.origin;
		this.res = res;

		this.uOrigin = uniform( terrain.origin ).setName( 'terOrigin' );
		this.uSize = uniform( terrain.size ).setName( 'terSize' );

		if ( shoreField ) this.setShoreField( shoreField );
		this._initSunShadow();
		this.timings = { ...maps.ms, detail: this.detailTexture.userData.ms, total: performance.now() - t0 };

	}

	// ------------------------------------------------------------ heightfield sun shadow

	// World-space shadow map of the terrain for the key light: for every 4 m cell the lowest height
	// that still sees the light past the terrain (max over the march of h(q) - t * tan(elevation))
	// and the distance to that occluder (penumbra width). One texture fetch gives soft, long hill
	// shadows for any point - terrain, rocks, buildings, plants - far beyond the shadow map range.
	_initSunShadow() {

		const tex = new THREE.StorageTexture( SUN_N, SUN_N );
		tex.type = THREE.HalfFloatType;
		tex.format = THREE.RGBAFormat;
		// nearest + textureLoad (manual bilinear in sunShadowAt): no sampler binding, because every lit
		// material includes this lookup and WebGPU allows only 16 samplers per shader stage
		tex.magFilter = tex.minFilter = THREE.NearestFilter;
		tex.generateMipmaps = false;
		tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
		tex.name = 'terrainSunShadow';
		this.sunShadowTexture = tex;
		this.uSunBake = uniform( new THREE.Vector3( 0, 1, 0 ) ).setName( 'terSunBake' );
		this.uSunBaked = uniform( 0 ).setName( 'terSunBaked' ); // 0 until the first bake: everything lit
		this._sunBaked = false;

		const heights = this.shadowHeightTexture;
		const invTexel = 1 / this.shadowTexel;
		this.sunShadowKernel = Fn( () => {

			const ij = vec2( globalId.xy );
			const xz = ij.add( 0.5 ).div( SUN_N ).mul( this.uSize ).add( this.uOrigin ).toVar();
			const L = this.uSunBake;
			const lxz = max( length( L.xz ), 1e-4 );
			const dir = L.xz.div( lxz ).toVar();
			const tanEl = L.y.div( lxz ).toVar();
			const top = float( - 1e4 ).toVar(), occ = float( 0 ).toVar();
			const t = float( SUN_T0 ).toVar(), dt = float( SUN_DT0 ).toVar();
			Loop( SUN_STEPS, () => {

				const q = xz.add( dir.mul( t ) );
				const hq = texture( heights, this.uvOf( q ) ).level( log2( max( dt.mul( invTexel ), 1 ) ) ).x;
				const v = hq.sub( t.mul( tanEl ) );
				If( v.greaterThan( top ), () => {

					top.assign( v );
					occ.assign( t );

				} );
				dt.mulAssign( SUN_GROWTH );
				t.addAssign( dt );

			} );

			textureStore( tex, uvec2( globalId.xy ), vec4( top, occ, 0, 1 ) );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Terrain Sun Shadow' );

	}

	// Re-bake the sun shadow map when the key light moved (cheap: ~6 M texture fetches).
	updateSunShadow( renderer, force = false ) {

		const L = G.sunDir.value;
		if ( ! force && this._sunBaked && L.angleTo( this.uSunBake.value ) < 0.0015 ) return false;
		this.uSunBake.value.copy( L );
		renderer.compute( this.sunShadowKernel, [ SUN_N / 8, SUN_N / 8, 1 ] );
		this._sunBaked = true;
		this.uSunBaked.value = 1;
		return true;

	}

	// TSL: 0 (in the terrain's shadow) .. 1 (lit) for a world position, soft penumbra that widens
	// with the occluder distance
	sunShadowAt( p ) {

		const N = this.sunShadowTexture.image.width;
		const st = clamp( this.uvOf( p.xz ).mul( N ).sub( 0.5 ), 0, N - 1.001 );
		const i = ivec2( floor( st ) );
		const t = fract( st );
		const tex = this.sunShadowTexture;
		const s = mix(
			mix( textureLoad( tex, i ), textureLoad( tex, i.add( ivec2( 1, 0 ) ) ), t.x ),
			mix( textureLoad( tex, i.add( ivec2( 0, 1 ) ) ), textureLoad( tex, i.add( ivec2( 1, 1 ) ) ), t.x ),
			t.y
		);
		const w = s.y.mul( 0.012 ).add( 0.35 );
		return mix( float( 1 ), smoothstep( w.negate(), w, p.y.sub( s.x ) ), this.uSunBaked );

	}

	setShoreField( f ) {

		this.shoreRes = f.res;
		if ( ! this.shoreTexture ) {

			this.shoreTexture = new THREE.DataTexture( f.data, f.res, f.res, THREE.RGBAFormat, THREE.FloatType );
			this.shoreTexture.magFilter = this.shoreTexture.minFilter = THREE.NearestFilter;
			this.shoreTexture.generateMipmaps = false;

		} else {

			this.shoreTexture.image.data.set( f.data );

		}

		this.shoreTexture.needsUpdate = true;

	}

	// ------------------------------------------------------------ TSL

	// exact bilinear height at world xz (matches TerrainData.heightAt)
	heightAt( xz ) {

		const res = this.res;
		const f = xz.sub( this.uOrigin ).div( this.uSize ).mul( res ).sub( 0.5 );
		const fc = clamp( f, 0, res - 1.001 );
		const i = floor( fc );
		const t = fract( fc );
		const ii = ivec2( i );
		const tex = this.heightTexture;
		const a = textureLoad( tex, ii ).x;
		const b = textureLoad( tex, ii.add( ivec2( 1, 0 ) ) ).x;
		const c = textureLoad( tex, ii.add( ivec2( 0, 1 ) ) ).x;
		const d = textureLoad( tex, ii.add( ivec2( 1, 1 ) ) ).x;
		const h = mix( mix( a, b, t.x ), mix( c, d, t.x ), t.y );
		// outside the domain: deep ocean floor
		const outside = f.x.lessThan( 0 ).or( f.y.lessThan( 0 ) ).or( f.x.greaterThan( res - 1 ) ).or( f.y.greaterThan( res - 1 ) );
		return select( outside, float( - 90 ), h );

	}

	uvOf( xz ) {

		return xz.sub( this.uOrigin ).div( this.uSize );

	}

	// filtered normal (xz components), rock mask, ambient occlusion (`level`: explicit mip, e.g. in
	// the vertex stage)
	normalRock( xz, level = null ) {

		let s = texture( this.normalTexture, this.uvOf( xz ) );
		if ( level !== null ) s = s.level( level );
		return vec4( s.xy.mul( 2 ).sub( 1 ), s.z, s.w );

	}

	// loose sand, worn ground / paths, gullies (land) or seagrass (seabed), seabed rubble
	splat( xz ) {

		return texture( this.splatTexture, this.uvOf( xz ) );

	}

	// shore field: (T, dirX, dirZ, exposure), bilinear via loads (float32 data)
	shoreSample( xz ) {

		const res = this.shoreRes;
		const f = xz.sub( this.uOrigin ).div( this.uSize ).mul( res ).sub( 0.5 );
		const fc = clamp( f, 0, res - 1.001 );
		const i = floor( fc );
		const t = fract( fc );
		const ii = ivec2( i );
		const tex = this.shoreTexture;
		const a = textureLoad( tex, ii );
		const b = textureLoad( tex, ii.add( ivec2( 1, 0 ) ) );
		const c = textureLoad( tex, ii.add( ivec2( 0, 1 ) ) );
		const d = textureLoad( tex, ii.add( ivec2( 1, 1 ) ) );
		return mix( mix( a, b, t.x ), mix( c, d, t.x ), t.y );

	}

}
