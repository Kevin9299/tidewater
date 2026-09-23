import * as THREE from 'three/webgpu';
import {
	Fn, If, float, vec2, vec3, vec4, ivec2, uvec2, clamp, floor, fract, mix, max, abs, sqrt, dot, exp, smoothstep, saturate,
	textureLoad, textureStore, uniform, globalId, positionWorld, normalWorld,
} from 'three/tsl';
import { G } from '../core/Globals.js';
import { SceneLighting } from './SceneLighting.js';

// Sunlight bounced off the ground (one diffuse bounce, the dominant indirect light on a sunny beach):
// sunlit coral sand lights the underside of the pier, the eaves, the boat hull, palm trunks and the
// undersides of rocks and fronds.
//
// Bake (compute, only when the key light moves): a 512^2 map over the terrain domain (4 m texels,
// aligned with the terrain sun shadow map) of the light the ground reflects per unit of sun
// irradiance: albedo (sand / grass / forest / rock from the terrain masks, the sea as the seabed seen
// through the water column) x cos(sun, ground normal) x heightfield sun shadow, minus what the
// environment map's own ground (the atmosphere's dark planet surface) already gives, then blurred
// (binomial 5x5, ~4 m). Alpha: height of the bouncing surface (ground or sea level).
//
// Shading: irradiance = sun colour x cloud shadow x map(xz) x lower-hemisphere view factor of the
// normal ((1 - N.y) / 2) x fade with height above the bouncing surface, added to the indirect
// diffuse light (so the diffuse albedo, the material AO and the underwater tint all apply). One
// bilinear lookup via texel loads (no sampler) and only for normals that face down or sideways.
const N = 512;
// objects around the receiver shade part of the ground it sees (their shadows are not in the map)
const SURROUND = 0.6;
// the environment map's lower hemisphere: the atmosphere's planet ground (Atmosphere.groundAlbedo)
const ENV_GROUND = 0.08;

export const GroundBounce = {
	strength: uniform( 1 ).setName( 'giBounce' ),
	lookup: null, // ( builder ) -> vec3 irradiance node, or null
};

export function installGroundBounce( { terrain, clouds = null } ) {

	const make = ( name ) => {

		const tex = new THREE.StorageTexture( N, N );
		tex.type = THREE.HalfFloatType;
		tex.format = THREE.RGBAFormat;
		tex.magFilter = tex.minFilter = THREE.NearestFilter;
		tex.generateMipmaps = false;
		tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
		tex.name = name;
		return tex;

	};

	const raw = make( 'groundBounceRaw' );
	const map = make( 'groundBounce' );
	const uSun = uniform( new THREE.Vector3( 0, 1, 0 ) ).setName( 'giSun' );
	const uReady = uniform( 0 ).setName( 'giReady' );
	const lin = ( r, g, b ) => {

		const c = new THREE.Color().setRGB( r, g, b, THREE.SRGBColorSpace );
		return vec3( c.r, c.g, c.b );

	};

	const bakeRaw = Fn( () => {

		const ij = vec2( globalId.xy );
		const xz = ij.add( 0.5 ).div( N ).mul( terrain.uSize ).add( terrain.uOrigin ).toVar();
		const h = terrain.heightAt( xz ).toVar();
		const nr = terrain.normalRock( xz, 1 );
		const n = vec3( nr.x, sqrt( saturate( float( 1 ).sub( dot( nr.xy, nr.xy ) ) ) ), nr.y );
		const slope = float( 1 ).sub( n.y );
		const sp = terrain.splat( xz ).level( 1 );
		const L = uSun;
		// land cover (same masks as the terrain material, coarsely)
		const rockW = saturate( smoothstep( 0.35, 0.6, nr.z.mul( 0.7 ).add( smoothstep( 0.3, 0.62, slope ).mul( 0.5 ) ) ) );
		const sandW = smoothstep( 0.3, 0.72, sp.x ).mul( float( 1 ).sub( rockW ) );
		const jungleW = saturate( smoothstep( 9, 24, h ).add( smoothstep( 0.18, 0.36, slope ) ) );
		const veg = mix( lin( 0.3, 0.36, 0.16 ), lin( 0.12, 0.17, 0.07 ), jungleW );
		// dry coral sand, darker where damp near the water line
		const sand = lin( 0.86, 0.79, 0.66 ).mul( mix( 0.6, 1.0, smoothstep( 0.2, 1.2, h ) ) );
		const land = mix( mix( veg, sand, sandW ), vec3( 0.08, 0.075, 0.065 ), rockW );
		// sea: the seabed (sand in the bay) seen through the water column, down and back up
		const depth = max( G.seaLevel.sub( h ), 0 );
		const seabed = mix( lin( 0.62, 0.58, 0.48 ), lin( 0.25, 0.3, 0.22 ), smoothstep( 0.3, 0.7, sp.y ) );
		const sea = seabed.mul( exp( G.waterAbsorption.add( G.waterScattering ).mul( depth ).mul( - 2.4 ) ) ).mul( 0.9 ).add( vec3( 0.01, 0.025, 0.03 ) );
		const wet = smoothstep( 0.05, - 0.1, h.sub( G.seaLevel ) );
		const albedo = mix( land, sea, wet );
		const nSurf = mix( n, vec3( 0, 1, 0 ), wet );
		// heightfield sun shadow at the ground (same filtering as TerrainGPU.sunShadowAt)
		const top = max( h, G.seaLevel );
		const vis = terrain.sunShadowAt( vec3( xz.x, top, xz.y ) );
		const E = max( dot( nSurf, L ), 0 ).mul( vis ).mul( smoothstep( - 0.02, 0.05, L.y ) );
		const out = max( albedo.mul( E ).sub( float( ENV_GROUND ).mul( max( L.y, 0 ) ) ), vec3( 0 ) );
		textureStore( raw, uvec2( globalId.xy ), vec4( out, top ) );

	} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Ground Bounce Bake' );

	const blur = Fn( () => {

		const c = ivec2( globalId.xy );
		const sum = vec4( 0 ).toVar();
		const w = [ 1, 4, 6, 4, 1 ];
		for ( let y = - 2; y <= 2; y ++ ) for ( let x = - 2; x <= 2; x ++ ) {

			sum.addAssign( textureLoad( raw, clamp( c.add( ivec2( x, y ) ), ivec2( 0 ), ivec2( N - 1 ) ) ).mul( w[ x + 2 ] * w[ y + 2 ] / 256 ) );

		}

		textureStore( map, uvec2( globalId.xy ), sum );

	} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Ground Bounce Blur' );

	// re-bake whenever the terrain's sun shadow is re-baked (the key light moved)
	const bake = ( renderer ) => {

		uSun.value.copy( G.sunDir.value );
		renderer.compute( bakeRaw, [ N / 8, N / 8, 1 ] );
		renderer.compute( blur, [ N / 8, N / 8, 1 ] );
		uReady.value = 1;

	};

	let done = false;
	const orig = terrain.updateSunShadow.bind( terrain );
	terrain.updateSunShadow = ( renderer, force = false ) => {

		const baked = orig( renderer, force );
		if ( baked || ! done ) bake( renderer );
		done = true;
		return baked;

	};

	GroundBounce.lookup = ( builder ) => {

		const mat = builder.material;
		if ( ! mat || mat.isWaterMaterial || mat.groundBounce === false ) return null;
		const P = positionWorld;
		const Nw = normalWorld;
		const view = saturate( Nw.y.mul( - 0.5 ).add( 0.5 ) ).toVar();
		const E = vec3( 0 ).toVar();
		If( view.greaterThan( 0.03 ).and( P.y.greaterThan( G.seaLevel.sub( 0.3 ) ) ).and( GroundBounce.strength.greaterThan( 0 ) ).and( uReady.greaterThan( 0 ) ), () => {

			const st = clamp( terrain.uvOf( P.xz ).mul( N ).sub( 0.5 ), 0, N - 1.001 );
			const i = ivec2( floor( st ) );
			const t = fract( st );
			const s = mix(
				mix( textureLoad( map, i ), textureLoad( map, i.add( ivec2( 1, 0 ) ) ), t.x ),
				mix( textureLoad( map, i.add( ivec2( 0, 1 ) ) ), textureLoad( map, i.add( ivec2( 1, 1 ) ) ), t.x ),
				t.y
			);
			const above = P.y.sub( s.w );
			// nothing from a surface above the receiver (under water: the sea surface is above), a
			// little less high up where more of the view is taken by other objects
			const fade = smoothstep( - 0.3, 0.3, above ).mul( mix( 0.4, 1.0, smoothstep( 30, 4, above ) ) );
			// cloud shadow over the surroundings: one texel of the cloud shadow map is plenty
			let cloud = float( 1 );
			if ( clouds && clouds.shadowMap ) {

				const uv = P.xz.sub( clouds.shadowCenter ).div( clouds.shadowSize ).add( 0.5 );
				const c = textureLoad( clouds.shadowMap, ivec2( uv.clamp( 0, 0.9999 ).mul( clouds.shadowMap.image.width ) ) ).x;
				const e = abs( uv.sub( 0.5 ) );
				cloud = mix( float( 1 ), c, clouds.shadowStrength.mul( smoothstep( 0.5, 0.42, max( e.x, e.y ) ) ) );

			}

			// a face turned away from a low sun looks at the ground its object shades (a long shadow)
			const Lh = G.sunDir.xz.div( max( G.sunDir.xz.length(), 1e-3 ) );
			const selfShade = float( 1 ).sub( saturate( dot( Nw.xz, Lh ).negate() ).mul( saturate( G.sunDir.y.oneMinus() ) ).mul( 0.6 ) );
			E.assign( s.xyz.mul( G.sunColor ).mul( cloud ).mul( view.mul( fade ).mul( selfShade ).mul( GroundBounce.strength.mul( SURROUND ) ) ) );

		} );
		return E;

	};

	SceneLighting.bounce = GroundBounce.lookup;
	return { bake, map, raw };

}
