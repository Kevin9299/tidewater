import { UniformBlock, ShaderModule, SceneLighting } from '../engine/webgpu.js';
import { commonModule } from '../engine/render/wgsl/common.js';
import { surfaceModule } from '../engine/render/wgsl/lighting.js';

// Installs the lighting hooks that every scene material uses:
//  - direct sun: attenuated along the refracted sun path through the water column (Beer-Lambert),
//    modulated by caustics that follow the waves above (swell + shore waves tilt the light, surf
//    foam and bubbles shade the floor)
//  - ambient: attenuated + tinted with depth
// Everything runs only for fragments that can be under water.
//
// Per material (material.underwaterLighting -> define UNDERWATER_LIGHTING): 'full' (2, default:
// terrain, reef, rocks - caustics, waves, foam shading), 'lite' (1, water column attenuation only:
// pier, boat hull, village) or 'none' (0, never under water: plants). The water itself (IS_WATER)
// is shaded by WaterMaterial and gets no modulation here.
//
// WGSL: installs `fn hookDirectModulation( P, N ) -> vec3f` and `fn hookAmbientModulation( P, N ) -> vec3f`
// (module prefix `underwater` for the helpers: underwaterLongWaves, underwaterMeanLevel).
// Note: unlike TSL, the modules' bindings (FFT maps, caustics, terrain, shore, clouds) are declared in
// every material that includes the lighting hooks, whatever its mode (the code itself is #if'd away).
export function installUnderwaterLighting( { fft, caustics, clouds = null, terrain = null, shore = null, surface = null, shoreSim = null } ) {

	const params = new UniformBlock( 'UnderwaterParams', {
		// highest the water can reach on the shore: the swash run-up grows with the surf height. Terrain
		// and props above it (the dry beach) skip the wave evaluation entirely.
		reach: [ 'f32', 3.0 ],
	}, { label: 'underwater' } );
	params.onBeforePack = () => {

		params.fields.reach.value = shore && shore.amplitude ? shore.amplitude.value * 1.5 + 1.2 : 3.0;

	};

	const T = !! terrain;
	const S = !! ( shore && terrain );
	const deps = [ commonModule, surfaceModule, fft.module, terrain && terrain.module, S && shore.module, shoreSim && shoreSim.module,
		caustics && caustics.module, clouds && clouds.module, surface && surface.attenuationModule ];

	const C3 = Math.min( 3, fft.cascades ), C2 = Math.min( 2, fft.cascades );

	const helpers = new ShaderModule( {
		name: 'underwater',
		deps,
		uniforms: params,
		uniformName: 'underwater',
		code: /* wgsl */`
struct UnderwaterLongWaves { height: f32, slope: vec2f, foam: f32 };

// long waves at xz: height, slope and foam from the coarse FFT cascades and the shore waves
fn underwaterLongWaves( xz: vec2f ) -> UnderwaterLongWaves {
	let seaDepth = ${ T ? 'frame.seaLevel - terrainHeightAt( xz )' : '50.0' };
	var h = 0.0;
	var slope = vec2f( 0.0 );
	for ( var c = 0; c < ${ C3 }; c++ ) {
		let uv = xz / ocean.sizes[ c ].x;
		let att = ${ surface ? 'waterSurfaceCascadeAttenuation( c, seaDepth )' : '1.0' };
		h += textureSampleLevel( oceanDisplacement, smpLinearRepeat, uv, c, 2.0 ).y * att;
		if ( c < 2 ) {
			let d = textureSampleLevel( oceanDerivatives, smpLinearRepeat, uv, c, 2.0 );
			slope += vec2f( d.x, d.y ) * att;
		}
	}

	var foam = 0.0;
${ S ? `	{
		let sw = shoreEvaluate( xz, seaDepth, terrainHeightAt( xz ) );
		h += sw.disp.y;
		let n = sw.nShore;
		slope += - vec2f( n.x, n.z ) / max( n.y, 0.25 );
		foam += sw.foam;
	}` : '' }
${ shoreSim ? '	foam += shoreSimSample( xz ).x * 0.8;' : '' }
	// waves only exist over water: none over dry land
${ T ? '	h = h * smoothstep( 0.0, 1.0, seaDepth ) - smoothstep( 0.0, -0.4, seaDepth ) * 10.0;' : '' }
	var o: UnderwaterLongWaves;
	o.height = frame.seaLevel + h;
	o.slope = slope;
	o.foam = foam;
	return o;
}

fn underwaterSigT() -> vec3f { return frame.waterAbsorption + frame.waterScattering; }

// mean water level from the two longest FFT cascades (one texture binding)
fn underwaterMeanLevel( P: vec3f ) -> f32 {
	var h = 0.0;
	for ( var c = 0; c < ${ C2 }; c++ ) {
		h += textureSampleLevel( oceanDisplacement, smpLinearRepeat, P.xz / ocean.sizes[ c ].x, c, 3.0 ).y;
	}
${ T ? '	h *= smoothstep( 0.0, 3.0, frame.seaLevel - terrainHeightAt( P.xz ) );' : '' }
	return frame.seaLevel + h;
}
`,
	} );

	const direct = new ShaderModule( {
		name: 'hook-directModulation-underwater',
		deps: [ helpers ],
		code: /* wgsl */`
fn hookDirectModulation( P: vec3f, N: vec3f ) -> vec3f {
#if IS_WATER
	return vec3f( 1.0 );
#else
	var result = vec3f( 1.0 );
#if UNDERWATER_LIGHTING == 2
	// the pixel's footprint on the ground plane, to filter the caustics over it (taken here, in
	// uniform control flow, ahead of the branches below)
	let gdx = dpdx( P.xz );
	let gdy = dpdy( P.xz );
#endif
#if UNDERWATER_LIGHTING != 0
	// cheap reject: above anything the water reaches
	if ( P.y < frame.seaLevel + underwater.reach ) {
#if UNDERWATER_LIGHTING == 2
		let lw = underwaterLongWaves( P.xz );
		let lwHeight = lw.height;
#else
		let lwHeight = underwaterMeanLevel( P );
#endif
		let d = max( lwHeight - P.y, 0.0 );
		if ( d > 0.0 ) {
			let under = smoothstep( 0.0, 0.08, d );
			let Ls = refract( - frame.sunDir, vec3f( 0.0, 1.0, 0.0 ), 1.0 / 1.333 );
			let mu = max( - Ls.y, 0.15 );
			let atten = exp( - underwaterSigT() * d / mu );
#if UNDERWATER_LIGHTING == 2
			let caust = ${ caustics ? 'causticsSample( P, d, lw.slope, sat( lw.foam ), gdx, gdy )' : 'vec3f( 1.0 )' };
#else
			let caust = vec3f( 1.0 );
#endif
			result = mix( vec3f( 1.0 ), atten * caust, under );
		}
	}
#endif
	let cloudShadow = ${ clouds ? 'cloudsShadow( P.xz )' : '1.0' };
	// hills shadowing the island and the bay at low sun (terrain heightfield shadow; the terrain and
	// the rocks apply it in their own lighting model)
#if HILL_SHADOW_SELF
	let hill = 1.0;
#else
	let hill = ${ T ? 'terrainSunShadowAt( P )' : '1.0' };
#endif
	return result * cloudShadow * hill;
#endif
}
`,
	} );

	const ambient = new ShaderModule( {
		name: 'hook-ambientModulation-underwater',
		deps: [ helpers ],
		code: /* wgsl */`
fn hookAmbientModulation( P: vec3f, N: vec3f ) -> vec3f {
	var result = vec3f( 1.0 );
#if !IS_WATER
#if UNDERWATER_LIGHTING != 0
	if ( P.y < frame.seaLevel + underwater.reach ) {
		// the ambient term only needs the mean water level (no shore evaluation)
		let d = max( underwaterMeanLevel( P ) - P.y, 0.0 );
		let under = smoothstep( 0.0, 0.1, d );
		// diffuse downwelling light: effective path ~1.2x depth, plus a little in-scattered blue
		let atten = exp( - underwaterSigT() * d * 1.2 ) * 0.85 + vec3f( 0.0, 0.02, 0.04 ) * exp( d * -0.1 );
		result = mix( vec3f( 1.0 ), atten, under );
	}
#endif
#endif
	return result;
}
`,
	} );

	SceneLighting.set( 'directModulation', direct );
	SceneLighting.set( 'ambientModulation', ambient );
	return { helpers, direct, ambient, params };

}
