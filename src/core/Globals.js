import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';

// Shared uniforms used across many shaders. Values are written by the CPU
// every frame (time, sun, camera state) so every system sees the same state.
export const G = {
	time: uniform( 0 ).setName( 'gTime' ), // simulation time (s)
	dt: uniform( 1 / 60 ).setName( 'gDt' ),
	seaLevel: uniform( 0 ).setName( 'gSeaLevel' ),

	// Sun (or moon at night) direction, pointing toward the light.
	sunDir: uniform( new THREE.Vector3( 0.3, 0.6, - 0.7 ).normalize() ).setName( 'gSunDir' ),
	// Radiance-scaled irradiance of the sun at sea level after atmospheric extinction.
	sunColor: uniform( new THREE.Color( 1, 1, 1 ) ).setName( 'gSunColor' ),
	// Hemispherical sky irradiance at sea level (cosine-weighted, divided by PI).
	skyIrradiance: uniform( new THREE.Color( 0.3, 0.4, 0.6 ) ).setName( 'gSkyIrr' ),
	// Average horizon sky color (used for fog / aerial perspective fallback).
	horizonColor: uniform( new THREE.Color( 0.6, 0.7, 0.8 ) ).setName( 'gHorizon' ),

	// Water optical properties (per meter).
	waterAbsorption: uniform( new THREE.Vector3( 0.42, 0.075, 0.035 ) ).setName( 'gWaterAbs' ),
	waterScattering: uniform( new THREE.Vector3( 0.012, 0.018, 0.024 ) ).setName( 'gWaterScat' ),

	// 1 when the whole view is under water: the camera is deeper than the near clip plane can reach
	// (closer to the surface the view can be split by the waterline on the lens)
	cameraUnderwater: uniform( 0 ).setName( 'gCamUnder' ),
	// water surface height at the camera (read back from the GPU query, 1-3 frames late)
	cameraWaterHeight: uniform( 0 ).setName( 'gCamWaterH' ),
	exposure: uniform( 1 ).setName( 'gExposure' ),

	// Wind (drives vegetation sway, flags, spray drift, clouds). windDir is the direction the wind blows toward.
	windDir: uniform( new THREE.Vector2( 0.35, 0.94 ).normalize() ).setName( 'gWindDir' ),
	windSpeed: uniform( 7 ).setName( 'gWindSpeed' ), // m/s at 10 m height

	// 0 = day, 1 = full night. Drives emissive windows / lanterns.
	night: uniform( 0 ).setName( 'gNight' ),
};

export const GRAVITY = 9.81;
