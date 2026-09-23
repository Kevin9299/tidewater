import { ShaderModule, UniformBlock } from '../../gpu/Shader.js';
import { commonModule } from './common.js';
import { Matrix4, Vector4 } from '../../math/index.js';

// Scene lighting (the former SceneLightingModel on top of three's PhysicalLightingModel):
//   - sun / moon: GGX specular + Lambert diffuse, cascaded shadow maps (PCSS on the near cascade)
//   - image based lighting from the environment probe (prefiltered cube + SH irradiance)
//   - optional clearcoat and sheen lobes (material defines CLEARCOAT / SHEEN)
//   - scene hooks installed by systems (each defaults to a neutral stub):
//       fn hookDirectModulation( P: vec3f, N: vec3f ) -> vec3f   caustics, water column, clouds, hill shadow
//       fn hookAmbientModulation( P: vec3f, N: vec3f ) -> vec3f  underwater tint / attenuation
//       fn hookContactShadow( P: vec3f, N: vec3f ) -> f32        screen-space contact shadow of the sun
//       fn hookBounce( P: vec3f, N: vec3f ) -> vec3f             ground bounce irradiance (already / PI)
//       fn hookLocalLights( s: Surface, P, N, V, acc: ptr<function, LightAccum> )   point / spot lights
//       fn hookEnvSpecular( R: vec3f, roughness: f32 ) -> vec3f  IBL radiance (Environment probe)
//       fn hookEnvDiffuse( N: vec3f ) -> vec3f                   IBL irradiance / PI
//   Hooks see the material's defines (UNDERWATER_LIGHTING = 0 none / 1 lite / 2 full, HILL_SHADOW_SELF,
//   LOCAL_LIGHTS_CHEAP, IS_WATER, ...) through the preprocessor.

export const SceneLighting = {

	// name -> ShaderModule defining the hook function
	hooks: {},
	version: 0,

	set( name, module ) {

		this.hooks[ name ] = module;
		this.version ++;

	},

	modules() {

		return Object.values( this.hooks ).filter( Boolean );

	},

};

const HOOK_DEFAULTS = {
	directModulation: 'fn hookDirectModulation( P: vec3f, N: vec3f ) -> vec3f { return vec3f( 1.0 ); }',
	ambientModulation: 'fn hookAmbientModulation( P: vec3f, N: vec3f ) -> vec3f { return vec3f( 1.0 ); }',
	contactShadow: 'fn hookContactShadow( P: vec3f, N: vec3f ) -> f32 { return 1.0; }',
	bounce: 'fn hookBounce( P: vec3f, N: vec3f ) -> vec3f { return vec3f( 0.0 ); }',
	localLights: 'fn hookLocalLights( s: Surface, P: vec3f, N: vec3f, V: vec3f, acc: ptr<function, LightAccum> ) {}',
	envSpecular: 'fn hookEnvSpecular( R: vec3f, roughness: f32 ) -> vec3f { let t = sat( R.y * 0.5 + 0.5 ); return mix( frame.horizonColor * 0.6, frame.skyIrradiance * PI, t ) * frame.envIntensity; }',
	envDiffuse: 'fn hookEnvDiffuse( N: vec3f ) -> vec3f { return mix( frame.horizonColor * 0.25, frame.skyIrradiance, N.y * 0.5 + 0.5 ) * frame.envIntensity; }',
};

// stub modules for hooks nobody installed
export function hookModules() {

	const out = [];
	for ( const k in HOOK_DEFAULTS ) {

		const m = SceneLighting.hooks[ k ];
		if ( m ) out.push( m );
		else out.push( stub( k ) );

	}

	return out;

}

const _stubs = {};
function stub( k ) {

	return _stubs[ k ] || ( _stubs[ k ] = new ShaderModule( { name: 'hook-' + k + '-default', deps: [ surfaceModule ], code: HOOK_DEFAULTS[ k ] } ) );

}

// ---------------------------------------------------------------------------------- shadows

export const MAX_CASCADES = 4;

export const ShadowUniforms = new UniformBlock( 'SunShadow', {
	matrices: [ 'mat4x4f[4]', [ new Matrix4(), new Matrix4(), new Matrix4(), new Matrix4() ] ],
	// per cascade: x = far split (view distance), y = texel size (world m), z = normal bias (m), w = depth range (m)
	cascades: [ 'vec4f[4]', [ new Vector4(), new Vector4(), new Vector4(), new Vector4() ] ],
	count: [ 'u32', 0 ],
	mapSize: [ 'f32', 2048 ],
	bias: [ 'f32', 0.00002 ],
	fade: [ 'f32', 1 ],
	// cascades below this index use the contact-hardening (PCSS) filter
	pcssCascades: [ 'u32', 1 ],
	sunAngularDiameter: [ 'f32', 0.0093 ],
	enabled: [ 'f32', 0 ],
	pad: [ 'f32', 0 ],
} );

let _shadowMap = null;
// the renderer's cascade texture array (depth32float), set by Shadows.js
export function setShadowMap( tex ) {

	_shadowMap = tex;

}

export const shadowModule = new ShaderModule( {
	name: 'sunShadow',
	deps: [ commonModule ],
	uniforms: ShadowUniforms,
	uniformName: 'shadowParams',
	bindings: {
		sunShadowMap: { texture: () => _shadowMap, viewDimension: '2d-array' },
	},
	code: /* wgsl */`
fn shadowCascadeOf( viewDist: f32 ) -> i32 {
	for ( var i = 0; i < i32( shadowParams.count ); i++ ) { if ( viewDist < shadowParams.cascades[ i ].x ) { return i; } }
	return -1;
}

fn _shadowTap( uv: vec2f, layer: i32, z: f32 ) -> f32 {
	return textureSampleCompareLevel( sunShadowMap, smpShadow, uv, layer, z );
}

// shadow map depth (0 near .. 1 far, standard Z) at uv
fn _shadowDepth( uv: vec2f, layer: i32 ) -> f32 {
	let dim = vec2f( textureDimensions( sunShadowMap ) );
	let px = vec2i( clamp( uv * dim, vec2f( 0.0 ), dim - 1.0 ) );
	return textureLoad( sunShadowMap, px, layer, 0 );
}

fn sunShadowCascade( P: vec3f, N: vec3f, c: i32, noise: f32 ) -> f32 {
	let info = shadowParams.cascades[ c ];
	let Pb = P + N * info.z;
	let sc = shadowParams.matrices[ c ] * vec4f( Pb, 1.0 );
	let uvz = vec3f( sc.x * 0.5 + 0.5, 0.5 - sc.y * 0.5, sc.z );
	if ( any( uvz.xy < vec2f( 0.0 ) ) || any( uvz.xy > vec2f( 1.0 ) ) || uvz.z > 1.0 ) { return 1.0; }
	let z = uvz.z - shadowParams.bias;
	let texel = 1.0 / shadowParams.mapSize;
	let phi = noise * TWO_PI;
	var radius = texel * 1.5;
	if ( u32( c ) < shadowParams.pcssCascades ) {
		// blocker search -> penumbra = occluder distance * sun diameter (in this cascade's texels)
		let searchUV = min( 30.0 * shadowParams.sunAngularDiameter / info.y, 24.0 ) * texel;
		var blockers = 0.0; var sumZ = 0.0;
		for ( var i = 0; i < 12; i++ ) {
			let o = vogelDiskSample( i, 12, phi ) * searchUV;
			let d = _shadowDepth( uvz.xy + o, c );
			if ( d < z ) { blockers += 1.0; sumZ += d; }
		}
		if ( blockers < 0.5 ) { return 1.0; }
		let dz = ( z - sumZ / blockers ) * info.w; // metres between occluder and receiver
		radius = clamp( dz * shadowParams.sunAngularDiameter / info.y * texel, texel * 1.2, texel * 32.0 );
	}
	var sum = 0.0;
	for ( var i = 0; i < 16; i++ ) {
		sum += _shadowTap( uvz.xy + vogelDiskSample( i, 16, phi + 1.7 ) * radius, c, z );
	}
	return sum / 16.0;
}

// visibility of the sun at P (1 = lit); pixel = fragment coordinate for the dither
fn sunShadow( P: vec3f, N: vec3f, pixel: vec2f ) -> f32 {
	if ( shadowParams.enabled < 0.5 ) { return 1.0; }
	let dist = dot( P - frame.cameraPos, - vec3f( frame.view[ 0 ][ 2 ], frame.view[ 1 ][ 2 ], frame.view[ 2 ][ 2 ] ) );
	let c = shadowCascadeOf( dist );
	if ( c < 0 ) { return 1.0; }
	let noise = interleavedGradientNoise( pixel + f32( frame.frameIndex % 64u ) * 5.588238 );
	var s = sunShadowCascade( P, N, c, noise );
	// blend into the next cascade (or out to unshadowed) over the last 10% of this one
	let far = shadowParams.cascades[ c ].x;
	let near = select( 0.0, shadowParams.cascades[ max( c - 1, 0 ) ].x, c > 0 );
	let t = sat( ( dist - mix( near, far, 0.9 ) ) / ( far - mix( near, far, 0.9 ) ) );
	if ( t > 0.0 && shadowParams.fade > 0.5 ) {
		let next = select( 1.0, sunShadowCascade( P, N, c + 1, noise ), c + 1 < i32( shadowParams.count ) );
		s = mix( s, next, t );
	}
	return s;
}
`,
} );

// ---------------------------------------------------------------------------------- surface + BRDF

export const surfaceModule = new ShaderModule( {
	name: 'surface',
	deps: [ commonModule ],
	code: /* wgsl */`
struct Surface {
	albedo: vec3f,
	alpha: f32,
	normal: vec3f,      // world space, shading normal
	roughness: f32,
	emissive: vec3f,
	metalness: f32,
	translucency: vec3f, // fraction of the direct light transmitted through thin foliage (x lightColor)
	ao: f32,
	sheenColor: vec3f,
	specularIntensity: f32,
	clearcoat: f32,
	clearcoatRoughness: f32,
	sheenRoughness: f32,
	ior: f32,
	clearcoatNormal: vec3f,
	envIntensity: f32,
};

fn defaultSurface( N: vec3f ) -> Surface {
	var s: Surface;
	s.albedo = vec3f( 1.0 ); s.alpha = 1.0; s.normal = N; s.roughness = 1.0; s.metalness = 0.0;
	s.emissive = vec3f( 0.0 ); s.translucency = vec3f( 0.0 ); s.ao = 1.0; s.sheenColor = vec3f( 0.0 );
	s.specularIntensity = 1.0; s.clearcoat = 0.0; s.clearcoatRoughness = 0.0; s.sheenRoughness = 1.0;
	s.ior = 1.5; s.clearcoatNormal = N; s.envIntensity = 1.0;
	return s;
}

struct LightAccum {
	directDiffuse: vec3f,
	directSpecular: vec3f,
	indirectDiffuse: vec3f,
	indirectSpecular: vec3f,
};

fn F_Schlick( f0: vec3f, f90: f32, dotVH: f32 ) -> vec3f {
	let fresnel = exp2( ( -5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + f90 * fresnel;
}
fn V_GGX_SmithCorrelated( alpha: f32, dotNL: f32, dotNV: f32 ) -> f32 {
	let a2 = alpha * alpha;
	let gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * dotNV * dotNV );
	let gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * dotNL * dotNL );
	return 0.5 / max( gv + gl, EPS );
}
fn D_GGX( alpha: f32, dotNH: f32 ) -> f32 {
	let a2 = alpha * alpha;
	let d = dotNH * dotNH * ( a2 - 1.0 ) + 1.0;
	return INV_PI * a2 / ( d * d );
}
fn BRDF_GGX( L: vec3f, V: vec3f, N: vec3f, f0: vec3f, f90: f32, roughness: f32 ) -> vec3f {
	let alpha = roughness * roughness;
	let H = normalize( L + V );
	let dotNL = sat( dot( N, L ) ); let dotNV = sat( dot( N, V ) );
	let dotNH = sat( dot( N, H ) ); let dotVH = sat( dot( V, H ) );
	return F_Schlick( f0, f90, dotVH ) * V_GGX_SmithCorrelated( alpha, dotNL, dotNV ) * D_GGX( alpha, dotNH );
}
// Charlie sheen (Estevez & Kulla)
fn D_Charlie( roughness: f32, dotNH: f32 ) -> f32 {
	let a = roughness * roughness;
	let invA = 1.0 / a;
	let cos2h = dotNH * dotNH;
	let sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invA ) * pow( sin2h, invA * 0.5 ) / ( 2.0 * PI );
}
fn V_Neubelt( dotNV: f32, dotNL: f32 ) -> f32 { return sat( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) ); }
fn BRDF_Sheen( L: vec3f, V: vec3f, N: vec3f, color: vec3f, roughness: f32 ) -> vec3f {
	let H = normalize( L + V );
	return color * D_Charlie( roughness, sat( dot( N, H ) ) ) * V_Neubelt( sat( dot( N, V ) ), sat( dot( N, L ) ) );
}
// analytical approximation of the split-sum DFG term (Karis)
fn DFGApprox( dotNV: f32, roughness: f32 ) -> vec2f {
	let c0 = vec4f( -1.0, -0.0275, -0.572, 0.022 );
	let c1 = vec4f( 1.0, 0.0425, 1.04, -0.04 );
	let r = roughness * c0 + c1;
	let a004 = min( r.x * r.x, exp2( -9.28 * dotNV ) ) * r.x + r.y;
	return vec2f( -1.04, 1.04 ) * a004 + r.zw;
}
// multi-scattering specular energy compensation (Fdez-Aguera), as three's computeMultiscattering
fn multiscatter( N: vec3f, V: vec3f, specColor: vec3f, specF90: f32, roughness: f32, single: ptr<function, vec3f>, multi: ptr<function, vec3f> ) {
	let fab = DFGApprox( sat( dot( N, V ) ), roughness );
	let Fr = specColor;
	let FssEss = Fr * fab.x + specF90 * fab.y;
	let Ess = fab.x + fab.y;
	let Ems = 1.0 - Ess;
	let Favg = Fr + ( 1.0 - Fr ) * 0.047619;
	let Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	*single += FssEss;
	*multi += Fms * Ems;
}
`,
} );

// Full lighting of a surface: returns outgoing radiance (before fog / post).
export const lightingModule = new ShaderModule( {
	name: 'lighting',
	deps: [ commonModule, surfaceModule, shadowModule ],
	code: /* wgsl */`
fn shadeSurface( s: Surface, P: vec3f, V: vec3f, pixel: vec2f ) -> vec3f {
	let N = s.normal;
	let rough = clamp( s.roughness, 0.03, 1.0 );
	let diffuseColor = s.albedo * ( 1.0 - s.metalness );
	let specF0 = mix( vec3f( 0.04 ) * s.specularIntensity, s.albedo, s.metalness );
	let specF90 = mix( s.specularIntensity, 1.0, s.metalness );
	var acc: LightAccum;
	acc.directDiffuse = vec3f( 0.0 ); acc.directSpecular = vec3f( 0.0 );
	acc.indirectDiffuse = vec3f( 0.0 ); acc.indirectSpecular = vec3f( 0.0 );

	// ---- sun / moon
	let L = frame.sunDir;
	let dotNL = sat( dot( N, L ) );
	var lightColor = frame.sunColor * hookDirectModulation( P, N );
	let geomN = N;
	let shadow = sunShadow( P, geomN, pixel ) * hookContactShadow( P, N );
	lightColor *= shadow;
	let irradiance = dotNL * lightColor;
	acc.directDiffuse += irradiance * diffuseColor * INV_PI;
	acc.directSpecular += irradiance * BRDF_GGX( L, V, N, specF0, specF90, rough );
#if SHEEN
	acc.directSpecular += irradiance * BRDF_Sheen( L, V, N, s.sheenColor, max( s.sheenRoughness, 0.07 ) );
#endif
	// thin-surface transmission (foliage): lit from behind as well
	acc.directDiffuse += s.translucency * lightColor;
#if CLEARCOAT
	let ccN = s.clearcoatNormal;
	let ccNL = sat( dot( ccN, L ) );
	let ccSpec = ccNL * lightColor * BRDF_GGX( L, V, ccN, vec3f( 0.04 ), 1.0, clamp( s.clearcoatRoughness, 0.03, 1.0 ) );
#endif

	// ---- local lights (lanterns, windows, boat lights, flashlight)
	hookLocalLights( s, P, N, V, &acc );

	// ---- indirect: environment + ground bounce
	// (three's PhysicalLightingModel: env irradiance goes through the multiscatter-compensated
	// diffuse; the ground bounce is plain Lambert)
	let envIrr = hookEnvDiffuse( N ) * PI * s.envIntensity;
	let R = reflect( -V, N );
	let Rr = normalize( mix( R, N, rough * rough ) );
	let radiance = hookEnvSpecular( Rr, rough ) * s.envIntensity;
	var single = vec3f( 0.0 ); var multi = vec3f( 0.0 );
	multiscatter( N, V, specF0, specF90, rough, &single, &multi );
	let totalScatter = single + multi;
	let diffuseMS = diffuseColor * ( 1.0 - max( max( totalScatter.r, totalScatter.g ), totalScatter.b ) );
	acc.indirectSpecular += radiance * single + multi * envIrr * INV_PI;
	acc.indirectDiffuse += diffuseMS * envIrr * INV_PI + hookBounce( P, N ) * diffuseColor;

	// ambient occlusion (specular occlusion after Lagarde)
	let dotNV = sat( dot( N, V ) );
	let specAO = sat( pow( dotNV + s.ao, exp2( -16.0 * rough - 1.0 ) ) - 1.0 + s.ao );
	acc.indirectDiffuse *= s.ao;
	acc.indirectSpecular *= specAO;
	let amb = hookAmbientModulation( P, N );
	acc.indirectDiffuse *= amb;
	acc.indirectSpecular *= amb;

	var color = acc.directDiffuse + acc.directSpecular + acc.indirectDiffuse + acc.indirectSpecular;
#if SHEEN
	color += s.sheenColor * envIrr * INV_PI * 0.5 * s.ao * amb;
#endif
#if CLEARCOAT
	let ccNV = sat( dot( ccN, V ) );
	let Fcc = F_Schlick( vec3f( 0.04 ), 1.0, ccNV ) * s.clearcoat;
	let ccRad = hookEnvSpecular( reflect( -V, ccN ), clamp( s.clearcoatRoughness, 0.03, 1.0 ) ) * amb * specAO;
	color = color * ( 1.0 - Fcc ) + ( ccSpec * s.clearcoat + ccRad * Fcc );
#endif
	return color + s.emissive;
}
`,
} );
