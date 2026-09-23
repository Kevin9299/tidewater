import { PhysicalLightingModel } from 'three/webgpu';
import { diffuseContribution } from 'three/tsl';

// Hooks installed by the water system at startup:
//   directModulation(builder)  -> vec3 node multiplying direct (sun) light: caustics, water column, cloud shadow
//   ambientModulation(builder) -> vec3 node multiplying indirect light: underwater tint/attenuation
//   contactShadow(builder, lightColor) -> float visibility of the key light (ContactShadows.js)
//   bounce(builder)            -> vec3 irradiance of sunlight bounced off the ground (GroundBounce.js)
export const SceneLighting = {
	directModulation: null,
	ambientModulation: null,
	contactShadow: null,
	bounce: null,
	localLights: null, // ( model, builder ): lanterns, windows, boat lights, flashlight (LocalLights.js)
};

export class SceneLightingModel extends PhysicalLightingModel {

	constructor( material, f = {} ) {

		super( !! f.clearcoat, !! f.sheen, !! f.iridescence, !! f.anisotropy, !! f.transmission, !! f.dispersion );
		this.material = material;

	}

	direct( data, builder ) {

		const mod = SceneLighting.directModulation ? SceneLighting.directModulation( builder ) : null;
		if ( mod ) data = { ...data, lightColor: data.lightColor.mul( mod ) };
		const light = data.lightNode && data.lightNode.light;
		if ( SceneLighting.contactShadow && light && light.isDirectionalLight ) {

			const cs = SceneLighting.contactShadow( builder, data.lightColor );
			if ( cs ) data = { ...data, lightColor: data.lightColor.mul( cs ) };

		}

		super.direct( data, builder );
		// light transmitted through thin foliage; lightColor already carries the shadow term
		const tr = this.material && this.material.translucencyNode;
		if ( tr ) builder.context.reflectedLight.directDiffuse.addAssign( tr( data.lightColor ) );

	}

	indirectDiffuse( builder ) {

		super.indirectDiffuse( builder );
		// sunlight bounced off the ground: before the material's AO and the underwater modulation
		const E = SceneLighting.bounce ? SceneLighting.bounce( builder ) : null;
		if ( E ) builder.context.reflectedLight.indirectDiffuse.addAssign( E.mul( diffuseContribution ).mul( 1 / Math.PI ) );

	}

	indirect( builder ) {

		// local lights: one loop over a small uniform array, without the sun-only terms of direct()
		if ( SceneLighting.localLights ) SceneLighting.localLights( this, builder );
		super.indirect( builder );
		const amb = SceneLighting.ambientModulation ? SceneLighting.ambientModulation( builder ) : null;
		if ( amb ) {

			const { reflectedLight } = builder.context;
			reflectedLight.indirectDiffuse.mulAssign( amb );
			reflectedLight.indirectSpecular.mulAssign( amb );

		}

	}

}
