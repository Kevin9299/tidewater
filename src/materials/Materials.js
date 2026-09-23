import * as THREE from 'three/webgpu';
import { SceneLightingModel } from './SceneLighting.js';

// All world objects use these material classes so they share underwater lighting
// (caustics, water-column attenuation, tinted ambient) and wetness handling.
// They accept exactly the same parameters as MeshStandardNodeMaterial /
// MeshPhysicalNodeMaterial (colorNode, roughnessNode, normalNode, positionNode, ...).

export class SceneMaterial extends THREE.MeshStandardNodeMaterial {

	constructor( params ) {

		super( params );
		this.isSceneMaterial = true;

	}

	setupLightingModel() {

		return new SceneLightingModel( this );

	}

}

export class ScenePhysicalMaterial extends THREE.MeshPhysicalNodeMaterial {

	constructor( params ) {

		super( params );
		this.isSceneMaterial = true;

	}

	setupLightingModel() {

		return new SceneLightingModel( this, {
			clearcoat: this.useClearcoat,
			sheen: this.useSheen,
			iridescence: this.useIridescence,
			anisotropy: this.useAnisotropy,
			transmission: this.useTransmission,
			dispersion: this.useDispersion,
		} );

	}

}

export const standard = ( params = {} ) => new SceneMaterial( params );
export const physical = ( params = {} ) => new ScenePhysicalMaterial( params );
