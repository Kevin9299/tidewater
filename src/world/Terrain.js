import * as THREE from 'three/webgpu';
import {
	Fn, If, float, int, vec2, vec3, vec4, normalize, mix, smoothstep, saturate, clamp, max, min, sin, dot, sqrt, pow, fract,
	positionWorld, positionGeometry, attribute, transformNormalToView, uniform, length, texture, fwidth, dFdx,
	dFdy, cameraPosition,
} from 'three/tsl';
import { CDLOD } from '../core/CDLOD.js';
import { standard } from '../materials/Materials.js';
import { SceneLightingModel } from '../materials/SceneLighting.js';
import { G } from '../core/Globals.js';
import { WORLD } from './WorldLayout.js';
import { srgb, rot2, perturbNormal, rockSurface, saturation, meadowTone, MEADOW } from './terrain/TerrainShading.js';
import { gustAt, windStrength } from './vegetation/VegNodes.js';
import { FADE_T0 as GRASS_FADE } from './vegetation/GrassField.js';

export { perturbNormal };

// Multiplies the key directional light by the soft heightfield sun shadow (long hill shadows
// beyond the shadow map range and from terrain outside the view).
export class TerrainLightingModel extends SceneLightingModel {

	constructor( material, sunShadow, grass = null ) {

		super( material );
		this.sunShadow = grass ? sunShadow.mul( grass ) : sunShadow;

	}

	direct( data, builder ) {

		const light = data.lightNode && data.lightNode.light;
		if ( light && light.isDirectionalLight ) data = { ...data, lightColor: data.lightColor.mul( this.sunShadow ) };
		super.direct( data, builder );

	}

}

// Island terrain: CDLOD mesh displaced by the heightmap, with a procedural material that blends
// coral sand (dry / damp / wet, wind ripples, shell grit, wrack line, swash marks), the seabed
// (sand with ripple fields, seagrass meadows, rubble heads), tropical lawn, jungle floor and canopy,
// landslide scars, worn dirt paths and weathered volcanic rock (triplanar, faceted blocks, bedding,
// lichen, moss, splash-zone zonation, rain streaks). Detail comes from one small tileable height
// texture sampled at several scales / rotations; normals use surface-gradient bump mapping; the
// baked horizon AO feeds material.aoNode. Rock is only evaluated where it can appear. The key light
// is multiplied by the baked heightfield sun shadow (long, soft hill shadows at low sun).
export class Terrain {

	// gridSize 40 / rangeFactor 2.0: 0.2 m vertices at the camera, ~80 quads per LOD range
	// (~9-25 % finer than 32 / 2.3 at every distance) for 0.1-0.2 M triangles.
	// sunShadow: apply the heightfield sun shadow here (pass false if it is applied to every scene
	// material through SceneLighting.directModulation). renderer: optional, otherwise it is picked
	// up on the first render (the sun shadow map is baked from update()).
	constructor( { scene, terrainData, terrainGPU, gridSize = 40, rangeFactor = 2.0, sunShadow = true, renderer = null } ) {

		this.data = terrainData;
		this.gpu = terrainGPU;
		const half = terrainData.size / 2;

		this.lod = new CDLOD( {
			gridSize, leafSize: 8, levels: 9,
			heightBounds: ( x0, z0, x1, z1 ) => terrainData.boundsFor( x0, z0, x1, z1 ),
			center: { x: - half, z: - half, size: terrainData.size },
			rangeFactor,
		} );

		this.wetness = null; // optional TSL fn (xz, h) -> vec2(wetness, foamResidue), set by the shore system

		const gpu = terrainGPU;
		const mat = this.material = standard( { roughness: 0.9, metalness: 0 } );
		mat.name = 'Terrain';

		// Morph toward the view camera in every pass: the shadow passes render the same surface
		// (with `cameraPosition` they would morph toward the light camera instead).
		this.uViewPos = uniform( new THREE.Vector3() ).setName( 'terViewPos' );
		mat.positionNode = Fn( () => this._vertexPosition() )();
		mat.castShadowPositionNode = mat.positionNode;

		this.renderer = renderer;
		// tall grass shades itself when the sun is low (the meadow darkens and gains contrast)
		const grassShadow = () => this.meadowNode ? mix( float( 1 ), smoothstep( - 0.05, 0.45, G.sunDir.y ).mul( 0.35 ).add( 0.62 ), this.meadowNode ) : null;
		if ( sunShadow ) mat.setupLightingModel = () => new TerrainLightingModel( mat, gpu.sunShadowAt( positionWorld ), grassShadow() );

		this.params = {
			wetDarken: uniform( 0.58 ).setName( 'terWetDark' ),
		};

		this._surface = () => this._buildSurface();
		this.finalizeMaterial();

		this.mesh = new THREE.Mesh( this.lod.geometry, mat );
		this.mesh.frustumCulled = false;
		this.mesh.receiveShadow = true;
		this.mesh.castShadow = true;
		this.mesh.name = 'Terrain';
		this.mesh.onBeforeRender = ( r ) => {

			if ( ! this.renderer ) this.renderer = r;

		};
		scene.add( this.mesh );

	}

	// CDLOD vertex (same lattice snapping and geomorph as CDLOD.vertexNodes) using uViewPos
	_vertexPosition() {

		const lod = this.lod, gpu = this.gpu;
		const node = attribute( 'nodeData', 'vec4' );
		const level = int( node.w );
		const h = lod.uSpacing.element( level );
		const p = node.xy.add( positionGeometry.xz.mul( node.z ) );
		const idx = p.div( h ).add( 1e-3 ).floor();
		const snapped = idx.mul( h );
		const y0 = gpu.heightAt( snapped );
		const dist = this.uViewPos.sub( vec3( snapped.x, y0, snapped.y ) ).length();
		const m = lod.uMorph.element( level );
		const morphK = clamp( dist.sub( m.x ).mul( m.y ), 0, 1 );
		const odd = fract( idx.mul( 0.5 ) ).mul( 2 );
		const xz = snapped.sub( odd.mul( h ).mul( morphK ) );
		return vec3( xz.x, gpu.heightAt( xz ), xz.y );

	}

	_buildSurface() {

		const gpu = this.gpu;
		const det = gpu.detailTexture;
		const outRough = float( 0.9 ).toVar( 'terRough' );
		const outN = vec3( 0, 1, 0 ).toVar( 'terNormal' );
		const outAO = float( 1 ).toVar( 'terAO' );
		const meadowW = float( 0 ).toVar( 'terMeadow' );

		const albedoNode = Fn( () => {

			const p = positionWorld;
			const xz = p.xz;
			const h = p.y;

			// ---- data maps
			const nr = gpu.normalRock( xz ).toVar();
			const N0 = normalize( vec3( nr.x, sqrt( max( float( 1 ).sub( nr.x.mul( nr.x ) ).sub( nr.y.mul( nr.y ) ), 0.0025 ) ), nr.y ) ).toVar();
			const sp = gpu.splat( xz ).toVar();
			const slope = float( 1 ).sub( N0.y ).toVar();
			const camDist = length( cameraPosition.sub( p ) ).toVar();
			const dpdx = dFdx( p ).toVar(), dpdy = dFdy( p ).toVar();
			const fwY = fwidth( h ).toVar();

			// ---- deep seabed (seen through the water: no rock, land cover, swash or wind ripples),
			// the same colour / relief / AO as the full path below, from 6 detail samples instead of ~17
			// detail samples shared by both paths (sampled in uniform control flow: no derivative seams
			// where the paths meet)
			const macroA = texture( det, rot2( xz, 0.7 ).div( 173 ) ).w.toVar();
			const macroB = texture( det, rot2( xz, 2.1 ).div( 47 ) ).w.toVar();
			const macro = macroA.mul( 0.6 ).add( macroB.mul( 0.4 ) ).toVar();
			const dN = texture( det, xz.div( 1.9 ) ).toVar();
			const dM = texture( det, rot2( xz, 1.3 ).div( 6.7 ).add( 0.21 ) ).toVar();
			const dF = texture( det, rot2( xz, 2.4 ).div( 0.63 ).add( 0.53 ) ).toVar();
			const grain = min( dN.z, 0.62 ).toVar();
			const grainF = min( dF.z, 0.62 );
			const seabedPath = h.lessThan( - 0.8 ).and( nr.z.lessThan( 0.05 ) ).and( slope.lessThan( 0.18 ) );
			const albedoOut = vec3( 0 ).toVar( 'terAlbedo' );
			If( seabedPath, () => {

				const underW = float( 1 );
				const sandW = smoothstep( 0.3, 0.72, sp.x.add( dM.z.sub( 0.5 ).mul( 0.5 ) ).add( macro.sub( 0.5 ).mul( 0.35 ) ) );
				// ---- seabed: sand with ripple fields, seagrass meadows, rubble heads
				const depth = h.negate();
				const rc = WORLD.reef.center;
				const reefD = length( xz.sub( vec2( rc.x, rc.z ) ) );
				const reefW = float( 1 ).sub( smoothstep( WORLD.reef.radius * 0.5, WORLD.reef.radius * 1.15, reefD.add( macro.sub( 0.5 ).mul( 30 ) ) ) );
				let under = mix( srgb( 0.84, 0.78, 0.64 ), srgb( 0.72, 0.7, 0.58 ), smoothstep( 1.0, 9.0, depth ) );
				under = under.mul( dM.w.sub( 0.5 ).mul( 0.14 ).add( 1 ) ).mul( grain.sub( 0.45 ).mul( 0.25 ).add( 1 ) );
				// megaripple fields (~0.75 m) across the swell, troughs collect darker shell hash; not in
				// the swash zone or the first metre of depth
				const sw = WORLD.swellDir;
				const swDir = vec2( sw.x, sw.y );
				const ph2 = dot( xz, swDir ).mul( 6.2832 / 0.75 ).add( dM.w.mul( 16 ) ).add( macroB.mul( 24 ) );
				const fade2 = float( 1 ).sub( smoothstep( 0.5, 2.0, fwidth( ph2 ) ) );
				const rip2 = pow( sin( ph2 ).mul( 0.5 ).add( 0.5 ), 1.4 );
				const fieldW = smoothstep( 0.42, 0.62, macroB.add( dM.w.sub( 0.5 ).mul( 0.35 ) ) ).mul( smoothstep( 0.9, 2.0, depth ) ).mul( float( 1 ).sub( reefW ) ).toVar();
				under = under.mul( rip2.sub( 0.55 ).mul( 0.22 ).mul( fieldW ).mul( fade2 ).add( 1 ) );
				// small wave ripples (~0.16 m) everywhere below the swash
				const ph3 = dot( xz, swDir ).mul( 6.2832 / 0.16 ).add( dM.w.mul( 26 ) ).add( dN.w.mul( 5 ) );
				const fade3 = float( 1 ).sub( smoothstep( 0.6, 2.2, fwidth( ph3 ) ) );
				const rip3 = pow( sin( ph3 ).mul( 0.5 ).add( 0.5 ), 1.5 );
				// seagrass meadows: ragged edges, blade streaks leaning with the wave surge, epiphyte tips
				// (the fringe breaks up into clumps: noise at three scales thresholds the soft splat edge)
				const clumps = dM.w.sub( 0.5 ).mul( 0.5 ).add( dN.y.sub( 0.45 ).mul( 0.4 ) ).add( macroB.sub( 0.5 ).mul( 0.3 ) );
				const seagrassW = smoothstep( 0.3, 0.55, sp.z.add( clumps ) ).mul( underW ).mul( smoothstep( 0.3, 0.9, depth ) ).toVar();
				const swPerp = vec2( swDir.y.negate(), swDir.x );
				const blades = texture( det, vec2( dot( xz, swDir ).div( 2.6 ), dot( xz, swPerp ).div( 0.35 ) ) ).y.toVar();
				let meadow = mix( srgb( 0.12, 0.16, 0.07 ), srgb( 0.27, 0.29, 0.15 ), smoothstep( 0.35, 0.75, blades ) );
				meadow = mix( meadow, srgb( 0.24, 0.2, 0.11 ), smoothstep( 0.55, 0.8, dM.y.add( macroB.sub( 0.5 ).mul( 0.4 ) ) ).mul( 0.5 ) );
				// sparse at the fringe: sand shows between the blades; thinner, paler patches inside
				meadow = mix( under, meadow, smoothstep( 0.3, 0.85, sp.z.add( clumps.mul( 0.5 ) ) ).mul( 0.35 ).add( 0.65 ) );
				meadow = mix( meadow, mix( meadow, under, 0.45 ), smoothstep( 0.58, 0.8, macroB.add( dM.w.sub( 0.5 ).mul( 0.4 ) ) ) );
				under = mix( under, meadow, seagrassW );
				// rubble heads: coral rubble and rock turfed with algae, pink coralline crusts
				const rubbleW = smoothstep( 0.3, 0.6, sp.w.add( dN.x.sub( 0.5 ).mul( 0.4 ) ).add( dM.w.sub( 0.5 ).mul( 0.3 ) ) ).mul( underW ).toVar();
				let rubble = mix( srgb( 0.2, 0.19, 0.15 ), srgb( 0.36, 0.33, 0.26 ), smoothstep( 0.3, 0.7, dN.x ) );
				rubble = mix( rubble, srgb( 0.2, 0.24, 0.1 ), smoothstep( 0.5, 0.7, dF.y ).mul( 0.6 ) );
				rubble = mix( rubble, srgb( 0.58, 0.38, 0.44 ), smoothstep( 0.62, 0.74, dM.x ).mul( 0.6 ) );
				under = mix( under, rubble, rubbleW );
				// reef flat: coral rubble and pink crusts toward the reef
				under = mix( under, mix( srgb( 0.56, 0.50, 0.44 ), srgb( 0.60, 0.43, 0.46 ), smoothstep( 0.45, 0.7, dM.x ) ), reefW.mul( 0.7 ).mul( smoothstep( 0.4, 0.6, dN.x ) ) );

				// damp sand below the berm (wet = 0 under water)
				const dampMottle = smoothstep( 0.3, 0.7, dM.w.add( dN.y.sub( 0.45 ).mul( 0.6 ) ).add( macroB.sub( 0.5 ).mul( 0.4 ) ) );
				const wetK = smoothstep( 1.7, 0.5, h.add( dM.w.mul( 0.3 ) ) ).mul( sandW ).mul( mix( float( 0.22 ), float( 0.5 ), dampMottle ) );
				const wetAlbedo = saturation( under.mul( this.params.wetDarken ), 1.15 ).mul( vec3( 0.97, 0.98, 1.0 ) );
				albedoOut.assign( mix( under, wetAlbedo, wetK ) );
				outRough.assign( 0.75 );
				const waveR = rip3.mul( 0.012 ).mul( fade3 ).mul( smoothstep( - 0.3, - 0.9, h ) ).mul( float( 1 ).sub( seagrassW ) );
				const megaR = rip2.mul( 0.035 ).mul( fade2 ).mul( fieldW ).mul( float( 1 ).sub( seagrassW ) );
				const sandH = grain.mul( 0.004 ).add( grainF.mul( 0.003 ) ).add( waveR ).add( megaR );
				const seabedH = seagrassW.mul( blades.mul( 0.05 ).add( 0.08 ) ).add( rubbleW.mul( dN.x.mul( 0.07 ).add( dF.x.mul( 0.015 ) ) ) );
				outN.assign( perturbNormal( N0, sandH.add( seabedH ), 1.0 ) );
				outAO.assign( nr.w.mul( float( 1 ).sub( seagrassW.mul( 0.3 ) ) ).mul( float( 1 ).sub( rubbleW.mul( smoothstep( 0.55, 0.2, dN.x ) ).mul( 0.35 ) ) ) );
				meadowW.assign( 0 );

			} ).Else( () => {

				// ---- detail samples (tileable heights: x rock, y soil, z sand, w fbm)
				// sand grain without the pebble peaks (pebbles are drawn separately where they belong)

				// ---- downslope streaks (rock flutes, hanging vegetation, landslide scars): the detail
				// texture stretched vertically on the two vertical projection planes
				const sw4 = N0.xz.abs().pow( vec2( 4 ) );
				const swN = sw4.div( sw4.x.add( sw4.y ).add( 1e-5 ) );
				const stA = texture( det, vec2( p.z.div( 9.3 ), h.div( 37 ) ) ), stB = texture( det, vec2( p.x.div( 9.3 ).add( 0.5 ), h.div( 37 ).add( 0.3 ) ) );
				const streak = stA.w.mul( swN.x ).add( stB.w.mul( swN.y ) ).toVar();
				const scar = stA.y.mul( swN.x ).add( stB.y.mul( swN.y ) );

				// ---- rock: only evaluated where the rock mask or the slope allow it. Exposure follows the
				// form: steep faces, convex spurs and ridges (high AO) go bare, gully floors (low AO,
				// drainage lines) keep soil and plants; noise and fall-line streaks break the outline up.
				// Around the bare rock a band of scree / dark soil and moss; plants creep over it.
				const gully = sp.z.mul( smoothstep( - 0.5, 0.5, h ) ).toVar();
				const rockAlbedo = vec3( 0.2 ).toVar(), rockRough = float( 0.8 ).toVar(), rockHd = float( 0 ).toVar();
				const rockW = float( 0 ).toVar(), screeW = float( 0 ).toVar();
				If( nr.z.greaterThan( 0.06 ).or( slope.greaterThan( 0.3 ) ), () => {

					const R = rockSurface( { tex: det, p, N: N0, h, macro, grad: { dpdx, dpdy, fwY } } );
					const cliffK = smoothstep( 0.28, 0.55, slope );
					const convex = smoothstep( 0.5, 0.85, nr.w );
					const rv = nr.z.mul( 0.7 ).add( smoothstep( 0.3, 0.62, slope ).mul( 0.5 ) ).add( convex.mul( 0.14 ) ).sub( gully.mul( 0.4 ) )
						.add( R.height.sub( 0.45 ).mul( 0.35 ) ).add( dM.w.sub( 0.5 ).mul( 0.34 ) ).add( dN.w.sub( 0.5 ).mul( 0.22 ) )
						.add( streak.sub( 0.5 ).mul( 1.0 ).mul( cliffK ) ).toVar();
					// fades to 0 at the branch boundary: no step along the slope / mask iso-lines
					const branchK = max( smoothstep( 0.06, 0.18, nr.z ), smoothstep( 0.3, 0.42, slope ) );
					rockW.assign( smoothstep( 0.5, 0.68, rv ).mul( branchK ) );
					screeW.assign( smoothstep( 0.28, 0.52, rv ).mul( branchK ).mul( float( 1 ).sub( rockW ) ) );
					// weathered basalt: darker and browner than the sea-cliff palette, streaked; moss and
					// ferns on the ledges and on the less steep parts of the faces
					// dark wet stains down the fall line, paler dry ribs between them
					const stain = smoothstep( 0.5, 0.7, streak ).mul( cliffK );
					// dark, weathered basalt (the island's inland rock is darker and browner than the pale
					// sea-cliff palette), stained down the fall line
					let basalt = R.albedo.mul( vec3( 0.36, 0.34, 0.31 ) ).mul( float( 1 ).sub( stain.mul( 0.45 ) ) ).mul( smoothstep( 0.42, 0.25, streak ).mul( cliffK ).mul( 0.15 ).add( 1 ) );
					// soil and humus caught in the joints and hollows of the rock (low relief), so the face
					// reads as fractured stone instead of a smooth plate
					const joints = smoothstep( 0.42, 0.22, R.height.add( dN.w.sub( 0.5 ).mul( 0.25 ) ) );
					basalt = mix( basalt, mix( srgb( 0.13, 0.1, 0.07 ), srgb( 0.2, 0.2, 0.1 ), dF.y ), joints.mul( 0.7 ) );
					// moss / small plants on every ledge and on the less steep parts, more near the edges
					const ledgeMoss = smoothstep( 0.4, 0.75, N0.y.add( dN.y.sub( 0.45 ).mul( 0.6 ) ) ).mul( smoothstep( 0.25, 0.55, macroB.add( dM.y.mul( 0.4 ) ) ) );
					const fringe = smoothstep( 0.5, 0.58, rv ).mul( smoothstep( 0.8, 0.6, rv ) ).mul( smoothstep( 0.3, 0.6, dM.w.add( dN.y.mul( 0.4 ) ) ) );
					const mossK = saturate( max( ledgeMoss.mul( 0.75 ), fringe.mul( 0.8 ) ).add( R.moss.mul( 0.3 ) ).add( joints.mul( 0.25 ) ) );
					rockAlbedo.assign( mix( basalt, mix( srgb( 0.12, 0.17, 0.05 ), srgb( 0.22, 0.26, 0.09 ), dF.y ), mossK ) );
					rockRough.assign( R.rough );
					// craggier than the boulders: the big blocks and plates stand out from afar
					rockHd.assign( R.hd.mul( 2.2 ).add( R.height.mul( 0.6 ) ) );

				} );

				// ---- weights
				const notRock = float( 1 ).sub( rockW ).toVar();
				const underW = smoothstep( 0.12, - 0.6, h ).toVar();
				const landW = float( 1 ).sub( underW );
				const sandW = smoothstep( 0.3, 0.72, sp.x.add( dM.z.sub( 0.5 ).mul( 0.5 ) ).add( macro.sub( 0.5 ).mul( 0.35 ) ) ).mul( notRock ).toVar();
				const pathW = smoothstep( 0.28, 0.62, sp.y.add( dN.y.sub( 0.45 ).mul( 0.4 ) ).add( dM.w.sub( 0.5 ).mul( 0.25 ) ) ).mul( notRock ).mul( landW )
					.mul( float( 1 ).sub( smoothstep( 50, 220, camDist ).mul( 0.85 ) ) ).toVar();
				// forest on the higher / steeper ground and in the gullies, tall-grass meadow on the valley
				// floor and around the village (same classification as the vegetation's land cover)
				const jungleW = saturate( smoothstep( 9, 24, h.add( macro.sub( 0.5 ).mul( 18 ) ) ).add( smoothstep( 0.18, 0.36, slope ) ).add( gully.mul( 0.6 ) ) ).toVar();
				// landslide scars: raw red-brown laterite in streaks down steep slopes, rare
				const lateriteW = smoothstep( 0.62, 0.74, scar.add( macroB.sub( 0.5 ).mul( 0.3 ) ) ).mul( smoothstep( 0.3, 0.42, slope ) )
					.mul( smoothstep( 0.52, 0.66, macro ) ).mul( notRock ).mul( 0.85 );

				// ---- beach sand: pale coral sand, drifts of warmer / coarser sand, grain
				const dryK = smoothstep( 0.8, 3.0, h );
				let sand = mix( srgb( 0.83, 0.75, 0.6 ), srgb( 0.9, 0.84, 0.72 ), smoothstep( 0.3, 0.72, macro.add( dryK.mul( 0.2 ) ) ) );
				sand = mix( sand, srgb( 0.84, 0.72, 0.55 ), smoothstep( 0.55, 0.8, dM.w.add( macroB.sub( 0.5 ).mul( 0.6 ) ) ).mul( 0.45 ) );
				sand = sand.mul( dM.w.sub( 0.5 ).mul( 0.16 ).add( 1 ) ).mul( dN.w.sub( 0.5 ).mul( 0.1 ).add( 1 ) );
				sand = sand.mul( grain.sub( 0.45 ).mul( 0.3 ).add( 0.97 ) ).mul( grainF.sub( 0.45 ).mul( 0.2 ).add( 1 ) );
				// disturbed / trodden patches: slightly darker, coarser sand (footfall, crabs, wind scour)
				const trod = smoothstep( 0.52, 0.7, dN.y.add( dM.y.sub( 0.5 ).mul( 0.6 ) ) ).mul( dryK );
				sand = sand.mul( float( 1 ).sub( trod.mul( 0.07 ) ) );
				// the high-water band collects shell grit, coral bits and dried seaweed
				const hw = h.add( dM.w.sub( 0.5 ).mul( 0.5 ) );
				const wrackBand = smoothstep( 1.15, 1.4, hw ).mul( smoothstep( 2.1, 1.7, hw ) );
				const pebDensity = smoothstep( 0.62, 0.85, macroB.add( dM.w.mul( 0.3 ) ) ).mul( 0.2 ).add( wrackBand.mul( smoothstep( 0.3, 0.6, dM.y ) ) );
				const pebW = smoothstep( 0.66, 0.8, dN.z ).mul( saturate( pebDensity ) ).mul( smoothstep( 0.6, 0.9, sp.x ) ).mul( landW )
					.mul( float( 1 ).sub( smoothstep( 12, 35, camDist ) ) ).toVar();
				const pebCol = mix( mix( srgb( 0.86, 0.82, 0.74 ), srgb( 0.78, 0.64, 0.6 ), smoothstep( 0.45, 0.75, dM.x ) ), srgb( 0.36, 0.33, 0.3 ), smoothstep( 0.74, 0.82, dM.y ) );
				sand = mix( sand, pebCol, pebW.mul( 0.75 ) );
				const wrack = wrackBand.mul( smoothstep( 0.58, 0.72, dN.y ) ).mul( smoothstep( 0.4, 0.6, macroB ) );
				sand = mix( sand, srgb( 0.24, 0.18, 0.11 ), wrack.mul( 0.8 ) );
				// trampled sand along the paths
				sand = sand.mul( float( 1 ).sub( pathW.mul( 0.07 ) ) );

				// wind ripples on the dry sand: crests across the wind (bent by the fbm), wavelength
				// 8-13 cm, fading where trodden; visible in the albedo too (finer sand on the crests)
				const wd = G.windDir;
				const ph1 = dot( xz, wd ).mul( 6.2832 ).div( mix( float( 0.08 ), float( 0.13 ), macroB ) ).add( dM.w.mul( 24 ) ).add( dN.w.mul( 7 ) );
				const fade1 = float( 1 ).sub( smoothstep( 0.6, 2.2, fwidth( ph1 ) ) );
				const rip1 = pow( sin( ph1 ).mul( 0.5 ).add( 0.5 ), 1.6 );
				const windK = fade1.mul( smoothstep( 1.5, 2.2, h ) ).mul( float( 1 ).sub( pathW ) ).mul( float( 1 ).sub( trod.mul( 0.7 ) ) )
					.mul( smoothstep( 0.2, 0.5, macroB.add( dM.w.mul( 0.3 ) ) ) ).toVar();
				sand = sand.mul( rip1.sub( 0.5 ).mul( 0.12 ).mul( windK ).add( 1 ) );

				// ---- seabed: sand with ripple fields, seagrass meadows, rubble heads
				const depth = h.negate();
				const rc = WORLD.reef.center;
				const reefD = length( xz.sub( vec2( rc.x, rc.z ) ) );
				const reefW = float( 1 ).sub( smoothstep( WORLD.reef.radius * 0.5, WORLD.reef.radius * 1.15, reefD.add( macro.sub( 0.5 ).mul( 30 ) ) ) );
				let under = mix( srgb( 0.84, 0.78, 0.64 ), srgb( 0.72, 0.7, 0.58 ), smoothstep( 1.0, 9.0, depth ) );
				under = under.mul( dM.w.sub( 0.5 ).mul( 0.14 ).add( 1 ) ).mul( grain.sub( 0.45 ).mul( 0.25 ).add( 1 ) );
				// megaripple fields (~0.75 m) across the swell, troughs collect darker shell hash; not in
				// the swash zone or the first metre of depth
				const sw = WORLD.swellDir;
				const swDir = vec2( sw.x, sw.y );
				const ph2 = dot( xz, swDir ).mul( 6.2832 / 0.75 ).add( dM.w.mul( 16 ) ).add( macroB.mul( 24 ) );
				const fade2 = float( 1 ).sub( smoothstep( 0.5, 2.0, fwidth( ph2 ) ) );
				const rip2 = pow( sin( ph2 ).mul( 0.5 ).add( 0.5 ), 1.4 );
				const fieldW = smoothstep( 0.42, 0.62, macroB.add( dM.w.sub( 0.5 ).mul( 0.35 ) ) ).mul( smoothstep( 0.9, 2.0, depth ) ).mul( float( 1 ).sub( reefW ) ).toVar();
				under = under.mul( rip2.sub( 0.55 ).mul( 0.22 ).mul( fieldW ).mul( fade2 ).add( 1 ) );
				// small wave ripples (~0.16 m) everywhere below the swash
				const ph3 = dot( xz, swDir ).mul( 6.2832 / 0.16 ).add( dM.w.mul( 26 ) ).add( dN.w.mul( 5 ) );
				const fade3 = float( 1 ).sub( smoothstep( 0.6, 2.2, fwidth( ph3 ) ) );
				const rip3 = pow( sin( ph3 ).mul( 0.5 ).add( 0.5 ), 1.5 );
				// seagrass meadows: ragged edges, blade streaks leaning with the wave surge, epiphyte tips
				// (the fringe breaks up into clumps: noise at three scales thresholds the soft splat edge)
				const clumps = dM.w.sub( 0.5 ).mul( 0.5 ).add( dN.y.sub( 0.45 ).mul( 0.4 ) ).add( macroB.sub( 0.5 ).mul( 0.3 ) );
				const seagrassW = smoothstep( 0.3, 0.55, sp.z.add( clumps ) ).mul( underW ).mul( smoothstep( 0.3, 0.9, depth ) ).toVar();
				const swPerp = vec2( swDir.y.negate(), swDir.x );
				const blades = texture( det, vec2( dot( xz, swDir ).div( 2.6 ), dot( xz, swPerp ).div( 0.35 ) ) ).y.toVar();
				let meadow = mix( srgb( 0.12, 0.16, 0.07 ), srgb( 0.27, 0.29, 0.15 ), smoothstep( 0.35, 0.75, blades ) );
				meadow = mix( meadow, srgb( 0.24, 0.2, 0.11 ), smoothstep( 0.55, 0.8, dM.y.add( macroB.sub( 0.5 ).mul( 0.4 ) ) ).mul( 0.5 ) );
				// sparse at the fringe: sand shows between the blades; thinner, paler patches inside
				meadow = mix( under, meadow, smoothstep( 0.3, 0.85, sp.z.add( clumps.mul( 0.5 ) ) ).mul( 0.35 ).add( 0.65 ) );
				meadow = mix( meadow, mix( meadow, under, 0.45 ), smoothstep( 0.58, 0.8, macroB.add( dM.w.sub( 0.5 ).mul( 0.4 ) ) ) );
				under = mix( under, meadow, seagrassW );
				// rubble heads: coral rubble and rock turfed with algae, pink coralline crusts
				const rubbleW = smoothstep( 0.3, 0.6, sp.w.add( dN.x.sub( 0.5 ).mul( 0.4 ) ).add( dM.w.sub( 0.5 ).mul( 0.3 ) ) ).mul( underW ).toVar();
				let rubble = mix( srgb( 0.2, 0.19, 0.15 ), srgb( 0.36, 0.33, 0.26 ), smoothstep( 0.3, 0.7, dN.x ) );
				rubble = mix( rubble, srgb( 0.2, 0.24, 0.1 ), smoothstep( 0.5, 0.7, dF.y ).mul( 0.6 ) );
				rubble = mix( rubble, srgb( 0.58, 0.38, 0.44 ), smoothstep( 0.62, 0.74, dM.x ).mul( 0.6 ) );
				under = mix( under, rubble, rubbleW );
				// reef flat: coral rubble and pink crusts toward the reef
				under = mix( under, mix( srgb( 0.56, 0.50, 0.44 ), srgb( 0.60, 0.43, 0.46 ), smoothstep( 0.45, 0.7, dM.x ) ), reefW.mul( 0.7 ).mul( smoothstep( 0.4, 0.6, dN.x ) ) );
				sand = mix( sand, under, underW );

				// ---- ground: tall-grass meadow (tone shared with the grass field), forest floor and, from
				// afar, the forest canopy; laterite scars
				const V = normalize( cameraPosition.sub( p ) );
				const NdV = saturate( dot( N0, V ) ).toVar();
				const mt = meadowTone( macroA, macroB, slope, N0.z, dM.w.mul( 0.65 ).add( dN.w.mul( 0.35 ) ) );
				// clumps (1-3 m) and tussocks, blade-scale grain
				const clump = dM.w.mul( 0.6 ).add( dN.y.mul( 0.4 ) ).toVar();
				// seen from afar the tussocks and their shadowed gaps are what makes tall grass read as
				// grass (not lawn): the clump contrast grows with distance as the blades fade out
				const clumpK = mix( float( 0.34 ), float( 0.95 ), smoothstep( 40, 140, camDist ) );
				let lawn = mt.tone.mul( clump.sub( 0.5 ).mul( clumpK ).add( 1 ) ).mul( dF.y.sub( 0.4 ).mul( 0.22 ).add( 1 ) );
				lawn = lawn.mul( mix( float( 1 ), smoothstep( 0.25, 0.55, dN.y.mul( 0.5 ).add( dM.y.mul( 0.5 ) ) ).mul( 0.35 ).add( 0.72 ), smoothstep( 50, 160, camDist ) ) );
				// grass combed along the wind: long streaks (anisotropic sample of the fbm channel)
				const combUV = vec2( dot( xz, G.windDir ).div( 7.5 ), dot( xz, vec2( G.windDir.y.negate(), G.windDir.x ) ).div( 0.9 ) );
				const comb = texture( det, combUV.add( vec2( 0.31, 0.77 ) ) ).w;
				lawn = lawn.mul( comb.sub( 0.5 ).mul( 0.3 ).add( 1 ) );
				// seen from above the dark soil shows between the clumps; at grazing angles blade sides
				// cover everything (lighter, more saturated)
				const gapK = smoothstep( 0.3, 0.95, NdV ).mul( smoothstep( 0.62, 0.3, clump ) );
				lawn = mix( lawn, MEADOW.soil, gapK.mul( 0.45 ) );
				lawn = mix( lawn, saturation( lawn.mul( 1.12 ), 1.15 ), smoothstep( 0.45, 0.1, NdV ).mul( 0.6 ) );
				// travelling gusts flatten the grass: the paler blade backs show as waves (same gust
				// field as the grass blades)
				const gust = gustAt( xz ).mul( saturate( windStrength.mul( 0.5 ) ) ).toVar();
				lawn = mix( lawn, lawn.mul( vec3( 1.25, 1.22, 1.06 ) ).add( 0.01 ), gust.mul( 0.6 ) );
				// bare trodden soil in places, sandy soil toward the beach
				lawn = mix( lawn, srgb( 0.4, 0.33, 0.23 ), smoothstep( 0.72, 0.84, dN.y.add( dM.y.sub( 0.5 ).mul( 0.5 ) ) ).mul( 0.25 ) );
				lawn = mix( lawn, srgb( 0.60, 0.52, 0.38 ), saturate( sp.x.mul( 1.6 ) ).mul( smoothstep( 0.45, 0.62, dN.z.add( dM.y.mul( 0.3 ) ) ) ).mul( 0.7 ) );
				// inside the geometric grass field (GrassField) the ground is only seen between the blades:
				// the shaded base of the sward, dark and brownish with dead leaves; it hands over to the
				// sward's own look (above) where the blades thin out
				const grassHere = smoothstep( 2.5, 4.5, h ).mul( float( 1 ).sub( smoothstep( 0.45, 0.85, jungleW ) ) ).mul( float( 1 ).sub( saturate( sp.x.mul( 1.6 ) ) ) );
				const fieldK = float( 1 ).sub( smoothstep( GRASS_FADE[ 0 ], GRASS_FADE[ 1 ], length( p.xz.sub( cameraPosition.xz ) ) ) ).mul( grassHere ).toVar();
				const swardBase = mix( mt.tone.mul( 0.4 ), MEADOW.soil, 0.4 ).mul( dN.y.sub( 0.45 ).mul( 0.6 ).add( 1 ) ).mul( dF.y.sub( 0.4 ).mul( 0.3 ).add( 1 ) );
				lawn = mix( lawn, swardBase, fieldK.mul( 0.85 ) );
				const litter = mix( srgb( 0.2, 0.15, 0.09 ), srgb( 0.34, 0.25, 0.13 ), dF.y );
				let jungle = mix( srgb( 0.1, 0.16, 0.05 ), litter, smoothstep( 0.52, 0.7, dN.y ) );
				jungle = mix( jungle, srgb( 0.12, 0.1, 0.06 ), gully.mul( 0.3 ) );
				const far = smoothstep( 40, 160, camDist );
				let cover = mix( srgb( 0.08, 0.13, 0.04 ), srgb( 0.17, 0.24, 0.07 ), smoothstep( 0.3, 0.7, macro ) );
				cover = mix( cover, srgb( 0.27, 0.29, 0.12 ), smoothstep( 0.66, 0.84, macroB.add( dM.w.mul( 0.2 ) ) ).mul( 0.45 ) );
				jungle = mix( jungle, cover, far.mul( 0.8 ) );
				jungle = jungle.mul( macro.sub( 0.5 ).mul( 0.3 ).add( 1 ) );
				// canopy: seen from a distance (or on slopes too steep for the trees) the forest reads as
				// a carpet of lumpy crowns with dark gaps
				const crowns = texture( det, rot2( xz, 0.9 ).div( 61 ) ).w;
				const crownsB = texture( det, rot2( xz, 2.3 ).div( 13 ).add( 0.37 ) ).w;
				const canopyH = smoothstep( 0.32, 0.7, crowns.mul( 0.45 ).add( crownsB.mul( 0.4 ) ).add( dM.w.mul( 0.15 ) ) ).toVar();
				const canopyW = jungleW.mul( max( smoothstep( 0.2, 0.4, slope ), smoothstep( 90, 260, camDist ) ) ).mul( smoothstep( 25, 70, camDist ) ).mul( notRock ).mul( float( 1 ).sub( screeW.mul( 0.7 ) ) ).toVar();
				let canopy = mix( srgb( 0.05, 0.08, 0.025 ), mix( srgb( 0.14, 0.21, 0.06 ), srgb( 0.22, 0.27, 0.09 ), macroB ), canopyH );
				// steep faces: the canopy hangs in streaks down the fall line
				canopy = canopy.mul( streak.sub( 0.5 ).mul( 0.5 ).mul( smoothstep( 0.3, 0.5, slope ) ).add( 1 ) );
				jungle = mix( jungle, canopy, canopyW );
				let ground = mix( lawn, jungle, jungleW );
				// around the bare rock: dark humus, stones and moss, with the surrounding plants creeping
				// in (a soft, noisy band; no speckle)
				const creepIn = smoothstep( 0.4, 0.75, dM.w.add( dN.y.sub( 0.45 ).mul( 0.5 ) ).add( macroB.sub( 0.5 ).mul( 0.3 ) ) );
				let scree = mix( srgb( 0.16, 0.13, 0.1 ), srgb( 0.27, 0.24, 0.2 ), smoothstep( 0.45, 0.75, dM.x.add( dN.x.sub( 0.5 ).mul( 0.3 ) ) ) );
				scree = mix( scree, mix( srgb( 0.13, 0.18, 0.06 ), srgb( 0.22, 0.26, 0.09 ), dF.y ), smoothstep( 0.45, 0.7, dM.y.add( dN.y.sub( 0.45 ).mul( 0.5 ) ) ).mul( 0.7 ) );
				ground = mix( ground, scree, screeW.mul( float( 1 ).sub( creepIn.mul( 0.7 ) ) ) );
				const laterite = mix( srgb( 0.42, 0.25, 0.16 ), srgb( 0.52, 0.36, 0.24 ), dM.w ).mul( dN.y.sub( 0.4 ).mul( 0.3 ).add( 1 ) );
				ground = mix( ground, laterite, lateriteW );

				// ---- worn dirt paths / trampled ground (darker, redder soil in the forest)
				let dirt = mix( srgb( 0.38, 0.31, 0.23 ), srgb( 0.5, 0.43, 0.32 ), dM.w ).mul( dN.y.sub( 0.4 ).mul( 0.35 ).add( 0.95 ) );
				dirt = mix( dirt, srgb( 0.3, 0.22, 0.15 ), jungleW.mul( 0.7 ) );
				dirt = mix( dirt, srgb( 0.56, 0.53, 0.48 ), smoothstep( 0.72, 0.82, dN.z ).mul( 0.5 ) );
				// grass creeping onto the trail, a grassy strip between the two worn ruts
				const creep = smoothstep( 0.45, 0.7, dN.y.add( dF.y.sub( 0.45 ).mul( 0.5 ) ) ).mul( smoothstep( 0.9, 0.5, sp.y ) );
				dirt = mix( dirt, lawn, creep.mul( 0.8 ) );

				// ---- combine
				meadowW.assign( float( 1 ).sub( jungleW ).mul( float( 1 ).sub( pathW ) ).mul( notRock ).mul( float( 1 ).sub( sandW ) ).mul( landW ).mul( float( 1 ).sub( screeW ) ) );
				let albedo = mix( ground, dirt, pathW );
				albedo = mix( albedo, sand, max( sandW, underW.mul( notRock ) ) );
				albedo = mix( albedo, rockAlbedo, rockW );

				// ---- wetness (swash zone) from the shore system, else a static damp band
				const wetFoam = this.wetness ? this.wetness( xz, h ) : vec2( smoothstep( 0.5, 0.0, h ), 0 );
				const wet = saturate( wetFoam.x ).mul( float( 1 ).sub( jungleW.mul( 0.8 ) ) ).mul( landW ).toVar();
				// damp sand below the berm: darker in mottled, drying patches even when the swash has not
				// reached it lately
				const dampMottle = smoothstep( 0.3, 0.7, dM.w.add( dN.y.sub( 0.45 ).mul( 0.6 ) ).add( macroB.sub( 0.5 ).mul( 0.4 ) ) );
				const damp = smoothstep( 1.7, 0.5, h.add( dM.w.mul( 0.3 ) ) ).mul( sandW ).mul( mix( float( 0.22 ), float( 0.5 ), dampMottle ) );
				// sand dries in mottled patches; backwash leaves faint rills down the slope
				const mottle = smoothstep( 0.25, 0.75, dM.w.add( dN.y.sub( 0.45 ).mul( 0.5 ) ) );
				const dryEdge = smoothstep( 0.0, 0.6, wet ).mul( smoothstep( 1.0, 0.6, wet ) );
				const wetK = max( wet, damp ).mul( float( 1 ).sub( dryEdge.mul( mottle ).mul( 0.6 ) ) ).mul( notRock ).toVar();
				const slopeDir = normalize( N0.xz.add( vec2( 1e-4, 0 ) ) );
				const rill = texture( det, vec2( dot( xz, slopeDir ).div( 3.2 ), dot( xz, vec2( slopeDir.y.negate(), slopeDir.x ) ).div( 0.3 ) ) ).w;
				const rillK = smoothstep( 0.2, 0.9, wet ).mul( smoothstep( 0.05, 0.6, h ) ).mul( sandW );
				const wetAlbedo = saturation( albedo.mul( this.params.wetDarken ), 1.15 ).mul( vec3( 0.97, 0.98, 1.0 ) ).mul( rill.sub( 0.5 ).mul( 0.25 ).mul( rillK ).add( 1 ) );
				albedo = mix( albedo, wetAlbedo, wetK );
				// swash marks: thin wavy lines of grit left at the limits of earlier uprushes
				const sl = h.add( dM.w.mul( 0.18 ) ).add( dN.w.mul( 0.05 ) ).div( 0.13 );
				const slD = fract( sl ).sub( 0.5 ).abs();
				const slW = fwidth( sl ).add( 1e-4 );
				const swashLine = smoothstep( slW.mul( 1.5 ).add( 0.04 ), 0.0, slD ).mul( smoothstep( 0.2, 0.4, h ) ).mul( smoothstep( 1.6, 1.2, h ) )
					.mul( smoothstep( 0.45, 0.65, dM.y.add( macroB.sub( 0.5 ).mul( 0.4 ) ) ) ).mul( sandW ).mul( float( 1 ).sub( smoothstep( 0.8, 2.0, slW.mul( 10 ) ) ) );
				albedo = mix( albedo.mul( float( 1 ).sub( swashLine.mul( 0.3 ) ) ), srgb( 0.9, 0.88, 0.84 ), swashLine.mul( smoothstep( 0.6, 0.75, dF.z ) ).mul( 0.5 ) );
				// foam residue: lacy patterns stranded on the sand
				const lace = smoothstep( 0.42, 0.18, texture( det, rot2( xz, 0.4 ).div( 0.9 ) ).x ).mul( 0.7 ).add( 0.3 );
				const residue = saturate( wetFoam.y ).mul( lace ).mul( landW ).mul( notRock ).toVar();
				albedo = mix( albedo, srgb( 0.88, 0.9, 0.9 ), residue );

				// ---- roughness
				let rough = mix( float( 0.88 ), float( 0.93 ), sandW );
				rough = mix( rough, float( 0.9 ), pathW );
				rough = mix( rough, rockRough, rockW );
				// wet sand has a film of water: glossy while fresh, satin as it drains
				rough = mix( rough, mix( float( 0.42 ), float( 0.16 ), wet ), wetK );
				rough = mix( rough, float( 0.7 ), residue );
				rough = mix( rough, float( 0.75 ), underW );
				outRough.assign( rough );

				// ---- micro relief for the normal
				const windR = rip1.mul( 0.005 ).mul( windK );
				const waveR = rip3.mul( 0.012 ).mul( fade3 ).mul( smoothstep( - 0.3, - 0.9, h ) ).mul( float( 1 ).sub( seagrassW ) );
				const megaR = rip2.mul( 0.035 ).mul( fade2 ).mul( fieldW ).mul( float( 1 ).sub( seagrassW ) );
				const sandH = grain.mul( 0.004 ).add( grainF.mul( 0.003 ) ).add( pebW.mul( 0.004 ) ).add( windR ).add( waveR ).add( megaR ).mul( float( 1 ).sub( wet.mul( 0.6 ) ) )
					.add( rill.mul( 0.006 ).mul( rillK ) );
				// seagrass canopy stands proud of the sand with a ragged scarp; rubble is knobbly
				const seabedH = seagrassW.mul( blades.mul( 0.05 ).add( 0.08 ) ).add( rubbleW.mul( dN.x.mul( 0.07 ).add( dF.x.mul( 0.015 ) ) ) );
				const groundH = dN.y.mul( mix( float( 0.05 ), float( 0.035 ), jungleW ) ).add( dF.y.mul( 0.012 ) ).add( dM.y.mul( 0.045 ) ).add( canopyH.mul( canopyW ).mul( 2.5 ) )
					.add( clump.mul( 0.12 ).mul( float( 1 ).sub( jungleW ) ) ).add( comb.mul( 0.03 ).mul( float( 1 ).sub( jungleW ) ) );
				const screeH = dN.z.mul( 0.04 ).add( dM.x.mul( 0.06 ) );
				const dirtH = dN.z.mul( 0.012 ).add( dN.y.mul( 0.01 ) );
				let hd = mix( mix( groundH, screeH, screeW ), dirtH, pathW );
				hd = mix( hd, sandH.add( seabedH ), max( sandW, underW.mul( notRock ) ) );
				hd = mix( hd, rockHd, rockW );
				outN.assign( perturbNormal( N0, hd, 1.0 ) );

				// ---- ambient occlusion: baked horizon + cavity, plus litter / crevices / seagrass canopy
				const aoDetail = mix( float( 1 ), dN.y.mul( 0.5 ).add( 0.7 ), jungleW.mul( float( 1 ).sub( sandW ) ).mul( notRock ).mul( landW ) )
					.mul( mix( float( 1 ), smoothstep( 0.2, 0.7, clump ).mul( 0.35 ).add( 0.65 ), meadowW ) );
				outAO.assign( nr.w.mul( aoDetail ).mul( float( 1 ).sub( seagrassW.mul( 0.3 ) ) ).mul( float( 1 ).sub( rubbleW.mul( smoothstep( 0.55, 0.2, dN.x ) ).mul( 0.35 ) ) )
					.mul( mix( float( 1 ), smoothstep( 0.15, 0.6, canopyH ).mul( 0.6 ).add( 0.4 ), canopyW ) ) );


				albedoOut.assign( albedo );

			} );

			return albedoOut;

		} )();

		return { albedo: albedoNode, rough: outRough, N: outN, ao: outAO, meadow: meadowW };

	}

	// (Re)build the fragment nodes. Called again once the shore system provides wetness.
	finalizeMaterial() {

		const s = this._surface();
		const mat = this.material;
		this.meadowNode = s.meadow;
		mat.colorNode = vec4( s.albedo, 1 );
		mat.roughnessNode = s.rough;
		mat.normalNode = transformNormalToView( s.N );
		mat.aoNode = saturate( s.ao );
		mat.needsUpdate = true;

	}

	update( camera ) {

		camera.getWorldPosition( this.uViewPos.value );
		this.lod.update( camera );
		if ( this.renderer ) this.gpu.updateSunShadow( this.renderer );

	}

}
