import { vec2, vec3, float, max, exp, mix, smoothstep, refract, positionWorld, texture, If, saturate, dFdx, dFdy } from 'three/tsl';
import { G } from '../core/Globals.js';
import { SceneLighting } from '../materials/SceneLighting.js';

// Installs the lighting hooks that every SceneMaterial uses:
//  - direct sun: attenuated along the refracted sun path through the water column (Beer-Lambert),
//    modulated by caustics that follow the waves above (swell + shore waves tilt the light, surf
//    foam and bubbles shade the floor)
//  - ambient: attenuated + tinted with depth
// Everything runs only for fragments that can be under water.
//
// Per material (material.underwaterLighting): 'full' (default: terrain, reef, rocks - caustics,
// waves, foam shading), 'lite' (water column attenuation only: pier, boat hull, village) or
// 'none' (never under water: plants). Each extra texture here is a sampler in every material that
// uses the hook, and WebGPU allows only 16 per stage.
export function installUnderwaterLighting( { fft, caustics, clouds = null, terrain = null, shore = null, surface = null, shoreSim = null } ) {

	// shared base nodes (one texture + sampler binding each): these hooks go into every scene
	// material, several of which are close to the 16-sampler limit
	const dispTex = texture( fft.displacementTexture );
	const derivTex = texture( fft.derivativeTexture );

	// long waves at xz: height, slope and foam from the coarse FFT cascades and the shore waves
	const longWaves = ( xz ) => {

		const seaDepth = terrain ? G.seaLevel.sub( terrain.heightAt( xz ) ).toVar() : float( 50 );
		const h = float( 0 ).toVar();
		const slope = vec2( 0 ).toVar();
		for ( let c = 0; c < Math.min( 3, fft.cascades ); c ++ ) {

			const uv = xz.div( fft.sizes[ c ] );
			const att = surface ? surface.cascadeAttenuation( c, seaDepth ) : float( 1 );
			h.addAssign( dispTex.sample( uv ).depth( c ).level( 2 ).y.mul( att ) );
			if ( c < 2 ) {

				const d = derivTex.sample( uv ).depth( c ).level( 2 );
				slope.addAssign( vec2( d.x, d.y ).mul( att ) );

			}

		}

		const foam = float( 0 ).toVar();
		if ( shore && terrain ) {

			const sw = shore.evaluate( xz, seaDepth, terrain.heightAt( xz ) );
			h.addAssign( sw.disp.y );
			const n = sw.nShore;
			slope.addAssign( vec2( n.x, n.z ).div( max( n.y, 0.25 ) ).negate() );
			foam.addAssign( sw.foam );

		}

		if ( shoreSim ) foam.addAssign( shoreSim.sample( xz ).x.mul( 0.8 ) );
		// waves only exist over water: none over dry land
		if ( terrain ) h.assign( h.mul( smoothstep( 0.0, 1.0, seaDepth ) ).sub( smoothstep( 0.0, - 0.4, seaDepth ).mul( 10 ) ) );
		return { height: G.seaLevel.add( h ), slope, foam };

	};

	const sigT = () => G.waterAbsorption.add( G.waterScattering );

	// mean water level from the two longest FFT cascades (one texture binding)
	const meanLevel = ( P ) => {

		let h = float( 0 );
		for ( let c = 0; c < Math.min( 2, fft.cascades ); c ++ ) h = h.add( dispTex.sample( P.xz.div( fft.sizes[ c ] ) ).depth( c ).level( 3 ).y );
		if ( terrain ) h = h.mul( smoothstep( 0.0, 3.0, G.seaLevel.sub( terrain.heightAt( P.xz ) ) ) );
		return G.seaLevel.add( h );

	};

	const mode = ( builder ) => ( builder.material && builder.material.underwaterLighting ) || 'full';

	// highest the water can reach on the shore: the swash run-up grows with the surf height. Terrain
	// and props above it (the dry beach) skip the wave evaluation entirely.
	const reach = () => G.seaLevel.add( shore ? shore.amplitude.mul( 1.5 ).add( 1.2 ) : float( 3.0 ) );

	SceneLighting.directModulation = ( builder ) => {

		if ( builder.material && builder.material.isWaterMaterial ) return null;
		const m = mode( builder );
		const P = positionWorld;
		const result = vec3( 1 ).toVar();
		// the pixel's footprint on the ground plane, to filter the caustics over it (taken here, in
		// uniform control flow, ahead of the branches below)
		const grad = caustics && m === 'full' ? { dx: dFdx( P.xz ).toVar(), dy: dFdy( P.xz ).toVar() } : null;
		// cheap reject: above anything the water reaches
		if ( m !== 'none' ) If( P.y.lessThan( reach() ), () => {

			const lw = m === 'full' ? longWaves( P.xz ) : { height: meanLevel( P ) };
			const d = max( lw.height.sub( P.y ), 0 ).toVar();
			If( d.greaterThan( 0 ), () => {

				const under = smoothstep( 0.0, 0.08, d );
				const Ls = refract( G.sunDir.negate(), vec3( 0, 1, 0 ), 1 / 1.333 );
				const mu = max( Ls.y.negate(), 0.15 );
				const atten = exp( sigT().mul( d ).div( mu ).negate() );
				const caust = caustics && m === 'full' ? caustics.sample( P, d, null, { slope: lw.slope, foam: saturate( lw.foam ), grad } ) : vec3( 1 );
				result.assign( mix( vec3( 1 ), atten.mul( caust ), under ) );

			} );

		} );

		const cloudShadow = clouds ? clouds.shadow( P.xz ) : float( 1 );
		// hills shadowing the island and the bay at low sun (terrain heightfield shadow; the terrain and
		// the rocks apply it in their own lighting model)
		const mat = builder.material;
		const hill = terrain && terrain.sunShadowAt && ! ( mat && mat.appliesHillShadow ) ? terrain.sunShadowAt( P ) : float( 1 );
		return result.mul( cloudShadow ).mul( hill );

	};

	SceneLighting.ambientModulation = ( builder ) => {

		if ( builder.material && builder.material.isWaterMaterial ) return null;
		if ( mode( builder ) === 'none' ) return null;
		const P = positionWorld;
		const result = vec3( 1 ).toVar();
		If( P.y.lessThan( reach() ), () => {

			// the ambient term only needs the mean water level (no shore evaluation)
			const d = max( meanLevel( P ).sub( P.y ), 0 );
			const under = smoothstep( 0.0, 0.1, d );
			// diffuse downwelling light: effective path ~1.2x depth, plus a little in-scattered blue
			const atten = exp( sigT().mul( d ).mul( 1.2 ).negate() ).mul( 0.85 ).add( vec3( 0.0, 0.02, 0.04 ).mul( exp( d.mul( - 0.1 ) ) ) );
			result.assign( mix( vec3( 1 ), atten, under ) );

		} );

		return result;

	};

}
