import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, vec2, vec3, vec4, texture, positionGeometry, instanceIndex, varyingProperty,
	normalize, refract, max, min, abs, clamp, mix, smoothstep, exp, exp2, saturate, pow, dFdx, dFdy, length,
} from 'three/tsl';
import { G } from '../core/Globals.js';

// Caustics by rasterized photon splatting (as in Evan Wallace's "WebGL Water").
//
// A fine grid covering one FFT tile is drawn into an offscreen target. Each vertex is a point on
// the real wave surface; the sun ray is refracted through the surface normal there and followed
// down to a plane `D` metres below, and the vertex is placed at that landing point. The fragment
// writes (area on the surface / area on the floor) with additive blending, which is exactly the
// light concentration, so focusing folds form the bright caustic networks physically. The grid
// is drawn 9 times offset by +-1 tile so the result tiles seamlessly (no finite area).
//
// Two focal planes are rendered (R = shallow, G = deep) and blended by the real depth at lookup.
// A second, larger tile from the next cascade adds broad focusing so the result never repeats.
class CausticLayer {

	constructor( fft, cascade, { res, grid, depths, name, slopeLevel } ) {

		this.fft = fft;
		this.cascade = cascade;
		this.tile = fft.sizes[ cascade ];
		this.res = res;
		this.depths = depths;

		this.target = new THREE.RenderTarget( res, res, { type: THREE.HalfFloatType, depthBuffer: false, generateMipmaps: true } );
		this.target.texture.name = name;
		this.target.texture.wrapS = this.target.texture.wrapT = THREE.RepeatWrapping;
		this.target.texture.minFilter = THREE.LinearMipmapLinearFilter;
		this.target.texture.magFilter = THREE.LinearFilter;
		this.target.texture.generateMipmaps = true;
		// the floor is mostly seen at grazing angles: filter along the view, stay sharp across it
		this.target.texture.anisotropy = 8;

		const geo = new THREE.PlaneGeometry( 1, 1, grid, grid );
		geo.translate( 0.5, 0.5, 0 ); // [0,1]^2
		const inst = new THREE.InstancedBufferGeometry();
		inst.index = geo.index;
		inst.setAttribute( 'position', geo.getAttribute( 'position' ) );
		inst.instanceCount = 9;

		this.scene = new THREE.Scene();
		this.camera = new THREE.OrthographicCamera( 0, 1, 1, 0, - 1, 1 );
		this.meshes = [];

		const L = this.tile;
		const deriv = fft.derivativeTexture;

		for ( let k = 0; k < depths.length; k ++ ) {

			const D = depths[ k ];
			const vOld = varyingProperty( 'vec2', 'vCauOld' );
			const vNew = varyingProperty( 'vec2', 'vCauNew' );
			const mat = new THREE.MeshBasicNodeMaterial();
			mat.transparent = true;
			mat.blending = THREE.CustomBlending;
			mat.blendEquation = THREE.AddEquation;
			mat.blendSrc = THREE.OneFactor;
			mat.blendDst = THREE.OneFactor;
			mat.depthTest = false;
			mat.depthWrite = false;
			mat.side = THREE.DoubleSide;

			mat.vertexNode = Fn( () => {

				const uv = positionGeometry.xy;
				const d = texture( deriv, uv ).depth( cascade ).level( slopeLevel );
				const s = vec2( d.x.div( max( d.z.add( 1 ), 0.3 ) ), d.y.div( max( d.w.add( 1 ), 0.3 ) ) );
				const n = normalize( vec3( s.x.negate(), 1, s.y.negate() ) );
				const T = refract( G.sunDir.negate(), n, 1 / 1.333 );
				const tDown = max( T.y.negate(), 0.15 );
				// flat-surface refraction offset is removed so the pattern stays registered with the
				// entry point (the lookup re-applies it with the real depth)
				const T0 = refract( G.sunDir.negate(), vec3( 0, 1, 0 ), 1 / 1.333 );
				const off = T.xz.div( tDown ).sub( T0.xz.div( max( T0.y.negate(), 0.15 ) ) ).mul( D );
				const p = uv.mul( L );
				const q = p.add( off );
				vOld.assign( p );
				vNew.assign( q );
				// 3x3 copies shifted by whole tiles for seamless wrapping
				const i = float( instanceIndex );
				const ox = i.mod( 3 ).sub( 1 );
				const oy = i.div( 3 ).floor().sub( 1 );
				const ndc = q.div( L ).add( vec2( ox, oy ) ).mul( 2 ).sub( 1 );
				return vec4( ndc.x, ndc.y, 0, 1 );

			} )();

			mat.colorNode = Fn( () => {

				// area ratio between the surface patch and its image on the floor
				const ao = abs( dFdx( vOld ).x.mul( dFdy( vOld ).y ).sub( dFdx( vOld ).y.mul( dFdy( vOld ).x ) ) );
				const an = abs( dFdx( vNew ).x.mul( dFdy( vNew ).y ).sub( dFdx( vNew ).y.mul( dFdy( vNew ).x ) ) );
				// soft limit: a single nearly-folded cell must not become a flat white hot spot (the
				// finite sun disk spreads real caustic peaks to a few times the mean anyway)
				const I = ao.div( max( an.add( ao.mul( 1 / 8 ) ), 1e-9 ) );
				return vec4( k === 0 ? I : 0, k === 1 ? I : 0, 0, 1 );

			} )();

			const mesh = new THREE.Mesh( inst, mat );
			mesh.frustumCulled = false;
			this.scene.add( mesh );
			this.meshes.push( mesh );

		}

	}

	render( renderer ) {

		const prevTarget = renderer.getRenderTarget();
		const prevClear = renderer.getClearColor( new THREE.Color() );
		const prevAlpha = renderer.getClearAlpha();
		renderer.setRenderTarget( this.target );
		renderer.setClearColor( 0x000000, 0 );
		renderer.clear();
		renderer.render( this.scene, this.camera );
		renderer.setRenderTarget( prevTarget );
		renderer.setClearColor( prevClear, prevAlpha );

	}

}

export class Caustics {

	constructor( renderer, fft ) {

		this.renderer = renderer;
		this.fft = fft;
		this.strength = uniform( 0.75 ).setName( 'cauStrength' );
		this.detail = null; // SeaDetail: rougher water in gusts focuses more, slicks less
		const fine = fft.cascades - 1;
		// fine networks (finest cascade, ripples < ~11 cm filtered out: they defocus immediately)
		this.fine = new CausticLayer( fft, fine, { res: 512, grid: 256, depths: [ 1.2, 4.0 ], name: 'causticsFine', slopeLevel: 1 } );
		// broad focusing from the next cascade; different tile size -> no visible repetition
		this.broad = new CausticLayer( fft, fine - 1, { res: 256, grid: 128, depths: [ 3.0, 9.0 ], name: 'causticsBroad', slopeLevel: 0.5 } );
		// shared base nodes: all lookups reuse one texture + sampler binding per layer
		this.fineTex = texture( this.fine.target.texture );
		this.broadTex = texture( this.broad.target.texture );

	}

	update() {

		this.fine.render( this.renderer );
		this.broad.render( this.renderer );

	}

	// Caustic light factor (vec3, mean ~1) at a world point `depth` meters below the surface.
	// Options:
	//   slope: vec2 slope (dh/dx, dh/dz) of the long waves above (swell, shore waves). Their
	//          refraction tilts the light, so the whole network sways as each wave passes.
	//   foam:  surface foam / bubble coverage above (0..1): diffuses the light, kills the network
	//   grad:  { dx, dy } change of world xz across one pixel (fragment derivatives): the networks are
	//          filtered over the pixel's footprint, anisotropically. Without it the fixed blur level
	//          aliases into crawling noise on distant or grazing floors.
	sample( worldPos, depth, level = null, { slope = null, foam = null, grad = null } = {} ) {

		// the light reaching this point entered the water up-sun along the refracted sun ray
		const n = slope ? normalize( vec3( slope.x.negate(), 1, slope.y.negate() ) ) : vec3( 0, 1, 0 );
		const Ls = refract( G.sunDir.negate(), n, 1 / 1.333 );
		const tDown = max( Ls.y.negate(), 0.15 );
		const entry = worldPos.xz.sub( Ls.xz.mul( depth.div( tDown ) ) ).toVar();

		// deeper -> softer (finite sun disk + forward scattering)
		const blur = level !== null ? float( level ) : clamp( depth.mul( 0.4 ).sub( 0.2 ), 0, 3 );
		const wD = saturate( depth.sub( 1.2 ).div( 2.8 ) ); // blend between the two focal planes

		// the blur level is the least filtering; with a footprint, each gradient is stretched to at
		// least that level's texel size
		const fetch = ( layer, layerTex, uv, lvl ) => {

			if ( grad === null ) return layerTex.sample( uv ).level( lvl );
			const minLen = exp2( lvl ).div( layer.res );
			const stretch = ( g ) => g.mul( max( minLen.div( max( length( g ), 1e-9 ) ), 1 ) );
			return layerTex.sample( uv ).grad( stretch( grad.dx.div( layer.tile ) ), stretch( grad.dy.div( layer.tile ) ) );

		};

		const lookup = ( layerTex, uv ) => {

			const t = fetch( this.fine, layerTex, uv, blur );
			return mix( t.x, t.y, wD );

		};

		// slight chromatic dispersion: each color lands a little apart along the sun direction
		const disp = normalize( Ls.xz.add( vec2( 1e-4, 0 ) ) ).mul( depth.mul( 0.0035 ) );
		const uvF = entry.div( this.fine.tile );
		const r = lookup( this.fineTex, uvF.add( disp.div( this.fine.tile ) ) );
		const g = lookup( this.fineTex, uvF );
		const b = lookup( this.fineTex, uvF.sub( disp.div( this.fine.tile ) ) );
		const broad = fetch( this.broad, this.broadTex, entry.div( this.broad.tile ), float( 1.5 ) );
		const br = mix( broad.x, broad.y, saturate( depth.div( 9 ) ) );
		const c = vec3( r, g, b ).mul( mix( float( 1 ), br, 0.6 ) );

		// no caustics right at the surface, strongest in the first metres, fading with depth
		let k = smoothstep( 0.03, 0.5, depth ).mul( exp( depth.mul( - 0.06 ) ) ).mul( this.strength );
		if ( this.detail ) {

			const det = this.detail.sample( entry );
			k = k.mul( mix( float( 0.55 ), float( 1.25 ), det.gust ) ).mul( float( 1 ).sub( det.slick.mul( 0.6 ) ) );

		}

		if ( foam ) k = k.mul( float( 1 ).sub( saturate( foam ) ) );
		const result = mix( vec3( 1 ), c, k );
		// foam and bubble clouds scatter the light back up: the floor under them is shaded
		return foam ? result.mul( float( 1 ).sub( saturate( foam ).mul( 0.6 ) ) ) : result;

	}

}
