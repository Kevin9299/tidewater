import * as THREE from 'three/webgpu';
import {
	Fn, If, float, vec2, vec3, vec4, attribute, texture, positionGeometry, positionWorld, cameraPosition, smoothstep, property, varyingProperty,
	normalize, cross, dot, abs, max, mix, floor, fract, clamp, select, sin, cos, cameraViewMatrix, positionViewDirection, normalWorldGeometry, faceDirection,
} from 'three/tsl';
import { physical } from '../../materials/Materials.js';
import { windStrength, windDir3, gustAt, uCamPos, uLodRange, UP, LOD_BAND } from './VegNodes.js';
import { bayer4 } from '../../materials/LODFade.js';
import { translucency } from './VegMaterials.js';
import { G } from '../../core/Globals.js';

// Octahedral impostors for the broadleaf trees and shrubs.
//
// At startup (first update with a renderer) every plant variant is rendered from N x N directions
// on the upper hemisphere (hemi-octahedral layout) into two atlases:
//   A: leaf brightness structure, leaf (1) / bark (0) flag, per-card colour random, coverage
//   B: plant-local normal * 0.5 + 0.5, exposure (ambient occlusion)
// Colours are applied at runtime with the same species palette as the near geometry, and the
// variants use the same lobe tables, so the far plants keep the near plants' crowns.
// At runtime each plant is one camera-facing quad; the fragment shader picks the three frames
// around the view direction (barycentric blend) and re-projects the view ray onto each frame's
// plane (exact for any view direction, non-uniform instance scale included).

export const OCT_N = 6; // frames per side
const BLEND_DIST = 100; // m: three-frame blending within, nearest frame beyond
const THIN = [ 140, 320 ]; // m: distance range over which the forest is thinned
const THIN_FRACTION = 0.5; // share of the crowns removed at the far end
const SHRUB_MAX = 180; // m: shrub impostors are dropped beyond (too small to matter)
const FRAME_PX = 128;

// hemi-octahedral decode: (u, v) in [-1, 1]^2 -> unit direction with y >= 0
export function octDecode( u, v, out = new THREE.Vector3() ) {

	const x = ( u - v ) * 0.5, z = ( u + v ) * 0.5;
	return out.set( x, 1 - Math.abs( x ) - Math.abs( z ), z ).normalize();

}

export function frameBasis( d ) {

	const right = new THREE.Vector3().crossVectors( new THREE.Vector3( 0, 1, 0 ), d );
	if ( right.lengthSq() < 1e-8 ) right.set( 1, 0, 0 );
	right.normalize();
	const up = new THREE.Vector3().crossVectors( d, right ).normalize();
	return { right, up };

}

// TSL counterparts
const octDecodeT = ( u, v ) => {

	const x = u.sub( v ).mul( 0.5 ), z = u.add( v ).mul( 0.5 );
	return normalize( vec3( x, float( 1 ).sub( abs( x ) ).sub( abs( z ) ), z ) );

};

const octEncodeT = ( d ) => {

	const p = d.div( abs( d.x ).add( abs( d.y ) ).add( abs( d.z ) ) );
	return vec2( p.x.add( p.z ), p.z.sub( p.x ) );

};

// Bakes and draws the impostors of a set of plant groups (group 0 trees, group 1 shrubs).
export class ImpostorAtlas {

	// groups: [ { variants: [ BufferGeometry ], center: Vector3 (on the plant axis), radius (frame
	// size), rh (horizontal radius), hv (half height) } ], bakeMaterials: { albedo, normal }
	constructor( groups, bakeMaterials ) {

		this.groups = groups;
		this.bakeMaterials = bakeMaterials;
		let v = 0;
		for ( const g of groups ) {

			g.variantBase = v;
			v += g.variants.length;

		}

		this.variantCount = v;
		const W = v * OCT_N * FRAME_PX, H = OCT_N * FRAME_PX;
		const make = ( name ) => {

			const rt = new THREE.RenderTarget( W, H, {
				type: THREE.UnsignedByteType, depthBuffer: true, generateMipmaps: true,
				minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter,
			} );
			rt.texture.name = name;
			rt.texture.colorSpace = THREE.NoColorSpace;
			return rt;

		};

		this.rtA = make( 'vegImpostorA' );
		this.rtB = make( 'vegImpostorB' );
		this.width = W;
		this.height = H;
		this.baked = false;

	}

	// frame transforms: local plant -> atlas plane (cell centre) for variant v, frame (i, j)
	_frameMatrices( g, vi ) {

		const mats = [];
		const R = g.radius, C = g.center;
		for ( let j = 0; j < OCT_N; j ++ ) for ( let i = 0; i < OCT_N; i ++ ) {

			const d = octDecode( - 1 + 2 * i / ( OCT_N - 1 ), - 1 + 2 * j / ( OCT_N - 1 ) );
			const { right, up } = frameBasis( d );
			const cx = ( ( g.variantBase + vi ) * OCT_N + i + 0.5 ) * 2 * R, cy = ( j + 0.5 ) * 2 * R;
			const m = new THREE.Matrix4().set(
				right.x, right.y, right.z, - C.dot( right ) + cx,
				up.x, up.y, up.z, - C.dot( up ) + cy,
				d.x, d.y, d.z, - C.dot( d ),
				0, 0, 0, 1 );
			mats.push( m );

		}

		return mats;

	}

	bake( renderer ) {

		const t0 = performance.now();
		const scene = new THREE.Scene();
		const prevTarget = renderer.getRenderTarget();
		const prevClear = renderer.getClearColor( new THREE.Color() );
		const prevAlpha = renderer.getClearAlpha();
		const prevAuto = renderer.autoClear;

		for ( const [ rt, mat ] of [ [ this.rtA, this.bakeMaterials.albedo ], [ this.rtB, this.bakeMaterials.normal ] ] ) {

			renderer.setRenderTarget( rt );
			renderer.setClearColor( 0x000000, 0 );
			renderer.autoClear = false;
			renderer.clear();
			for ( const g of this.groups ) {

				// the atlas plane is one orthographic view: every cell of this group is 2R wide
				const R = g.radius;
				const cam = new THREE.OrthographicCamera( 0, this.variantCount * OCT_N * 2 * R, OCT_N * 2 * R, 0, 0.1, 6 * R + 10 );
				cam.position.set( 0, 0, 3 * R + 2 );
				cam.lookAt( 0, 0, - 1 );
				cam.updateMatrixWorld();
				cam.updateProjectionMatrix();
				g.variants.forEach( ( geo, vi ) => {

					const mats = this._frameMatrices( g, vi );
					const mesh = new THREE.InstancedMesh( geo, mat, mats.length );
					mats.forEach( ( m, k ) => mesh.setMatrixAt( k, m ) );
					mesh.frustumCulled = false;
					scene.add( mesh );
					renderer.render( scene, cam );
					scene.remove( mesh );
					mesh.dispose();

				} );

			}

		}

		renderer.setRenderTarget( prevTarget );
		renderer.setClearColor( prevClear, prevAlpha );
		renderer.autoClear = prevAuto;
		this.baked = true;
		this.bakeMs = performance.now() - t0;

	}

	// Runtime material. groupParams: per group { center, radius } in the same order as groups;
	// isGroup1: TSL bool (per instance) selecting group 1 (shrubs) over group 0 (trees);
	// variantOf( seed, isGroup1 ): TSL variant index within the group;
	// colorOf( { seed, cr, leaf, bright, isGroup1 } ): TSL linear albedo.
	createMaterial( { isGroup1, variantOf, colorOf, nearDist } ) {

		const mat = physical( { side: THREE.DoubleSide, specularIntensity: 0.12 } );
		mat.name = 'veg-impostor';
		const [ g0, g1 ] = this.groups;
		const iPos = attribute( 'iPos', 'vec4' );
		const iDat = attribute( 'iDat', 'vec4' );
		const g1Flag = isGroup1( iDat );
		const R = select( g1Flag, float( g1.radius ), float( g0.radius ) );
		const Rh = select( g1Flag, float( g1.rh ), float( g0.rh ) );
		const Hv = select( g1Flag, float( g1.hv ), float( g0.hv ) );
		const Cy = select( g1Flag, float( g1.center.y ), float( g0.center.y ) );
		const vBase = select( g1Flag, float( g1.variantBase ), float( g0.variantBase ) );

		// crown sway offset (xyz) and the effective scale (w) for the fragment stage
		const vImp = varyingProperty( 'vec4', 'vVegImp' );

		// ---- vertex: camera-facing quad around the plant centre, swaying with the wind
		mat.positionNode = Fn( () => {

			const base = iPos.xyz;
			const sy = abs( iDat.y );
			// LOD window: from the near plant's hand-over distance to the fade-out, else collapsed
			const d = uCamPos.sub( base ).length();
			const vis = select( d.greaterThanEqual( nearDist( g1Flag ).mul( 1 - LOD_BAND / 2 ) ).and( d.lessThan( select( g1Flag, float( SHRUB_MAX ), uLodRange.z ) ) ), float( 1 ), float( 0 ) );
			// far away the forest is thinned out: fewer, proportionally larger crowns (grown about
			// the base) keep the canopy closed
			const thin = smoothstep( THIN[ 0 ], THIN[ 1 ], d );
			const keep = select( fract( iDat.w.mul( 91.7 ) ).lessThan( thin.mul( THIN_FRACTION ) ), float( 0 ), float( 1 ) );
			const grow = thin.mul( 1 / Math.sqrt( 1 - THIN_FRACTION ) - 1 ).add( 1 );
			const s = iPos.w.mul( grow );
			const C = base.add( vec3( 0, Cy.mul( s ).mul( sy ), 0 ) );
			// sway of the whole crown (matches the near plants' trunk sway amplitude)
			const w = windStrength;
			const g = gustAt( base.xz );
			const ph0 = iDat.w.mul( 6.2832 );
			const sway = w.mul( w ).mul( 0.009 ).mul( g.mul( 0.8 ).add( 0.3 ) )
				.add( sin( G.time.mul( 0.9 ).add( ph0 ) ).mul( w ).mul( 0.0045 ).mul( g.add( 0.4 ) ) ).mul( iDat.z ).mul( 0.45 );
			const swayV = windDir3.mul( sway );
			const Cs = C.add( swayV );
			vImp.assign( vec4( swayV, s ) );
			const toCam = normalize( cameraPosition.sub( Cs ) );
			const right = normalize( cross( UP, toCam ).add( vec3( 1e-4, 0, 0 ) ) );
			const up = cross( toCam, right );
			// quad fitted to the plant's projected extent: its horizontal radius across, from above
			// the crown disc, from the side the (stretched) height
			const k = s.mul( vis ).mul( keep );
			const ty = abs( toCam.y );
			const halfW = Rh.mul( k );
			const halfH = Hv.mul( sy ).mul( max( float( 1 ).sub( ty.mul( ty ) ), 0 ).sqrt() ).add( Rh.mul( ty ) ).mul( k );
			const p = positionGeometry;
			return Cs.add( right.mul( p.x.mul( halfW ) ).add( up.mul( p.y.mul( halfH ) ) ) );

		} )();

		// ---- fragment: frame selection + re-projection
		// shared between mask, colour and normal: plain properties (a toVar() initialiser would be
		// re-emitted in every branch that reads them)
		const vA = property( 'vec4', 'impA' ), vB = property( 'vec4', 'impB' );
		const atlasA = texture( this.rtA.texture ), atlasB = texture( this.rtB.texture );
		const cells = this.variantCount * OCT_N;
		mat.maskNode = Fn( () => {

			const base = iPos.xyz;
			const s = vImp.w;
			const sy = abs( iDat.y );
			const yaw = iDat.x;
			const cyw = cos( yaw ), syw = sin( yaw );
			const C = base.add( vec3( 0, Cy.mul( s ).mul( sy ), 0 ) ).add( vImp.xyz );
			// world -> plant-local (unstretched, centred): rotate by -yaw, divide by the scale
			const toLocal = ( v ) => vec3( v.x.mul( cyw ).sub( v.z.mul( syw ) ), v.y, v.x.mul( syw ).add( v.z.mul( cyw ) ) ).div( vec3( s, s.mul( sy ), s ) );
			const O = toLocal( cameraPosition.sub( C ) ).toVar();
			const D = toLocal( positionWorld.sub( cameraPosition ) ).toVar();
			const vdir = normalize( O );
			const vd = normalize( vec3( vdir.x, max( vdir.y, 0.02 ), vdir.z ) );
			const g = octEncodeT( vd ).mul( 0.5 ).add( 0.5 ).mul( OCT_N - 1 );
			const gi = floor( clamp( g, vec2( 0 ), vec2( OCT_N - 1.001 ) ) );
			const f = g.sub( gi );
			const upper = f.x.add( f.y ).greaterThan( 1 );
			// triangle of the cell containing g, barycentric weights (materialised before the If
			// below: a value first built inside one branch would read as zero in the other)
			const i0 = select( upper, gi.add( 1 ), gi ).toVar();
			const i1 = select( upper, gi.add( vec2( 0, 1 ) ), gi.add( vec2( 1, 0 ) ) ).toVar();
			const i2 = select( upper, gi.add( vec2( 1, 0 ) ), gi.add( vec2( 0, 1 ) ) ).toVar();
			const w0 = select( upper, f.x.add( f.y ).sub( 1 ), float( 1 ).sub( f.x ).sub( f.y ) ).toVar();
			const w1 = select( upper, float( 1 ).sub( f.x ), f.x ).toVar();
			const w2 = select( upper, float( 1 ).sub( f.y ), f.y ).toVar();
			const variant = vBase.add( variantOf( iDat.w, g1Flag ) ).toVar();
			const Rf = R.toVar();
			const sampleFrame = ( ij ) => {

				const d = octDecodeT( ij.x.div( OCT_N - 1 ).mul( 2 ).sub( 1 ), ij.y.div( OCT_N - 1 ).mul( 2 ).sub( 1 ) );
				const right = normalize( cross( UP, d ) );
				const up = cross( d, right );
				const t = dot( O, d ).negate().div( dot( D, d ) );
				const P = O.add( D.mul( t ) );
				const a = dot( P, right ).div( Rf ), b = dot( P, up ).div( Rf );
				const cu = variant.mul( OCT_N ).add( ij.x ).add( a.mul( 0.5 ).add( 0.5 ) );
				const cv = ij.y.add( b.mul( 0.5 ).add( 0.5 ) );
				const inCell = abs( a ).lessThan( 1 ).and( abs( b ).lessThan( 1 ) );
				// render targets are stored top row first: flip v
				const st = vec2( cu.div( cells ), float( 1 ).sub( cv.div( OCT_N ) ) );
				const k = select( inCell, float( 1 ), float( 0 ) );
				return { A: atlasA.sample( st ).mul( k ), B: atlasB.sample( st ).mul( k ) };

			};

			// near: blend the three frames around the view direction; far: the dominant frame only
			const blend = uCamPos.sub( base ).length().lessThan( BLEND_DIST );
			If( blend, () => {

				const s0 = sampleFrame( i0 ), s1 = sampleFrame( i1 ), s2 = sampleFrame( i2 );
				vA.assign( s0.A.mul( w0 ).add( s1.A.mul( w1 ) ).add( s2.A.mul( w2 ) ) );
				vB.assign( s0.B.mul( w0 ).add( s1.B.mul( w1 ) ).add( s2.B.mul( w2 ) ) );

			} ).Else( () => {

				const iMax = select( w0.greaterThanEqual( max( w1, w2 ) ), i0, select( w1.greaterThanEqual( w2 ), i1, i2 ) );
				const sm = sampleFrame( iMax );
				vA.assign( sm.A );
				vB.assign( sm.B );

			} );

			// cross-fade from the near geometry (incoming level of the band around nearDist)
			const nd = nearDist( g1Flag );
			const fade = smoothstep( nd.mul( 1 - LOD_BAND / 2 ), nd.mul( 1 + LOD_BAND / 2 ), uCamPos.sub( base ).length() );
			return vA.w.greaterThan( 0.42 ).and( bayer4().lessThan( fade ) );

		} )();

		mat.colorNode = Fn( () => {

			const cov = max( vA.w, 1e-3 );
			const bright = vA.x.div( cov ), leaf = vA.y.div( cov ), cr = vA.z.div( cov );
			// exposure (baked crown AO): the inside and underside of the crown in deep shade
			const ex = vB.w.div( cov );
			return colorOf( { seed: iDat.w, cr, leaf, bright, isGroup1: g1Flag } ).mul( mix( float( 0.26 ), float( 1 ), ex.mul( ex ) ) ).mul( bright.mul( 0.6 ).add( 0.7 ) );

		} )();

		mat.normalNode = Fn( () => {

			const yaw = iDat.x;
			const cyw = cos( yaw ), syw = sin( yaw );
			const sy = abs( iDat.y );
			// B is written where A has coverage: un-premultiply the filtered edges by A's coverage
			const nl = vB.xyz.div( max( vA.w, 1e-3 ) ).mul( 2 ).sub( 1 );
			// local -> world: inverse-transpose of the stretch, then the yaw rotation
			const ns = vec3( nl.x, nl.y.div( sy ), nl.z );
			const nw = normalize( vec3( ns.x.mul( cyw ).add( ns.z.mul( syw ) ), ns.y, ns.z.mul( cyw ).sub( ns.x.mul( syw ) ) ) );
			return normalize( cameraViewMatrix.mul( vec4( nw, 0 ) ).xyz.add( positionViewDirection.mul( 0.12 ) ) );

		} )();

		mat.roughnessNode = float( 0.85 );
		// backlit crowns glow at the edges (light through the leaves), like the near canopy
		mat.translucencyNode = ( lightColor ) => translucency( mat.colorNode, normalWorldGeometry.mul( faceDirection ), 0.35, lightColor );
		mat.metalnessNode = float( 0 );
		return mat;

	}

}

// Quad geometry for the impostor instances (corners at +-1)
export function buildImpostorQuad() {

	const g = new THREE.BufferGeometry();
	g.setAttribute( 'position', new THREE.Float32BufferAttribute( [ - 1, - 1, 0, 1, - 1, 0, 1, 1, 0, - 1, 1, 0 ], 3 ) );
	g.setAttribute( 'normal', new THREE.Float32BufferAttribute( [ 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1 ], 3 ) );
	g.setIndex( [ 0, 1, 2, 0, 2, 3 ] );
	g.boundingSphere = new THREE.Sphere( new THREE.Vector3(), 1e7 );
	return g;

}

