import * as THREE from 'three/webgpu';
import {
	Fn, float, int, vec2, vec3, attribute, uniform, uniformArray, varyingProperty, positionLocal, normalLocal, texture,
	sin, cos, fract, floor, dot, mix, smoothstep, max, abs, length, normalize, cross, select,
} from 'three/tsl';
import { bayer4 } from '../../materials/LODFade.js';
import { G } from '../../core/Globals.js';
import { getDetailTexture } from '../terrain/DetailTextures.js';
import { LOBE_TABLE, TREE_VARIANTS, SHRUB_VARIANTS } from './PlantGeometry.js';

// Shared TSL building blocks for all vegetation: hashing, value noise, wind and the
// plant deformation used by palms, young palms, bananas and ferns.

// Position of the *main* camera. The shadow pass has its own cameraPosition, so all
// distance based effects (LOD split, fades) use this uniform to keep shadows consistent.
export const uCamPos = uniform( new THREE.Vector3( 0, 10, 0 ) ).setName( 'vegCamPos' );

// Integrated gust-field offset (m). Integrated on the CPU so that changes of wind speed
// or direction never make the gust pattern jump.
export const uGustOffset = uniform( new THREE.Vector2() ).setName( 'vegGustOffset' );

// Per-object LOD window (x: visible from, y: fade-out start, z: fade-out end).
// Read per draw so one material can serve meshes with different LOD ranges.
export const uLodRange = uniform( new THREE.Vector3( 0, 1e6, 1e6 ) )
	.onObjectUpdate( ( { object } ) => object.userData.lodRange || _noLod )
	.setName( 'vegLodRange' );
const _noLod = new THREE.Vector3( 0, 1e6, 1e6 );

export const UP = vec3( 0, 1, 0 );

// Dave Hoskins' sine-free hash, [0, 1)
export const hash12 = /*@__PURE__*/ Fn( ( [ p ] ) => {

	const p3 = fract( vec3( p.x, p.y, p.x ).mul( 0.1031 ) ).toVar();
	p3.addAssign( dot( p3, p3.yzx.add( 33.33 ) ) );
	return fract( p3.x.add( p3.y ).mul( p3.z ) );

} ).setLayout( { name: 'vegHash12', type: 'float', inputs: [ { name: 'p', type: 'vec2' } ] } );

export const vnoise = /*@__PURE__*/ Fn( ( [ p ] ) => {

	const i = floor( p );
	const f = fract( p );
	const u = f.mul( f ).mul( f.mul( - 2 ).add( 3 ) );
	const a = hash12( i );
	const b = hash12( i.add( vec2( 1, 0 ) ) );
	const c = hash12( i.add( vec2( 0, 1 ) ) );
	const d = hash12( i.add( vec2( 1, 1 ) ) );
	return mix( mix( a, b, u.x ), mix( c, d, u.x ), u.y );

} ).setLayout( { name: 'vegNoise', type: 'float', inputs: [ { name: 'p', type: 'vec2' } ] } );

// Wind -------------------------------------------------------------------------------

// 0 (calm) .. 2.5 (storm). Tiny floor so nothing is ever perfectly frozen.
export const windStrength = G.windSpeed.mul( 0.1 ).max( 0.03 );
export const windDir3 = vec3( G.windDir.x, 0, G.windDir.y );
export const windPerp3 = vec3( G.windDir.y.negate(), 0, G.windDir.x );

// Travelling gust field in [0, 1]: noise of (worldXZ - windDir * t * speed), where the
// time-integrated offset comes from uGustOffset. Two fetches of the detail texture's fbm channel
// (~35 m and ~15 m gust cells); the terrain uses the same function for its wind sheen.
const detail = texture( getDetailTexture() );
export const gustAt = ( xz ) => {

	const p = xz.sub( uGustOffset );
	const n = detail.sample( p.div( 140 ) ).level( 0 ).w.mul( 0.62 ).add( detail.sample( p.div( 61 ).add( 0.37 ) ).level( 0 ).w.mul( 0.38 ) );
	return smoothstep( 0.46, 0.6, n );

};

// Rotation taking +Y to the unit vector T, applied to v.
export const rotUpTo = /*@__PURE__*/ Fn( ( [ v, T ] ) => {

	const k = vec3( T.z, 0, T.x.negate() );
	const c1 = cross( k, v );
	return v.add( c1 ).add( cross( k, c1 ).div( T.y.add( 1 ) ) );

} ).setLayout( { name: 'vegRotUpTo', type: 'vec3', inputs: [ { name: 'v', type: 'vec3' }, { name: 'T', type: 'vec3' } ] } );

// Crown variants: the per-instance variant picks a row of the lobe table (per-lobe size, 0 =
// dropped). Shared by the near canopy geometry and the impostor bake, so near and far crowns match.
const lobeTable = uniformArray( Array.from( LOBE_TABLE ), 'float' ).setName( 'vegLobeTable' );
export const variantOf = ( seed, isShrub ) => floor( fract( seed.mul( 7.77 ) ).mul( select( isShrub, float( SHRUB_VARIANTS ), float( TREE_VARIANTS ) ) ) );
export const lobeScale = ( seed, li, isShrub ) => {

	const row = variantOf( seed, isShrub ).add( select( isShrub, float( TREE_VARIANTS ), float( 0 ) ) );
	return lobeTable.element( int( row.mul( 8 ).add( max( li, 0 ) ) ) );

};

// LOD window factor for an instance at `base` (0 = hidden, 1 = full size). Hard switches (a window
// that starts at x > 0, or ends with z - y < 5 cm) are widened by a cross-fade band where both
// levels are drawn and dissolve into each other (lodDither, Bayer screen-door); soft windows shrink.
export const LOD_BAND = 0.12; // share of the switch distance
export const lodScale = ( base ) => {

	const d = length( uCamPos.sub( base ) );
	const inside = d.greaterThanEqual( uLodRange.x.mul( 1 - LOD_BAND / 2 ) );
	const hardEnd = uLodRange.z.sub( uLodRange.y ).lessThan( 0.05 );
	const endK = select( hardEnd, select( d.lessThan( uLodRange.y.mul( 1 + LOD_BAND / 2 ) ), float( 1 ), float( 0 ) ), float( 1 ).sub( smoothstep( uLodRange.y, uLodRange.z, d ) ) );
	return select( inside, endK, float( 0 ) );

};

// Fragment-stage keep test of the LOD cross-fade for an instance at `base` (true: keep the pixel)
export const lodDither = ( base ) => {

	const d = length( uCamPos.sub( base ) );
	const t = bayer4();
	const x = uLodRange.x, y = uLodRange.y;
	const fadeIn = select( x.greaterThan( 0 ), smoothstep( x.mul( 1 - LOD_BAND / 2 ), x.mul( 1 + LOD_BAND / 2 ), d ), float( 1 ) );
	const fadeOut = select( uLodRange.z.sub( y ).lessThan( 0.05 ), smoothstep( y.mul( 1 - LOD_BAND / 2 ), y.mul( 1 + LOD_BAND / 2 ), d ), float( 0 ) );
	return t.lessThan( fadeIn ).and( t.greaterThanEqual( fadeOut ) );

};

// Varyings written by the plant deformation.
export const vTrunkY = varyingProperty( 'float', 'vVegTrunkY' ); // height along the stem (m)
export const vTrunkT = varyingProperty( 'vec3', 'vVegTrunkT' ); // stem axis (world)

// Plant deformation shared by palms / young palms / bananas / ferns.
//
// Geometry conventions (local space, applied after the instance matrix = T * Ry * S):
//   aMat.x == 0  -> stem vertex: aVeg.x = height fraction u, x/z = radial offset.
//   aMat.x >= 1  -> crown vertex: position is relative to the crown centre.
//   aVeg = (u, s along frond, flutter weight, phase)
// Per instance:
//   iPos = (base.xyz, scale), iDat = (lean azimuth, lean (fraction of H), stem height H (m), seed)
export const plantDeform = /*@__PURE__*/ Fn( () => {

	// NOTE: every value shared between the stem and crown paths is materialised with
	// toVar() up front. TSL compiles select() with statement-producing branches into
	// if/else blocks, and a sub-expression first built inside one branch would otherwise
	// be unassigned (zero) when reused by the other branch.
	const iPos = attribute( 'iPos', 'vec4' );
	const iDat = attribute( 'iDat', 'vec4' );
	const veg = attribute( 'aVeg', 'vec4' );
	const aMat = attribute( 'aMat', 'vec4' );
	const part = aMat.x.toVar();

	const base = iPos.xyz.toVar();
	const sc = iPos.w.toVar();
	const leanDir = vec3( cos( iDat.x ), 0, sin( iDat.x ) ).toVar();
	const lean = iDat.y.toVar();
	const H = iDat.z.toVar();
	// merged geometries (understory): the instance's plant kind is the integer part of the seed;
	// vertices of the other plants collapse onto the base
	const seed = fract( iDat.w ).toVar();
	const kindI = floor( iDat.w ).toVar();
	const kindV = attribute( 'aLobe', 'vec4' ).w.negate().sub( 1 );
	const keepKind = kindV.lessThan( 0.5 ).or( abs( kindV.sub( kindI ) ).lessThan( 0.5 ) );

	const u = veg.x.toVar();
	const s = veg.y.toVar();
	const flut = veg.z.toVar();
	const ph = veg.w.toVar();

	const P = positionLocal.toVar();
	const N0 = normalLocal.toVar();
	const t = G.time;
	const w = windStrength.toVar();
	const g = gustAt( base.xz ).toVar();
	const ph0 = seed.mul( 6.2832 ).toVar();

	// stem sway: horizontal offset of the top as a fraction of H
	const sway = w.mul( w ).mul( 0.014 ).mul( g.mul( 0.9 ).add( 0.3 ) )
		.add( sin( t.mul( 0.83 ).add( ph0 ) ).mul( w ).mul( 0.0065 ).mul( g.add( 0.45 ) ) ).toVar();
	const swayP = sin( t.mul( 0.61 ).add( ph0.mul( 1.7 ) ) ).mul( w ).mul( 0.0028 ).toVar();
	const windOff = windDir3.mul( sway ).add( windPerp3.mul( swayP ) ).toVar();

	// stem curve: mix of a straight tilt and a "banana" curve (vertical at the top)
	const c = fract( seed.mul( 7.31 ) ).toVar();
	const f = mix( u, u.mul( u.oneMinus().add( 1 ) ), c );
	const df = mix( float( 1 ), u.oneMinus().mul( 2 ), c );
	const off = leanDir.mul( lean.mul( f ) ).add( windOff.mul( u.mul( u ) ) );
	const T = normalize( UP.add( leanDir.mul( lean.mul( df ) ) ).add( windOff.mul( u.mul( 2 ) ) ) ).toVar();

	const radial = vec3( P.x.sub( base.x ), 0, P.z.sub( base.z ) );
	const axisPt = base.add( vec3( 0, u.mul( H ), 0 ) ).add( off.mul( H ) );
	const stemPos = axisPt.add( rotUpTo( radial, T ) ).toVar();
	const stemN = rotUpTo( N0, T ).toVar();

	// crown: follows the stem top, tilted half as much as the stem tip
	const Ttop = normalize( UP.add( leanDir.mul( lean.mul( c.oneMinus() ) ) ).add( windOff.mul( 2 ) ) );
	const Ttilt = normalize( UP.add( Ttop ) ).toVar();
	const C = base.add( vec3( 0, H, 0 ) ).add( leanDir.mul( lean ).add( windOff ).mul( H ) ).toVar();
	const o1 = rotUpTo( P.sub( base ), Ttilt ).toVar();
	const N1 = rotUpTo( N0, Ttilt ).toVar();

	// frond / leaf motion: gust bending + slow bounce + leaflet flutter, applied as a
	// length preserving rotation about the crown centre so fronds stream downwind in storms.
	const s2 = s.mul( s ).toVar();
	const bend = s2.mul( sc ).mul( 4.5 ).mul(
		w.mul( w ).mul( 0.16 ).mul( g.mul( 0.8 ).add( 0.35 ) )
			.add( sin( t.mul( ph.mul( 0.5 ).add( 1.3 ) ).add( ph.mul( 23 ) ).add( ph0 ) ).mul( w ).mul( 0.075 ).mul( g.add( 0.3 ) ) ) ).toVar();
	const bounce = sin( t.mul( ph.add( 2.1 ) ).add( ph.mul( 41 ) ) ).mul( s2 ).mul( w ).mul( sc ).mul( 0.1 );
	const flutter = flut.mul( sc ).mul(
		sin( t.mul( ph.mul( 5 ).add( 11 ) ).add( ph.mul( 60 ) ).add( s.mul( 9 ) ) ).mul( w.mul( 0.045 ).add( 0.008 ) )
			.add( sin( t.mul( 23 ).add( ph.mul( 13 ) ).add( s.mul( 17 ) ) ).mul( max( w.sub( 1 ), 0 ).mul( 0.05 ) ) ) );
	const disp = windDir3.mul( bend ).add( UP.mul( bounce.sub( bend.mul( 0.3 ) ) ) ).add( N1.mul( flutter ) );
	const L = length( o1 );
	const o2 = normalize( o1.add( disp ).add( vec3( 0, 1e-5, 0 ) ) ).mul( L );
	const crownPos = C.add( o2 ).toVar();

	// the dead (hanging) frond is only kept on some palms
	const hideDead = part.greaterThan( 0.5 ).and( part.lessThan( 1.5 ) ).and( aMat.y.greaterThan( 0.95 ) ).and( fract( seed.mul( 13.7 ) ).greaterThan( 0.4 ) );
	const crownPos2 = select( hideDead, C, crownPos ).toVar();

	const isStem = part.lessThan( 0.5 );
	const pos = select( isStem, stemPos, crownPos2 ).toVar();

	normalLocal.assign( select( isStem, stemN, N1 ) );
	vTrunkY.assign( u.mul( H ) );
	vTrunkT.assign( T );

	// LOD window / distance fade: shrink around the base (ferns fade out earlier)
	const dCam = length( uCamPos.sub( base ) );
	const fernFade = select( kindI.greaterThan( 2.5 ), float( 1 ).sub( smoothstep( UNDER_FERN_FADE[ 0 ], UNDER_FERN_FADE[ 1 ], dCam ) ), float( 1 ) );
	const k = lodScale( base ).mul( fernFade ).mul( select( keepKind, float( 1 ), float( 0 ) ) );
	return base.add( pos.sub( base ).mul( k ) );

} );

// distance window (m) of the ferns inside the merged understory mesh
export const UNDER_FERN_FADE = [ 42, 58 ];
