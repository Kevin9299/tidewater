import * as THREE from 'three/webgpu';
import {
	float, vec2, vec3, normalize, mix, smoothstep, max, min, abs, dot, cross, dFdx, dFdy, sign, fwidth, length,
	positionWorld, texture, luminance,
} from 'three/tsl';

// Shared TSL building blocks for the terrain and the scattered rocks.

// sRGB triplet -> linear vec3 constant
export const srgb = ( r, g, b ) => {

	const c = new THREE.Color().setRGB( r, g, b, THREE.SRGBColorSpace );
	return vec3( c.r, c.g, c.b );

};

// 2D rotation of a vec2 node by a constant angle
export const rot2 = ( v, a ) => {

	const c = Math.cos( a ), s = Math.sin( a );
	return vec2( v.x.mul( c ).sub( v.y.mul( s ) ), v.x.mul( s ).add( v.y.mul( c ) ) );

};

// Mikkelsen surface-gradient bump: perturb world normal N by the scalar height field hd
// (screen-space derivatives, so any mix of projections / scales works). Robustness for terrain:
//  - the tilt is limited to ~55 degrees (at grazing angles |det| collapses and the unclamped
//    gradient would swing the normal into the tangent plane: 'chrome' patches on steep faces)
//  - the bump fades out where the screen-space frame is degenerate (the thin sliver triangles
//    of CDLOD geomorphing, which otherwise light up as bright lines along the grid) or where the
//    rendered facet disagrees with N (sub-texel crags)
export const perturbNormal = ( N, hd, scale = 1 ) => {

	const p = positionWorld;
	const dpdx = dFdx( p ), dpdy = dFdy( p );
	const dhdx = dFdx( hd ).mul( scale ), dhdy = dFdy( hd ).mul( scale );
	const r1 = cross( dpdy, N );
	const r2 = cross( N, dpdx );
	const det = dot( dpdx, r1 );
	const ad = abs( det );
	const grad = r1.mul( dhdx ).add( r2.mul( dhdy ) ).mul( sign( det ) );
	const area = length( cross( dpdx, dpdy ) );
	const frame = area.div( max( length( dpdx ).mul( length( dpdy ) ), 1e-20 ) ); // sin of the footprint angle
	const facet = ad.div( max( area, 1e-20 ) ); // cos between the facet and N
	const k = smoothstep( 0.12, 0.35, frame ).mul( smoothstep( 0.3, 0.6, facet ) );
	const g = grad.mul( min( float( 1 ), ad.mul( 1.4 ).div( max( length( grad ), 1e-20 ) ) ) ).mul( k );
	return normalize( N.mul( max( ad, 1e-20 ) ).sub( g ) );

};

// triplanar blend weights (sharp)
export const triWeights = ( N ) => {

	const a = abs( N );
	const w = a.mul( a ).mul( a.mul( a ) );
	return w.div( w.x.add( w.y ).add( w.z ) );

};

// one channel-set of the detail texture, triplanar. With `grad` ({ dpdx, dpdy } of the world
// position) the samples use explicit gradients so they may run in non-uniform control flow.
export const triplanar = ( tex, p, w, tile, grad = null ) => {

	const s = 1 / tile;
	const sample = ( uv, sw ) => grad ? texture( tex, uv ).grad( grad.dpdx[ sw ].mul( s ), grad.dpdy[ sw ].mul( s ) ) : texture( tex, uv );
	const x = sample( p.zy.mul( s ), 'zy' );
	const y = sample( p.xz.mul( s ).add( 0.37 ), 'xz' );
	const z = sample( p.xy.mul( s ).add( 0.71 ), 'xy' );
	return x.mul( w.x ).add( y.mul( w.y ) ).add( z.mul( w.z ) );

};

// ---- palette (sRGB picked from photo references, stored linear)
export const PALETTE = {
	rockDark: srgb( 0.15, 0.145, 0.135 ),
	rockMid: srgb( 0.3, 0.285, 0.265 ),
	rockLight: srgb( 0.48, 0.455, 0.42 ),
	rockWarm: srgb( 0.44, 0.37, 0.30 ),
	lichenPale: srgb( 0.70, 0.70, 0.64 ),
	lichenOrange: srgb( 0.78, 0.50, 0.20 ),
	blackZone: srgb( 0.075, 0.075, 0.07 ),
	barnacle: srgb( 0.78, 0.76, 0.70 ),
	algae: srgb( 0.20, 0.27, 0.10 ),
	coralline: srgb( 0.62, 0.44, 0.46 ),
	moss: srgb( 0.19, 0.29, 0.08 ),
	mossDry: srgb( 0.3, 0.34, 0.14 ),
};

// Weathered volcanic rock seen on the headlands, sea stacks and boulders.
//   p world position, N world normal (geometric / macro), h height above sea level
//   macro: 0..1 large scale variation, seed: per object variation (0..1)
// Returns { albedo, rough, hd (bump height, m), moss (0..1), wetness }
// With `grad` = { dpdx, dpdy, fwY } every sample uses explicit gradients (branch safe).
export function rockSurface( { tex, p, N, h, macro, seed = float( 0.5 ), mossAmount = float( 1 ), grad = null } ) {

	const w = triWeights( N );
	// big blocks (4 m cells), plates (0.9 m) and grain / chips
	const big = triplanar( tex, p, w, 27, grad );
	const mid = triplanar( tex, p, w, 6.1, grad );
	const fine = triplanar( tex, p, w, 1.3, grad );
	// pixel footprint (m): features smaller than a few pixels fade out instead of sparkling
	const px = grad ? max( length( grad.dpdx ), length( grad.dpdy ) ) : length( fwidth( p ) );
	const fineK = float( 1 ).sub( smoothstep( 0.006, 0.02, px ) ).toVar();
	const midK = float( 1 ).sub( smoothstep( 0.03, 0.1, px ) );
	const hr = big.x.mul( 0.45 ).add( mid.x.mul( 0.35 ) ).add( fine.x.sub( 0.5 ).mul( fineK ).add( 0.5 ).mul( 0.2 ) ).toVar();

	// layered lava flows / bedding on steep faces: irregular bands (1D lookup of the fbm channel
	// along the height, warped), faded out once a band gets thinner than a few pixels
	const steep = float( 1 ).sub( smoothstep( 0.55, 0.85, N.y ) );
	const bandY = p.y.add( mid.w.mul( 2.5 ) ).add( macro.mul( 6 ) );
	const fwY = grad ? grad.fwY : fwidth( bandY );
	const bandUV = vec2( bandY.div( 14 ), seed.mul( 0.37 ).add( 0.13 ) );
	const strata = grad ? texture( tex, bandUV ).grad( vec2( fwY.div( 14 ), 0 ), vec2( 0, 0 ) ).w : texture( tex, bandUV ).w;
	const strataAA = float( 1 ).sub( smoothstep( 0.15, 0.6, fwY ) );
	const tone = hr.mul( 0.9 ).add( macro.sub( 0.5 ).mul( 0.7 ) ).add( strata.sub( 0.5 ).mul( 0.8 ).mul( steep ).mul( strataAA ) ).add( seed.sub( 0.5 ).mul( 0.3 ) );
	let col = mix( PALETTE.rockDark, PALETTE.rockMid, smoothstep( 0.1, 0.5, tone ) );
	col = mix( col, PALETTE.rockLight, smoothstep( 0.5, 0.85, tone ) );
	// iron staining / warm weathering in patches
	col = mix( col, PALETTE.rockWarm, smoothstep( 0.62, 0.8, mid.w.add( macro.mul( 0.3 ) ) ).mul( 0.18 ) );
	// joints between the big blocks, fainter between plates
	col = col.mul( smoothstep( 0.05, 0.25, big.x ).mul( 0.35 ).add( 0.65 ) ).mul( smoothstep( 0.05, 0.25, mid.x ).mul( 0.15 ).add( 0.85 ) );
	// rain streaks: dark stains running down steep faces, paler bands between (the fbm channel
	// stretched vertically on the two vertical projection planes)
	const sUVa = vec2( p.z.div( 3.1 ), p.y.div( 41 ) ), sUVb = vec2( p.x.div( 3.1 ).add( 0.5 ), p.y.div( 41 ).add( 0.3 ) );
	const sa = grad ? texture( tex, sUVa ).grad( vec2( grad.dpdx.z.div( 3.1 ), grad.dpdx.y.div( 41 ) ), vec2( grad.dpdy.z.div( 3.1 ), grad.dpdy.y.div( 41 ) ) ) : texture( tex, sUVa );
	const sb = grad ? texture( tex, sUVb ).grad( vec2( grad.dpdx.x.div( 3.1 ), grad.dpdx.y.div( 41 ) ), vec2( grad.dpdy.x.div( 3.1 ), grad.dpdy.y.div( 41 ) ) ) : texture( tex, sUVb );
	const sw4 = N.xz.abs().pow( vec2( 4 ) );
	const stainS = sa.w.mul( sw4.x ).add( sb.w.mul( sw4.y ) ).div( sw4.x.add( sw4.y ).add( 1e-5 ) );
	const stain = smoothstep( 0.52, 0.72, stainS ).mul( steep );
	col = col.mul( float( 1 ).sub( stain.mul( 0.4 ) ) ).mul( smoothstep( 0.35, 0.2, stainS ).mul( steep ).mul( 0.12 ).add( 1 ) );

	// lichens on the dry upper faces
	const dry = smoothstep( 2.6, 4.0, h );
	const lichen = smoothstep( 0.6, 0.78, mid.y ).mul( smoothstep( 0.2, 0.7, N.y ) ).mul( dry ).mul( smoothstep( 0.45, 0.65, macro ) );
	col = mix( col, PALETTE.lichenPale, lichen.mul( 0.45 ) );
	col = mix( col, PALETTE.lichenOrange, smoothstep( 0.8, 0.88, mid.y ).mul( dry ).mul( smoothstep( 0.5, 0.8, N.y ) ).mul( 0.3 ) );

	// moss / grass on ledges and tops, ferns hanging along the bedding planes of steep faces
	const ledge = smoothstep( 0.55, 0.7, strata ).mul( steep ).mul( strataAA ).mul( smoothstep( 0.4, 0.6, mid.w.add( macro.sub( 0.5 ).mul( 0.4 ) ) ) );
	const moss = max( smoothstep( 0.62, 0.9, N.y.add( big.x.sub( 0.5 ).mul( 0.5 ) ).add( macro.sub( 0.5 ).mul( 0.3 ) ) ), ledge.mul( 0.8 ) )
		.mul( smoothstep( 2.5, 5.0, h ) ).mul( mossAmount ).toVar();
	col = mix( col, mix( PALETTE.moss, PALETTE.mossDry, mid.w ), moss.mul( 0.9 ) );

	// shoreline zonation: black lichen band (splash zone), barnacles and algae in the intertidal
	const splash = smoothstep( 0.5, 1.0, h ).mul( smoothstep( 2.8, 1.8, h.add( mid.w.mul( 1.2 ) ) ) ).mul( smoothstep( 0.35, 0.6, macro.add( mid.w.mul( 0.3 ) ) ) );
	col = mix( col, PALETTE.blackZone, splash.mul( 0.55 ) );
	// sun-bleached, weathered upper faces
	col = mix( col, PALETTE.rockLight, smoothstep( 0.35, 0.95, N.y ).mul( smoothstep( 1.5, 3.0, h ) ).mul( 0.3 ) );
	const inter = smoothstep( - 0.7, - 0.2, h ).mul( smoothstep( 0.7, 0.2, h ) );
	const barn = smoothstep( 0.62, 0.72, fine.z ).mul( inter ).mul( fineK );
	col = mix( col, PALETTE.algae, inter.mul( smoothstep( 0.4, 0.6, mid.y ) ).mul( 0.6 ) );
	col = mix( col, PALETTE.barnacle, barn.mul( 0.8 ) );
	// below the water: algae films and pink coralline crusts
	const sub = smoothstep( - 0.3, - 1.2, h );
	col = mix( col, mix( PALETTE.algae, PALETTE.coralline, smoothstep( 0.45, 0.7, mid.w ) ), sub.mul( 0.55 ) );

	// wet below the swash line (dark, glossy)
	const wet = smoothstep( 1.0, 0.25, h.add( mid.w.mul( 0.3 ) ) ).toVar();
	col = col.mul( mix( float( 1 ), float( 0.55 ), wet ) );

	const rough = mix( mix( float( 0.88 ), float( 0.8 ), steep ), float( 0.45 ), wet ).add( moss.mul( 0.06 ) );
	// relief (m): tilted blocks and plates with bevelled joints, then grain
	const hd = big.x.mul( 0.25 ).add( mid.x.mul( 0.07 ).mul( midK.mul( 0.6 ).add( 0.4 ) ) ).add( fine.x.mul( 0.012 ).mul( fineK ).mul( float( 1 ).sub( wet.mul( 0.6 ) ) ) ).add( barn.mul( 0.005 ) );
	return { albedo: col, rough, hd, moss, wet, height: hr };

}

// saturation helper
export const saturation = ( c, s ) => mix( vec3( luminance( c ) ), c, s );

// ---- tropical meadow (tall guinea / elephant grass)
// The tone is shared by the terrain and the grass field (blade base colour): both evaluate it from
// the same inputs, so the geometric grass fades into the ground without a visible boundary.
export const MEADOW = {
	lush: srgb( 0.13, 0.2, 0.05 ),
	green: srgb( 0.25, 0.32, 0.1 ),
	olive: srgb( 0.36, 0.37, 0.14 ),
	yellow: srgb( 0.5, 0.46, 0.2 ),
	straw: srgb( 0.62, 0.54, 0.33 ),
	soil: srgb( 0.17, 0.13, 0.08 ),
};

//   mA, mB: detail fbm channel at the 173 m / 47 m scales (~0.5 +- 0.1): the samples at
//   rot2( xz, 0.7 ) / 173 and rot2( xz, 2.1 ) / 47 that the terrain takes anyway; slope: 1 - N.y;
//   south: N.z (the sun side); detail: optional finer fbm (~0.5 +- 0.1) that breaks up the patches
// Returns { tone, dry, lush }: mostly fresh green grass with olive, sun-bleached yellow and a few
// straw-dry patches (more on exposed slopes), darker lush grass in the damp patches (the hollows
// are darkened further by the AO).
export const meadowTone = ( mA, mB, slope, south, detail = null ) => {

	let m = mA.mul( 0.55 ).add( mB.mul( 0.45 ) ).add( slope.mul( 0.25 ) ).add( south.mul( 0.04 ) );
	if ( detail ) m = m.add( detail.sub( 0.5 ).mul( 0.28 ) );
	const olive = smoothstep( 0.52, 0.6, m );
	const yellow = smoothstep( 0.6, 0.67, m );
	const straw = smoothstep( 0.66, 0.73, m.add( mB.sub( 0.5 ).mul( 0.2 ) ) );
	const lush = smoothstep( 0.46, 0.37, mB.mul( 0.7 ).add( mA.mul( 0.3 ) ).add( slope.mul( 0.2 ) ).add( detail ? detail.sub( 0.5 ).mul( 0.2 ) : 0 ) );
	let c = mix( MEADOW.green, MEADOW.olive, olive );
	c = mix( c, MEADOW.yellow, yellow.mul( 0.8 ) );
	c = mix( c, MEADOW.straw, straw.mul( 0.55 ) );
	c = mix( c, MEADOW.lush, lush.mul( 0.75 ) );
	return { tone: c, dry: olive.mul( 0.4 ).add( yellow.mul( 0.6 ) ), lush };

};
