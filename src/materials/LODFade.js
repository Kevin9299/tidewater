import { Fn, float, uvec2, screenCoordinate, frameId, Discard } from 'three/tsl';

// Screen-door cross-fade between levels of detail (and for anything that appears or disappears
// with distance), so nothing switches over in one frame.
//
// Both levels are drawn during a transition band. Each discards the pixels where the ordered-dither
// (Bayer 4x4) threshold says it isn't visible: the incoming level keeps the pixels below `fade`,
// the outgoing one the pixels at or above the SAME threshold, so together they cover every pixel
// exactly once (no holes, no double-drawn pixels). The pattern shifts every frame, so the TAA
// resolves the 16 dither levels into a smooth blend.
//
// fade: 0..1, the share of this level that is visible (the same value for both levels of one
// instance: ramp it with distance across the band on the CPU or in the vertex stage).

// Bayer 4x4 threshold in (0, 1): bit-interleaved formula of
//   0  8  2 10 / 12  4 14  6 / 3 11  1  9 / 15  7 13  5
export const bayer4 = Fn( () => {

	const f = frameId;
	// shift the pattern by a different offset every frame (all 16 over 16 frames)
	const p = uvec2( screenCoordinate.xy ).add( uvec2( f.mul( 3 ), f.shiftRight( 2 ).mul( 1 ) ) );
	const x0 = p.x.bitAnd( 1 ), x1 = p.x.shiftRight( 1 ).bitAnd( 1 );
	const y0 = p.y.bitAnd( 1 ), y1 = p.y.shiftRight( 1 ).bitAnd( 1 );
	const v = x0.bitXor( y0 ).shiftLeft( 3 ).bitOr( y0.shiftLeft( 2 ) ).bitOr( x1.bitXor( y1 ).shiftLeft( 1 ) ).bitOr( y1 );
	return float( v ).add( 0.5 ).div( 16 );

} );

// Call in the fragment stage (e.g. at the top of a colorNode / opacity Fn). Nothing is discarded
// at fade >= 1 (incoming) or fade <= 0 (outgoing).
export function lodFadeDiscard( fade, outgoing = false ) {

	const t = bayer4();
	Discard( outgoing ? t.lessThan( fade ) : t.greaterThanEqual( fade ) );

}

// Fade factor across a distance band [start, end] (0 before, 1 after), for the level that takes over
// at `end`. Use on the CPU when bucketing instances.
export function bandFade( dist, start, end ) {

	const t = ( dist - start ) / Math.max( end - start, 1e-6 );
	return t <= 0 ? 0 : t >= 1 ? 1 : t * t * ( 3 - 2 * t );

}
