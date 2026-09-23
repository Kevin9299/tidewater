// Procedural humpback skin (TSL), evaluated once offline by bake_textures.mjs into the texture
// atlas: albedo (linear), roughness and relief height (m). Inputs are the rest-pose position P
// (whale frame, metres), the part id and the shading coordinates (body: s along the length,
// ring 0 ventral .. 0.5 dorsal .. 1; flippers: span r, airfoil loop 0 TE-upper .. 0.5 LE .. 1
// TE-lower; flukes: x / half span, same loop).
import {
	Fn, float, vec3, vec4, mix, smoothstep, sin, cos, abs, min, max, clamp, saturate, select, If,
	mx_noise_float, mx_worley_noise_float, mx_worley_noise_vec2, floor, fract, dot, pow, exp, length,
} from 'three/tsl';

const TL = 14.5;

export function makeSkin( manifest ) {

	const snoutY = float( manifest.snoutY );
	// lip (mouth) line height over s (fit of anatomy.LIP), metres in the whale frame
	const lipY = ( s ) => snoutY.add( float( - 0.014 ).sub( s.mul( 0.07 ) ).sub( s.mul( s ).mul( 0.42 ) ).mul( TL ) );
	const worley2 = ( p ) => mx_worley_noise_vec2( p ); // (F1, F2)

	// one knob field: distance (m) to the nearest knob centre of a jittered 3D lattice
	const knobs = ( P, cell, seed ) => mx_worley_noise_float( P.div( cell ).add( seed ) ).mul( cell );

	// barnacle cluster: acorn barnacles (crowned rings with a dark centre) packed by a worley field,
	// pink-ish skin and whale lice around them
	const barnacles = ( P, cluster, out ) => {

		const w = worley2( P.mul( 26 ) ); // ~4 cm cells
		const f1 = w.x, edge = w.y.sub( w.x );
		const on = smoothstep( 0.05, 0.25, cluster ).toVar();
		const shell = smoothstep( 0.48, 0.32, f1 ).mul( on );
		const pit = smoothstep( 0.2, 0.08, f1 ).mul( on );
		const plates = saturate( sin( w.x.mul( 60 ) ).mul( 0.5 ).add( 0.5 ) );
		out.h.addAssign( shell.mul( 0.028 ).sub( pit.mul( 0.022 ) ).add( smoothstep( 0.0, 0.08, edge ).oneMinus().mul( on ).mul( - 0.007 ) ) );
		const col = mix( vec3( 0.52, 0.5, 0.45 ), vec3( 0.68, 0.66, 0.6 ), plates ).mul( float( 1 ).sub( pit.mul( 0.85 ) ) );
		// skin around the cluster: pink-ish, abraded; whale lice (orange) in the gaps
		const halo = smoothstep( 0.0, 0.2, cluster ).mul( float( 1 ).sub( on ) );
		out.c.assign( mix( out.c, vec3( 0.32, 0.2, 0.19 ), halo.mul( 0.55 ) ) );
		const lice = smoothstep( 0.35, 0.2, mx_worley_noise_float( P.mul( 70 ) ) ).mul( smoothstep( 0.1, 0.3, cluster ) ).mul( smoothstep( 0.2, 0.5, mx_noise_float( P.mul( 9 ).add( 4.2 ) ) ) );
		out.c.assign( mix( out.c, vec3( 0.55, 0.26, 0.12 ), lice.mul( 0.8 ) ) );
		out.h.addAssign( lice.mul( 0.003 ) );
		out.c.assign( mix( out.c, col, shell ) );
		out.r.assign( mix( out.r, 0.92, shell.max( lice ) ) );

	};

	const body = ( P, uv, out ) => {

		const s = uv.x, ring = uv.y;
		const up = ring.mul( Math.PI * 2 ).cos().negate(); // -1 ventral .. 1 dorsal
		const lat = min( ring, float( 1 ).sub( ring ) ); // 0 ventral midline .. 0.5 dorsal
		const n1 = mx_noise_float( P.mul( 0.45 ) ).toVar();
		const n2 = mx_noise_float( P.mul( 1.9 ).add( 7.1 ) ).toVar();
		const n3 = mx_noise_float( P.mul( 7.0 ).add( 3.3 ) ).toVar();
		const n4 = mx_noise_float( P.mul( 23.0 ).add( 1.7 ) ).toVar();
		// back: slate black with a faint blue-grey bloom and darker saddle; flanks a touch lighter
		const c = mix( vec3( 0.014, 0.016, 0.02 ), vec3( 0.032, 0.036, 0.044 ), saturate( n1.mul( 0.6 ).add( 0.45 ).add( n2.mul( 0.25 ) ) ) ).toVar();
		c.mulAssign( mix( 1.0, 1.35, smoothstep( 0.4, - 0.3, up ) ) );
		// throat and belly: white with black mottling and a ragged border (individual pattern)
		const border = float( - 0.05 ).add( n1.mul( 0.4 ) ).add( n2.mul( 0.18 ) ).add( n3.mul( 0.06 ) ).sub( smoothstep( 0.3, 0.62, s ).mul( 0.35 ) ).add( smoothstep( 0.1, 0.02, s ).mul( 0.2 ) );
		const along = smoothstep( 0.008, 0.04, s ).mul( float( 1 ).sub( smoothstep( 0.55, 0.72, s.add( n1.mul( 0.06 ) ) ) ) );
		const white = smoothstep( 0.05, - 0.05, up.sub( border ) ).mul( along ).toVar();
		const blot = smoothstep( 0.38, 0.28, mx_noise_float( P.mul( vec3( 1.4, 1.4, 0.7 ) ).add( 11.0 ) ).add( n3.mul( 0.08 ) ) );
		white.mulAssign( blot );
		const wcol = vec3( 0.62, 0.64, 0.64 ).mul( n3.mul( 0.06 ).add( n4.mul( 0.03 ) ).add( 0.94 ) );
		c.assign( mix( c, wcol, white ) );
		out.r.assign( float( 0.62 ).add( n3.mul( 0.06 ) ).add( white.mul( 0.05 ) ) );
		// ventral grooves: broad rounded ridges and narrow grooves, chin to umbilicus
		const pleatOn = smoothstep( 0.006, 0.02, s ).mul( float( 1 ).sub( smoothstep( 0.46, 0.53, s.add( n2.mul( 0.02 ) ) ) ) )
			.mul( float( 1 ).sub( smoothstep( 0.15, 0.2, lat.sub( s.mul( 0.1 ) ) ) ) ).toVar();
		const gph = lat.mul( Math.PI * 70 ).add( n2.mul( 0.6 ) ).add( s.mul( 3.0 ) );
		const g = abs( sin( gph ) );
		out.h.addAssign( g.pow( 0.3 ).sub( 1 ).mul( 0.042 ).mul( pleatOn ) );
		const groove = float( 1 ).sub( smoothstep( 0.0, 0.3, g ) ).mul( pleatOn );
		c.assign( mix( c, c.mul( 0.35 ), groove.mul( 0.7 ) ) );
		// lip line: dark crease; eye: eyelid folds
		const ly = lipY( s );
		const onLip = float( 1 ).sub( smoothstep( 0.235, 0.25, s ) ).mul( smoothstep( 0.005, 0.02, s ) ).mul( smoothstep( 0.05, 0.2, abs( P.x ) ) );
		const crease = exp( P.y.sub( ly ).div( 0.025 ).pow( 2 ).negate() ).mul( onLip );
		c.assign( mix( c, vec3( 0.006, 0.006, 0.007 ), crease.mul( 0.9 ) ) );
		out.h.addAssign( crease.mul( - 0.012 ) );
		// tubercles: rows along the rostrum (midline and above the lip) and scattered on the lower jaw
		const head = float( 1 ).sub( smoothstep( 0.16, 0.19, s ) ).mul( smoothstep( 0.0, 0.012, s ) );
		const dMid = abs( P.x ).div( 0.12 );
		const aboveLip = P.y.sub( ly );
		const rowMid = exp( dMid.pow( 2 ).negate() ).mul( up.greaterThan( 0 ).select( 1, 0 ) );
		const rowLip = exp( aboveLip.sub( 0.18 ).div( 0.12 ).pow( 2 ).negate() );
		const jaw = exp( aboveLip.add( 0.16 ).div( 0.16 ).pow( 2 ).negate() ).mul( float( 1 ).sub( smoothstep( 0.1, 0.14, s ) ) ).add( smoothstep( 0.05, 0.015, s ).mul( up.lessThan( 0.2 ).select( 1, 0 ) ) );
		const mask = saturate( rowMid.add( rowLip ).add( jaw ) ).mul( head ).toVar();
		const kd = knobs( P, 0.34, 3.7 ); // m to the nearest knob centre
		const knob = smoothstep( 0.085, 0.0, kd ).pow( 0.7 ).mul( mask ).toVar();
		const follicle = smoothstep( 0.014, 0.005, kd ).mul( mask );
		out.h.addAssign( knob.mul( 0.065 ).sub( follicle.mul( 0.018 ) ) );
		c.assign( mix( c, c.mul( 0.7 ).add( vec3( 0.01 ) ), knob.mul( 0.5 ) ) );
		// barnacles: chin, the front of the throat, genital region
		const cl = mx_noise_float( P.mul( 1.3 ).add( 5.0 ) ).add( n3.mul( 0.3 ) );
		const clusterZone = smoothstep( 0.06, 0.012, s ).mul( up.lessThan( 0.1 ).select( 1, 0 ) )
			.add( smoothstep( 0.03, 0.0, abs( s.sub( 0.12 ) ).sub( 0.03 ) ).mul( smoothstep( - 0.7, - 0.9, up ) ).mul( 0.7 ) )
			.add( smoothstep( 0.02, 0.0, abs( s.sub( 0.66 ) ).sub( 0.01 ) ).mul( smoothstep( - 0.75, - 0.9, up ) ).mul( 0.6 ) );
		barnacles( P, cl.mul( clusterZone ).sub( 0.1 ), out );
		// scars: rake marks (parallel white lines), cookie-cutter ovals, healed barnacle rings
		const scarZone = smoothstep( 0.22, 0.35, s ).mul( smoothstep( - 0.6, - 0.2, up ) ).toVar();
		const rakeDir = P.y.mul( 38 ).add( P.z.mul( 16 ) ).add( n2.mul( 2.5 ) );
		const rake = smoothstep( 0.92, 0.985, sin( rakeDir ) ).mul( smoothstep( 0.62, 0.74, mx_noise_float( P.mul( vec3( 0.6, 0.6, 0.3 ) ).add( 21.0 ) ) ) ).mul( scarZone );
		const oval = smoothstep( 0.09, 0.05, mx_worley_noise_float( P.mul( vec3( 1.9, 2.6, 1.9 ) ).add( 4.0 ) ) ).mul( scarZone ).mul( smoothstep( 0.2, 0.4, mx_noise_float( P.mul( 0.8 ).add( 8.0 ) ) ) );
		const ringScar = smoothstep( 0.02, 0.0, abs( mx_worley_noise_float( P.mul( 7 ).add( 2.0 ) ).sub( 0.28 ) ) ).mul( scarZone ).mul( smoothstep( 0.45, 0.6, mx_noise_float( P.mul( 1.1 ).add( 13.0 ) ) ) );
		const scar = saturate( rake.mul( 0.85 ).add( oval.mul( 0.8 ) ).add( ringScar.mul( 0.6 ) ) );
		c.assign( mix( c, vec3( 0.36, 0.37, 0.37 ).mul( n4.mul( 0.1 ).add( 0.95 ) ), scar ) );
		out.h.addAssign( scar.mul( - 0.002 ) );
		out.r.assign( mix( out.r, 0.74, scar ) );
		// skin creases: behind the flipper roots and across the tail stock
		const pecZone = smoothstep( 0.27, 0.3, s ).mul( float( 1 ).sub( smoothstep( 0.36, 0.4, s ) ) ).mul( smoothstep( - 0.85, - 0.55, up ) ).mul( float( 1 ).sub( smoothstep( 0.0, 0.3, up ) ) );
		const pcrease = pow( abs( sin( P.z.mul( 26 ).add( P.y.mul( 9 ) ).add( n2.mul( 2.2 ) ) ) ), 8 );
		const tailZone = smoothstep( 0.7, 0.76, s ).mul( float( 1 ).sub( smoothstep( 0.9, 0.94, s ) ) );
		const tcrease = pow( abs( sin( P.z.mul( 16 ).add( n2.mul( 3.0 ) ) ) ), 10 );
		out.h.addAssign( pcrease.mul( pecZone ).mul( - 0.008 ).add( tcrease.mul( tailZone ).mul( - 0.006 ) ) );
		c.assign( mix( c, c.mul( 0.7 ), pcrease.mul( pecZone ).add( tcrease.mul( tailZone ) ).mul( 0.5 ) ) );
		// fine skin: wrinkles running around the body, pores
		const wr = sin( P.z.mul( 42 ).add( mx_noise_float( P.mul( 3.0 ) ).mul( 6 ) ) ).mul( 0.5 ).add( 0.5 );
		const n5 = mx_noise_float( P.mul( 70.0 ).add( 5.5 ) );
		out.h.addAssign( wr.mul( 0.0018 ).add( n4.mul( 0.0016 ) ).add( n5.mul( 0.0008 ) ) );
		out.c.assign( c );

	};

	const flipper = ( P, uv, out ) => {

		const r = uv.x, loop = uv.y;
		const lower = loop.greaterThan( 0.5 );
		const le = float( 1 ).sub( abs( loop.sub( 0.5 ) ).mul( 2 ) ); // 1 at the leading edge
		const n1 = mx_noise_float( P.mul( 1.2 ) ).toVar();
		const n2 = mx_noise_float( P.mul( 4.6 ).add( 2.0 ) ).toVar();
		const n3 = mx_noise_float( P.mul( 19.0 ) ).toVar();
		// underside and leading edge white; upper side dark with white mottles spreading from the base
		const upWhite = smoothstep( 0.8, 0.93, le.add( n2.mul( 0.08 ) ) ).max( smoothstep( 0.3, 0.5, n1.add( n2.mul( 0.35 ) ).sub( r.mul( 0.5 ) ).add( 0.05 ) ) );
		const loWhite = smoothstep( - 0.55, - 0.25, n1.add( n2.mul( 0.4 ) ).add( 0.25 ) );
		const white = select( lower, loWhite, upWhite ).mul( smoothstep( 0.05, 0.15, r ) ).toVar();
		const c = mix( vec3( 0.016, 0.018, 0.022 ), vec3( 0.64, 0.66, 0.66 ).mul( n3.mul( 0.05 ).add( 0.95 ) ), white ).toVar();
		out.r.assign( float( 0.62 ).add( n2.mul( 0.05 ) ) );
		out.c.assign( c );
		// barnacles on the leading-edge knobs and near the tip
		const zone = smoothstep( 0.82, 0.95, le ).mul( smoothstep( 0.5, 0.75, r ) ).add( smoothstep( 0.9, 0.97, r ).mul( 0.8 ) );
		barnacles( P, mx_noise_float( P.mul( 2.2 ).add( 9.0 ) ).add( 0.25 ).mul( zone ).sub( 0.12 ), out );
		out.h.addAssign( n3.mul( 0.0008 ) );

	};

	const fluke = ( P, uv, out ) => {

		const q = abs( uv.x ), loop = uv.y;
		const lower = loop.greaterThan( 0.5 );
		const n1 = mx_noise_float( P.mul( 1.0 ).add( 30.0 ) ).toVar();
		const n2 = mx_noise_float( P.mul( 4.0 ) ).toVar();
		const n3 = mx_noise_float( P.mul( 17.0 ) ).toVar();
		// underside: white, black trailing margin with dark points, black streaks from the notch,
		// scattered black spots and scratches (every whale's own pattern)
		const te = smoothstep( 0.9, 0.975, loop ).add( smoothstep( 0.1, 0.025, loop ) );
		const notch = smoothstep( 0.12, 0.03, q.add( n2.mul( 0.02 ) ) );
		const streak = smoothstep( 0.55, 0.7, sin( q.mul( 40 ).add( n1.mul( 4 ) ) ) ).mul( smoothstep( 0.3, 0.55, n1 ) );
		const spots = smoothstep( 0.1, 0.05, mx_worley_noise_float( P.mul( 3.5 ).add( 2.0 ) ) ).mul( smoothstep( 0.2, 0.4, n2 ) );
		const white = select( lower, float( 1 ).sub( saturate( te.mul( 1.3 ).add( notch ).add( streak.mul( 0.8 ) ).add( spots ).add( n2.mul( 0.15 ) ) ) ), 0 ).toVar();
		const c = mix( vec3( 0.015, 0.017, 0.02 ), vec3( 0.66, 0.67, 0.65 ).mul( n3.mul( 0.05 ).add( 0.95 ) ), white ).toVar();
		out.r.assign( float( 0.62 ).add( n2.mul( 0.05 ) ) );
		out.c.assign( c );
		barnacles( P, smoothstep( 0.8, 0.95, q ).mul( mx_noise_float( P.mul( 3.0 ) ).add( 0.3 ) ).sub( 0.12 ), out );
		out.h.addAssign( n3.mul( 0.0007 ) );

	};

	// call inside a Fn: returns { color: vec4( albedo, roughness ), height (m) }
	return ( P, part, uv ) => {

		const out = { c: vec3( 0 ).toVar( 'skC' ), r: float( 0.4 ).toVar( 'skR' ), h: float( 0 ).toVar( 'skH' ) };
		If( part.lessThan( 0.5 ), () => {

			body( P, uv, out );

		} ).ElseIf( part.lessThan( 2.5 ), () => {

			flipper( P, uv, out );

		} ).ElseIf( part.lessThan( 3.5 ), () => {

			fluke( P, uv, out );

		} ).Else( () => {

			// eye: very dark, glossy cornea
			out.c.assign( vec3( 0.004, 0.004, 0.005 ) );
			out.r.assign( 0.08 );

		} );
		return { color: vec4( out.c, out.r ), height: out.h };

	};

}
