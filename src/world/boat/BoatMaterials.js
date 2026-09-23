import * as THREE from 'three/webgpu';
import {
	attribute, positionLocal, positionView, normalView, faceDirection, uv, float, vec2, vec3, color,
	mix, smoothstep, step, fract, floor, sin, cos, abs, max, saturate, length, dot, fwidth, mod,
	mx_noise_float, mx_fractal_noise_float, uniform, exp, atan, TWO_PI, mrt, vec4,
} from 'three/tsl';
import { standard, physical } from '../../materials/Materials.js';
import { G } from '../../core/Globals.js';

// Shared attribute accessors (see GeoKit: color = linear albedo, aux = rough, metal, pattern, anim)
const vColor = attribute( 'color', 'vec3' );
const aux = attribute( 'aux', 'vec4' );
const isPattern = ( id ) => step( abs( aux.z.sub( id ) ), 0.5 );
const hash21 = ( p ) => fract( sin( dot( p, vec2( 127.1, 311.7 ) ) ).mul( 43758.5453 ) );
const lstep = ( a, b, x ) => smoothstep( a, b, x ); // a < b always (WGSL requires ordered edges)
const invstep = ( a, b, x ) => float( 1 ).sub( smoothstep( a, b, x ) );

// Mikkelsen surface-gradient bump mapping from a procedural height in meters.
function bumpNormal( height ) {

	const dpdx = positionView.dFdx();
	const dpdy = positionView.dFdy();
	const n = normalView;
	const r1 = dpdy.cross( n );
	const r2 = n.cross( dpdx );
	const det = dpdx.dot( r1 ).mul( faceDirection );
	const grad = det.sign().mul( height.dFdx().mul( r1 ).add( height.dFdy().mul( r2 ) ) );
	return det.abs().mul( n ).sub( grad ).add( n.mul( 1e-12 ) ).normalize();

}

// Molded non-skid: jittered pebbles on a ~1.8 cm lattice (uv in meters), faded when sub-pixel.
function nonSkidHeight() {

	const q = uv().mul( 55 );
	const c = floor( q );
	const f = fract( q ).sub( 0.5 );
	const j = vec2( hash21( c ), hash21( c.add( 17.31 ) ) ).sub( 0.5 ).mul( 0.3 );
	const d = length( f.sub( j ) );
	const pebble = invstep( 0.16, 0.36, d );
	const fade = invstep( 0.3, 0.8, fwidth( q.x ) );
	return pebble.mul( fade );

}

export class BoatMaterials {

	constructor( hullShape ) {

		this.navOn = uniform( 1 );
		this.flagPivot = uniform( new THREE.Vector3() );
		this.flagDir = uniform( new THREE.Vector3( 0, 0, - 1 ) );
		this.flagWind = uniform( 0.5 );

		this.hull = this.createHull( hullShape );
		this.gelcoat = this.createGelcoat();
		this.wood = this.createWood();
		this.fittings = this.createFittings();
		this.glass = this.createGlass();
		this.glow = this.createGlow();
		this.trap = this.createTrap();

	}

	// Hull exterior: antifouling / boot stripe / white topsides painted by height in the
	// boat frame (the hull mesh sits at the root with an identity transform).
	createHull( shape ) {

		const m = physical( { roughness: 0.25, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.1 } );
		m.name = 'boatHull';

		const p = positionLocal;
		const tS = saturate( p.z.sub( shape.zAft ).div( shape.length ) );
		const sheer = tS.pow( 2.2 ).mul( 0.62 ).add( 0.98 ).add( max( float( 1 ).sub( tS.div( 0.2 ) ), 0 ).pow( 2 ).mul( 0.04 ) );

		// boot top sweeps up slightly toward the bow
		const boot = p.y.sub( lstep( 0.8, 4.3, p.z ).mul( 0.06 ) );
		const aaB = fwidth( boot ).add( 1e-4 );
		const aboveBottom = lstep( aaB.negate(), aaB, boot.sub( 0.05 ) );
		const aboveStripe = lstep( aaB.negate(), aaB, boot.sub( 0.15 ) );

		const below = sheer.sub( p.y );
		const aaS = fwidth( below ).add( 1e-4 );
		const cove = lstep( aaS.negate(), aaS, below.sub( 0.1 ) ).mul( invstep( aaS.negate(), aaS, below.sub( 0.122 ) ) );

		const n1 = mx_fractal_noise_float( p.mul( vec3( 0.9, 2.4, 0.9 ) ), 3, 2.0, 0.5 );
		const streakN = mx_noise_float( vec3( p.z.mul( 7.0 ), p.y.mul( 0.45 ), p.x.mul( 7.0 ) ) );
		const scuffN = mx_noise_float( vec3( p.z.mul( 2.2 ), p.y.mul( 34 ), p.x.mul( 2.2 ) ) );

		let col = mix( color( 0x7a1d15 ), color( 0x0f1a30 ), aboveBottom );
		col = mix( col, color( 0xf2efe6 ), aboveStripe );
		col = mix( col, color( 0x0f1a30 ), cove );

		// waterline scum on the white, algae on the antifouling just below the boot top
		const scum = invstep( 0.16, n1.mul( 0.09 ).add( 0.42 ), boot ).mul( aboveStripe ).mul( saturate( n1.mul( 0.6 ).add( 0.7 ) ) );
		col = mix( col, color( 0x8b7b57 ), scum.mul( 0.32 ) );
		const algae = lstep( - 0.35, 0.03, boot ).mul( float( 1 ).sub( aboveBottom ) ).mul( saturate( n1.add( 0.45 ) ) );
		col = mix( col, color( 0x2c3a1f ), algae.mul( 0.45 ) );

		// faint vertical weathering streaks under the gunwale
		const streak = lstep( 0.3, 0.75, streakN ).mul( lstep( 0.05, 0.2, below ) ).mul( invstep( 0.4, 0.95, below ) ).mul( aboveStripe );
		col = mix( col, color( 0xa29579 ), streak.mul( 0.2 ) );

		// scuffs where traps come over the rail (starboard, aft of the wheelhouse)
		const side = lstep( - 0.2, 0.2, p.x.negate() ).mul( invstep( - 0.6, 0.2, p.z ) ).mul( lstep( - 3.7, - 2.8, p.z ) );
		const scuff = lstep( 0.45, 0.8, scuffN ).mul( lstep( 0.13, 0.18, below ) ).mul( invstep( 0.45, 0.7, below ) ).mul( side.mul( 0.8 ).add( 0.2 ) );
		col = mix( col, color( 0x5f5e59 ), scuff.mul( 0.55 ) );

		m.colorNode = col;
		m.roughnessNode = mix( mix( float( 0.75 ), float( 0.35 ), aboveBottom ), float( 0.2 ), aboveStripe ).add( scum.mul( 0.25 ) ).add( scuff.mul( 0.3 ) );
		m.clearcoatNode = aboveBottom.mul( float( 1 ).sub( scum.mul( 0.5 ) ) ).mul( float( 1 ).sub( scuff.mul( 0.6 ) ) );
		m.clearcoatRoughnessNode = float( 0.08 ).add( scum.mul( 0.3 ) );
		return m;

	}

	// White fiberglass (deck, lining, house, console); pattern 1 = molded non-skid.
	createGelcoat() {

		const m = standard( { roughness: 0.35, metalness: 0 } );
		m.name = 'boatGelcoat';
		const grip = isPattern( 1 );
		const h = nonSkidHeight().mul( grip );
		const p = positionLocal;
		const dirtN = mx_noise_float( p.mul( 1.7 ) );
		const dirt = saturate( dirtN.mul( 0.5 ).add( 0.35 ) ).mul( grip ).mul( 0.16 );
		// a little grime where walls meet the sole
		const corner = invstep( 0.35, 0.5, p.y ).mul( 0.08 ).mul( float( 1 ).sub( grip ) );
		m.colorNode = vColor.mul( float( 1 ).sub( h.mul( 0.07 ) ) ).mul( float( 1 ).sub( dirt ).sub( corner ) );
		m.roughnessNode = mix( aux.x, float( 0.72 ), grip ).add( h.mul( 0.1 ) );
		m.metalnessNode = aux.y;
		m.normalNode = bumpNormal( h.mul( 0.0008 ) );
		return m;

	}

	// Varnished teak/mahogany; grain follows uv.x. Pattern 1 = plain (brass, paint).
	createWood() {

		const m = standard( { roughness: 0.35, metalness: 0 } );
		m.name = 'boatWood';
		const w = uv();
		const g1 = mx_noise_float( vec3( w.x.mul( 0.8 ), w.y.mul( 30 ), 0.37 ) );
		const g2 = mx_noise_float( vec3( w.x.mul( 10 ), w.y.mul( 170 ), 5.1 ) );
		const rings = sin( w.y.mul( 150 ).add( g1.mul( 5 ) ).add( w.x.mul( 0.6 ) ) ).mul( 0.5 ).add( 0.5 );
		const grain = saturate( rings.mul( 0.45 ).add( g2.mul( 0.22 ) ).add( g1.mul( 0.22 ) ).add( 0.28 ) );
		const plain = isPattern( 1 );
		const wood = mix( color( 0x4a230f ), color( 0x9c5b2b ), grain );
		m.colorNode = mix( wood, vec3( 1 ), plain ).mul( vColor );
		m.roughnessNode = aux.x.add( g2.mul( 0.04 ).mul( float( 1 ).sub( plain ) ) );
		m.metalnessNode = aux.y;
		return m;

	}

	// Everything else opaque: stainless, bronze, painted metal, plastics, rope, vinyl, flag.
	// Pattern 1 = laid rope, 2 = flag (animated), 3 = whip antenna (animated sway).
	createFittings() {

		const m = standard( { roughness: 0.5, metalness: 0 } );
		m.name = 'boatFittings';
		const u = uv();

		const strand = sin( u.x.div( 0.07 ).add( u.y ).mul( TWO_PI.mul( 3 ) ) );
		const ropeShade = lstep( - 0.7, 0.7, strand ).mul( 0.4 ).add( 0.6 );

		const stripeIdx = floor( saturate( u.y ).mul( 12.999 ) );
		const red = float( 1 ).sub( mod( stripeIdx, 2 ) );
		const canton = step( u.x, 0.4 ).mul( step( 6 / 13, u.y ) );
		const sx = fract( u.x.div( 0.4 ).mul( 6 ) ).sub( 0.5 ), sy = fract( u.y.sub( 6 / 13 ).div( 7 / 13 ).mul( 5 ) ).sub( 0.5 );
		const star = invstep( 0.16, 0.24, length( vec2( sx, sy ) ) );
		const stripes = mix( color( 0xf4f1ea ), color( 0xb3172a ), red );
		const flag = mix( stripes, mix( color( 0x1c2a5c ), color( 0xf4f1ea ), star ), canton );

		let col = mix( vColor, vColor.mul( ropeShade ), isPattern( 1 ) );
		col = mix( col, flag, isPattern( 2 ) );
		m.colorNode = col;
		m.roughnessNode = aux.x;
		m.metalnessNode = aux.y;

		// vertex animation
		const p = positionLocal;
		const t = G.time;
		const wA = aux.w.mul( aux.w );
		const phase = p.x.mul( 3.1 ).add( p.z.mul( 1.7 ) );
		const gust = G.windSpeed.mul( 0.06 ).add( 0.5 );
		const sway = vec3( sin( t.mul( 1.9 ).add( phase ) ), 0, sin( t.mul( 1.37 ).add( phase.mul( 1.3 ) ) ).mul( 0.6 ) ).mul( wA.mul( 0.12 ).mul( gust ) );

		const rel = p.sub( this.flagPivot );
		const along = max( rel.z.negate(), 0 );
		const fu = aux.w; // 0 at the hoist .. 1 at the fly
		const droop = float( 1 ).sub( this.flagWind ).mul( 1.15 ).mul( fu.mul( 0.5 ).add( 0.5 ) );
		const lat = vec3( this.flagDir.z, 0, this.flagDir.x.negate() );
		const flutter = sin( fu.mul( 9 ).sub( t.mul( this.flagWind.mul( 9 ).add( 4 ) ) ).add( rel.y.mul( 4 ) ) )
			.mul( fu ).mul( this.flagWind.mul( 0.05 ).add( 0.015 ) );
		const flagPos = vec3( this.flagPivot.x, 0, this.flagPivot.z )
			.add( this.flagDir.mul( along.mul( cos( droop ) ) ) )
			.add( vec3( 0, rel.y.sub( along.mul( sin( droop ) ) ), 0 ) )
			.add( lat.mul( flutter ) );

		m.positionNode = mix( p.add( sway.mul( isPattern( 3 ) ) ), flagPos, isPattern( 2 ) );
		return m;

	}

	createGlass() {

		const m = physical( {
			color: 0xa9bec4, roughness: 0.05, metalness: 0, ior: 1.5,
			transparent: true, opacity: 0.25, side: THREE.DoubleSide, depthWrite: false,
		} );
		m.name = 'boatGlass';
		const u = uv();
		const spots = lstep( 0.45, 0.85, mx_noise_float( vec3( u.mul( 38 ), 3.3 ) ) );
		const haze = saturate( mx_noise_float( vec3( u.mul( 4 ), 9.1 ) ).mul( 0.5 ).add( 0.5 ) );
		const edge = invstep( 0.0, 0.18, u.y );
		const salt = saturate( spots.mul( 0.6 ).add( haze.mul( 0.25 ) ).mul( edge.mul( 0.8 ).add( 0.35 ) ) );
		m.opacityNode = float( 0.16 ).add( salt.mul( 0.22 ) );
		m.roughnessNode = float( 0.03 ).add( salt.mul( 0.35 ) );
		// the temporal resolve must reproject what is seen through the glass (the sea sliding past),
		// not the glass, which rides along with the helm camera: leave the velocity target untouched
		// (alpha 0 under the blending)
		m.mrtNode = mrt( { velocity: vec4( 0 ) } );
		return m;

	}

	// Emissive parts. aux.z selects: 0 nav light, 1 radar display, 2 chart plotter,
	// 3 gauge dial, 4 flood/spot light, 5 cabin dome light, 6 LCD.
	createGlow() {

		const m = standard( { color: 0x000000, roughness: 0.3, metalness: 0 } );
		m.name = 'boatGlow';
		const mode = aux.z;
		const is = ( id ) => step( abs( mode.sub( id ) ), 0.5 );
		const t = G.time;
		const night = G.night;
		const u = uv();

		// radar: head-up PPI with a 24 rpm sweep
		const q = u.sub( 0.5 ).mul( 2 );
		const r = length( q );
		const ang = atan( q.x, q.y );
		const da = fract( t.mul( 2.513 ).sub( ang ).div( TWO_PI ) );
		const trail = exp( da.mul( - 5 ) );
		const beam = invstep( 0.0, 0.01, da );
		const ringD = abs( fract( r.mul( 3 ).add( 0.5 ) ).sub( 0.5 ) );
		const ring = invstep( 0.012, 0.03, ringD );
		const landN = mx_fractal_noise_float( vec3( q.mul( 2.3 ).add( vec2( 1.7, 0.4 ) ), 0.5 ), 3, 2.0, 0.5 );
		const land = lstep( 0.18, 0.4, landN.add( q.x.mul( 0.35 ) ) ).mul( lstep( 0.3, 0.45, r ) );
		const cell = floor( q.mul( 7 ) );
		const blip = step( 0.93, hash21( cell ) ).mul( invstep( 0.1, 0.3, length( fract( q.mul( 7 ) ).sub( 0.5 ) ) ) ).mul( lstep( 0.2, 0.3, r ) );
		const heading = invstep( 0.004, 0.012, abs( q.x ) ).mul( step( 0, q.y ) );
		const inside = invstep( 0.96, 0.99, r );
		const echoes = saturate( land.add( blip ) ).mul( trail.mul( 0.75 ).add( 0.25 ) );
		const radar = vec3( 0.0, 0.012, 0.04 )
			.add( vec3( 0.05, 0.22, 0.28 ).mul( ring.mul( 0.5 ).add( heading.mul( 0.6 ) ) ) )
			.add( vec3( 1.0, 0.72, 0.12 ).mul( echoes ) )
			.add( vec3( 0.15, 0.9, 0.35 ).mul( beam.mul( 0.8 ).add( trail.mul( 0.08 ) ) ) )
			.mul( inside ).add( vec3( 0.01, 0.015, 0.02 ) ).mul( 1.3 );

		// chart plotter
		const cp = u.mul( vec2( 1.35, 1 ) );
		const cn = mx_fractal_noise_float( vec3( cp.mul( 2.6 ).add( vec2( 3.1, 1.2 ) ), 2.2 ), 4, 2.0, 0.5 );
		const shore = cn.add( u.x.sub( 0.62 ).mul( 1.1 ) );
		const isLand = lstep( 0.0, 0.015, shore );
		const shallow = lstep( - 0.25, 0.0, shore );
		const contour = invstep( 0.02, 0.05, abs( fract( shore.mul( 9 ) ).sub( 0.5 ) ) ).mul( float( 1 ).sub( isLand ) );
		const water = mix( vec3( 0.2, 0.42, 0.78 ), vec3( 0.62, 0.82, 0.97 ), shallow );
		const track = invstep( 0.003, 0.007, abs( u.x.sub( 0.5 ).sub( u.y.sub( 0.5 ).mul( 0.25 ) ) ) ).mul( step( 0.5, u.y ) );
		const boatIcon = invstep( 0.012, 0.022, length( u.sub( vec2( 0.5, 0.5 ) ) ) );
		const chart = mix( water, vec3( 0.88, 0.8, 0.55 ), isLand )
			.mul( float( 1 ).sub( contour.mul( 0.35 ) ) )
			.add( vec3( 0.9, 0.1, 0.8 ).mul( track ) )
			.add( vec3( 1.0, 0.35, 0.05 ).mul( boatIcon ) )
			.mul( 0.85 );

		// gauge dial (backlit ticks + needle)
		const gq = u.sub( 0.5 ).mul( 2 );
		const gr = length( gq );
		const ga = atan( gq.x, gq.y );
		const tickF = abs( fract( ga.div( TWO_PI ).mul( 24 ) ).sub( 0.5 ) );
		const ticks = invstep( 0.08, 0.15, tickF ).mul( lstep( 0.72, 0.76, gr ) ).mul( invstep( 0.86, 0.9, gr ) ).mul( invstep( 2.3, 2.4, abs( ga ) ) );
		const needleA = sin( t.mul( 0.7 ).add( positionLocal.x.mul( 13 ) ) ).mul( 0.06 ).add( positionLocal.x.mul( 3.7 ) ).add( 0.3 );
		const nd = vec2( sin( needleA ), cos( needleA ) );
		const along = dot( gq, nd );
		const perp = length( gq.sub( nd.mul( along ) ) );
		const needle = invstep( 0.025, 0.045, perp ).mul( step( - 0.1, along ) ).mul( invstep( 0.68, 0.72, along ) );
		const backlight = night.mul( 1.4 ).add( 0.25 );
		const gauge = vec3( 0.9, 0.95, 1.0 ).mul( ticks ).add( vec3( 1.0, 0.45, 0.08 ).mul( needle ) ).mul( backlight ).add( vec3( 0.004 ) );

		const nav = vColor.mul( this.navOn ).mul( night.mul( 7 ).add( 1.5 ) );
		const flood = vColor.mul( night.mul( 9 ).add( 0.02 ) );
		const dome = vColor.mul( night.mul( 2.2 ).add( 0.02 ) );
		const lcd = vColor.mul( night.mul( 0.5 ).add( 0.45 ) );

		m.colorNode = vColor.mul( 0.06 );
		m.emissiveNode = nav.mul( is( 0 ) )
			.add( radar.mul( is( 1 ) ) )
			.add( chart.mul( is( 2 ) ) )
			.add( gauge.mul( is( 3 ) ) )
			.add( flood.mul( is( 4 ) ) )
			.add( dome.mul( is( 5 ) ) )
			.add( lcd.mul( is( 6 ) ) );
		m.roughnessNode = aux.x;
		return m;

	}

	// Vinyl-coated wire traps: alpha-tested mesh. uv in meters, aux.xy = face size
	// (for the solid frame border), pattern 1 = diamond twine netting.
	createTrap() {

		const m = standard( { roughness: 0.55, metalness: 0, side: THREE.DoubleSide, alphaTest: 0.5 } );
		m.name = 'boatTrap';
		const u = uv();
		const net = isPattern( 1 );
		const diag = vec2( u.x.add( u.y ), u.x.sub( u.y ) ).mul( 0.7071 );
		const q = mix( u.div( 0.038 ), diag.div( 0.05 ), net );
		const f = abs( fract( q ).sub( 0.5 ) );
		const fw = fwidth( q );
		const half = mix( float( 0.5 - 0.0045 / 0.038 ), float( 0.5 - 0.002 / 0.05 ), net );
		const lineX = lstep( half.sub( fw.x ), half, f.x );
		const lineY = lstep( half.sub( fw.y ), half, f.y );
		const b = 0.014;
		const inner = step( b, u.x ).mul( step( b, u.y ) ).mul( step( u.x, aux.x.sub( b ) ) ).mul( step( u.y, aux.y.sub( b ) ) );
		const border = float( 1 ).sub( inner ).mul( float( 1 ).sub( net ) );
		m.opacityNode = max( max( lineX, lineY ), border );
		m.colorNode = vColor.mul( mix( float( 1 ), float( 0.8 ), border ) );
		return m;

	}

	setNavLights( on ) {

		this.navOn.value = on ? 1 : 0;

	}

	dispose() {

		for ( const k of [ 'hull', 'gelcoat', 'wood', 'fittings', 'glass', 'glow', 'trap' ] ) this[ k ].dispose();

	}

}
