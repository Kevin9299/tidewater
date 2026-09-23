import * as THREE from 'three/webgpu';
import {
	slab, loft, fanCap, box, roundedBox, cylinder, rod, sphere, torus, lathe, tube, prepare, mat4, alignY,
	auxVertices, paintVertices, mergePrepared,
} from './GeoKit.js';
import { houseHalfWidth, foredeckY, PALETTE } from './HullBuilder.js';
import { lerp } from './HullLines.js';
import { buoyGeometry } from './DeckGear.js';

const V = ( x, y, z ) => new THREE.Vector3( x, y, z );

// Wheelhouse dimensions (boat frame)
export const HOUSE = {
	wallT: 0.045,
	wsBottomY: 1.48, // windshield base
	wsTopY: 2.24,
	roofUnderY: 2.31,
	roofZ0: - 0.95,
	roofZ1: 1.34,
	winBottom: 1.56,
	winTop: 2.12,
	dash: { zFace: 0.98, yKnee: 1.08, zTop: 1.16, yTop: 1.4, zBack: 1.41, halfW: 1.1 },
	helmX: - 0.55,
	seatZ: 0.22,
};

const STAINLESS = { color: PALETTE.stainless, rough: 0.22, metal: 1 };
const BLACK_PLASTIC = { color: 0x1a1b1d, rough: 0.55, metal: 0 };
const WHITE_PAINT = { color: 0xf1f0eb, rough: 0.35, metal: 0 };
const FRAME = { color: 0xb9bdc1, rough: 0.4, metal: 1 };
const VINYL = { color: 0x1f2a3a, rough: 0.6, metal: 0 };

export function wsZ( L, y ) {

	return L.houseFront - ( y - HOUSE.wsBottomY ) * ( 0.25 / 0.76 );

}

export function wallX( L, z, y ) {

	return houseHalfWidth( L, z ) - Math.max( 0, y - 1.15 ) * 0.04;

}

export function roofTopY( x ) {

	return 2.375 + 0.035 * ( 1 - ( x / 1.35 ) ** 2 );

}

// Offset a convex polygon (list of [u, v]) outward by d.
function offsetPoly( poly, d ) {

	const n = poly.length;
	const ccw = area2( poly ) > 0;
	const lines = [];
	for ( let i = 0; i < n; i ++ ) {

		const a = poly[ i ], b = poly[ ( i + 1 ) % n ];
		const ex = b[ 0 ] - a[ 0 ], ey = b[ 1 ] - a[ 1 ];
		const l = Math.hypot( ex, ey );
		let nx = ey / l, ny = - ex / l; // right-hand normal (outward for CCW)
		if ( ! ccw ) {

			nx = - nx; ny = - ny;

		}

		lines.push( [ a[ 0 ] + nx * d, a[ 1 ] + ny * d, ex, ey ] );

	}

	const out = [];
	for ( let i = 0; i < n; i ++ ) {

		const l1 = lines[ ( i + n - 1 ) % n ], l2 = lines[ i ];
		const den = l1[ 2 ] * l2[ 3 ] - l1[ 3 ] * l2[ 2 ];
		const s = ( ( l2[ 0 ] - l1[ 0 ] ) * l2[ 3 ] - ( l2[ 1 ] - l1[ 1 ] ) * l2[ 2 ] ) / den;
		out.push( [ l1[ 0 ] + l1[ 2 ] * s, l1[ 1 ] + l1[ 3 ] * s ] );

	}

	return out;

}

function area2( poly ) {

	let a = 0;
	for ( let i = 0; i < poly.length; i ++ ) {

		const p = poly[ i ], q = poly[ ( i + 1 ) % poly.length ];
		a += p[ 0 ] * q[ 1 ] - q[ 0 ] * p[ 1 ];

	}

	return a;

}

// Quad from four corner points (counter-clockwise seen from the front) with 0..1 or metric UVs.
function quad( a, b, c, d, uvs = [ 0, 0, 1, 0, 1, 1, 0, 1 ] ) {

	const g = new THREE.BufferGeometry();
	g.setAttribute( 'position', new THREE.Float32BufferAttribute( [ a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z, d.x, d.y, d.z ], 3 ) );
	g.setAttribute( 'uv', new THREE.Float32BufferAttribute( uvs, 2 ) );
	g.setIndex( [ 0, 1, 2, 0, 2, 3 ] );
	g.computeVertexNormals();
	return g;

}

export function buildWheelhouse( kit, L, parts ) {

	buildWalls( kit, L );
	buildWindshield( kit, L );
	buildRoof( kit, L );
	buildConsole( kit, L, parts );
	buildSeating( kit, L );
	buildRoofGear( kit, L, parts );

}

// ------------------------------------------------------------------ side walls

function sideWindowHoles( L ) {

	const { winBottom: y0, winTop: y1 } = HOUSE;
	const front = ( y ) => wsZ( L, y ) - 0.1;
	return [
		[ [ 0.3, y0 ], [ 0.78, y0 ], [ 0.78, y1 ], [ 0.3, y1 ] ],
		[ [ 0.84, y0 ], [ front( y0 ), y0 ], [ front( y1 ), y1 ], [ 0.84, y1 ] ],
	];

}

function buildWalls( kit, L ) {

	const { wallT, roofUnderY } = HOUSE;
	const z0 = L.houseBack, z1 = L.houseFront;
	const outline = [];
	const NB = 8;
	for ( let i = 0; i <= NB; i ++ ) {

		const z = lerp( z0, z1, i / NB );
		outline.push( [ z, L.sheerY( L.tAtSheerZ( z ) ) ] );

	}

	outline.push( [ z1, HOUSE.wsBottomY ] );
	outline.push( [ wsZ( L, roofUnderY ), roofUnderY ] );
	outline.push( [ z0, roofUnderY ] );
	const holes = sideWindowHoles( L );

	for ( const s of [ 1, - 1 ] ) {

		const map = ( u, v, side ) => V( s * ( wallX( L, u, v ) - side * wallT ), v, u );
		kit.add( 'gelcoat', slab( outline, holes, map ), { color: PALETTE.gelcoat, rough: 0.3 } );

		for ( const h of holes ) {

			// aluminum frame ring through the wall, proud of both faces
			const outer = offsetPoly( h, 0.024 );
			const fmap = ( u, v, side ) => V( s * ( wallX( L, u, v ) + 0.008 - side * ( wallT + 0.016 ) ), v, u );
			kit.add( 'fittings', slab( outer, [ offsetPoly( h, - 0.006 ) ], fmap ), FRAME );

			// (no glass in the openings: the windows are left open)

		}

		// sliding-window latch
		const lz = 0.81, ly = 1.84;
		const lb = box( 0.02, 0.06, 0.025 );
		lb.translate( s * ( wallX( L, lz, ly ) - wallT - 0.012 ), ly, lz );
		kit.add( 'fittings', lb, BLACK_PLASTIC );

	}

	// house front below the windshield (coaming)
	const tF = L.tAtSheerZ( L.houseFront );
	const hw = houseHalfWidth( L, L.houseFront );
	const cOutline = [];
	for ( let i = 0; i <= 10; i ++ ) {

		const x = lerp( - hw, hw, i / 10 );
		cOutline.push( [ x, foredeckY( L, tF, x ) - 0.012 ] );

	}

	cOutline.push( [ hw, HOUSE.wsBottomY ], [ - hw, HOUSE.wsBottomY ] );
	kit.add( 'gelcoat', slab( cOutline, [], ( u, v, side ) => V( u, v, L.houseFront - side * 0.04 ) ), { color: PALETTE.gelcoat, rough: 0.3 } );

}

// ------------------------------------------------------------------ windshield

function buildWindshield( kit, L ) {

	const { wsBottomY, roofUnderY } = HOUSE;
	const rake = new THREE.Vector3( 0, 0.76, - 0.25 ).normalize();
	const normal = new THREE.Vector3( 0, 0.25, 0.76 ).normalize();
	const base = V( 0, wsBottomY, L.houseFront );
	const vTop = ( roofUnderY - wsBottomY ) / rake.y;
	const at = ( u, v ) => base.clone().addScaledVector( rake, v ).setX( u );
	const halfAt = ( v ) => {

		const p = at( 0, v );
		return wallX( L, p.z, p.y );

	};

	const xb = halfAt( 0 ), xt = halfAt( vTop );
	const outline = [ [ - xb, 0 ], [ xb, 0 ], [ xt, vTop ], [ - xt, vTop ] ];
	const v0 = 0.07, v1 = 0.75;
	const holes = [
		[ [ - 0.36, v0 ], [ 0.36, v0 ], [ 0.36, v1 ], [ - 0.36, v1 ] ],
		[ [ 0.42, v0 ], [ halfAt( v0 ) - 0.09, v0 ], [ halfAt( v1 ) - 0.09, v1 ], [ 0.42, v1 ] ],
		[ [ - 0.42, v0 ], [ - 0.42, v1 ], [ - halfAt( v1 ) + 0.09, v1 ], [ - halfAt( v0 ) + 0.09, v0 ] ],
	];
	const T = 0.05;
	const map = ( u, v, side ) => at( u, v ).addScaledVector( normal, - side * T );
	kit.add( 'gelcoat', slab( outline, holes, map ), { color: PALETTE.gelcoat, rough: 0.3 } );

	for ( const h of holes ) {

		const outer = offsetPoly( h, 0.022 );
		kit.add( 'fittings', slab( outer, [ offsetPoly( h, - 0.006 ) ], ( u, v, side ) => at( u, v ).addScaledVector( normal, 0.008 - side * ( T + 0.016 ) ) ), FRAME );
		// (no glass: open windows)

	}

	// pantograph wipers on the centre and starboard panes
	for ( const [ u, ang ] of [ [ 0.0, 0.25 ], [ - 0.72, 0.2 ] ] ) {

		const pivot = at( u, v0 + 0.03 ).addScaledVector( normal, 0.012 );
		const dir = rake.clone().applyAxisAngle( normal, ang );
		const blade = rod( pivot, pivot.clone().addScaledVector( dir, 0.52 ), 0.006, 5 );
		kit.add( 'fittings', blade, BLACK_PLASTIC );
		kit.add( 'fittings', rod( pivot.clone().addScaledVector( normal, - 0.01 ), pivot.clone().addScaledVector( normal, 0.012 ), 0.014, 8 ), BLACK_PLASTIC );

	}

}

// ------------------------------------------------------------------ roof

function roofHalf( L, z ) {

	const zc = Math.min( Math.max( z, L.houseBack ), 1.18 );
	let hw = wallX( L, zc, HOUSE.roofUnderY ) + 0.09;
	const rc = 0.22;
	const d = Math.min( z - HOUSE.roofZ0, HOUSE.roofZ1 - z );
	if ( d < rc ) hw -= rc - Math.sqrt( Math.max( 0, rc * rc - ( rc - d ) ** 2 ) );
	return hw;

}

function buildRoof( kit, L ) {

	const yb = HOUSE.roofUnderY;
	const N = 22;
	const profiles = [];
	for ( let i = 0; i <= N; i ++ ) {

		const f = i / N;
		// cluster stations near both ends for the rounded corners
		const g = 0.5 - 0.5 * Math.cos( Math.PI * f );
		const z = lerp( HOUSE.roofZ0, HOUSE.roofZ1, lerp( f, g, 0.6 ) );
		const hw = roofHalf( L, z );
		const half = [
			[ 0, yb ], [ hw * 0.5, yb ], [ hw - 0.03, yb ], [ hw - 0.008, yb + 0.008 ], [ hw, yb + 0.025 ],
			[ hw - 0.004, roofTopY( hw ) - 0.012 ], [ hw - 0.02, roofTopY( hw ) ], [ hw * 0.75, roofTopY( hw * 0.75 ) ],
			[ hw * 0.5, roofTopY( hw * 0.5 ) ], [ hw * 0.25, roofTopY( hw * 0.25 ) ], [ 0, roofTopY( 0 ) ],
		];
		const loop = half.map( ( p ) => V( p[ 0 ], p[ 1 ], z ) );
		for ( let k = half.length - 2; k >= 1; k -- ) loop.push( V( - half[ k ][ 0 ], half[ k ][ 1 ], z ) );
		profiles.push( loop );

	}

	const g = loft( profiles, { closed: true } );
	orientOutwardFn( g, ( p ) => V( p.x, p.y - 2.345, 0 ) );
	metricUV( g, ( p ) => [ p.x, p.z ] );
	auxVertices( g, ( p ) => [ 0.35, 0, p.y > yb + 0.03 ? 1 : 0, 0 ] );
	kit.add( 'gelcoat', g, { color: PALETTE.gelcoat } );
	kit.add( 'gelcoat', fanCap( profiles[ 0 ], V( 0, 0, - 1 ) ), { color: PALETTE.gelcoat, rough: 0.3 } );
	kit.add( 'gelcoat', fanCap( profiles[ N ], V( 0, 0, 1 ) ), { color: PALETTE.gelcoat, rough: 0.3 } );

	// stainless grab rails along the roof edges
	for ( const s of [ 1, - 1 ] ) {

		const x = s * 1.1;
		const y = roofTopY( 1.1 ) + 0.07;
		kit.add( 'fittings', rod( V( x, y, - 0.7 ), V( x, y, 0.95 ), 0.013, 8 ), STAINLESS );
		for ( const z of [ - 0.7, - 0.12, 0.45, 0.95 ] ) kit.add( 'fittings', rod( V( x, roofTopY( 1.1 ) - 0.01, z ), V( x, y, z ), 0.011, 6 ), STAINLESS );

	}

}

function orientOutwardFn( g, fn ) {

	const p = g.attributes.position, n = g.attributes.normal;
	let dot = 0;
	const a = V( 0, 0, 0 ), b = V( 0, 0, 0 );
	for ( let i = 0; i < p.count; i ++ ) {

		a.fromBufferAttribute( p, i ); b.fromBufferAttribute( n, i );
		dot += b.dot( fn( a ) );

	}

	if ( dot < 0 ) {

		const idx = Array.from( g.index.array );
		for ( let i = 0; i < idx.length; i += 3 ) {

			const t = idx[ i + 1 ]; idx[ i + 1 ] = idx[ i + 2 ]; idx[ i + 2 ] = t;

		}

		g.setIndex( idx );
		g.computeVertexNormals();

	}

}

function metricUV( g, fn ) {

	const p = g.attributes.position;
	const uv = new Float32Array( p.count * 2 );
	const v = V( 0, 0, 0 );
	for ( let i = 0; i < p.count; i ++ ) {

		v.fromBufferAttribute( p, i );
		const t = fn( v );
		uv[ i * 2 ] = t[ 0 ]; uv[ i * 2 + 1 ] = t[ 1 ];

	}

	g.setAttribute( 'uv', new THREE.Float32BufferAttribute( uv, 2 ) );

}

// ------------------------------------------------------------------ console, helm, instruments

// Slope of the instrument panel (from the knee to the dash top).
export function panelFrame() {

	const d = HOUSE.dash;
	const B = new THREE.Vector2( d.zFace, d.yKnee ), C = new THREE.Vector2( d.zTop, d.yTop );
	const dir = C.clone().sub( B ).normalize(); // (dz, dy)
	const along = V( 0, dir.y, dir.x ); // up the panel
	const normal = V( 0, dir.x, - dir.y ); // toward the helmsman (up and aft)
	const at = ( x, f ) => V( x, lerp( B.y, C.y, f ), lerp( B.x, C.x, f ) );
	return { along, normal, at, length: B.distanceTo( C ), angle: Math.atan2( dir.x, dir.y ) };

}

function buildConsole( kit, L, parts ) {

	const d = HOUSE.dash;
	// console body; its ends follow the hull lining where the flared hull narrows toward the bow
	const corners = [ [ d.zFace, L.deckY ], [ d.zFace, d.yKnee ], [ d.zTop, d.yTop ], [ d.zBack, d.yTop ], [ d.zBack, L.deckY ] ];
	const outline = [];
	for ( let i = 0; i < corners.length; i ++ ) {

		const a = corners[ i ], b = corners[ ( i + 1 ) % corners.length ];
		for ( let k = 0; k < 6; k ++ ) outline.push( [ lerp( a[ 0 ], b[ 0 ], k / 6 ), lerp( a[ 1 ], b[ 1 ], k / 6 ) ] );

	}

	const endX = ( z, y ) => Math.min( d.halfW, L.halfBreadth( L.tAtSheerZ( z ), Math.min( y, L.sheerY( L.tAtSheerZ( z ) ) ) ) - L.shell - 0.004 );
	kit.add( 'gelcoat', slab( outline, [], ( u, v, side ) => V( ( side ? - 1 : 1 ) * endX( u, v ), v, u ) ), { color: PALETTE.gelcoat, rough: 0.3 } );

	// anti-glare dash top and instrument panel
	const top = box( d.halfW * 2 - 0.01, 0.01, d.zBack - d.zTop );
	top.translate( 0, d.yTop + 0.005, ( d.zBack + d.zTop ) / 2 );
	kit.add( 'gelcoat', top, { color: 0x2a2c2f, rough: 0.85 } );

	const pf = panelFrame();
	const panel = box( d.halfW * 2 - 0.04, pf.length - 0.02, 0.012 );
	panel.applyMatrix4( mat4( 0, 0, 0, pf.angle, 0, 0 ) );
	panel.translate( ...pf.at( 0, 0.5 ).addScaledVector( pf.normal, 0.004 ).toArray() );
	kit.add( 'fittings', panel, { color: 0x1d1f22, rough: 0.7 } );

	// teak fiddle rail along the dash top edge
	const fr = box( d.halfW * 2 - 0.02, 0.03, 0.022 );
	fr.translate( 0, d.yTop + 0.02, d.zTop + 0.011 );
	kit.add( 'wood', fr, { rough: 0.3 } );

	// gauges (bezel + backlit dial)
	for ( const [ x, f ] of [ [ - 0.76, 0.72 ], [ - 0.34, 0.72 ], [ - 0.55, 0.9 ], [ - 0.12, 0.5 ], [ 0.02, 0.5 ] ] ) {

		const c = pf.at( x, f ).addScaledVector( pf.normal, 0.012 );
		const m = new THREE.Matrix4().makeBasis( V( - 1, 0, 0 ), pf.along, pf.normal ).setPosition( c );
		const bezel = torus( 0.043, 0.006, 4, 16 );
		bezel.applyMatrix4( m );
		kit.add( 'fittings', bezel, STAINLESS );
		const dial = new THREE.CircleGeometry( 0.042, 20 );
		// 3 mm proud of the panel face (was coplanar with it: z-fighting)
		dial.applyMatrix4( new THREE.Matrix4().makeBasis( V( - 1, 0, 0 ), pf.along, pf.normal ).setPosition( c.clone().addScaledVector( pf.normal, 0.001 ) ) );
		kit.add( 'glow', dial, { color: 0xffffff, rough: 0.2, pattern: 3 } );

	}

	// switch panel with rocker switches
	const sw = pf.at( 0.55, 0.45 ).addScaledVector( pf.normal, 0.012 );
	const swm = new THREE.Matrix4().makeBasis( V( - 1, 0, 0 ), pf.along, pf.normal ).setPosition( sw );
	const swPlate = box( 0.34, 0.1, 0.008 );
	swPlate.applyMatrix4( swm );
	kit.add( 'fittings', swPlate, { color: 0x2d3036, rough: 0.5 } );
	for ( let i = 0; i < 6; i ++ ) {

		const r = box( 0.03, 0.045, 0.015 );
		r.applyMatrix4( new THREE.Matrix4().makeBasis( V( - 1, 0, 0 ), pf.along, pf.normal ).setPosition( sw.clone().add( V( - 0.13 + i * 0.052, 0, 0 ) ).addScaledVector( pf.normal, 0.008 ) ) );
		kit.add( 'fittings', r, { color: 0x111214, rough: 0.4 } );
		const led = box( 0.008, 0.008, 0.004 );
		led.applyMatrix4( new THREE.Matrix4().makeBasis( V( - 1, 0, 0 ), pf.along, pf.normal ).setPosition( sw.clone().add( V( - 0.13 + i * 0.052, 0, 0 ) ).addScaledVector( pf.along, 0.035 ).addScaledVector( pf.normal, 0.006 ) ) );
		kit.add( 'glow', led, { color: i % 3 === 0 ? 0x33ff66 : 0xff5522, rough: 0.3, pattern: 6 } );

	}

	// helm: wheel shaft (static) and the wheel pivot (animated part built in BoatModel)
	const hub = pf.at( HOUSE.helmX, 0.45 );
	const center = hub.clone().addScaledVector( pf.normal, 0.14 );
	kit.add( 'fittings', rod( hub.clone().addScaledVector( pf.normal, - 0.01 ), center.clone().addScaledVector( pf.normal, - 0.03 ), 0.02, 10 ), STAINLESS );
	const helmBoss = cylinder( 0.05, 0.055, 0.025, 16 );
	helmBoss.applyMatrix4( alignY( hub.clone().addScaledVector( pf.normal, 0.008 ), pf.normal ) );
	kit.add( 'fittings', helmBoss, { color: 0x2b2d30, rough: 0.5 } );
	parts.wheelCenter = center;
	parts.wheelAxis = pf.normal.clone().negate(); // local +Z of the wheel points into the dash

	// throttle / shift control on the dash top, starboard of the wheel
	const tx = - 0.93, tz = 1.24;
	const tb = roundedBox( 0.1, 0.075, 0.15, 0.015, 1 );
	tb.translate( tx, d.yTop + 0.01 + 0.0375, tz );
	kit.add( 'fittings', tb, BLACK_PLASTIC );
	const tp = box( 0.085, 0.004, 0.13 );
	tp.translate( tx, d.yTop + 0.01 + 0.077, tz );
	kit.add( 'fittings', tp, STAINLESS );
	parts.throttlePivot = V( tx, d.yTop + 0.09, tz );

	// compass binnacle with a glass dome
	const cx = HOUSE.helmX, cz = 1.3;
	const cb = lathe( [ [ 0, 0 ], [ 0.07, 0 ], [ 0.072, 0.02 ], [ 0.062, 0.045 ], [ 0, 0.045 ] ], 20 );
	cb.translate( cx, d.yTop + 0.01, cz );
	kit.add( 'fittings', cb, BLACK_PLASTIC );
	const card = cylinder( 0.05, 0.05, 0.01, 20 );
	card.translate( cx, d.yTop + 0.06, cz );
	kit.add( 'fittings', card, { color: 0x2c2c2a, rough: 0.5 } );
	const lubber = box( 0.004, 0.02, 0.008 );
	lubber.translate( cx, d.yTop + 0.07, cz + 0.045 );
	kit.add( 'fittings', lubber, { color: 0xd0402a, rough: 0.5 } );
	const dome = sphere( 0.058, 20, 8, 0, Math.PI * 2, 0, Math.PI / 2 );
	dome.translate( cx, d.yTop + 0.053, cz );
	kit.add( 'glass', dome );

	// electronics: radar display (centre) and chart plotter (port)
	for ( const [ x, mode ] of [ [ - 0.06, 1 ], [ 0.38, 2 ] ] ) {

		const tilt = 0.3; // lean back so the screen faces the helmsman
		const w = 0.36, h = 0.27;
		const bodyM = mat4( x, d.yTop + 0.01 + h / 2 + 0.03, 1.29, tilt, 0, 0 );
		const body = roundedBox( w, h, 0.07, 0.018, 1 );
		body.applyMatrix4( bodyM );
		kit.add( 'fittings', body, BLACK_PLASTIC );
		const mount = box( 0.1, 0.05, 0.08 );
		mount.translate( x, d.yTop + 0.035, 1.3 );
		kit.add( 'fittings', mount, BLACK_PLASTIC );
		const screen = new THREE.PlaneGeometry( w - 0.05, h - 0.06 );
		screen.applyMatrix4( new THREE.Matrix4().makeRotationY( Math.PI ) );
		screen.applyMatrix4( mat4( 0, 0.012, - 0.037 ) ); // 2 mm proud of the bezel face
		screen.applyMatrix4( bodyM );
		// PlaneGeometry uvs are flipped horizontally after the Y rotation; restore them
		const uv = screen.attributes.uv;
		for ( let i = 0; i < uv.count; i ++ ) uv.setX( i, 1 - uv.getX( i ) );
		kit.add( 'glow', screen, { color: 0xffffff, rough: 0.15, pattern: mode } );

	}

	// overhead console with VHF radio above the windshield
	const oz = 1.02, oy = HOUSE.roofUnderY - 0.07;
	const oc = roundedBox( 0.9, 0.13, 0.3, 0.02, 1 );
	oc.applyMatrix4( mat4( - 0.15, oy, oz, - 0.25, 0, 0 ) );
	kit.add( 'fittings', oc, { color: 0x24262a, rough: 0.6 } );
	const face = mat4( - 0.15, oy, oz, - 0.25, 0, 0 ); // aft face tilted down toward the helm
	const vhf = box( 0.2, 0.06, 0.02 );
	vhf.applyMatrix4( new THREE.Matrix4().makeTranslation( - 0.18, 0, - 0.155 ) );
	vhf.applyMatrix4( face );
	kit.add( 'fittings', vhf, { color: 0x111214, rough: 0.4 } );
	const lcd = new THREE.PlaneGeometry( 0.09, 0.03 );
	lcd.applyMatrix4( new THREE.Matrix4().makeRotationY( Math.PI ) );
	lcd.applyMatrix4( new THREE.Matrix4().makeTranslation( - 0.2, 0.005, - 0.168 ) ); // 3 mm proud of the radio face
	lcd.applyMatrix4( face );
	kit.add( 'glow', lcd, { color: 0x7dff9a, rough: 0.2, pattern: 6 } );
	for ( const k of [ - 0.06, 0.1, 0.26 ] ) {

		const knob = cylinder( 0.012, 0.012, 0.02, 10 );
		knob.applyMatrix4( new THREE.Matrix4().makeRotationX( Math.PI / 2 ) );
		knob.applyMatrix4( new THREE.Matrix4().makeTranslation( k, 0, - 0.16 ) );
		knob.applyMatrix4( face );
		kit.add( 'fittings', knob, STAINLESS );

	}

	// microphone hanging from the VHF
	const mic = roundedBox( 0.05, 0.08, 0.03, 0.01, 1 );
	mic.translate( - 0.02, oy - 0.2, oz - 0.12 );
	kit.add( 'fittings', mic, BLACK_PLASTIC );
	kit.add( 'fittings', tube( [ V( - 0.05, oy - 0.04, oz - 0.15 ), V( - 0.08, oy - 0.1, oz - 0.14 ), V( - 0.04, oy - 0.14, oz - 0.13 ), V( - 0.02, oy - 0.16, oz - 0.12 ) ], 0.004, 16, 4 ), BLACK_PLASTIC );

	// cabin dome light
	const dl = cylinder( 0.08, 0.085, 0.012, 20 );
	dl.translate( 0, HOUSE.roofUnderY - 0.006, 0.42 );
	kit.add( 'glow', dl, { color: 0xffe6b8, rough: 0.4, pattern: 5 } );

	// cuddy door in the console face (port side)
	const door = box( 0.5, 0.62, 0.01 );
	door.translate( 0.55, 0.72, d.zFace - 0.004 );
	kit.add( 'fittings', door, { color: 0x121315, rough: 0.9 } );
	for ( const [ w, h, x, y ] of [ [ 0.56, 0.03, 0.55, 1.045 ], [ 0.56, 0.03, 0.55, 0.395 ], [ 0.03, 0.68, 0.285, 0.72 ], [ 0.03, 0.68, 0.815, 0.72 ] ] ) {

		const b = box( w, h, 0.022 );
		b.translate( x, y, d.zFace - 0.011 );
		kit.add( 'wood', b, { rough: 0.3 } );

	}

}

// ------------------------------------------------------------------ seating

function buildSeating( kit, L ) {

	const x = HOUSE.helmX, z = HOUSE.seatZ, y0 = L.deckY;
	const baseP = cylinder( 0.18, 0.19, 0.012, 20 );
	baseP.translate( x, y0 + 0.006, z );
	kit.add( 'fittings', baseP, STAINLESS );
	kit.add( 'fittings', rod( V( x, y0, z ), V( x, 0.92, z ), 0.042, 14 ), STAINLESS );
	const ring = torus( 0.17, 0.011, 6, 28 );
	ring.applyMatrix4( mat4( x, 0.62, z, Math.PI / 2, 0, 0 ) );
	kit.add( 'fittings', ring, STAINLESS );
	for ( let i = 0; i < 3; i ++ ) {

		const a = i * Math.PI * 2 / 3 + 0.5;
		kit.add( 'fittings', rod( V( x + Math.cos( a ) * 0.04, 0.62, z + Math.sin( a ) * 0.04 ), V( x + Math.cos( a ) * 0.165, 0.62, z + Math.sin( a ) * 0.165 ), 0.008, 6 ), STAINLESS );

	}

	const pan = roundedBox( 0.44, 0.045, 0.4, 0.015, 1 );
	pan.translate( x, 0.94, z );
	kit.add( 'fittings', pan, { color: 0x2b2e33, rough: 0.5 } );
	const cushion = roundedBox( 0.46, 0.085, 0.42, 0.035, 1 );
	cushion.translate( x, 1.0, z );
	kit.add( 'fittings', cushion, VINYL );
	const back = roundedBox( 0.44, 0.3, 0.075, 0.03, 1 );
	back.applyMatrix4( mat4( x, 1.22, z - 0.2, - 0.17, 0, 0 ) );
	kit.add( 'fittings', back, VINYL );
	kit.add( 'fittings', rod( V( x, 0.96, z - 0.17 ), V( x, 1.12, z - 0.2 ), 0.018, 8 ), STAINLESS );

	// companion bench on the port side
	const bench = roundedBox( 0.42, 0.4, 0.75, 0.02, 1 );
	bench.translate( 0.86, y0 + 0.2, 0.45 );
	kit.add( 'gelcoat', bench, { color: PALETTE.gelcoat, rough: 0.35 } );
	const bc = roundedBox( 0.42, 0.07, 0.73, 0.03, 1 );
	bc.translate( 0.86, y0 + 0.435, 0.45 );
	kit.add( 'fittings', bc, VINYL );

}

// ------------------------------------------------------------------ roof gear: mast, radars, lights, antennas

function buildRoofGear( kit, L, parts ) {

	const yr = roofTopY( 0 );
	const mz = - 0.5;

	// mast
	const mb = box( 0.18, 0.025, 0.18 );
	mb.translate( 0, yr + 0.005, mz );
	kit.add( 'fittings', mb, WHITE_PAINT );
	kit.add( 'fittings', rod( V( 0, yr, mz ), V( 0, 3.9, mz ), 0.042, 14, 0.03 ), WHITE_PAINT );

	// radome on a forward bracket
	const ry = 3.02;
	const arm = box( 0.1, 0.035, 0.34 );
	arm.translate( 0, ry - 0.02, mz + 0.19 );
	kit.add( 'fittings', arm, WHITE_PAINT );
	const plate = box( 0.34, 0.015, 0.34 );
	plate.translate( 0, ry, mz + 0.3 );
	kit.add( 'fittings', plate, WHITE_PAINT );
	kit.add( 'fittings', rod( V( 0, ry - 0.02, mz + 0.04 ), V( 0, ry - 0.2, mz + 0.01 ), 0.012, 6 ), WHITE_PAINT );
	const radome = lathe( [ [ 0, 0 ], [ 0.285, 0 ], [ 0.3, 0.018 ], [ 0.3, 0.095 ], [ 0.29, 0.14 ], [ 0.245, 0.188 ], [ 0.16, 0.222 ], [ 0.07, 0.234 ], [ 0, 0.236 ] ], 24 );
	radome.translate( 0, ry + 0.008, mz + 0.3 );
	kit.add( 'fittings', radome, { color: 0xf3f2ee, rough: 0.4 } );
	const band = cylinder( 0.302, 0.302, 0.02, 24, 1, true );
	band.translate( 0, ry + 0.03, mz + 0.3 );
	kit.add( 'fittings', band, { color: 0x3a3d42, rough: 0.5 } );

	// spreader with side lights, masthead light
	const sy = 3.45;
	const spreader = box( 0.78, 0.03, 0.045 );
	spreader.translate( 0, sy - 0.05, mz );
	kit.add( 'fittings', spreader, WHITE_PAINT );
	for ( const s of [ 1, - 1 ] ) {

		const hx = s * 0.4;
		const housing = box( 0.055, 0.07, 0.11 );
		housing.translate( hx, sy, mz + 0.02 );
		kit.add( 'fittings', housing, BLACK_PLASTIC );
		const lens = box( 0.014, 0.05, 0.085 );
		lens.translate( hx + s * 0.033, sy, mz + 0.03 );
		kit.add( 'glow', lens, { color: s > 0 ? 0xff1a0e : 0x14ff5a, rough: 0.2, pattern: 0 } );
		// inboard screen
		const scr = box( 0.004, 0.07, 0.12 );
		scr.translate( hx - s * 0.03, sy, mz + 0.02 );
		kit.add( 'fittings', scr, BLACK_PLASTIC );

	}

	const mhBase = cylinder( 0.036, 0.036, 0.03, 12 );
	mhBase.translate( 0, 3.915, mz );
	kit.add( 'fittings', mhBase, BLACK_PLASTIC );
	const mhLens = cylinder( 0.03, 0.03, 0.065, 12 );
	mhLens.translate( 0, 3.962, mz );
	kit.add( 'glow', mhLens, { color: 0xfff3dc, rough: 0.2, pattern: 0 } );
	const mhCap = cylinder( 0.034, 0.038, 0.016, 12 );
	mhCap.translate( 0, 4.003, mz );
	kit.add( 'fittings', mhCap, BLACK_PLASTIC );

	// VHF whips on ratchet mounts at the aft roof corners (sway in the vertex shader)
	for ( const s of [ 1, - 1 ] ) {

		const x = s * 1.02, z = - 0.78, y = roofTopY( 1.02 );
		const mount = roundedBox( 0.06, 0.07, 0.06, 0.01, 1 );
		mount.translate( x, y + 0.035, z );
		kit.add( 'fittings', mount, STAINLESS );
		const len = s > 0 ? 2.4 : 1.2;
		const whip = cylinder( 0.005, 0.013, len, 8, 10 );
		whip.translate( x, y + 0.07 + len / 2, z );
		auxVertices( whip, ( p ) => [ 0.3, 0, 3, Math.max( 0, ( p.y - y - 0.07 ) / len ) ] );
		kit.add( 'fittings', whip, { color: 0xf4f4f1 } );

	}

	// open-array radar pedestal (array is animated, built in BoatModel)
	const pz = 0.8;
	const ped = roundedBox( 0.26, 0.28, 0.3, 0.03, 1 );
	ped.translate( 0, yr + 0.14, pz );
	kit.add( 'fittings', ped, { color: 0xf3f2ee, rough: 0.4 } );
	parts.radarPivot = V( 0, yr + 0.3, pz );

	// spotlight and horn on the front of the roof
	const sx = 0.42, szz = 1.05, syy = roofTopY( 0.42 );
	kit.add( 'fittings', rod( V( sx, syy, szz ), V( sx, syy + 0.1, szz ), 0.03, 10 ), BLACK_PLASTIC );
	const head = cylinder( 0.065, 0.06, 0.15, 16 );
	head.applyMatrix4( mat4( sx, syy + 0.17, szz, Math.PI / 2, 0, 0 ) );
	kit.add( 'fittings', head, STAINLESS );
	const sl = new THREE.CircleGeometry( 0.058, 16 );
	sl.translate( sx, syy + 0.17, szz + 0.0755 );
	kit.add( 'glow', sl, { color: 0xfff1d6, rough: 0.2, pattern: 4 } );
	const horn = lathe( [ [ 0.0, 0 ], [ 0.018, 0.0 ], [ 0.016, 0.1 ], [ 0.024, 0.17 ], [ 0.05, 0.22 ], [ 0.046, 0.222 ], [ 0.0, 0.2 ] ], 16 );
	horn.applyMatrix4( mat4( - sx, roofTopY( 0.42 ) + 0.07, szz - 0.12, Math.PI / 2, 0, 0 ) );
	kit.add( 'fittings', horn, STAINLESS );
	kit.add( 'fittings', rod( V( - sx, roofTopY( 0.42 ), szz ), V( - sx, roofTopY( 0.42 ) + 0.07, szz ), 0.012, 6 ), STAINLESS );

	// the owner's buoy colours displayed on the roof
	const bx = 0.62, bz = - 0.72;
	const bracket = box( 0.12, 0.03, 0.12 );
	bracket.translate( bx, roofTopY( bx ) + 0.015, bz );
	kit.add( 'fittings', bracket, STAINLESS );
	for ( const [ g, opts ] of buoyGeometry() ) {

		g.translate( bx, roofTopY( bx ) + 0.03, bz );
		kit.add( 'fittings', g, opts );

	}

	// deck floodlights under the roof overhang, aimed at the work deck
	for ( const s of [ 1, - 1 ] ) {

		const fl = roundedBox( 0.13, 0.07, 0.08, 0.01, 1 );
		fl.translate( s * 0.55, HOUSE.roofUnderY - 0.035, HOUSE.roofZ0 + 0.14 );
		kit.add( 'fittings', fl, BLACK_PLASTIC );
		const lens = new THREE.PlaneGeometry( 0.11, 0.06 );
		lens.applyMatrix4( mat4( s * 0.55, HOUSE.roofUnderY - 0.071, HOUSE.roofZ0 + 0.14, Math.PI / 2 + 0.35, 0, 0 ) );
		kit.add( 'glow', lens, { color: 0xfff1d6, rough: 0.2, pattern: 4 } );

	}

	// life ring on the port house side
	const lx = wallX( L, - 0.02, 1.72 ) + 0.05;
	const ringG = torus( 0.235, 0.05, 8, 24 );
	ringG.applyMatrix4( mat4( lx, 1.72, - 0.02, 0, Math.PI / 2, 0 ) );
	paintVertices( ringG, ( p ) => {

		const a = Math.atan2( p.y - 1.72, p.z + 0.02 );
		return Math.abs( ( ( a / ( Math.PI / 2 ) ) % 1 + 1 ) % 1 - 0.5 ) > 0.4 ? 0xf2f0ea : 0xf25a12;

	} );
	kit.add( 'fittings', ringG, { rough: 0.6 } );
	const hook = box( 0.03, 0.05, 0.08 );
	hook.translate( lx - 0.03, 1.72 + 0.26, - 0.02 );
	kit.add( 'fittings', hook, STAINLESS );

	void parts;

}

// Steering wheel: varnished mahogany destroyer wheel with a brass hub.
// Local frame: +Z is the shaft axis (into the dash), spokes in the XY plane.
export function wheelGeometry() {

	const list = [];
	const wood = { rough: 0.3, metal: 0 };
	const R = 0.2;
	list.push( prepare( torus( R, 0.017, 8, 40 ), { ...wood, color: 0xe8b898 } ) );
	for ( let i = 0; i < 6; i ++ ) {

		const a = i * Math.PI / 3 + Math.PI / 6; // king spoke straight up when centred
		const dir = V( Math.cos( a ), Math.sin( a ), 0 );
		const spoke = cylinder( 0.009, 0.013, R - 0.05, 8 );
		spoke.applyMatrix4( alignY( dir.clone().multiplyScalar( 0.05 + ( R - 0.05 ) / 2 ), dir ) );
		list.push( prepare( spoke, { ...wood, color: 0xe8b898 } ) );
		const handle = lathe( [ [ 0.0, 0 ], [ 0.011, 0.0 ], [ 0.013, 0.02 ], [ 0.017, 0.045 ], [ 0.012, 0.06 ], [ 0.01, 0.07 ], [ 0.015, 0.078 ], [ 0.0, 0.086 ] ], 8 );
		handle.applyMatrix4( alignY( dir.clone().multiplyScalar( R - 0.005 ), dir ) );
		list.push( prepare( handle, { ...wood, color: 0xe8b898 } ) );

	}

	const hub = lathe( [ [ 0, - 0.03 ], [ 0.05, - 0.03 ], [ 0.058, - 0.012 ], [ 0.058, 0.012 ], [ 0.05, 0.028 ], [ 0.0, 0.03 ] ], 20 );
	hub.applyMatrix4( new THREE.Matrix4().makeRotationX( Math.PI / 2 ) );
	list.push( prepare( hub, { ...wood, color: 0xe8b898 } ) );
	const cap = sphere( 0.032, 16, 6, 0, Math.PI * 2, 0, Math.PI / 2 );
	cap.applyMatrix4( new THREE.Matrix4().makeRotationX( - Math.PI / 2 ) );
	cap.translate( 0, 0, - 0.028 );
	list.push( prepare( cap, { color: 0xc8a050, rough: 0.25, metal: 1, pattern: 1 } ) );
	return mergePrepared( list );

}

// Throttle lever, local origin at the pivot, lever pointing +Y (neutral).
export function throttleGeometry() {

	const list = [];
	list.push( prepare( rod( V( 0, - 0.01, 0 ), V( 0, 0.15, 0.015 ), 0.008, 8 ), STAINLESS ) );
	const knob = sphere( 0.022, 14, 10 );
	knob.scale( 1, 1.15, 1 );
	knob.translate( 0, 0.165, 0.016 );
	list.push( prepare( knob, BLACK_PLASTIC ) );
	const boss = cylinder( 0.02, 0.02, 0.05, 12 );
	boss.applyMatrix4( new THREE.Matrix4().makeRotationZ( Math.PI / 2 ) );
	list.push( prepare( boss, STAINLESS ) );
	return mergePrepared( list );

}

// Open-array radar antenna, local origin on the rotation axis at the pedestal top.
export function radarArrayGeometry() {

	const list = [];
	list.push( prepare( cylinder( 0.05, 0.06, 0.06, 14 ), { color: 0xf3f2ee, rough: 0.4 } ) );
	const bar = roundedBox( 1.05, 0.12, 0.1, 0.035, 2 );
	bar.translate( 0, 0.09, 0 );
	list.push( prepare( bar, { color: 0xf3f2ee, rough: 0.4 } ) );
	const stripe = box( 0.95, 0.028, 0.004 );
	stripe.translate( 0, 0.09, 0.051 );
	list.push( prepare( stripe, { color: 0x2c2f35, rough: 0.5 } ) );
	return mergePrepared( list );

}
