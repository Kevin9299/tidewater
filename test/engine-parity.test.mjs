// Parity check: runs the repo's pure-geometry builders once against three.js
// and once against src/engine (three specifiers remapped by a loader hook),
// then compares per-geometry digests. Run: node test/engine-parity.test.mjs

import { register } from 'node:module';
import { spawnSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const self = fileURLToPath( import.meta.url );
const engineURL = new URL( '../src/engine/index.js', import.meta.url ).href;

const HOOK = `
export async function resolve( spec, ctx, next ) {
	if ( spec === 'three' || spec === 'three/webgpu' || spec === 'three/addons/utils/BufferGeometryUtils.js' || spec === 'three/addons/geometries/RoundedBoxGeometry.js' ) {
		return { url: ${ JSON.stringify( engineURL ) }, shortCircuit: true };
	}
	return next( spec, ctx );
}`;

async function collect() {

	const src = ( p ) => pathToFileURL( fileURLToPath( new URL( '../src/' + p, import.meta.url ) ) ).href;
	const { mulberry32 } = await import( src( 'util/Noise.js' ) );
	const WH = await import( src( 'world/boat/Wheelhouse.js' ) );
	const RUN = await import( src( 'world/boat/Running.js' ) );
	const DG = await import( src( 'world/boat/DeckGear.js' ) );
	const BS = await import( src( 'world/wildlife/BirdShapes.js' ) );
	const CS = await import( src( 'world/wildlife/CritterShapes.js' ) );
	const CG = await import( src( 'world/fish/CreatureGeometry.js' ) );
	const RG = await import( src( 'world/terrain/RockGeometry.js' ) );
	const PG = await import( src( 'world/vegetation/PlantGeometry.js' ) );
	const RF = await import( src( 'world/reef/ReefGeometry.js' ) );
	const GK = await import( src( 'world/boat/GeoKit.js' ) );
	const HL = await import( src( 'world/boat/HullLines.js' ) );
	const HB = await import( src( 'world/boat/HullBuilder.js' ) );

	const cases = {
		panelFrame: () => WH.panelFrame(),
		wheel: () => WH.wheelGeometry(),
		throttle: () => WH.throttleGeometry(),
		radar: () => WH.radarArrayGeometry(),
		propeller: () => RUN.propellerGeometry(),
		rudder: () => RUN.rudderGeometry(),
		buoy: () => DG.buoyGeometry(),
		critters: () => CS.buildCritters(),
		ray: () => CG.rayGeometry(),
		turtle: () => CG.turtleGeometry(),
		palmNear: () => PG.buildPalmNear(),
		palmFar: () => PG.buildPalmFar(),
		fern: () => PG.buildFern(),
		banana: () => PG.buildBanana(),
		treeNear: () => PG.buildTreeNear(),
		shrub: () => PG.buildShrubNear(),
		canopy: () => PG.buildCanopyNear(),
		understory: () => PG.buildUnderstory(),
	};
	BS.SPECIES.forEach( ( sp, i ) => { cases[ 'bird' + i ] = () => BS.buildBird( sp ); } );
	RG.ROCK_STYLES.forEach( ( _, i ) => { cases[ 'rock' + i ] = () => RG.buildRockGeometry( i, 7 + i, 2 ); } );
	// the full boat: slab() triangulation may differ from earcut, so these
	// compare vertex data + triangle count + total area, not index values
	{

		const kit = new GK.GeoKit(), lines = new HL.HullLines(), parts = {};
		HB.buildHull( kit, lines );
		WH.buildWheelhouse( kit, lines, parts );
		DG.buildDeckGear( kit, lines, parts );
		for ( const name of kit.buckets.keys() ) cases[ '~boat-' + name ] = () => kit.merged( name );
		cases[ '~boat-hullVolume' ] = () => HB.buildHullVolume( lines );
		cases[ '~boat-keelVolume' ] = () => HB.keelVolume( lines );
		cases[ '~boat-samples' ] = () => ( { n: lines.buildHullSamples( 8 ) } );

	}

	for ( const name of Object.keys( RF ).filter( ( k ) => k.startsWith( 'create' ) ) ) cases[ name ] = () => RF[ name ]( mulberry32( 99 ), 0 );

	const out = {};
	for ( const [ name, fn ] of Object.entries( cases ) ) {

		try {

			out[ name ] = digest( fn() );

		} catch ( e ) {

			out[ name ] = { error: String( e && e.stack ? e.stack.split( '\n' ).slice( 0, 3 ).join( ' | ' ) : e ) };

		}

	}

	return out;

}

function digest( value ) {

	const geos = [], seen = new Set();
	const walk = ( v, depth ) => {

		if ( ! v || typeof v !== 'object' || seen.has( v ) || depth > 4 ) return;
		seen.add( v );
		if ( v.isBufferGeometry ) { geos.push( v ); return; }
		if ( v.isMesh || v.isObject3D ) { v.traverse?.( ( o ) => o.geometry && walk( o.geometry, depth + 1 ) ); return; }
		if ( ArrayBuffer.isView( v ) ) return;
		for ( const k of Object.keys( v ) ) walk( v[ k ], depth + 1 );

	};

	walk( value, 0 );
	return geos.map( ( g ) => {

		const d = { index: g.index ? [ g.index.count, sum( g.index ) ] : null, groups: g.groups.length, area: area( g ), attrs: {} };
		for ( const [ n, a ] of Object.entries( g.attributes ) ) d.attrs[ n ] = [ a.count, a.itemSize, sum( a ), sumAbs( a ) ];
		return d;

	} );

}

function area( g ) {

	const p = g.attributes.position;
	if ( ! p ) return 0;
	const n = g.index ? g.index.count : p.count, at = ( k ) => ( g.index ? g.index.getX( k ) : k );
	let s = 0;
	for ( let k = 0; k + 2 < n; k += 3 ) {

		const a = at( k ), b = at( k + 1 ), c = at( k + 2 );
		const ux = p.getX( b ) - p.getX( a ), uy = p.getY( b ) - p.getY( a ), uz = p.getZ( b ) - p.getZ( a );
		const vx = p.getX( c ) - p.getX( a ), vy = p.getY( c ) - p.getY( a ), vz = p.getZ( c ) - p.getZ( a );
		s += Math.hypot( uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx ) / 2;

	}

	return s;

}

function sum( a ) { let s = 0; for ( let i = 0; i < a.count; i ++ ) for ( let c = 0; c < a.itemSize; c ++ ) s += a.getComponent( i, c ) * ( 1 + ( i % 7 ) ); return s; }
function sumAbs( a ) { let s = 0; for ( let i = 0; i < a.count; i ++ ) for ( let c = 0; c < a.itemSize; c ++ ) s += Math.abs( a.getComponent( i, c ) ); return s; }

if ( process.argv[ 2 ] === '--child' ) {

	if ( process.argv[ 3 ] === 'engine' ) register( 'data:text/javascript,' + encodeURIComponent( HOOK ) );
	// the engine's typed getComponent is shared; three's BufferAttribute has it too
	const out = await collect();
	process.stdout.write( JSON.stringify( out ) );

} else {

	const run = ( mode ) => {

		const r = spawnSync( process.execPath, [ self, '--child', mode ], { encoding: 'utf8', maxBuffer: 1 << 28 } );
		if ( r.status !== 0 ) { console.error( r.stderr ); process.exit( 1 ); }
		return JSON.parse( r.stdout );

	};

	const ref = run( 'three' ), eng = run( 'engine' );
	let pass = 0, fail = 0;
	const rel = ( a, b ) => Math.abs( a - b ) <= 1e-6 * Math.max( 1, Math.abs( a ), Math.abs( b ) ) + 1e-6;

	for ( const name of Object.keys( ref ) ) {

		const a = ref[ name ], b = eng[ name ];
		let why = null;
		if ( a.error ) { console.log( `  SKIP ${ name } (fails under three too: ${ a.error })` ); continue; }
		if ( b.error ) why = 'engine error: ' + b.error;
		else if ( a.length !== b.length ) why = `geometry count ${ a.length } vs ${ b.length }`;
		else {

			for ( let i = 0; i < a.length && ! why; i ++ ) {

				const ga = a[ i ], gb = b[ i ];
				const loose = name.startsWith( '~' );
				if ( ! loose && JSON.stringify( ga.index && ga.index[ 0 ] ) !== JSON.stringify( gb.index && gb.index[ 0 ] ) ) why = `geo ${ i } index count`;
				else if ( ga.index && ! name.startsWith( '~' ) && ! rel( ga.index[ 1 ], gb.index[ 1 ] ) ) why = `geo ${ i } index values`;
				// slab() maps 2D triangles onto curved surfaces, so a different (valid)
				// triangulation changes the 3D area slightly; allow 0.1% there
				else if ( loose ? Math.abs( ga.area - gb.area ) > 1e-3 * ga.area : ! rel( ga.area, gb.area ) ) why = `geo ${ i } area ${ ga.area } vs ${ gb.area }`;
				else if ( ga.groups !== gb.groups ) why = `geo ${ i } groups`;
				else if ( Object.keys( ga.attrs ).join() !== Object.keys( gb.attrs ).join() ) why = `geo ${ i } attrs ${ Object.keys( ga.attrs ) } vs ${ Object.keys( gb.attrs ) }`;
				else for ( const n in ga.attrs ) {

					const x = ga.attrs[ n ], y = gb.attrs[ n ];
					// smooth normals depend on the triangulation; loose cases only compare counts for them
					const skipVals = loose && ( n === 'normal' || n === 'tangent' );
					if ( x[ 0 ] !== y[ 0 ] || x[ 1 ] !== y[ 1 ] || ( ! skipVals && ( ! rel( x[ 2 ], y[ 2 ] ) || ! rel( x[ 3 ], y[ 3 ] ) ) ) ) { why = `geo ${ i } attr ${ n }: ${ x } vs ${ y }`; break; }

				}

			}

		}

		if ( why ) { fail ++; console.log( `  FAIL ${ name }: ${ why }` ); } else pass ++;

	}

	console.log( `engine-parity: ${ pass } passed, ${ fail } failed` );
	if ( fail ) process.exit( 1 );

}
