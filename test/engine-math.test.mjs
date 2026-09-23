// Engine CPU math / scene / geometry checks. Run: node test/engine-math.test.mjs
// three.js is imported here ONLY as a numerical reference.

import * as E from '../src/engine/index.js';
import * as T from 'three';
import { RoundedBoxGeometry as TRoundedBox } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { mergeGeometries as tMerge } from 'three/addons/utils/BufferGeometryUtils.js';

let pass = 0, fail = 0;
const failures = [];

function check( name, ok, detail = '' ) {

	if ( ok ) pass ++; else { fail ++; failures.push( name + ( detail ? ' :: ' + detail : '' ) ); }

}

function close( a, b, eps = 1e-6 ) { return Math.abs( a - b ) <= eps * Math.max( 1, Math.abs( a ), Math.abs( b ) ); }

function arrClose( a, b, eps = 1e-5 ) {

	if ( a.length !== b.length ) return `length ${ a.length } vs ${ b.length }`;
	for ( let i = 0; i < a.length; i ++ ) if ( ! close( a[ i ], b[ i ], eps ) ) return `[${ i }] ${ a[ i ] } vs ${ b[ i ] }`;
	return null;

}

function checkArr( name, a, b, eps ) { const e = arrClose( Array.from( a ), Array.from( b ), eps ); check( name, e === null, e ); }

// deterministic RNG
let seed = 42;
const rnd = () => { seed = ( seed * 16807 ) % 2147483647; return seed / 2147483647; };
const rs = ( s = 1 ) => ( rnd() * 2 - 1 ) * s;

const ORDERS = [ 'XYZ', 'YXZ', 'ZXY', 'ZYX', 'YZX', 'XZY' ];

// ---------------------------------------------------------------- matrices
{

	for ( let k = 0; k < 50; k ++ ) {

		const p = new E.Vector3( rs( 10 ), rs( 10 ), rs( 10 ) );
		const q = new E.Quaternion( rs(), rs(), rs(), rs() ).normalize();
		const s = new E.Vector3( 0.1 + rnd() * 3, 0.1 + rnd() * 3, ( 0.1 + rnd() * 3 ) * ( k % 3 === 0 ? - 1 : 1 ) );
		const m = new E.Matrix4().compose( p, q, s );
		const tm = new T.Matrix4().compose( new T.Vector3( p.x, p.y, p.z ), new T.Quaternion( q.x, q.y, q.z, q.w ), new T.Vector3( s.x, s.y, s.z ) );
		checkArr( 'compose matches three', m.elements, tm.elements );

		const p2 = new E.Vector3(), q2 = new E.Quaternion(), s2 = new E.Vector3();
		m.decompose( p2, q2, s2 );
		const m2 = new E.Matrix4().compose( p2, q2, s2 );
		checkArr( 'compose/decompose round trip', m2.elements, m.elements, 1e-5 );

		const inv = m.clone().invert();
		checkArr( 'invert matches three', inv.elements, tm.clone().invert().elements, 1e-5 );
		checkArr( 'M * M^-1 = I', new E.Matrix4().multiplyMatrices( m, inv ).elements, new E.Matrix4().elements, 1e-5 );
		check( 'determinant matches three', close( m.determinant(), tm.determinant(), 1e-6 ) );

		const m3 = new E.Matrix3().getNormalMatrix( m ), tm3 = new T.Matrix3().getNormalMatrix( tm );
		checkArr( 'normal matrix matches three', m3.elements, tm3.elements, 1e-5 );

		const axis = new E.Vector3( rs(), rs(), rs() ).normalize(), ang = rs( 3 );
		checkArr( 'makeRotationAxis', new E.Matrix4().makeRotationAxis( axis, ang ).elements, new T.Matrix4().makeRotationAxis( new T.Vector3( axis.x, axis.y, axis.z ), ang ).elements );

	}

}

// ---------------------------------------------------------------- quaternion / euler
{

	for ( const order of ORDERS ) {

		for ( let k = 0; k < 40; k ++ ) {

			const x = rs( 3 ), y = rs( 1.5 ), z = rs( 3 );
			const e = new E.Euler( x, y, z, order ), te = new T.Euler( x, y, z, order );
			const q = new E.Quaternion().setFromEuler( e ), tq = new T.Quaternion().setFromEuler( te );
			checkArr( `setFromEuler ${ order }`, q.toArray(), tq.toArray() );

			const e2 = new E.Euler().setFromQuaternion( q, order );
			const te2 = new T.Euler().setFromQuaternion( tq, order );
			checkArr( `setFromQuaternion ${ order }`, [ e2.x, e2.y, e2.z ], [ te2.x, te2.y, te2.z ], 1e-5 );

			const q3 = new E.Quaternion().setFromEuler( e2 );
			check( `euler round trip ${ order }`, Math.abs( Math.abs( q3.dot( q ) ) - 1 ) < 1e-9 );

			const m = new E.Matrix4().makeRotationFromEuler( e ), tm = new T.Matrix4().makeRotationFromEuler( te );
			checkArr( `makeRotationFromEuler ${ order }`, m.elements, tm.elements );

		}

	}

	for ( let k = 0; k < 30; k ++ ) {

		const a = new E.Vector3( rs(), rs(), rs() ).normalize(), b = new E.Vector3( rs(), rs(), rs() ).normalize();
		const q = new E.Quaternion().setFromUnitVectors( a, b );
		const tq = new T.Quaternion().setFromUnitVectors( new T.Vector3( a.x, a.y, a.z ), new T.Vector3( b.x, b.y, b.z ) );
		checkArr( 'setFromUnitVectors', q.toArray(), tq.toArray() );
		checkArr( 'setFromUnitVectors maps a->b', a.clone().applyQuaternion( q ).toArray(), b.toArray(), 1e-6 );

		const q2 = new E.Quaternion( rs(), rs(), rs(), rs() ).normalize(), t = rnd();
		const tq2 = new T.Quaternion( q2.x, q2.y, q2.z, q2.w );
		checkArr( 'slerp', q.clone().slerp( q2, t ).toArray(), tq.clone().slerp( tq2, t ).toArray(), 1e-6 );
		checkArr( 'multiply', q.clone().multiply( q2 ).toArray(), tq.clone().multiply( tq2 ).toArray() );

		const m = new E.Matrix4().makeRotationFromQuaternion( q2 );
		checkArr( 'setFromRotationMatrix', new E.Quaternion().setFromRotationMatrix( m ).toArray(), new T.Quaternion().setFromRotationMatrix( new T.Matrix4().fromArray( m.elements ) ).toArray(), 1e-6 );

		const v = new E.Vector3( rs( 5 ), rs( 5 ), rs( 5 ) );
		checkArr( 'applyQuaternion', v.clone().applyQuaternion( q2 ).toArray(), new T.Vector3( v.x, v.y, v.z ).applyQuaternion( tq2 ).toArray() );

	}

	// antiparallel
	const q = new E.Quaternion().setFromUnitVectors( new E.Vector3( 0, 1, 0 ), new E.Vector3( 0, - 1, 0 ) );
	checkArr( 'setFromUnitVectors antiparallel', new E.Vector3( 0, 1, 0 ).applyQuaternion( q ).toArray(), [ 0, - 1, 0 ], 1e-6 );

}

// ---------------------------------------------------------------- Object3D / lookAt
{

	for ( let k = 0; k < 20; k ++ ) {

		const pos = [ rs( 10 ), rs( 10 ), rs( 10 ) ], tgt = [ rs( 10 ), rs( 10 ), rs( 10 ) ];
		const o = new E.Object3D(), to = new T.Object3D();
		o.position.set( ...pos ); to.position.set( ...pos );
		o.lookAt( ...tgt ); to.lookAt( ...tgt );
		checkArr( 'Object3D.lookAt', o.quaternion.toArray(), to.quaternion.toArray(), 1e-6 );
		checkArr( 'rotation synced from quaternion', [ o.rotation.x, o.rotation.y, o.rotation.z ], [ to.rotation.x, to.rotation.y, to.rotation.z ], 1e-5 );

		const c = new E.PerspectiveCamera(), tc = new T.PerspectiveCamera();
		c.position.set( ...pos ); tc.position.set( ...pos );
		c.lookAt( ...tgt ); tc.lookAt( ...tgt );
		checkArr( 'Camera.lookAt', c.quaternion.toArray(), tc.quaternion.toArray(), 1e-6 );
		c.updateMatrixWorld();
		const d = c.getWorldDirection( new E.Vector3() );
		const want = new E.Vector3( ...tgt ).sub( new E.Vector3( ...pos ) ).normalize();
		checkArr( 'camera looks at target', d.toArray(), want.toArray(), 1e-6 );

		// nested lookAt with rotated parent
		const p = new E.Group(), tp = new T.Group();
		p.rotation.set( 0.3, - 0.7, 0.2 ); tp.rotation.set( 0.3, - 0.7, 0.2 );
		p.position.set( 1, 2, 3 ); tp.position.set( 1, 2, 3 );
		const ch = new E.Mesh(), tch = new T.Mesh();
		p.add( ch ); tp.add( tch );
		ch.position.set( ...pos ); tch.position.set( ...pos );
		p.updateMatrixWorld(); tp.updateMatrixWorld();
		ch.lookAt( ...tgt ); tch.lookAt( ...tgt );
		checkArr( 'nested lookAt', ch.quaternion.toArray(), tch.quaternion.toArray(), 1e-6 );
		p.updateMatrixWorld( true ); tp.updateMatrixWorld( true );
		checkArr( 'matrixWorld hierarchy', ch.matrixWorld.elements, tch.matrixWorld.elements, 1e-6 );
		checkArr( 'getWorldPosition', ch.getWorldPosition( new E.Vector3() ).toArray(), tch.getWorldPosition( new T.Vector3() ).toArray(), 1e-6 );
		checkArr( 'worldToLocal', ch.worldToLocal( new E.Vector3( 1, 2, 3 ) ).toArray(), tch.worldToLocal( new T.Vector3( 1, 2, 3 ) ).toArray(), 1e-5 );

	}

	const o = new E.Object3D();
	o.rotation.y = 0.5;
	check( 'euler->quaternion sync', close( o.quaternion.y, Math.sin( 0.25 ) ) );
	o.rotateX( 0.3 ).translateZ( 2 );
	const to = new T.Object3D(); to.rotation.y = 0.5; to.rotateX( 0.3 ).translateZ( 2 );
	checkArr( 'rotateX/translateZ', o.position.toArray(), to.position.toArray() );

	// attach / applyMatrix4 / copy
	{

		const a = new E.Group(), b = new E.Group(), c = new E.Mesh();
		const ta = new T.Group(), tb = new T.Group(), tc = new T.Mesh();
		for ( const [ x, y ] of [ [ a, ta ], [ b, tb ] ] ) { x.position.set( 1, - 2, 3 ); y.position.set( 1, - 2, 3 ); }
		a.rotation.set( 0.4, 0.1, - 0.3, 'YXZ' ); ta.rotation.set( 0.4, 0.1, - 0.3, 'YXZ' );
		b.scale.set( 2, 1, 0.5 ); tb.scale.set( 2, 1, 0.5 );
		c.position.set( 0.3, 0.2, 0.1 ); tc.position.set( 0.3, 0.2, 0.1 );
		a.add( c ); ta.add( tc );
		b.attach( c ); tb.attach( tc );
		checkArr( 'attach keeps world transform', [ ...c.position.toArray(), ...c.quaternion.toArray(), ...c.scale.toArray() ], [ ...tc.position.toArray(), ...tc.quaternion.toArray(), ...tc.scale.toArray() ], 1e-6 );
		const m = new E.Matrix4().makeRotationY( 0.7 ).setPosition( 5, 0, 0 );
		c.applyMatrix4( m ); tc.applyMatrix4( new T.Matrix4().fromArray( m.elements ) );
		checkArr( 'applyMatrix4', [ ...c.position.toArray(), ...c.quaternion.toArray() ], [ ...tc.position.toArray(), ...tc.quaternion.toArray() ], 1e-6 );
		const d = new E.Mesh().copy( c );
		check( 'copy', d.position.equals( c.position ) && d.quaternion.equals( c.quaternion ) && d.geometry === c.geometry );
		c.rotation.order = 'ZYX'; tc.rotation.order = 'ZYX';
		checkArr( 'rotation.order change re-syncs quaternion', c.quaternion.toArray(), tc.quaternion.toArray(), 1e-6 );

	}

	const L = new E.Layers();
	L.set( 3 ); L.enable( 5 );
	check( 'layers set/enable', L.mask === ( ( 1 << 3 ) | ( 1 << 5 ) ) );
	const L2 = new E.Layers(); L2.set( 5 );
	check( 'layers test', L.test( L2 ) );
	L.disable( 5 ); check( 'layers disable', ! L.test( L2 ) );
	L.enableAll(); check( 'layers enableAll', L.isEnabled( 31 ) && L.isEnabled( 0 ) );

	const g = new E.Group(), a = new E.Mesh(), b = new E.Mesh();
	a.name = 'a'; b.name = 'b';
	g.add( a, b );
	check( 'add multiple', g.children.length === 2 && a.parent === g );
	check( 'getObjectByName', g.getObjectByName( 'b' ) === b );
	b.removeFromParent();
	check( 'removeFromParent', g.children.length === 1 && b.parent === null );
	let n = 0; g.traverse( () => n ++ ); check( 'traverse', n === 2 );
	a.visible = false; n = 0; g.traverseVisible( () => n ++ ); check( 'traverseVisible', n === 1 );
	const gc = g.clone(); check( 'clone recursive', gc.children.length === 1 && gc.children[ 0 ] !== a );

	const im = new E.InstancedMesh( new E.BoxGeometry(), null, 4 );
	const m = new E.Matrix4().makeTranslation( 1, 2, 3 );
	im.setMatrixAt( 2, m );
	checkArr( 'instanced get/setMatrixAt', im.getMatrixAt( 2, new E.Matrix4() ).elements, m.elements );
	im.setColorAt( 1, new E.Color( 0.5, 0.25, 1 ) );
	check( 'instanceColor', im.instanceColor.count === 4 && im.instanceColor.array[ 3 ] === 0.5 && im.instanceColor.array[ 0 ] === 1 );
	const v0 = im.instanceMatrix.version; im.instanceMatrix.needsUpdate = true;
	check( 'needsUpdate bumps version', im.instanceMatrix.version === v0 + 1 );
	im.computeBoundingBox();
	check( 'instanced bounding box', close( im.boundingBox.max.z, 3.5 ) && close( im.boundingBox.min.x, - 0.5 ) );

}

// ---------------------------------------------------------------- cameras / projection / frustum
{

	const cam = new E.PerspectiveCamera( 62, 16 / 9, 0.06, 60000 );
	cam.updateMatrixWorld();
	const ndc = ( z ) => new E.Vector3( 0, 0, - z ).applyMatrix4( cam.projectionMatrix ).z;
	check( 'near -> depth 1', close( ndc( 0.06 ), 1, 1e-9 ), ndc( 0.06 ) );
	check( 'far -> depth 0', Math.abs( ndc( 60000 ) ) < 1e-9, ndc( 60000 ) );
	check( 'mid depth in (0,1) decreasing', ndc( 1 ) < 1 && ndc( 1 ) > ndc( 100 ) && ndc( 100 ) > 0 );

	const tcam = new T.PerspectiveCamera( 62, 16 / 9, 0.06, 60000 );
	tcam.coordinateSystem = T.WebGPUCoordinateSystem;
	tcam._reversedDepth = true;
	tcam.updateProjectionMatrix();
	checkArr( 'perspective matches three (WebGPU reversed)', cam.projectionMatrix.elements, tcam.projectionMatrix.elements, 1e-9 );
	checkArr( 'projectionMatrixInverse', new E.Matrix4().multiplyMatrices( cam.projectionMatrix, cam.projectionMatrixInverse ).elements, new E.Matrix4().elements, 1e-6 );

	cam.setViewOffset( 1920, 1080, 0.37, - 0.21, 1920, 1080 );
	tcam.setViewOffset( 1920, 1080, 0.37, - 0.21, 1920, 1080 );
	checkArr( 'setViewOffset jitter matches three', cam.projectionMatrix.elements, tcam.projectionMatrix.elements, 1e-9 );
	cam.clearViewOffset(); tcam.clearViewOffset();
	checkArr( 'clearViewOffset', cam.projectionMatrix.elements, tcam.projectionMatrix.elements, 1e-9 );

	const inf = new E.PerspectiveCamera( 60, 1, 0.1, 1000 );
	inf.infiniteFar = true; inf.updateProjectionMatrix();
	const infZ = ( z ) => { const v = new E.Vector4( 0, 0, - z, 1 ).applyMatrix4( inf.projectionMatrix ); return v.z / v.w; };
	check( 'infinite far: near->1, 1e9->~0', close( infZ( 0.1 ), 1, 1e-9 ) && infZ( 1e9 ) < 1e-9 && infZ( 1e9 ) > 0 );

	const oc = new E.OrthographicCamera( - 10, 10, 5, - 5, 1, 101 );
	const toc = new T.OrthographicCamera( - 10, 10, 5, - 5, 1, 101 );
	toc.coordinateSystem = T.WebGPUCoordinateSystem; toc._reversedDepth = true; toc.updateProjectionMatrix();
	checkArr( 'ortho matches three (WebGPU reversed)', oc.projectionMatrix.elements, toc.projectionMatrix.elements, 1e-9 );
	const oz = ( z ) => new E.Vector3( 0, 0, - z ).applyMatrix4( oc.projectionMatrix ).z;
	check( 'ortho near->1 far->0', close( oz( 1 ), 1 ) && Math.abs( oz( 101 ) ) < 1e-9 );
	const ocn = new E.OrthographicCamera( - 1, 1, 1, - 1, 0, 2 ); ocn.reversedDepth = false; ocn.updateProjectionMatrix();
	check( 'ortho non-reversed near->0 far->1', Math.abs( new E.Vector3( 0, 0, 0 ).applyMatrix4( ocn.projectionMatrix ).z ) < 1e-9 && close( new E.Vector3( 0, 0, - 2 ).applyMatrix4( ocn.projectionMatrix ).z, 1 ) );

	// frustum culling with reversed-Z view-projection
	const c2 = new E.PerspectiveCamera( 60, 1.5, 0.5, 500 );
	c2.position.set( 3, 4, 5 );
	c2.lookAt( 3, 4, - 100 );
	c2.updateMatrixWorld();
	const vp = new E.Matrix4().multiplyMatrices( c2.projectionMatrix, c2.matrixWorldInverse );
	const f = new E.Frustum().setFromProjectionMatrix( vp, c2.coordinateSystem, c2.reversedDepth );
	const S = ( x, y, z, r ) => new E.Sphere( new E.Vector3( x, y, z ), r );
	check( 'frustum: in front visible', f.intersectsSphere( S( 3, 4, - 20, 1 ) ) );
	check( 'frustum: behind culled', ! f.intersectsSphere( S( 3, 4, 20, 1 ) ) );
	check( 'frustum: beyond far culled', ! f.intersectsSphere( S( 3, 4, 5 - 520, 1 ) ) );
	check( 'frustum: straddling far visible', f.intersectsSphere( S( 3, 4, 5 - 500.5, 1 ) ) );
	check( 'frustum: nearer than near culled', ! f.intersectsSphere( S( 3, 4, 5 - 0.2, 0.1 ) ) );
	check( 'frustum: far left culled', ! f.intersectsSphere( S( - 200, 4, - 20, 1 ) ) );
	check( 'frustum: box visible', f.intersectsBox( new E.Box3( new E.Vector3( 2, 3, - 30 ), new E.Vector3( 4, 5, - 28 ) ) ) );
	check( 'frustum: box above culled', ! f.intersectsBox( new E.Box3( new E.Vector3( 2, 300, - 30 ), new E.Vector3( 4, 305, - 28 ) ) ) );
	check( 'frustum: default args = reversed WebGPU', new E.Frustum().setFromProjectionMatrix( vp ).intersectsSphere( S( 3, 4, - 20, 1 ) ) && ! new E.Frustum().setFromProjectionMatrix( vp ).intersectsSphere( S( 3, 4, 20, 1 ) ) );

	// compare against three's frustum on random spheres
	const tc2 = new T.PerspectiveCamera( 60, 1.5, 0.5, 500 );
	tc2.coordinateSystem = T.WebGPUCoordinateSystem; tc2._reversedDepth = true; tc2.updateProjectionMatrix();
	tc2.position.set( 3, 4, 5 ); tc2.lookAt( 3, 4, - 100 ); tc2.updateMatrixWorld();
	const tvp = new T.Matrix4().multiplyMatrices( tc2.projectionMatrix, tc2.matrixWorldInverse );
	const tf = new T.Frustum().setFromProjectionMatrix( tvp, T.WebGPUCoordinateSystem, true );
	let agree = 0;
	for ( let k = 0; k < 2000; k ++ ) {

		const x = rs( 600 ), y = rs( 600 ), z = rs( 600 ), r = rnd() * 20;
		if ( f.intersectsSphere( S( x, y, z, r ) ) === tf.intersectsSphere( new T.Sphere( new T.Vector3( x, y, z ), r ) ) ) agree ++;

	}

	check( 'frustum agrees with three on 2000 spheres', agree === 2000, agree );

	// infinite far frustum never culls by distance
	const ci = new E.PerspectiveCamera( 60, 1, 0.1, 100 ); ci.infiniteFar = true; ci.updateProjectionMatrix(); ci.updateMatrixWorld();
	const fi = new E.Frustum().setFromProjectionMatrix( new E.Matrix4().multiplyMatrices( ci.projectionMatrix, ci.matrixWorldInverse ) );
	check( 'infinite-far frustum keeps distant', fi.intersectsSphere( S( 0, 0, - 1e7, 1 ) ) && ! fi.intersectsSphere( S( 0, 0, 10, 1 ) ) );
	check( 'infinite-far frustum box', fi.intersectsBox( new E.Box3( new E.Vector3( - 1, - 1, - 1e6 ), new E.Vector3( 1, 1, - 1e6 + 2 ) ) ) );

	// project()
	const pv = new E.Vector3( 3, 4, - 20 ).project( c2 );
	check( 'project center', close( pv.x, 0, 1e-9 ) && close( pv.y, 0, 1e-9 ) && pv.z > 0 && pv.z < 1 );
	const up = pv.clone().unproject( c2 );
	checkArr( 'unproject round trip', up.toArray(), [ 3, 4, - 20 ], 1e-6 );

}

// ---------------------------------------------------------------- geometry generators vs three
function cmpGeo( name, g, tg, eps = 1e-5 ) {

	for ( const attr of [ 'position', 'normal', 'uv' ] ) {

		const a = g.getAttribute( attr ), b = tg.getAttribute( attr );
		check( `${ name } has ${ attr }`, !! a === !! b );
		if ( a && b ) {

			check( `${ name } ${ attr } count`, a.count === b.count, `${ a.count } vs ${ b.count }` );
			checkArr( `${ name } ${ attr } values`, a.array, b.array, eps );

		}

	}

	check( `${ name } attribute order`, Object.keys( g.attributes ).join() === Object.keys( tg.attributes ).join(), Object.keys( g.attributes ) + ' vs ' + Object.keys( tg.attributes ) );
	check( `${ name } indexed`, ( g.index === null ) === ( tg.index === null ) );
	if ( g.index && tg.index ) checkArr( `${ name } index`, g.index.array, tg.index.array, 0 );
	check( `${ name } groups`, JSON.stringify( g.groups ) === JSON.stringify( tg.groups ), JSON.stringify( g.groups ) + ' vs ' + JSON.stringify( tg.groups ) );

	const uv = g.getAttribute( 'uv' );
	if ( uv ) {

		let mn = Infinity, mx = - Infinity;
		for ( const v of uv.array ) { mn = Math.min( mn, v ); mx = Math.max( mx, v ); }
		let tmn = Infinity, tmx = - Infinity;
		for ( const v of tg.getAttribute( 'uv' ).array ) { tmn = Math.min( tmn, v ); tmx = Math.max( tmx, v ); }
		check( `${ name } uv range`, close( mn, tmn, 1e-6 ) && close( mx, tmx, 1e-6 ), `[${ mn },${ mx }] vs [${ tmn },${ tmx }]` );

	}

}

{

	cmpGeo( 'Plane', new E.PlaneGeometry( 2, 3, 4, 5 ), new T.PlaneGeometry( 2, 3, 4, 5 ) );
	cmpGeo( 'Plane default', new E.PlaneGeometry(), new T.PlaneGeometry() );
	cmpGeo( 'Box', new E.BoxGeometry( 1, 2, 3, 2, 3, 4 ), new T.BoxGeometry( 1, 2, 3, 2, 3, 4 ) );
	cmpGeo( 'Box default', new E.BoxGeometry( 0.4, 0.2, 0.9 ), new T.BoxGeometry( 0.4, 0.2, 0.9 ) );
	cmpGeo( 'Sphere', new E.SphereGeometry( 2, 12, 7 ), new T.SphereGeometry( 2, 12, 7 ) );
	cmpGeo( 'Sphere partial', new E.SphereGeometry( 1, 10, 6, 0.3, 4, 0.2, 2 ), new T.SphereGeometry( 1, 10, 6, 0.3, 4, 0.2, 2 ) );
	cmpGeo( 'Cylinder', new E.CylinderGeometry( 0.43, 0.37, 0.36, 4, 1 ), new T.CylinderGeometry( 0.43, 0.37, 0.36, 4, 1 ) );
	cmpGeo( 'Cylinder open arc', new E.CylinderGeometry( 1, 2, 3, 9, 3, true, 0.5, 2 ), new T.CylinderGeometry( 1, 2, 3, 9, 3, true, 0.5, 2 ) );
	cmpGeo( 'Cylinder default', new E.CylinderGeometry(), new T.CylinderGeometry() );
	cmpGeo( 'Cone', new E.ConeGeometry( 0.12, 0.34, 4 ), new T.ConeGeometry( 0.12, 0.34, 4 ) );
	cmpGeo( 'Cone segs', new E.ConeGeometry( 1, 2, 7, 3 ), new T.ConeGeometry( 1, 2, 7, 3 ) );
	cmpGeo( 'Circle', new E.CircleGeometry( 0.042, 20 ), new T.CircleGeometry( 0.042, 20 ) );
	cmpGeo( 'Circle arc', new E.CircleGeometry( 2, 5, 0.3, 2 ), new T.CircleGeometry( 2, 5, 0.3, 2 ) );
	cmpGeo( 'Torus', new E.TorusGeometry( 1, 0.3, 8, 20, 5 ), new T.TorusGeometry( 1, 0.3, 8, 20, 5 ) );
	cmpGeo( 'Torus default', new E.TorusGeometry(), new T.TorusGeometry() );
	for ( const d of [ 0, 1, 2 ] ) cmpGeo( `Icosahedron d${ d }`, new E.IcosahedronGeometry( 1.5, d ), new T.IcosahedronGeometry( 1.5, d ) );

	const lp = [ [ 0, - 1 ], [ 0.5, - 0.8 ], [ 0.7, 0 ], [ 0.4, 0.6 ], [ 0, 1 ] ];
	cmpGeo( 'Lathe', new E.LatheGeometry( lp.map( ( p ) => new E.Vector2( ...p ) ), 9 ), new T.LatheGeometry( lp.map( ( p ) => new T.Vector2( ...p ) ), 9 ) );
	cmpGeo( 'Lathe arc', new E.LatheGeometry( lp.map( ( p ) => new E.Vector2( ...p ) ), 5, 0.2, 3 ), new T.LatheGeometry( lp.map( ( p ) => new T.Vector2( ...p ) ), 5, 0.2, 3 ) );

	for ( const [ w, h, d, s, r ] of [ [ 1, 1, 1, 2, 0.1 ], [ 0.6, 0.3, 1.2, 3, 0.05 ], [ 2, 1, 0.5, 1, 0.2 ] ] ) {

		cmpGeo( `RoundedBox ${ w }x${ h }x${ d } s${ s }`, new E.RoundedBoxGeometry( w, h, d, s, r ), new TRoundedBox( w, h, d, s, r ), 1e-4 );

	}

	// curves + tube
	const pts = [ [ 0, 0, 0 ], [ 1, 2, 0 ], [ 3, 2, 1 ], [ 4, 0, 3 ], [ 2, - 1, 4 ] ];
	for ( const type of [ 'centripetal', 'chordal', 'catmullrom' ] ) {

		for ( const closed of [ false, true ] ) {

			const c = new E.CatmullRomCurve3( pts.map( ( p ) => new E.Vector3( ...p ) ), closed, type, 0.3 );
			const tc = new T.CatmullRomCurve3( pts.map( ( p ) => new T.Vector3( ...p ) ), closed, type, 0.3 );
			const tag = `${ type } closed=${ closed }`;
			check( `curve length ${ tag }`, close( c.getLength(), tc.getLength(), 1e-9 ) );
			const pa = [], pb = [];
			for ( let i = 0; i <= 20; i ++ ) {

				const t = i / 20;
				pa.push( ...c.getPoint( t ).toArray(), ...c.getPointAt( t ).toArray(), ...c.getTangentAt( t ).toArray(), ...c.getTangent( t ).toArray() );
				pb.push( ...tc.getPoint( t ).toArray(), ...tc.getPointAt( t ).toArray(), ...tc.getTangentAt( t ).toArray(), ...tc.getTangent( t ).toArray() );

			}

			checkArr( `curve points ${ tag }`, pa, pb, 1e-9 );
			checkArr( `getSpacedPoints ${ tag }`, c.getSpacedPoints( 7 ).flatMap( ( p ) => p.toArray() ), tc.getSpacedPoints( 7 ).flatMap( ( p ) => p.toArray() ), 1e-9 );
			checkArr( `getPoints ${ tag }`, c.getPoints( 7 ).flatMap( ( p ) => p.toArray() ), tc.getPoints( 7 ).flatMap( ( p ) => p.toArray() ), 1e-9 );
			const fr = c.computeFrenetFrames( 16, closed ), tfr = tc.computeFrenetFrames( 16, closed );
			checkArr( `frenet normals ${ tag }`, fr.normals.flatMap( ( v ) => v.toArray() ), tfr.normals.flatMap( ( v ) => v.toArray() ), 1e-7 );
			checkArr( `frenet binormals ${ tag }`, fr.binormals.flatMap( ( v ) => v.toArray() ), tfr.binormals.flatMap( ( v ) => v.toArray() ), 1e-7 );
			cmpGeo( `Tube ${ tag }`, new E.TubeGeometry( c, 24, 0.2, 6, closed ), new T.TubeGeometry( tc, 24, 0.2, 6, closed ) );

		}

	}

	// computeVertexNormals / tangents / toNonIndexed / bounds
	const g = new E.SphereGeometry( 1, 8, 6 ), tg = new T.SphereGeometry( 1, 8, 6 );
	g.computeVertexNormals(); tg.computeVertexNormals();
	checkArr( 'computeVertexNormals indexed', g.attributes.normal.array, tg.attributes.normal.array, 1e-5 );
	g.computeTangents(); tg.computeTangents();
	checkArr( 'computeTangents', g.attributes.tangent.array, tg.attributes.tangent.array, 1e-4 );
	const ni = g.toNonIndexed(), tni = tg.toNonIndexed();
	checkArr( 'toNonIndexed position', ni.attributes.position.array, tni.attributes.position.array );
	ni.computeVertexNormals(); tni.computeVertexNormals();
	checkArr( 'computeVertexNormals flat', ni.attributes.normal.array, tni.attributes.normal.array, 1e-5 );
	const b = new E.BoxGeometry( 1, 2, 3 ).translate( 1, 0, 0 ).rotateY( 0.4 ).scale( 2, 1, 1 );
	const tb = new T.BoxGeometry( 1, 2, 3 ).translate( 1, 0, 0 ).rotateY( 0.4 ).scale( 2, 1, 1 );
	checkArr( 'transform position', b.attributes.position.array, tb.attributes.position.array );
	checkArr( 'transform normal', b.attributes.normal.array, tb.attributes.normal.array );
	b.computeBoundingBox(); tb.computeBoundingBox(); b.computeBoundingSphere(); tb.computeBoundingSphere();
	checkArr( 'boundingBox', [ ...b.boundingBox.min.toArray(), ...b.boundingBox.max.toArray() ], [ ...tb.boundingBox.min.toArray(), ...tb.boundingBox.max.toArray() ] );
	checkArr( 'boundingSphere', [ ...b.boundingSphere.center.toArray(), b.boundingSphere.radius ], [ ...tb.boundingSphere.center.toArray(), tb.boundingSphere.radius ] );
	b.center(); b.computeBoundingBox();
	check( 'center()', b.boundingBox.getCenter( new E.Vector3() ).length() < 1e-6 );

	let disposed = 0;
	b.addEventListener( 'dispose', () => disposed ++ );
	b.dispose();
	check( 'dispose event', disposed === 1 );

	// interleaved attributes
	const data = new Float32Array( [ 1, 2, 3, 9, 4, 5, 6, 9 ] );
	const ib = new E.InterleavedBuffer( data, 4 );
	const ia = new E.InterleavedBufferAttribute( ib, 3, 0 );
	check( 'interleaved count/get', ia.count === 2 && ia.getY( 1 ) === 5 && ia.getX( 0 ) === 1 );
	ia.setZ( 1, 7 ); check( 'interleaved set', data[ 6 ] === 7 );
	const v0 = ib.version; ia.needsUpdate = true; check( 'interleaved needsUpdate', ib.version === v0 + 1 );
	checkArr( 'interleaved clone de-interleaves', ia.clone().array, [ 1, 2, 3, 4, 5, 7 ] );

	// BufferAttribute helpers
	const at = new E.Float32BufferAttribute( [ 1, 0, 0, 0, 1, 0 ], 3 );
	at.applyMatrix4( new E.Matrix4().makeTranslation( 0, 0, 5 ) );
	check( 'attr applyMatrix4', at.getZ( 1 ) === 5 );
	at.setXYZ( 0, 7, 8, 9 ); check( 'attr setXYZ', at.getX( 0 ) === 7 && at.getZ( 0 ) === 9 );
	const na = new E.Uint8BufferAttribute( [ 255, 0 ], 1, true );
	check( 'normalized get', na.getX( 0 ) === 1 );
	const big = new E.BufferGeometry().setIndex( [ 0, 70000, 1 ] );
	check( 'setIndex uint32 when needed', big.index.array instanceof Uint32Array );
	check( 'setIndex uint16 default', new E.BufferGeometry().setIndex( [ 0, 1, 2 ] ).index.array instanceof Uint16Array );

}

// ---------------------------------------------------------------- mergeGeometries / mergeVertices
{

	const parts = [ new E.BoxGeometry( 1, 1, 1 ), new E.CylinderGeometry( 0.5, 0.5, 1, 8 ).translate( 2, 0, 0 ), new E.SphereGeometry( 0.5, 8, 6 ) ];
	const tparts = [ new T.BoxGeometry( 1, 1, 1 ), new T.CylinderGeometry( 0.5, 0.5, 1, 8 ).translate( 2, 0, 0 ), new T.SphereGeometry( 0.5, 8, 6 ) ];
	for ( const useGroups of [ false, true ] ) {

		const m = E.mergeGeometries( parts, useGroups ), tm = tMerge( tparts, useGroups );
		check( `merge vertex count g=${ useGroups }`, m.attributes.position.count === tm.attributes.position.count );
		check( `merge index count g=${ useGroups }`, m.index.count === tm.index.count );
		checkArr( `merge index g=${ useGroups }`, m.index.array, tm.index.array, 0 );
		checkArr( `merge positions g=${ useGroups }`, m.attributes.position.array, tm.attributes.position.array );
		check( `merge groups g=${ useGroups }`, JSON.stringify( m.groups ) === JSON.stringify( tm.groups ), JSON.stringify( m.groups ) + ' vs ' + JSON.stringify( tm.groups ) );

	}

	const nonIdx = [ new E.BoxGeometry().toNonIndexed(), new E.PlaneGeometry().toNonIndexed() ];
	const mn = E.mergeGeometries( nonIdx );
	check( 'merge non-indexed', mn.index === null && mn.attributes.position.count === 36 + 6 );
	const orig = console.error; console.error = () => {};
	check( 'merge mismatch -> null', E.mergeGeometries( [ new E.BoxGeometry(), new E.BoxGeometry().toNonIndexed() ] ) === null );
	console.error = orig;

	const welded = E.mergeVertices( new E.BoxGeometry().toNonIndexed() );
	check( 'mergeVertices box (with uv/normal) -> 24 verts', welded.attributes.position.count === 24 && welded.index.count === 36 );
	const g = new E.BoxGeometry(); g.deleteAttribute( 'normal' ); g.deleteAttribute( 'uv' );
	check( 'mergeVertices position-only box -> 8 verts', E.mergeVertices( g ).attributes.position.count === 8 );

}

// ---------------------------------------------------------------- color
{

	for ( const hex of [ 0x000000, 0xffffff, 0x3a7f22, 0x123456, 0xff8000, 0x0a0b0c ] ) {

		const c = new E.Color( hex ), tc = new T.Color( hex );
		checkArr( `Color(hex ${ hex.toString( 16 ) }) linear`, [ c.r, c.g, c.b ], [ tc.r, tc.g, tc.b ], 1e-6 );
		check( `getHex round trip ${ hex.toString( 16 ) }`, c.getHex() === hex && c.getHexString() === tc.getHexString() );
		const o = c.clone().offsetHSL( 0.03, - 0.05, 0.04 ), to = tc.clone().offsetHSL( 0.03, - 0.05, 0.04 );
		checkArr( `offsetHSL ${ hex.toString( 16 ) }`, [ o.r, o.g, o.b ], [ to.r, to.g, to.b ], 1e-6 );

	}

	const c = new E.Color().setRGB( 0.5, 0.2, 0.9, E.SRGBColorSpace ), tc = new T.Color().setRGB( 0.5, 0.2, 0.9, T.SRGBColorSpace );
	checkArr( 'setRGB sRGB', [ c.r, c.g, c.b ], [ tc.r, tc.g, tc.b ], 1e-6 );
	const h = new E.Color().setHSL( 0.6, 0.5, 0.4 ), th = new T.Color().setHSL( 0.6, 0.5, 0.4 );
	checkArr( 'setHSL', [ h.r, h.g, h.b ], [ th.r, th.g, th.b ], 1e-6 );
	const s = new E.Color( '#ff8000' ), ts = new T.Color( '#ff8000' );
	checkArr( 'setStyle hex', [ s.r, s.g, s.b ], [ ts.r, ts.g, ts.b ], 1e-6 );
	const l = new E.Color( 1, 0, 0 ).lerp( new E.Color( 0, 0, 1 ), 0.25 ).multiplyScalar( 2 );
	checkArr( 'lerp/multiplyScalar', [ l.r, l.g, l.b ], [ 1.5, 0, 0.5 ] );
	check( 'Color(r,g,b) is linear', new E.Color( 0.5, 0.25, 1 ).g === 0.25 );

}

// ---------------------------------------------------------------- misc
{

	const M = E.MathUtils;
	check( 'clamp', M.clamp( 5, 0, 1 ) === 1 && M.clamp( - 1, 0, 1 ) === 0 );
	check( 'smoothstep', M.smoothstep( 0.5, 0, 1 ) === 0.5 && M.smoothstep( 2, 0, 1 ) === 1 );
	check( 'mapLinear', M.mapLinear( 5, 0, 10, 100, 200 ) === 150 );
	check( 'euclideanModulo', M.euclideanModulo( - 1, 3 ) === 2 );
	check( 'pingpong', close( M.pingpong( 1.25, 1 ), 0.75 ) );
	check( 'damp', close( M.damp( 0, 1, 2, 0.1 ), T.MathUtils.damp( 0, 1, 2, 0.1 ) ) );
	check( 'inverseLerp', M.inverseLerp( 2, 4, 3 ) === 0.5 );
	check( 'deg/rad', close( M.radToDeg( M.degToRad( 37 ) ), 37 ) );
	const r1 = M.seededRandom( 7 ), r2 = M.seededRandom( 7 );
	check( 'seededRandom deterministic', r1 === r2 && r1 >= 0 && r1 < 1 );
	check( 'randInt range', [ ...Array( 200 ) ].every( () => { const v = M.randInt( 2, 4 ); return v >= 2 && v <= 4 && Number.isInteger( v ); } ) );
	check( 'uuid format', /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test( M.generateUUID() ) );

	for ( const v of [ 0, 1, - 1, 0.5, 3.14159, 65504, 1e-5, 6.1e-5, - 123.456, 0.1, 1 / 3, 2049, 1e-8 ] ) {

		check( `toHalfFloat ${ v }`, E.DataUtils.toHalfFloat( v ) === T.DataUtils.toHalfFloat( v ), `${ E.DataUtils.toHalfFloat( v ) } vs ${ T.DataUtils.toHalfFloat( v ) }` );
		check( `fromHalfFloat ${ v }`, close( E.DataUtils.fromHalfFloat( E.DataUtils.toHalfFloat( v ) ), T.DataUtils.fromHalfFloat( T.DataUtils.toHalfFloat( v ) ), 1e-9 ) );

	}

	// triangulation: area of triangles = outer - holes, indices valid
	const V = ( x, y ) => new E.Vector2( x, y );
	const triArea = ( P, tris ) => tris.reduce( ( s, [ a, b, c ] ) => s + Math.abs( ( P[ b ].x - P[ a ].x ) * ( P[ c ].y - P[ a ].y ) - ( P[ c ].x - P[ a ].x ) * ( P[ b ].y - P[ a ].y ) ) / 2, 0 );
	{

		const outer = [ V( 0, 0 ), V( 10, 0 ), V( 10, 6 ), V( 5, 8 ), V( 0, 6 ) ];
		const hole1 = [ V( 2, 2 ), V( 2, 4 ), V( 4, 4 ), V( 4, 2 ) ];
		const hole2 = [ V( 6, 1 ), V( 6, 3 ), V( 8, 3 ), V( 8, 1 ) ];
		const want = Math.abs( E.ShapeUtils.area( outer ) ) - 4 - 4;
		const tris = E.ShapeUtils.triangulateShape( outer.slice(), [ hole1.slice(), hole2.slice() ] );
		const P = outer.concat( hole1, hole2 );
		check( 'triangulate with holes area', close( triArea( P, tris ), want, 1e-9 ), `${ triArea( P, tris ) } vs ${ want }` );
		check( 'triangulate with holes count', tris.length === P.length + 2 * 2 - 2, tris.length );
		const tt = T.ShapeUtils.triangulateShape( outer.map( ( p ) => new T.Vector2( p.x, p.y ) ), [ hole1, hole2 ].map( ( h ) => h.map( ( p ) => new T.Vector2( p.x, p.y ) ) ) );
		check( 'triangulate count matches three', tt.length === tris.length );
		// winding follows the (CCW) contour
		check( 'triangulate CCW winding', tris.every( ( [ a, b, c ] ) => ( P[ b ].x - P[ a ].x ) * ( P[ c ].y - P[ a ].y ) - ( P[ c ].x - P[ a ].x ) * ( P[ b ].y - P[ a ].y ) > 0 ) );

	}

	{

		// concave (U shape), duplicated closing point, CW input
		const u = [ V( 0, 0 ), V( 0, 5 ), V( 1, 5 ), V( 1, 1 ), V( 4, 1 ), V( 4, 5 ), V( 5, 5 ), V( 5, 0 ), V( 0, 0 ) ];
		check( 'isClockWise', E.ShapeUtils.isClockWise( u ) === T.ShapeUtils.isClockWise( u.map( ( p ) => new T.Vector2( p.x, p.y ) ) ) );
		const tris = E.ShapeUtils.triangulateShape( u, [] );
		check( 'dup end point removed', u.length === 8 );
		check( 'concave area', close( triArea( u, tris ), Math.abs( E.ShapeUtils.area( u ) ), 1e-9 ) && tris.length === 6 );

	}

	{

		// circle with circular hole (the boat "porthole" case)
		const outer = [], hole = [];
		for ( let i = 0; i < 48; i ++ ) { const a = i / 48 * Math.PI * 2; outer.push( V( Math.cos( a ) * 2, Math.sin( a ) * 1.2 ) ); }
		for ( let i = 0; i < 24; i ++ ) { const a = - i / 24 * Math.PI * 2; hole.push( V( 0.5 + Math.cos( a ) * 0.4, 0.1 + Math.sin( a ) * 0.4 ) ); }
		const want = Math.abs( E.ShapeUtils.area( outer ) ) - Math.abs( E.ShapeUtils.area( hole ) );
		const tris = E.ShapeUtils.triangulateShape( outer, [ hole ] );
		check( 'porthole area', close( triArea( outer.concat( hole ), tris ), want, 1e-9 ), `${ triArea( outer.concat( hole ), tris ) } vs ${ want }` );

	}

	const tm = new E.Timer();
	tm.update( 1000 ); tm.update( 1016 );
	check( 'Timer delta', close( tm.getDelta(), 0.016 ) );
	tm.update( 1050 ); check( 'Timer elapsed', tm.getElapsed() > 0.049 );

	const box = new E.Box3().setFromPoints( [ new E.Vector3( 1, 2, 3 ), new E.Vector3( - 1, 0, 5 ) ] );
	check( 'Box3 containsPoint', box.containsPoint( new E.Vector3( 0, 1, 4 ) ) && ! box.containsPoint( new E.Vector3( 0, 3, 4 ) ) );
	const bs = box.getBoundingSphere( new E.Sphere() );
	check( 'Box3 bounding sphere', close( bs.radius, Math.sqrt( 12 ) / 2 ) );
	const sp = new E.Sphere( new E.Vector3(), 1 ).applyMatrix4( new E.Matrix4().makeScale( 1, 3, 2 ) );
	check( 'Sphere applyMatrix4', sp.radius === 3 );
	const ray = new E.Ray( new E.Vector3( 0, 0, 5 ), new E.Vector3( 0, 0, - 1 ) );
	const hit = ray.intersectSphere( new E.Sphere( new E.Vector3(), 1 ), new E.Vector3() );
	check( 'Ray intersectSphere', hit && close( hit.z, 1 ) );
	check( 'Ray intersectBox', close( ray.intersectBox( new E.Box3( new E.Vector3( - 1, - 1, - 1 ), new E.Vector3( 1, 1, 1 ) ), new E.Vector3() ).z, 1 ) );
	const sph = new E.Spherical().setFromVector3( new E.Vector3( 1, 2, 3 ) );
	checkArr( 'Spherical round trip', new E.Vector3().setFromSpherical( sph ).toArray(), [ 1, 2, 3 ], 1e-9 );

}

console.log( `engine-math: ${ pass } passed, ${ fail } failed` );
if ( fail ) {

	for ( const f of failures.slice( 0, 60 ) ) console.log( '  FAIL ' + f );
	process.exit( 1 );

}
