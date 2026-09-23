import * as THREE from 'three/webgpu';
import { Fn, uniform, vec4, positionWorld, mrt } from 'three/tsl';

// Motion vectors for geometry that doesn't move in the world (only the camera does).
// three's velocity node tracks previous instance matrices with an extra vertex buffer, which
// pushes heavily instanced materials (vegetation) past the 8 vertex buffer limit. Static
// instances don't need it: reproject the world position with last frame's camera instead.
const currViewProj = uniform( new THREE.Matrix4() ).setName( 'velCurrViewProj' );
export const prevViewProj = uniform( new THREE.Matrix4() ).setName( 'velPrevViewProj' );
const _m = new THREE.Matrix4();
let _hasPrev = false;

export const staticVelocity = Fn( () => {

	const p = vec4( positionWorld, 1 );
	const c = currViewProj.mul( p );
	const q = prevViewProj.mul( p );
	return c.xy.div( c.w ).sub( q.xy.div( q.w ) );

} )();

export const staticVelocityMRT = mrt( { velocity: staticVelocity } );

// Call once per frame before the (jittered) scene render, with the unjittered camera.
export function updateCameraVelocity( camera ) {

	camera.updateMatrixWorld();
	_m.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse );
	if ( _hasPrev ) prevViewProj.value.copy( currViewProj.value );
	else prevViewProj.value.copy( _m );
	currViewProj.value.copy( _m );
	_hasPrev = true;

}

// Use the static velocity for every material under root.
export function useStaticVelocity( root ) {

	root.traverse( ( o ) => {

		if ( ! o.material ) return;
		for ( const m of Array.isArray( o.material ) ? o.material : [ o.material ] ) m.mrtNode = staticVelocityMRT;

	} );

}
