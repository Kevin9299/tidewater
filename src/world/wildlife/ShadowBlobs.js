import * as THREE from 'three/webgpu';
import { Fn, vec2, vec3, vec4, attribute, varyingProperty, mrt, exp, dot, max, length, select, float } from 'three/tsl';
import { G } from '../../core/Globals.js';
import { LAYERS } from '../../core/SceneRenderer.js';
import { InstanceRecords, instancedMesh } from './Kit.js';

// Soft contact shadows under the small animals on the ground. Crabs and sanderlings are only a
// few centimetres tall: the sun shadow map (and its normal bias) can't resolve them, and without
// a shadow they seem to hover. One instanced quad each, drawn in the late (transparent) pass and
// darkening what is below (premultiplied black; the velocity target is left untouched): a round
// occlusion blob right under the body plus a sun shadow stretched away from the key light.
//
// Record (2 vec4): ( ground position, radius ), ( height of the body above the ground, darkness,
// ground slope dh/dx, dh/dz ) - the quad lies on the local slope

export class ShadowBlobs {

	constructor( { capacity = 160 } = {} ) {

		this.records = new InstanceRecords( 'blobInstances', capacity, 2 );
		const R = this.records;
		const g = new THREE.PlaneGeometry( 2, 2 );
		g.rotateX( - Math.PI / 2 );
		const vUV = varyingProperty( 'vec2', 'vBlobUV' );
		const vInfo = varyingProperty( 'vec4', 'vBlobInfo' );

		const mat = new THREE.MeshBasicNodeMaterial();
		mat.name = 'ContactShadows';
		mat.transparent = true;
		mat.premultipliedAlpha = true;
		mat.depthWrite = false;
		mat.positionNode = Fn( () => {

			const r0 = R.field( 0 ), r1 = R.field( 1 );
			// the quad covers the blob and its sun shadow (offset away from the light, stretched)
			const L = G.sunDir;
			const lh = max( length( L.xz ), 1e-3 );
			const dir = L.xz.div( lh ).negate();
			const reach = r1.x.mul( lh ).div( max( L.y, 0.12 ) ).min( r0.w.mul( 4 ) );
			const p = attribute( 'position', 'vec3' );
			const side = vec2( dir.y, dir.x.negate() ); // (keeps the winding: the quad faces up)
			const s = r0.w.mul( 1.6 );
			const along = p.z.mul( s.add( reach.mul( 0.5 ) ) ).add( reach.mul( 0.5 ) );
			const xz = r0.xz.add( dir.mul( along ) ).add( side.mul( p.x.mul( s ) ) );
			vUV.assign( vec2( along, p.x.mul( s ) ) );
			vInfo.assign( vec4( r0.w, reach, r1.y, L.y ) );
			const off = xz.sub( r0.xz );
			return vec3( xz.x, r0.y.add( off.x.mul( r1.z ) ).add( off.y.mul( r1.w ) ).add( 0.012 ), xz.y );

		} )();

		mat.colorNode = vec3( 0 );
		mat.opacityNode = Fn( () => {

			const r = vInfo.x, reach = vInfo.y, dark = vInfo.z;
			const q = vUV.div( r );
			// occlusion right below, the sun shadow stretched along the light
			const ao = exp( dot( q, q ).mul( - 2.2 ) ).mul( 0.55 );
			const t = vUV.x.div( max( reach, 1e-3 ) ).clamp( 0, 1 );
			const c = vec2( vUV.x.sub( reach.mul( t ) ), vUV.y ).div( r );
			const sun = exp( dot( c, c ).mul( - 1.6 ) ).mul( 0.6 ).mul( select( vInfo.w.greaterThan( 0 ), float( 1 ), float( 0 ) ) );
			return ao.max( sun ).mul( dark ).mul( G.night.oneMinus().mul( 0.6 ).add( 0.4 ) );

		} )();

		mat.mrtNode = mrt( { velocity: vec4( 0 ) } );
		this.material = mat;
		this.mesh = instancedMesh( 'ContactShadows', g, mat, R, {} );
		this.mesh.castShadow = false;
		this.mesh.receiveShadow = false;
		this.mesh.layers.set( LAYERS.TRANSPARENT );
		this.mesh.renderOrder = - 1;

	}

	begin() {

		this.records.begin();

	}

	// ground slope: dh/dx, dh/dz at the blob (the quad follows it)
	add( x, y, z, radius, height, darkness = 1, sx = 0, sz = 0 ) {

		const o = this.records.push();
		if ( o < 0 ) return;
		const d = this.records.data;
		d[ o ] = x; d[ o + 1 ] = y; d[ o + 2 ] = z; d[ o + 3 ] = radius;
		d[ o + 4 ] = height; d[ o + 5 ] = darkness; d[ o + 6 ] = sx; d[ o + 7 ] = sz;

	}

	commit() {

		this.records.commit();

	}

}
