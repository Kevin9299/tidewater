import * as THREE from 'three/webgpu';
import {
	Fn, float, int, vec2, vec3, vec4, attribute, positionLocal, normalLocal, positionWorld, varyingProperty,
	uniformArray, uniform, mix, smoothstep, sin, abs, fract, floor, max, min, clamp, normalize, cross, dot,
	select, cameraProjectionMatrix, cameraViewMatrix, mrt, mx_noise_float, mx_worley_noise_float, bumpMap,
	saturate, pow, exp, sqrt, If, normalView, positionView, texture, uv,
} from 'three/tsl';
import { physical } from '../../materials/Materials.js';
import { staticVelocity } from '../../post/CameraVelocity.js';
import { LAYERS } from '../../core/SceneRenderer.js';
import { SPRAY } from '../../fx/Spray.js';
import { WhaleBrain } from './WhaleBrain.js';
import { loadTexture } from './WhaleTextures.js';

// Humpback whale (Megaptera novaeangliae), ~14.5 m.
//
// Geometry: authoring/whale (anatomy tables -> parametric lofts, baked to public/models/whale:
// three levels of detail, one draw call each, one shared material).
// Rig: a chain of K spine frames from the snout to the fluke tips plus a joint per flipper,
// posed on the CPU every frame (WhaleBrain: route, surfacing, blows, fluke-up dive) and
// applied in the vertex shader (each vertex follows the frame at its rest axial position).
// The shader poses every vertex twice (this frame and the last) for exact motion vectors.
// Skin: procedural (dark slate back, mottled white throat and belly, ventral grooves,
// tubercles, barnacles, scars, white flipper undersides and fluke pattern), SceneLighting with
// full underwater lighting (caustics, depth attenuation).

const K = 40; // spine frames
const LOD_DIST = [ 48, 170 ]; // m: lod0 -> lod1 -> lod2
const MAX_DIST = 1800;

const _q = new THREE.Quaternion();
const _q2 = new THREE.Quaternion();
const _v = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _m = new THREE.Matrix4();
const _x = new THREE.Vector3( 1, 0, 0 );
const _t = new THREE.Vector3();

export class Whale {

	constructor( { scene, terrain, query = null, spray = null, url = ( ( import.meta.env && import.meta.env.BASE_URL ) || '/' ) + 'models/whale/' } ) {

		this.scene = scene;
		this.terrain = terrain;
		this.query = query;
		this.spray = spray;
		this.url = url;
		this.ready = false;
		this.group = new THREE.Group();
		this.group.name = 'Whale';
		this.brain = new WhaleBrain( { terrain, query } );
		this.frames = null;
		this.meshes = [];
		this.lod = - 1;
		this._blowAcc = 0;
		this._dropAcc = 0;
		this._slaps = 0;

	}

	async load() {

		if ( this._loading ) return this._loading;
		this._loading = ( async () => {

			const [ manifest, bin ] = await Promise.all( [
				fetch( this.url + 'humpback.json' ).then( ( r ) => r.json() ),
				fetch( this.url + 'humpback.bin' ).then( ( r ) => r.arrayBuffer() ),
			] );
			this.manifest = manifest;
			if ( manifest.textures ) {

				[ this.albedoTex, this.heightTex ] = await Promise.all( [
					loadTexture( this.url + manifest.textures.albedo, true ),
					loadTexture( this.url + manifest.textures.height, false ),
				] );

			}

			this._setupRig( manifest );
			this.material = this._createMaterial( manifest );
			for ( const lv of manifest.levels ) {

				const geo = new THREE.BufferGeometry();
				geo.setAttribute( 'position', new THREE.BufferAttribute( new Float32Array( bin, lv.position, lv.vertices * 3 ), 3 ) );
				geo.setAttribute( 'normal', new THREE.BufferAttribute( new Int16Array( bin, lv.normal, lv.vertices * 3 ), 3, true ) );
				geo.setAttribute( 'uv', new THREE.BufferAttribute( new Float32Array( bin, lv.uv, lv.vertices * 2 ), 2 ) );
				geo.setAttribute( 'uv1', new THREE.BufferAttribute( new Float32Array( bin, lv.uv1, lv.vertices * 2 ), 2 ) );
				geo.setAttribute( 'rig', new THREE.BufferAttribute( new Float32Array( bin, lv.rig, lv.vertices * 4 ), 4 ) );
				geo.setIndex( new THREE.BufferAttribute( new Uint32Array( bin, lv.index, lv.indices ), 1 ) );
				// posed in world space by the vertex shader: the bounds follow the whale (see update)
				geo.boundingSphere = new THREE.Sphere( new THREE.Vector3(), 11 );
				geo.boundingBox = new THREE.Box3( new THREE.Vector3( - 11, - 11, - 11 ), new THREE.Vector3( 11, 11, 11 ) );
				const mesh = new THREE.Mesh( geo, this.material );
				mesh.name = 'Whale_' + lv.name;
				mesh.castShadow = true;
				mesh.receiveShadow = true;
				mesh.layers.set( LAYERS.OPAQUE );
				mesh.visible = false;
				mesh.matrixAutoUpdate = false;
				this.meshes.push( mesh );
				this.group.add( mesh );

			}

			this.scene.add( this.group );
			this.ready = true;
			this.update( 0, null );

		} )();
		return this._loading;

	}

	// ------------------------------------------------------------------ rig

	_setupRig( m ) {

		const zHead = m.snoutZ + 0.15;
		const zTail = m.notchZ - 1.2; // past the fluke tips
		this.zHead = zHead;
		this.dz = ( zHead - zTail ) / ( K - 1 );
		const cz = m.centerline.z, cy = m.centerline.y; // z descending
		const ycAt = ( z ) => {

			if ( z >= cz[ 0 ] ) return cy[ 0 ];
			for ( let i = 1; i < cz.length; i ++ ) if ( z >= cz[ i ] ) return cy[ i - 1 ] + ( cy[ i ] - cy[ i - 1 ] ) * ( z - cz[ i - 1 ] ) / ( cz[ i ] - cz[ i - 1 ] );
			return cy[ cy.length - 1 ];

		};

		this.rest = [];
		for ( let k = 0; k < K; k ++ ) {

			const z = zHead - k * this.dz;
			this.rest.push( { z, y: ycAt( z ) } );

		}

		this.rootIndex = Math.round( zHead / this.dz ); // frame nearest to z = 0
		// uniform arrays: this frame [0, K), last frame [K, 2K)
		this.uPos = uniformArray( new Array( 2 * K ).fill( 0 ).map( () => new THREE.Vector4() ), 'vec4' ).setName( 'whalePos' );
		this.uRot = uniformArray( new Array( 2 * K ).fill( 0 ).map( () => new THREE.Vector4( 0, 0, 0, 1 ) ), 'vec4' ).setName( 'whaleRot' );
		this.uFlip = uniformArray( [ new THREE.Vector4( 1, 0, 0, 0 ), new THREE.Vector4( 1, 0, 0, 0 ), new THREE.Vector4( 1, 0, 0, 0 ), new THREE.Vector4( 1, 0, 0, 0 ) ], 'vec4' ).setName( 'whaleFlip' );
		this.pos = [];
		this.rot = [];
		for ( let k = 0; k < K; k ++ ) {

			this.pos.push( new THREE.Vector3() );
			this.rot.push( new THREE.Quaternion() );

		}

		this.pecRoot = [ m.pectoral.left.origin, m.pectoral.right.origin ].map( ( o ) => new THREE.Vector3( ...o ) );
		this.pecBasis = [ m.pectoral.left.basis, m.pectoral.right.basis ].map( ( b ) => new THREE.Matrix3().set( b[ 0 ][ 0 ], b[ 0 ][ 1 ], b[ 0 ][ 2 ], b[ 1 ][ 0 ], b[ 1 ][ 1 ], b[ 1 ][ 2 ], b[ 2 ][ 0 ], b[ 2 ][ 1 ], b[ 2 ][ 2 ] ) );
		this.blowLocal = new THREE.Vector3( ...m.blowhole );
		this._hasPrev = false;

	}

	// Pose the spine frames from the brain's state (root pose, path history, stroke, arch).
	_pose() {

		const b = this.brain;
		const rest = this.rest;
		const ri = this.rootIndex;
		// last frame's pose -> previous slots
		const P = this.uPos.array, R = this.uRot.array;
		for ( let k = 0; k < K; k ++ ) {

			P[ K + k ].copy( P[ k ] );
			R[ K + k ].copy( R[ k ] );

		}

		const Qr = b.quaternion;
		// orientations
		const rots = this.rot;
		for ( let k = 0; k < K; k ++ ) {

			const z = rest[ k ].z;
			const q = rots[ k ];
			if ( z >= 0 ) {

				// head and chest: rigid with the root plus a little recoil against the stroke
				const recoil = - b.strokeAmp * 0.1 * Math.sin( b.strokePhase + 0.6 ) * Math.min( 1, z / 5 );
				q.copy( Qr ).multiply( _q.setFromAxisAngle( _x, recoil + b.headPitch * Math.min( 1, z / 4 ) ) );

			} else {

				const d = - z;
				// follow the path (the body bends along the route: turns, rolling over at the surface)
				b.pathRotation( d, _q2 );
				q.copy( Qr ).slerp( _q2, b.follow );
				// stroke: a travelling wave of pitch, growing toward the flukes
				const L = 9.6;
				const u = Math.min( d / L, 1.15 );
				const env = u * u * 1.1 + 0.05 * u;
				const ph = b.strokePhase - d * 0.52;
				let beta = b.strokeAmp * env * ( Math.sin( ph ) + 0.22 * Math.sin( 2 * ph ) );
				// the flukes lead the tail stock (angle of attack)
				if ( d > 7.6 ) beta += b.strokeAmp * 2.0 * Math.sin( ph + 1.35 ) * Math.min( 1, ( d - 7.6 ) / 1.2 );
				// arch: the back humps (both ends down) when rolling over and before the fluke-up dive
				beta += b.arch * 3.0 * Math.min( d / 6, 1.0 );
				q.multiply( _q.setFromAxisAngle( _x, beta ) );

			}

		}

		// positions: integrate the chain outward from the root
		const pos = this.pos;
		pos[ ri ].copy( b.position ).add( _v.set( 0, rest[ ri ].y, rest[ ri ].z ).applyQuaternion( rots[ ri ] ) );
		for ( let k = ri - 1; k >= 0; k -- ) {

			_q.copy( rots[ k ] ).slerp( rots[ k + 1 ], 0.5 );
			_v.set( 0, rest[ k ].y - rest[ k + 1 ].y, rest[ k ].z - rest[ k + 1 ].z ).applyQuaternion( _q );
			pos[ k ].copy( pos[ k + 1 ] ).add( _v );

		}

		for ( let k = ri + 1; k < K; k ++ ) {

			_q.copy( rots[ k ] ).slerp( rots[ k - 1 ], 0.5 );
			_v.set( 0, rest[ k ].y - rest[ k - 1 ].y, rest[ k ].z - rest[ k - 1 ].z ).applyQuaternion( _q );
			pos[ k ].copy( pos[ k - 1 ] ).add( _v );

		}

		// frame k maps rest point (x, y, z) -> pos_k + rot_k * (x, y - yc_k, z - z_k)
		for ( let k = 0; k < K; k ++ ) {

			P[ k ].set( pos[ k ].x, pos[ k ].y, pos[ k ].z, rest[ k ].y );
			const q = rots[ k ];
			// keep the quaternion hemisphere continuous along the chain (shader nlerp)
			if ( k > 0 && q.x * R[ k - 1 ].x + q.y * R[ k - 1 ].y + q.z * R[ k - 1 ].z + q.w * R[ k - 1 ].w < 0 ) R[ k ].set( - q.x, - q.y, - q.z, - q.w );
			else R[ k ].set( q.x, q.y, q.z, q.w );

		}

		// flippers: axis-angle relative to the bind pose
		const F = this.uFlip.array;
		F[ 2 ].copy( F[ 0 ] );
		F[ 3 ].copy( F[ 1 ] );
		for ( let s = 0; s < 2; s ++ ) {

			b.flipperRotation( s, _q );
			const ang = 2 * Math.acos( Math.min( 1, Math.abs( _q.w ) ) );
			const sgn = _q.w < 0 ? - 1 : 1;
			const sn = Math.sqrt( Math.max( 1e-12, 1 - _q.w * _q.w ) );
			F[ s ].set( _q.x / sn * sgn, _q.y / sn * sgn, _q.z / sn * sgn, ang );
			if ( ang < 1e-5 ) F[ s ].set( 1, 0, 0, 0 );

		}

		if ( ! this._hasPrev ) {

			for ( let k = 0; k < K; k ++ ) {

				P[ K + k ].copy( P[ k ] );
				R[ K + k ].copy( R[ k ] );

			}

			F[ 2 ].copy( F[ 0 ] );
			F[ 3 ].copy( F[ 1 ] );
			this._hasPrev = true;

		}

	}

	// world position of a rest-frame point (CPU copy of the vertex shader)
	toWorld( local, out ) {

		const fi = THREE.MathUtils.clamp( ( this.zHead - local.z ) / this.dz, 0, K - 1.001 );
		const i = Math.floor( fi ), t = fi - i;
		const r0 = this.rest[ i ], r1 = this.rest[ i + 1 ];
		const yc = r0.y + ( r1.y - r0.y ) * t;
		_q.copy( this.rot[ i ] ).slerp( this.rot[ i + 1 ], t );
		out.set( local.x, local.y - yc, 0 ).applyQuaternion( _q );
		_t.copy( this.pos[ i ] ).lerp( this.pos[ i + 1 ], t );
		return out.add( _t );

	}

	// ------------------------------------------------------------------ per frame

	update( dt, camera ) {

		if ( ! this.ready ) return;
		const b = this.brain;
		b.update( dt );
		this._pose();
		this.uWater.value = b.water;
		this.uWet.value = Math.exp( - b.wetAge / 12 );

		// level of detail and culling (the posed mesh lives in world space: move its bounds)
		let lod = 0, visible = true;
		if ( camera ) {

			const d = camera.position.distanceTo( b.position );
			const camUnder = camera.position.y < b.water;
			// hysteresis: no popping back and forth at the switch distances
			const cur = this._lod ?? 0;
			const up = ( i ) => LOD_DIST[ i ] * 1.08, dn = ( i ) => LOD_DIST[ i ] * 0.92;
			lod = cur;
			while ( lod < 2 && d > up( lod ) ) lod ++;
			while ( lod > 0 && d < dn( lod - 1 ) ) lod --;
			this._lod = lod;
			if ( d > MAX_DIST ) visible = false;
			// lost in the blue: underwater visibility is a few tens of metres
			if ( camUnder && d > 120 ) visible = false;
			// from above the water a deep whale is invisible beyond a short distance
			if ( ! camUnder && b.backDepth > 7 && d > 70 ) visible = false;

		}

		for ( let i = 0; i < this.meshes.length; i ++ ) {

			const m = this.meshes[ i ];
			m.visible = visible && i === lod;
			m.geometry.boundingSphere.center.copy( b.position );

		}

		this.lod = visible ? lod : - 1;
		if ( dt > 0 ) this._effects( dt );

	}

	// blow spout and water streaming off the flukes (Spray particles)
	_effects( dt ) {

		const b = this.brain, spray = this.spray;
		if ( ! spray ) return;
		// flipper slap: a sheet of spray where the flipper comes down on the water
		if ( b.slaps !== this._slaps ) {

			this._slaps = b.slaps;
			const side = b.slap ? b.slap.side : 0;
			const r = this.pecRoot[ side ];
			const p = this.toWorld( _v.set( r.x + ( side === 0 ? 3.2 : - 3.2 ), r.y, r.z - 1.2 ), _v2 );
			p.y = b.water + 0.1;
			for ( let k = 0; k < 4; k ++ ) {

				const vel = new THREE.Vector3( ( Math.random() - 0.5 ) * 3, 4.5 + Math.random() * 3, ( Math.random() - 0.5 ) * 3 );
				spray.emit( p, vel, 18, 0.25, SPRAY.SPRAY, { spread: 2.4, jitter: 0.6, life: 2.2, sizeJitter: 0.8 } );
				spray.emit( p, vel, 30, 0.022, SPRAY.DROPLET, { spread: 3, jitter: 0.5, life: 2 } );

			}

		}

		if ( b.blow > 0 ) {

			const p = this.toWorld( this.blowLocal, _v2 );
			if ( p.y > b.water - 0.3 ) {

				p.y = Math.max( p.y, b.water ) + 0.15;
				// the column: dense cloud of drops, a haze that drifts downwind, and heavier drops
				const fwd = _v.set( Math.sin( b.yaw ), 0, Math.cos( b.yaw ) );
				const vUp = 7.5 + 2.5 * b.blow;
				const vel = new THREE.Vector3( fwd.x * 0.8, vUp, fwd.z * 0.8 );
				this._blowAcc += dt * 60 * b.blow;
				const n = Math.floor( this._blowAcc );
				this._blowAcc -= n;
				if ( n > 0 ) {

					spray.emit( p, vel, 7 * n, 0.3, SPRAY.SPRAY, { spread: 3.2, jitter: 0.25, life: 2.2, sizeJitter: 0.8 } );
					spray.emit( p, vel.clone().multiplyScalar( 0.75 ), 10 * n, 0.7, SPRAY.MIST, { spread: 2.6, jitter: 0.35, life: 5.5, sizeJitter: 0.6 } );
					spray.emit( p, vel.clone().multiplyScalar( 0.9 ), 12 * n, 0.018, SPRAY.DROPLET, { spread: 3.5, jitter: 0.2, life: 2.2 } );

				}

			}

		}

		// flukes lifting clear: sheets of water pour off the trailing edge
		if ( b.flukeUp > 0.05 ) {

			this._dropAcc += dt * 60;
			const n = Math.floor( this._dropAcc );
			this._dropAcc -= n;
			if ( n > 0 ) {

				const nz = this.manifest.notchZ;
				for ( const x of [ - 1.9, - 1.2, - 0.5, 0.5, 1.2, 1.9 ] ) {

					const zte = nz - 0.1 - 0.35 * ( Math.abs( x ) / 2.2 ) ** 2;
					const p = this.toWorld( _v.set( x, this.rest[ K - 1 ].y, zte ), _v2 );
					if ( p.y < b.water + 0.2 ) continue;
					spray.emit( p, new THREE.Vector3( 0, - 0.5, 0 ), 5 * n, 0.02, SPRAY.DROPLET, { spread: 0.5, jitter: 0.12, life: 1.6 } );

				}

			}

		}

	}

	// ------------------------------------------------------------------ material

	_createMaterial( m ) {

		const zHead = float( this.zHead ), dz = float( this.dz );
		if ( this.heightTex ) this._hT = texture( this.heightTex );
		const uPos = this.uPos, uRot = this.uRot, uFlip = this.uFlip;
		const rig = attribute( 'rig', 'vec4' );
		const vRest = varyingProperty( 'vec3', 'vWhaleRest' );
		const vDelta = varyingProperty( 'vec3', 'vWhaleDelta' );
		const vPart = varyingProperty( 'float', 'vWhalePart' );
		const vUV = varyingProperty( 'vec2', 'vWhaleUV' );
		const pl = this.pecRoot[ 0 ], pr = this.pecRoot[ 1 ];
		const rootL = vec3( pl.x, pl.y, pl.z ), rootR = vec3( pr.x, pr.y, pr.z );

		const rotateQ = ( q, v ) => v.add( cross( q.xyz, cross( q.xyz, v ).add( v.mul( q.w ) ) ).mul( 2 ) );
		const rotAxis = ( ax, ang, v ) => {

			// Rodrigues
			const c = ang.cos(), s = ang.sin();
			return v.mul( c ).add( cross( ax, v ).mul( s ) ).add( ax.mul( dot( ax, v ).mul( float( 1 ).sub( c ) ) ) );

		};

		// pose a rest point (and normal) with frame set `o` (0 = this frame, K = last frame)
		const pose = ( p, n, o, withNormal ) => {

			const part = rig.y;
			const isL = part.greaterThan( 0.5 ).and( part.lessThan( 1.5 ) );
			const isR = part.greaterThan( 1.5 ).and( part.lessThan( 2.5 ) );
			const isPec = isL.or( isR );
			const fl = select( isL, uFlip.element( int( o === 0 ? 0 : 2 ) ), uFlip.element( int( o === 0 ? 1 : 3 ) ) );
			const root = select( isL, rootL, rootR );
			// flippers flex: the tip turns a little further than the root
			const ang = fl.w.mul( rig.z.mul( 0.35 ).add( 0.8 ) ).mul( select( isPec, 1, 0 ) );
			const pp = root.add( rotAxis( fl.xyz, ang, p.sub( root ) ) ).toVar();
			const fi = clamp( zHead.sub( rig.x ).div( dz ), 0, K - 1.001 );
			const i0 = int( floor( fi ) ), t = fract( fi );
			const P0 = uPos.element( i0.add( o ) ), P1 = uPos.element( i0.add( o + 1 ) );
			const Q0 = uRot.element( i0.add( o ) ), Q1 = uRot.element( i0.add( o + 1 ) );
			const q = normalize( mix( Q0, Q1, t ) );
			const yc = mix( P0.w, P1.w, t );
			const off = vec3( pp.x, pp.y.sub( yc ), pp.z.sub( rig.x ) );
			const world = mix( P0.xyz, P1.xyz, t ).add( rotateQ( q, off ) );
			if ( ! withNormal ) return { world };
			const nn = rotateQ( q, rotAxis( fl.xyz, ang, n ) );
			return { world, normal: nn };

		};

		const mat = physical( { roughness: 0.62, metalness: 0 } );
		mat.name = 'Whale';
		mat.underwaterLighting = 'full';
		// real relief: tubercles, pleats and barnacles displace the finest mesh (rest space)
		const relief = () => {

			if ( ! this.heightTex ) return positionLocal;
			const [ h0, h1 ] = m.textures.heightRange;
			const h = this._hT.sample( attribute( 'uv', 'vec2' ) ).level( 0 );
			const d = h.r.mul( 0.99611 ).add( h.g.mul( 0.00389 ) ).mul( h1 - h0 ).add( h0 );
			return positionLocal.add( normalLocal.mul( d ) );

		};

		mat.positionNode = Fn( () => {

			const p = relief().toVar();
			const cur = pose( p, normalLocal, 0, true );
			const prev = pose( p, normalLocal, K, false );
			vDelta.assign( cur.world.sub( prev.world ) );
			normalLocal.assign( cur.normal );
			return cur.world;

		} )();
		// the shadow pass needs only this frame's pose
		mat.castShadowPositionNode = Fn( () => pose( relief(), normalLocal, 0, false ).world )();

		// motion vectors: camera motion plus the whale's own motion (swimming, flexing)
		const own = Fn( () => {

			const vp = cameraProjectionMatrix.mul( cameraViewMatrix );
			const c = vp.mul( vec4( positionWorld, 1 ) );
			const q = vp.mul( vec4( positionWorld.sub( vDelta ), 1 ) );
			return c.xy.div( c.w ).sub( q.xy.div( q.w ) );

		} )();
		mat.mrtNode = mrt( { velocity: staticVelocity.add( own ) } );

		this._skinShading( mat, m );
		return mat;

	}

	// Baked skin (authoring/whale/bake_textures.mjs): albedo + roughness, and relief height
	// (16 bit in R/G) applied as a bump in view space from three height taps.
	_skinShading( mat, m ) {

		this.uWater = uniform( 0 ).setName( 'whaleWater' );
		this.uWet = uniform( 0 ).setName( 'whaleWet' ); // freshness of the wet film (1 = just surfaced)
		const above = smoothstep( - 0.1, 0.2, positionWorld.y.sub( this.uWater ) );
		const film = above.mul( this.uWet );
		// in water the skin's Fresnel reflectance is far weaker than in air (n 1.33 vs 1.0)
		mat.specularIntensityNode = mix( 0.45, 1.0, above );
		if ( ! this.albedoTex ) {

			mat.colorNode = vec3( 0.03, 0.033, 0.038 );
			mat.roughnessNode = mix( 0.62, 0.3, film );
			return;

		}

		const aT = texture( this.albedoTex );
		const hT = this._hT;
		const uvA = uv();
		const albedo = aT.sample( uvA );
		mat.colorNode = albedo.rgb;
		// matte skin (0.55-0.75) in the water; a thin glossy film only where it is out of the water
		// and freshly wet
		mat.roughnessNode = mix( albedo.a, albedo.a.mul( 0.48 ), film );
		const range = m.textures.heightRange[ 1 ] - m.textures.heightRange[ 0 ];
		const height = ( u ) => {

			const h = hT.sample( u );
			return h.r.mul( 0.99611 ).add( h.g.mul( 0.00389 ) ).mul( range );

		};

		mat.normalNode = Fn( () => {

			const dux = uvA.dFdx(), duy = uvA.dFdy();
			const h0 = height( uvA );
			const dhx = height( uvA.add( dux ) ).sub( h0 ).mul( 1.6 );
			const dhy = height( uvA.add( duy ) ).sub( h0 ).mul( 1.6 );
			const N = normalView.toVar();
			const dpx = positionView.dFdx(), dpy = positionView.dFdy();
			const r1 = cross( dpy, N ), r2 = cross( N, dpx );
			const det = dot( dpx, r1 );
			const grad = r1.mul( dhx ).add( r2.mul( dhy ) ).mul( det.sign() );
			return normalize( abs( det ).mul( N ).sub( grad ) );

		} )();

	}

}
