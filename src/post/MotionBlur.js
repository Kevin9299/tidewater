import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, vec2, vec3, vec4, int, ivec2, uint, instancedArray, workgroupArray, workgroupBarrier,
	localId, workgroupId, globalId, textureLoad, texture, If, Loop, max, min, floor, fract, dot, clamp, length, mix, sqrt, select,
} from 'three/tsl';

// Camera + object motion blur: reconstruction filter (McGuire et al. 2012, "A Reconstruction Filter
// for Plausible Motion Blur") with the Call of Duty: Advanced Warfare refinements (Jimenez 2014):
//   1. tile max: per 20x20 output-pixel tile the longest velocity (and the shortest, for the fast path)
//   2. neighbour max: the longest over the 3x3 tiles around each tile (every streak reaching a pixel
//      starts within one tile of it, as a streak is capped to one tile length each way)
//   3. gather, inlined in the final pass: ~4-12 jittered samples (interleaved gradient noise, new
//      offset every frame) along the neighbourhood's dominant motion and the pixel's own motion, with
//      soft depth comparisons: a moving foreground smears over the background, a static foreground
//      stays sharp over a moving background. Tiles whose velocities are all alike (camera rotation)
//      take a fast colour-only path; tiles with no motion (< 0.5 px) skip it and stay pixel-identical.
// Runs on the temporally resolved image at the output resolution, before bloom and the lens effects.
// Velocity: current - previous NDC, unjittered (so the TAA jitter never blurs).
const TILE = 20; // tile side in output pixels = maximum blur radius (streak <= 2 * TILE)
const WG = 10; // tile-max threads per side, each reducing a 2x2 pixel block
const MAX_TILES_X = 400, MAX_TILES_Y = 224; // up to 8000 x 4480 output pixels

export class MotionBlur {

	constructor( { velocityTexture, depthTexture, color } ) {

		// fraction of the frame time the shutter is open: 0.5 = 180 degree shutter, 0 = off
		this.shutter = uniform( 0.5 ).setName( 'mbShutter' );
		this.outSize = uniform( new THREE.Vector2( 1, 1 ) ).setName( 'mbOutSize' );
		this.tileCount = uniform( new THREE.Vector2( 1, 1 ) ).setName( 'mbTiles' );
		this.frameIndex = uniform( 0 ).setName( 'mbFrame' );
		// sky pixels (depth 0): reproject their far-plane points with last frame's unjittered camera
		this.skyReproj = uniform( new THREE.Matrix4() ).setName( 'mbSkyReproj' );
		this._vp = new THREE.Matrix4();
		this._prevVP = new THREE.Matrix4();
		this._hasPrev = false;
		this.color = color;
		this.velocity = texture( velocityTexture );
		this.depth = texture( depthTexture );
		this.tileMax = instancedArray( MAX_TILES_X * MAX_TILES_Y, 'vec4' ).setName( 'mbTileMax' );
		this.neighborMax = instancedArray( MAX_TILES_X * MAX_TILES_Y, 'vec4' ).setName( 'mbNeighborMax' );
		this._size = new THREE.Vector2();
		this._frame = 0;
		this._buildKernels( velocityTexture );

	}

	// NDC motion (current - previous) of the pixel at uv with reversed-Z depth d (sky = 0)
	_ndcVelocity( vel, uv, d ) {

		const ndc = uv.mul( 2 ).sub( 1 ).mul( vec2( 1, - 1 ) );
		const prev = this.skyReproj.mul( vec4( ndc, 0, 1 ) );
		return select( d.greaterThan( 1e-7 ), vel.xy, ndc.sub( prev.xy.div( prev.w ) ) );

	}

	// NDC velocity -> half streak vector in output pixels (shutter applied, capped at one tile)
	_toPx( vel ) {

		const v = vel.xy.mul( this.outSize ).mul( vec2( 0.5, - 0.5 ) ).mul( this.shutter.mul( 0.5 ) );
		const l = length( v );
		return v.mul( min( float( 1 ), float( TILE ).div( max( l, 1e-6 ) ) ) );

	}

	_buildKernels( velocityTexture ) {

		const velTex = texture( velocityTexture );
		const depthTex = this.depth;
		const N = WG * WG;

		this.tileKernel = Fn( () => {

			const shared = workgroupArray( 'vec4', N );
			const t = localId.y.mul( WG ).add( localId.x );
			const inSize = ivec2( velTex.size() );
			const best = vec2( 0 ).toVar(), bestL = float( 0 ).toVar(), minL = float( 1e8 ).toVar();
			for ( const [ ox, oy ] of [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ], [ 1, 1 ] ] ) {

				const p = vec2( workgroupId.xy.mul( TILE ).add( localId.xy.mul( 2 ) ).add( ivec2( ox, oy ) ) ).add( 0.5 );
				const q = ivec2( p.div( this.outSize ).mul( vec2( inSize ) ) ).min( inSize.sub( 1 ) );
				const v = this._toPx( this._ndcVelocity( textureLoad( velTex, q ), p.div( this.outSize ), textureLoad( depthTex, q ).x ) ).toVar();
				const l = dot( v, v );
				If( l.greaterThan( bestL ), () => {

					bestL.assign( l );
					best.assign( v );

				} );
				minL.assign( min( minL, l ) );

			}

			shared.element( t ).assign( vec4( best, bestL, minL ) );
			workgroupBarrier();
			for ( let s = 64; s > 0; s >>= 1 ) {

				If( t.lessThan( uint( s ) ).and( t.add( uint( s ) ).lessThan( uint( N ) ) ), () => {

					const a = shared.element( t ).toVar();
					const b = shared.element( t.add( uint( s ) ) ).toVar();
					const m = select( b.z.greaterThan( a.z ), b, a );
					shared.element( t ).assign( vec4( m.xyz, min( a.w, b.w ) ) );

				} );
				workgroupBarrier();

			}

			If( t.equal( uint( 0 ) ), () => {

				const r = shared.element( uint( 0 ) );
				const idx = int( workgroupId.y ).mul( MAX_TILES_X ).add( int( workgroupId.x ) );
				// xy = longest half streak (px), z = its length, w = shortest length in the tile
				this.tileMax.element( idx ).assign( vec4( r.xy, sqrt( r.z ), sqrt( r.w ) ) );

			} );

		} )().computeKernel( [ WG, WG, 1 ] ).setName( 'Motion Blur Tiles' );

		this.neighborKernel = Fn( () => {

			const tx = int( globalId.x ), ty = int( globalId.y );
			const nx = int( this.tileCount.x ), ny = int( this.tileCount.y );
			If( tx.lessThan( nx ).and( ty.lessThan( ny ) ), () => {

				const best = vec4( 0 ).toVar(), minL = float( 1e8 ).toVar();
				for ( let dy = - 1; dy <= 1; dy ++ ) for ( let dx = - 1; dx <= 1; dx ++ ) {

					const x = clamp( tx.add( dx ), int( 0 ), nx.sub( 1 ) ), y = clamp( ty.add( dy ), int( 0 ), ny.sub( 1 ) );
					const m = this.tileMax.element( y.mul( MAX_TILES_X ).add( x ) ).toVar();
					let take = m.z.greaterThan( best.z );
					if ( dx !== 0 && dy !== 0 ) {

						// a diagonal tile only matters when its streak runs toward this tile (Jimenez 2014)
						const d = vec2( - dx, - dy ).mul( Math.SQRT1_2 );
						take = take.and( dot( m.xy, d ).abs().greaterThan( m.z.mul( 0.7 ) ) );

					}

					If( take, () => {

						best.assign( m );

					} );
					minL.assign( min( minL, m.w ) );

				}

				// xy = dominant half streak (px), z = its length, w = shortest streak in the neighbourhood
				this.neighborMax.element( ty.mul( MAX_TILES_X ).add( tx ) ).assign( vec4( best.xyz, minL ) );

			} );

		} )().computeKernel( [ 8, 8, 1 ] ).setName( 'Motion Blur Neighbourhood' );

	}

	// Motion-blurred colour at uv. sharpColor = the (sharpened) unblurred colour at uv, returned
	// unchanged where nothing moves; the blur itself reads the unsharpened resolved image, so
	// sharpening never acts on blurred pixels.
	apply( sharpColor, uvIn ) {

		return Fn( () => {

			const out = vec3( sharpColor ).toVar();
			If( this.shutter.greaterThan( 0 ), () => {

				const pix = uvIn.mul( this.outSize );
				const tile = ivec2( floor( pix.div( TILE ) ) ).clamp( ivec2( 0 ), ivec2( this.tileCount ).sub( 1 ) );
				const nm = this.neighborMax.element( tile.y.mul( MAX_TILES_X ).add( tile.x ) ).toVar();
				const nmLen = nm.z;
				If( nmLen.greaterThanEqual( 0.5 ), () => {

					const inSize = vec2( this.velocity.size() );
					const inMax = ivec2( inSize ).sub( 1 );
					const texel = ( uv ) => ivec2( uv.mul( inSize ) ).clamp( ivec2( 0 ), inMax );
					const cX = texel( uvIn );
					const dX = this.depth.load( cX ).x.toVar();
					const vX = this._toPx( this._ndcVelocity( this.velocity.load( cX ), uvIn, dX ) ).toVar();
					const lenX = length( vX ).toVar();
					// interleaved gradient noise, shifted every frame
					const pn = floor( pix ).add( this.frameIndex.mul( 5.588238 ) );
					const jitter = fract( fract( dot( pn, vec2( 0.06711056, 0.00583715 ) ) ).mul( 52.9829189 ) ).sub( 0.5 );
					// sample pairs: ~3 px apart along the streak, 2..6 pairs
					const pairs = int( clamp( nmLen.div( 3 ).ceil(), 2, 6 ) );
					const invSize = vec2( 1 ).div( this.outSize );
					const acc = vec3( 0 ).toVar();
					const fade = nmLen.sub( 0.5 ).saturate(); // continuous with the early-out below 0.5 px

					If( nm.w.greaterThan( nmLen.mul( 0.75 ) ), () => {

						// fast path: all streaks in the neighbourhood alike (camera rotation, distant scenery):
						// a plain directional average along the pixel's own motion
						Loop( { start: int( 0 ), end: pairs, type: 'int', condition: '<' }, ( { i } ) => {

							const tt = float( i ).add( 0.5 ).add( jitter ).div( float( pairs ) );
							const o = vX.mul( tt ).mul( invSize );
							acc.addAssign( this.color.sample( uvIn.add( o ) ).rgb );
							acc.addAssign( this.color.sample( uvIn.sub( o ) ).rgb );

						} );
						out.assign( mix( sharpColor, acc.div( float( pairs ).mul( 2 ) ), fade ) );

					} ).Else( () => {

						// full reconstruction: soft depth classification of each sample against the centre
						const wSum = float( 0 ).toVar();
						const dirX = select( lenX.greaterThan( 0.5 ), vX, nm.xy );
						Loop( { start: int( 0 ), end: pairs, type: 'int', condition: '<' }, ( { i } ) => {

							const tt = float( i ).add( 0.5 ).add( jitter ).div( float( pairs ) );
							// alternate between the neighbourhood's dominant direction and the pixel's own (Jimenez 2014)
							const dir = select( i.bitAnd( 1 ).equal( 1 ), dirX, nm.xy );
							const off = dir.mul( tt );
							const offLen = length( off );
							for ( const sgn of [ 1, - 1 ] ) {

								const uvY = uvIn.add( off.mul( invSize ).mul( sgn ) );
								const tY = texel( uvY );
								const dY = this.depth.load( tY ).x.toVar();
								const lenY = length( this._toPx( this._ndcVelocity( this.velocity.load( tY ), uvY, dY ) ) );
								// reversed-Z depth ~ 1/distance: q > 0 when the sample is farther than the centre
								const q = dX.sub( dY ).div( max( max( dX, dY ), 1e-12 ) );
								const behind = q.mul( 40 ).add( 0.5 ).saturate();
								// a sample behind the centre shows through the centre's own streak; one in front
								// covers the centre when its streak reaches it (1 px soft cylinders)
								const spreadX = lenX.sub( max( offLen.sub( 1 ), 0 ) ).saturate();
								const spreadY = lenY.sub( max( offLen.sub( 1 ), 0 ) ).saturate();
								const w = behind.mul( spreadX ).add( behind.oneMinus().mul( spreadY ) );
								acc.addAssign( this.color.sample( uvY ).rgb.mul( w ) );
								wSum.addAssign( w );

							}

						} );
						// whatever the samples don't cover is this pixel's own (sharpened) colour
						const n = float( pairs ).mul( 2 );
						const res = acc.div( n ).add( sharpColor.mul( wSum.div( n ).oneMinus().max( 0 ) ) );
						out.assign( mix( sharpColor, res, fade ) );

					} );

				} );

			} );
			return out;

		} )();

	}

	// Per frame before the TAA jitter is applied (unjittered camera matrices).
	updateCamera( camera ) {

		this._vp.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse );
		if ( ! this._hasPrev ) this._prevVP.copy( this._vp );
		this._hasPrev = true;
		// previous view-projection * inverse current: far-plane NDC -> last frame's clip position
		this.skyReproj.value.copy( this._vp ).invert().premultiply( this._prevVP );
		this._prevVP.copy( this._vp );

	}

	// Per frame, after the scene render (velocity + depth ready) and before the post pipeline.
	compute( renderer ) {

		if ( this.shutter.value <= 0 ) return;
		renderer.getDrawingBufferSize( this._size );
		const tx = Math.min( MAX_TILES_X, Math.ceil( this._size.x / TILE ) );
		const ty = Math.min( MAX_TILES_Y, Math.ceil( this._size.y / TILE ) );
		this.outSize.value.copy( this._size );
		this.tileCount.value.set( tx, ty );
		this._frame = ( this._frame + 1 ) % 64;
		this.frameIndex.value = this._frame;
		renderer.compute( this.tileKernel, [ tx, ty, 1 ] );
		renderer.compute( this.neighborKernel, [ Math.ceil( tx / 8 ), Math.ceil( ty / 8 ), 1 ] );

	}

}
