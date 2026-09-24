import { RenderTarget } from '../engine/gpu/Texture.js';
import { LAYERS, DEPTH_FORMAT } from '../engine/render/SceneRenderer.js';
import { Box3, Vector3 } from '../engine/math/index.js';

// Refraction source for the water: the scene below the water surface only, rendered right after the
// opaque pass (same jittered camera, so the temporal resolve treats it like the scene) at a fraction
// of the internal resolution. The water samples it at the end point of its refracted view ray.
//
// The opaque colour copy can't serve for this: wherever something above the water (the pier deck,
// rails, posts, the boat) is in front of the refracted end point, the seabed there was never drawn.
// Faking it from nearby pixels smeared and repeated the seabed next to every rail. Here fragments
// higher than sea level + CLIP_MARGIN are discarded (engine REFRACTION_CLIP define), so those objects
// never enter the image; submerged geometry (seabed, reef, rocks, piles, hull bottoms, fish) is shaded
// with its full material and lighting.
//
// Colour target: rgb = lit colour, a = coverage (0 where nothing below the water was drawn: the water
// then falls back to the opaque copy). Depth: reversed-Z, 0 = nothing.
const CLIP_MARGIN = 0.4; // m above sea level (wave troughs and crests move the real surface around it)

const _box = new Box3();
const _v = new Vector3();

export class RefractionPass {

	constructor( { meshRenderer, scene, camera, sceneRenderer, scale = 0.5 } ) {

		this.meshRenderer = meshRenderer;
		this.scene = scene;
		this.camera = camera;
		this.sceneRenderer = sceneRenderer;
		this.scale = scale;
		this.enabled = true;
		this.target = new RenderTarget( 1, 1, { colors: [ 'rgba16float' ], depth: DEPTH_FORMAT, label: 'refraction' } );
		this.texture = this.target.texture;
		this.depthTexture = this.target.depthTexture;
		this._clearColors = [ [ 0, 0, 0, 0 ] ];
		// plants never reach under the water (underwaterLighting 'none'); objects whose world bounding box
		// stays above the clip height (houses, roofs, the vendors, most props) are skipped before drawing.
		// Instanced meshes are drawn (their bounds are per instance).
		this._filter = ( o ) => {

			const m = o.material;
			if ( m.underwaterLighting === 'none' || m.isWaterMaterial ) return false;
			if ( o.isInstancedMesh || ! o.geometry.attributes.position ) return true;
			return _box.copy( this._localBox( o ) ).applyMatrix4( o.matrixWorld ).min.y < this._clipY;

		};

		// local bounds of the part of the geometry an object draws (merged batches like the village draw
		// ranges of one shared geometry: the whole geometry's bounds would include the pier piles)
		this._boxes = new WeakMap();

		this._clipY = 0;
		this._defines = { REFRACTION_CLIP: 1, REFRACTION_CLIP_MARGIN: CLIP_MARGIN };

	}

	_localBox( o ) {

		const g = o.geometry, dr = g.drawRange || { start: 0, count: Infinity };
		const c = this._boxes.get( o );
		if ( c && c.geometry === g && c.start === dr.start && c.count === dr.count && c.version === ( g.attributes.position.version || 0 ) ) return c.box;
		const box = new Box3();
		const pos = g.attributes.position;
		if ( ( dr.start === 0 && ( dr.count === Infinity || dr.count === null ) ) ) {

			if ( ! g.boundingBox ) g.computeBoundingBox();
			box.copy( g.boundingBox );

		} else {

			const idx = g.index ? g.index.array : null;
			const n = idx ? idx.length : pos.count;
			const end = Math.min( n, dr.start + ( dr.count ?? Infinity ) );
			for ( let i = dr.start; i < end; i ++ ) {

				const v = idx ? idx[ i ] : i;
				_v.set( pos.getX( v ), pos.getY( v ), pos.getZ( v ) );
				box.expandByPoint( _v );

			}

		}

		this._boxes.set( o, { geometry: g, start: dr.start, count: dr.count, version: pos.version || 0, box } );
		return box;

	}

	render( seaLevel ) {

		const sr = this.sceneRenderer;
		const w = Math.max( 1, Math.round( sr.width * this.scale ) ), h = Math.max( 1, Math.round( sr.height * this.scale ) );
		if ( this.target.width !== w || this.target.height !== h ) this.target.setSize( w, h );
		this._clipY = seaLevel + CLIP_MARGIN;
		const rt = this.target;
		this.meshRenderer.render( this.scene, {
			label: 'refraction',
			kind: 'color',
			camera: this.camera,
			colorViews: [ rt.texture.view() ],
			colorFormats: rt.formats,
			clearColors: this._clearColors,
			depthView: rt.depthTexture.view(),
			depthFormat: DEPTH_FORMAT,
			clearDepth: 0,
			layerMask: this.enabled ? 1 << LAYERS.OPAQUE : 0,
			filter: this._filter,
			defines: this._defines,
		} );

	}

}
