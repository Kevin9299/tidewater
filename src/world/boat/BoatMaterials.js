import { Vector3, DoubleSide } from '../../engine/index.js';
import { ShaderModule } from '../../engine/gpu/Shader.js';
import { commonModule } from '../../engine/render/wgsl/common.js';
import { standard, physical } from '../../materials/Materials.js';

// WGSL port of the TSL node materials. The TSL helpers live in `boatModule`:
//   fn boatHash21( p: vec2f ) -> f32              sin-hash (the TSL file's own hash21, not common's)
//   fn boatLstep( a, b, x ) / boatInvstep( a, b, x )   smoothstep / 1 - smoothstep (a < b always)
//   fn boatIsPattern( auxZ: f32, id: f32 ) -> f32
//   fn boatBumpNormal( P, N, height ) -> vec3f   Mikkelsen surface-gradient bump (world space)
//   fn boatNonSkidHeight( uv: vec2f ) -> f32
// Shared attribute accessors (see GeoKit: color = linear albedo, aux = rough, metal, pattern, anim):
// `color` is the standard vertex colour (in.color / v.color), `aux` a material attribute copied to
// the `vAux` varying; `vLocal` carries positionLocal (the model-space position) to the fragment.

const boatModule = new ShaderModule( {
	name: 'boatMaterials',
	deps: [ commonModule ],
	code: /* wgsl */`
fn boatIsPattern( auxZ: f32, id: f32 ) -> f32 { return step( abs( auxZ - id ), 0.5 ); }
fn boatHash21( p: vec2f ) -> f32 { return fract( sin( dot( p, vec2f( 127.1, 311.7 ) ) ) * 43758.5453 ); }
// a < b always (WGSL requires ordered edges)
fn boatLstep( a: f32, b: f32, x: f32 ) -> f32 { return smoothstep( a, b, x ); }
fn boatInvstep( a: f32, b: f32, x: f32 ) -> f32 { return 1.0 - smoothstep( a, b, x ); }

// Mikkelsen surface-gradient bump mapping from a procedural height in meters.
// (TSL worked in view space with faceDirection; here P / N are world space and N already faces
// the viewer on double-sided materials, so faceDirection is folded into N.)
fn boatBumpNormal( P: vec3f, N: vec3f, height: f32 ) -> vec3f {
	let dPdx = dpdx( P );
	let dPdy = dpdy( P );
	let n = N;
	let r1 = cross( dPdy, n );
	let r2 = cross( n, dPdx );
	let det = dot( dPdx, r1 );
	let grad = sign( det ) * ( dpdx( height ) * r1 + dpdy( height ) * r2 );
	return normalize( abs( det ) * n - grad + n * 1e-12 );
}

// Molded non-skid: jittered pebbles on a ~1.8 cm lattice (uv in meters), faded when sub-pixel.
fn boatNonSkidHeight( uv: vec2f ) -> f32 {
	let q = uv * 55.0;
	let c = floor( q );
	let f = fract( q ) - 0.5;
	let j = ( vec2f( boatHash21( c ), boatHash21( c + 17.31 ) ) - 0.5 ) * 0.3;
	let d = length( f - j );
	let pebble = boatInvstep( 0.16, 0.36, d );
	let fade = boatInvstep( 0.3, 0.8, fwidth( q.x ) );
	return pebble * fade;
}
`,
} );

// vertex snippet shared by all boat materials: aux and the local position to the fragment
const AUX_VERTEX = /* wgsl */`
	o.vAux = v.aux;
	o.vLocal = v.position;
`;

const COMMON = () => ( {
	// the TSL read attribute( 'color' ) directly; the engine only binds it with vertexColors
	// (every snippet sets s.albedo itself, so mat.color * in.color never leaks through)
	vertexColors: true,
	modules: [ boatModule ],
	attributes: { aux: 'vec4f' },
	varyings: { vAux: 'vec4f', vLocal: 'vec3f' },
	vertex: AUX_VERTEX,
} );

// hex colour -> linear WGSL vec3f literal (TSL color( 0x.. ) is an sRGB hex converted to linear)
function col( hex ) {

	const c = ( v ) => {

		v /= 255;
		return v <= 0.04045 ? v / 12.92 : Math.pow( ( v + 0.055 ) / 1.055, 2.4 );

	};
	return `vec3f( ${ f( c( ( hex >> 16 ) & 255 ) ) }, ${ f( c( ( hex >> 8 ) & 255 ) ) }, ${ f( c( hex & 255 ) ) } )`;

}

// JS number -> WGSL float literal
function f( x ) {

	const s = String( + x.toPrecision( 9 ) );
	return s.includes( '.' ) || s.includes( 'e' ) ? s : s + '.0';

}

export class BoatMaterials {

	constructor( hullShape ) {

		this.hull = this.createHull( hullShape );
		this.gelcoat = this.createGelcoat();
		this.wood = this.createWood();
		this.fittings = this.createFittings();
		this.glass = this.createGlass();
		this.glow = this.createGlow();
		this.trap = this.createTrap();

		// uniform handles ({ value }), as the TSL uniform() nodes were
		this.navOn = this.glow.uniforms.navOn;
		this.flagPivot = this.fittings.uniforms.flagPivot;
		this.flagDir = this.fittings.uniforms.flagDir;
		this.flagWind = this.fittings.uniforms.flagWind;

	}

	// Hull exterior: antifouling / boot stripe / white topsides painted by height in the
	// boat frame (the hull mesh sits at the root with an identity transform).
	createHull( shape ) {

		const m = physical( { roughness: 0.25, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.1, ...COMMON() } );
		m.name = 'boatHull';

		m.surface = /* wgsl */`
	let p = in.vs.vLocal;
	let tS = sat( ( p.z - ${ f( shape.zAft ) } ) / ${ f( shape.length ) } );
	let sheer = pow( tS, 2.2 ) * 0.62 + 0.98 + pow( max( 1.0 - tS / 0.2, 0.0 ), 2.0 ) * 0.04;

	// boot top sweeps up slightly toward the bow
	let boot = p.y - boatLstep( 0.8, 4.3, p.z ) * 0.06;
	let aaB = fwidth( boot ) + 1e-4;
	let aboveBottom = boatLstep( -aaB, aaB, boot - 0.05 );
	let aboveStripe = boatLstep( -aaB, aaB, boot - 0.15 );

	let below = sheer - p.y;
	let aaS = fwidth( below ) + 1e-4;
	let cove = boatLstep( -aaS, aaS, below - 0.1 ) * boatInvstep( -aaS, aaS, below - 0.122 );

	let n1 = mx_fractal_noise_float3( p * vec3f( 0.9, 2.4, 0.9 ), 3, 2.0, 0.5 );
	let streakN = mx_noise_float3( vec3f( p.z * 7.0, p.y * 0.45, p.x * 7.0 ) );
	let scuffN = mx_noise_float3( vec3f( p.z * 2.2, p.y * 34.0, p.x * 2.2 ) );

	var c = mix( ${ col( 0x7a1d15 ) }, ${ col( 0x0f1a30 ) }, aboveBottom );
	c = mix( c, ${ col( 0xf2efe6 ) }, aboveStripe );
	c = mix( c, ${ col( 0x0f1a30 ) }, cove );

	// waterline scum on the white, algae on the antifouling just below the boot top
	let scum = boatInvstep( 0.16, n1 * 0.09 + 0.42, boot ) * aboveStripe * sat( n1 * 0.6 + 0.7 );
	c = mix( c, ${ col( 0x8b7b57 ) }, scum * 0.32 );
	let algae = boatLstep( -0.35, 0.03, boot ) * ( 1.0 - aboveBottom ) * sat( n1 + 0.45 );
	c = mix( c, ${ col( 0x2c3a1f ) }, algae * 0.45 );

	// faint vertical weathering streaks under the gunwale
	let streak = boatLstep( 0.3, 0.75, streakN ) * boatLstep( 0.05, 0.2, below ) * boatInvstep( 0.4, 0.95, below ) * aboveStripe;
	c = mix( c, ${ col( 0xa29579 ) }, streak * 0.2 );

	// scuffs where traps come over the rail (starboard, aft of the wheelhouse)
	let side = boatLstep( -0.2, 0.2, -p.x ) * boatInvstep( -0.6, 0.2, p.z ) * boatLstep( -3.7, -2.8, p.z );
	let scuff = boatLstep( 0.45, 0.8, scuffN ) * boatLstep( 0.13, 0.18, below ) * boatInvstep( 0.45, 0.7, below ) * ( side * 0.8 + 0.2 );
	c = mix( c, ${ col( 0x5f5e59 ) }, scuff * 0.55 );

	s.albedo = c;
	s.roughness = mix( mix( 0.75, 0.35, aboveBottom ), 0.2, aboveStripe ) + scum * 0.25 + scuff * 0.3;
	s.clearcoat = aboveBottom * ( 1.0 - scum * 0.5 ) * ( 1.0 - scuff * 0.6 );
	s.clearcoatRoughness = 0.08 + scum * 0.3;
`;
		return m;

	}

	// White fiberglass (deck, lining, house, console); pattern 1 = molded non-skid.
	createGelcoat() {

		const m = standard( { roughness: 0.35, metalness: 0, ...COMMON() } );
		m.name = 'boatGelcoat';
		m.surface = /* wgsl */`
	let aux = in.vs.vAux;
	let vColor = in.color.rgb;
	let grip = boatIsPattern( aux.z, 1.0 );
	let h = boatNonSkidHeight( in.uv ) * grip;
	let p = in.vs.vLocal;
	let dirtN = mx_noise_float3( p * 1.7 );
	let dirt = sat( dirtN * 0.5 + 0.35 ) * grip * 0.16;
	// a little grime where walls meet the sole
	let corner = boatInvstep( 0.35, 0.5, p.y ) * 0.08 * ( 1.0 - grip );
	s.albedo = vColor * ( 1.0 - h * 0.07 ) * ( 1.0 - dirt - corner );
	s.roughness = mix( aux.x, 0.72, grip ) + h * 0.1;
	s.metalness = aux.y;
	s.normal = boatBumpNormal( in.P, in.N, h * 0.0008 );
`;
		return m;

	}

	// Varnished teak/mahogany; grain follows uv.x. Pattern 1 = plain (brass, paint).
	createWood() {

		const m = standard( { roughness: 0.35, metalness: 0, ...COMMON() } );
		m.name = 'boatWood';
		m.surface = /* wgsl */`
	let aux = in.vs.vAux;
	let w = in.uv;
	let g1 = mx_noise_float3( vec3f( w.x * 0.8, w.y * 30.0, 0.37 ) );
	let g2 = mx_noise_float3( vec3f( w.x * 10.0, w.y * 170.0, 5.1 ) );
	let rings = sin( w.y * 150.0 + g1 * 5.0 + w.x * 0.6 ) * 0.5 + 0.5;
	let grain = sat( rings * 0.45 + g2 * 0.22 + g1 * 0.22 + 0.28 );
	let plain = boatIsPattern( aux.z, 1.0 );
	let wood = mix( ${ col( 0x4a230f ) }, ${ col( 0x9c5b2b ) }, grain );
	s.albedo = mix( wood, vec3f( 1.0 ), plain ) * in.color.rgb;
	s.roughness = aux.x + g2 * 0.04 * ( 1.0 - plain );
	s.metalness = aux.y;
`;
		return m;

	}

	// Everything else opaque: stainless, bronze, painted metal, plastics, rope, vinyl, flag.
	// Pattern 1 = laid rope, 2 = flag (animated), 3 = whip antenna (animated sway).
	createFittings() {

		const m = standard( {
			roughness: 0.5, metalness: 0, ...COMMON(),
			uniforms: {
				flagPivot: [ 'vec3f', new Vector3() ],
				flagDir: [ 'vec3f', new Vector3( 0, 0, - 1 ) ],
				flagWind: [ 'f32', 0.5 ],
			},
		} );
		m.name = 'boatFittings';

		m.surface = /* wgsl */`
	let aux = in.vs.vAux;
	let vColor = in.color.rgb;
	let u = in.uv;

	let strand = sin( ( u.x / 0.07 + u.y ) * ( TWO_PI * 3.0 ) );
	let ropeShade = boatLstep( -0.7, 0.7, strand ) * 0.4 + 0.6;

	let stripeIdx = floor( sat( u.y ) * 12.999 );
	let red = 1.0 - ( stripeIdx - 2.0 * floor( stripeIdx / 2.0 ) );
	let canton = step( u.x, 0.4 ) * step( ${ f( 6 / 13 ) }, u.y );
	let sx = fract( u.x / 0.4 * 6.0 ) - 0.5;
	let sy = fract( ( u.y - ${ f( 6 / 13 ) } ) / ${ f( 7 / 13 ) } * 5.0 ) - 0.5;
	let star = boatInvstep( 0.16, 0.24, length( vec2f( sx, sy ) ) );
	let stripes = mix( ${ col( 0xf4f1ea ) }, ${ col( 0xb3172a ) }, red );
	let flag = mix( stripes, mix( ${ col( 0x1c2a5c ) }, ${ col( 0xf4f1ea ) }, star ), canton );

	var c = mix( vColor, vColor * ropeShade, boatIsPattern( aux.z, 1.0 ) );
	c = mix( c, flag, boatIsPattern( aux.z, 2.0 ) );
	s.albedo = c;
	s.roughness = aux.x;
	s.metalness = aux.y;
`;

		// vertex animation
		m.vertex = AUX_VERTEX + /* wgsl */`
	let aux = v.aux;
	let p = v.position;
	let t = frame.time;
	let wA = aux.w * aux.w;
	let phase = p.x * 3.1 + p.z * 1.7;
	let gust = frame.windSpeed * 0.06 + 0.5;
	let sway = vec3f( sin( t * 1.9 + phase ), 0.0, sin( t * 1.37 + phase * 1.3 ) * 0.6 ) * ( wA * 0.12 * gust );

	let rel = p - mat.flagPivot;
	let along = max( -rel.z, 0.0 );
	let fu = aux.w; // 0 at the hoist .. 1 at the fly
	let droop = ( 1.0 - mat.flagWind ) * 1.15 * ( fu * 0.5 + 0.5 );
	let lat = vec3f( mat.flagDir.z, 0.0, -mat.flagDir.x );
	let flutter = sin( fu * 9.0 - t * ( mat.flagWind * 9.0 + 4.0 ) + rel.y * 4.0 ) * fu * ( mat.flagWind * 0.05 + 0.015 );
	let flagPos = vec3f( mat.flagPivot.x, 0.0, mat.flagPivot.z )
		+ mat.flagDir * ( along * cos( droop ) )
		+ vec3f( 0.0, rel.y - along * sin( droop ), 0.0 )
		+ lat * flutter;

	v.position = mix( p + sway * boatIsPattern( aux.z, 3.0 ), flagPos, boatIsPattern( aux.z, 2.0 ) );
`;
		return m;

	}

	createGlass() {

		const m = physical( {
			color: 0xa9bec4, roughness: 0.05, metalness: 0, ior: 1.5,
			transparent: true, opacity: 0.25, side: DoubleSide, depthWrite: false,
			// the temporal resolve must reproject what is seen through the glass (the sea sliding past),
			// not the glass, which rides along with the helm camera: leave the velocity target untouched
			// (engine: velocity weight 0 in the blended pass keeps what is behind)
			velocityWeight: 0,
			...COMMON(),
		} );
		m.name = 'boatGlass';
		m.surface = /* wgsl */`
	let u = in.uv;
	let spots = boatLstep( 0.45, 0.85, mx_noise_float3( vec3f( u * 38.0, 3.3 ) ) );
	let haze = sat( mx_noise_float3( vec3f( u * 4.0, 9.1 ) ) * 0.5 + 0.5 );
	let edge = boatInvstep( 0.0, 0.18, u.y );
	let salt = sat( ( spots * 0.6 + haze * 0.25 ) * ( edge * 0.8 + 0.35 ) );
	s.albedo = mat.color; // no vertex colours on the glass (three: vertexColors off)
	s.alpha = 0.16 + salt * 0.22;
	s.roughness = 0.03 + salt * 0.35;
`;
		m.output = /* wgsl */`
	r.velocity = vec4f( 0.0 );
`;
		return m;

	}

	// Emissive parts. aux.z selects: 0 nav light, 1 radar display, 2 chart plotter,
	// 3 gauge dial, 4 flood/spot light, 5 cabin dome light, 6 LCD.
	createGlow() {

		const m = standard( { color: 0x000000, roughness: 0.3, metalness: 0, ...COMMON(), uniforms: { navOn: [ 'f32', 1 ] } } );
		m.name = 'boatGlow';
		m.surface = /* wgsl */`
	let aux = in.vs.vAux;
	let vColor = in.color.rgb;
	let mode = aux.z;
	let t = frame.time;
	let night = frame.night;
	let u = in.uv;

	// radar: head-up PPI with a 24 rpm sweep
	let q = ( u - 0.5 ) * 2.0;
	let r = length( q );
	let ang = atan2( q.x, q.y );
	let da = fract( ( t * 2.513 - ang ) / TWO_PI );
	let trail = exp( da * -5.0 );
	let beam = boatInvstep( 0.0, 0.01, da );
	let ringD = abs( fract( r * 3.0 + 0.5 ) - 0.5 );
	let ring = boatInvstep( 0.012, 0.03, ringD );
	let landN = mx_fractal_noise_float3( vec3f( q * 2.3 + vec2f( 1.7, 0.4 ), 0.5 ), 3, 2.0, 0.5 );
	let land = boatLstep( 0.18, 0.4, landN + q.x * 0.35 ) * boatLstep( 0.3, 0.45, r );
	let cell = floor( q * 7.0 );
	let blip = step( 0.93, boatHash21( cell ) ) * boatInvstep( 0.1, 0.3, length( fract( q * 7.0 ) - 0.5 ) ) * boatLstep( 0.2, 0.3, r );
	let heading = boatInvstep( 0.004, 0.012, abs( q.x ) ) * step( 0.0, q.y );
	let inside = boatInvstep( 0.96, 0.99, r );
	let echoes = sat( land + blip ) * ( trail * 0.75 + 0.25 );
	let radar = ( ( vec3f( 0.0, 0.012, 0.04 )
		+ vec3f( 0.05, 0.22, 0.28 ) * ( ring * 0.5 + heading * 0.6 )
		+ vec3f( 1.0, 0.72, 0.12 ) * echoes
		+ vec3f( 0.15, 0.9, 0.35 ) * ( beam * 0.8 + trail * 0.08 ) )
		* inside + vec3f( 0.01, 0.015, 0.02 ) ) * 1.3;

	// chart plotter
	let cp = u * vec2f( 1.35, 1.0 );
	let cn = mx_fractal_noise_float3( vec3f( cp * 2.6 + vec2f( 3.1, 1.2 ), 2.2 ), 4, 2.0, 0.5 );
	let shore = cn + ( u.x - 0.62 ) * 1.1;
	let isLand = boatLstep( 0.0, 0.015, shore );
	let shallow = boatLstep( -0.25, 0.0, shore );
	let contour = boatInvstep( 0.02, 0.05, abs( fract( shore * 9.0 ) - 0.5 ) ) * ( 1.0 - isLand );
	let water = mix( vec3f( 0.2, 0.42, 0.78 ), vec3f( 0.62, 0.82, 0.97 ), shallow );
	let track = boatInvstep( 0.003, 0.007, abs( u.x - 0.5 - ( u.y - 0.5 ) * 0.25 ) ) * step( 0.5, u.y );
	let boatIcon = boatInvstep( 0.012, 0.022, length( u - vec2f( 0.5, 0.5 ) ) );
	let chart = ( mix( water, vec3f( 0.88, 0.8, 0.55 ), isLand ) * ( 1.0 - contour * 0.35 )
		+ vec3f( 0.9, 0.1, 0.8 ) * track
		+ vec3f( 1.0, 0.35, 0.05 ) * boatIcon ) * 0.85;

	// gauge dial (backlit ticks + needle)
	let gq = ( u - 0.5 ) * 2.0;
	let gr = length( gq );
	let ga = atan2( gq.x, gq.y );
	let tickF = abs( fract( ga / TWO_PI * 24.0 ) - 0.5 );
	let ticks = boatInvstep( 0.08, 0.15, tickF ) * boatLstep( 0.72, 0.76, gr ) * boatInvstep( 0.86, 0.9, gr ) * boatInvstep( 2.3, 2.4, abs( ga ) );
	let lx = in.vs.vLocal.x; // positionLocal.x
	let needleA = sin( t * 0.7 + lx * 13.0 ) * 0.06 + lx * 3.7 + 0.3;
	let nd = vec2f( sin( needleA ), cos( needleA ) );
	let along = dot( gq, nd );
	let perp = length( gq - nd * along );
	let needle = boatInvstep( 0.025, 0.045, perp ) * step( -0.1, along ) * boatInvstep( 0.68, 0.72, along );
	let backlight = night * 1.4 + 0.25;
	let gauge = ( vec3f( 0.9, 0.95, 1.0 ) * ticks + vec3f( 1.0, 0.45, 0.08 ) * needle ) * backlight + vec3f( 0.004 );

	let nav = vColor * mat.navOn * ( night * 7.0 + 1.5 );
	let flood = vColor * ( night * 9.0 + 0.02 );
	let dome = vColor * ( night * 2.2 + 0.02 );
	let lcd = vColor * ( night * 0.5 + 0.45 );

	s.albedo = vColor * 0.06;
	s.emissive = nav * boatIsPattern( mode, 0.0 )
		+ radar * boatIsPattern( mode, 1.0 )
		+ chart * boatIsPattern( mode, 2.0 )
		+ gauge * boatIsPattern( mode, 3.0 )
		+ flood * boatIsPattern( mode, 4.0 )
		+ dome * boatIsPattern( mode, 5.0 )
		+ lcd * boatIsPattern( mode, 6.0 );
	s.roughness = aux.x;
`;
		return m;

	}

	// Vinyl-coated wire traps: alpha-tested mesh. uv in meters, aux.xy = face size
	// (for the solid frame border), pattern 1 = diamond twine netting.
	createTrap() {

		const m = standard( { roughness: 0.55, metalness: 0, side: DoubleSide, alphaTest: 0.5, ...COMMON() } );
		m.name = 'boatTrap';
		m.surface = /* wgsl */`
	let aux = in.vs.vAux;
	let u = in.uv;
	let net = boatIsPattern( aux.z, 1.0 );
	let diag = vec2f( u.x + u.y, u.x - u.y ) * 0.7071;
	let q = mix( u / 0.038, diag / 0.05, net );
	let fr = abs( fract( q ) - 0.5 );
	let fw = fwidth( q );
	let half = mix( ${ f( 0.5 - 0.0045 / 0.038 ) }, ${ f( 0.5 - 0.002 / 0.05 ) }, net );
	let lineX = boatLstep( half - fw.x, half, fr.x );
	let lineY = boatLstep( half - fw.y, half, fr.y );
	let b = 0.014;
	let inner = step( b, u.x ) * step( b, u.y ) * step( u.x, aux.x - b ) * step( u.y, aux.y - b );
	let border = ( 1.0 - inner ) * ( 1.0 - net );
	s.alpha = max( max( lineX, lineY ), border );
	s.albedo = in.color.rgb * mix( 1.0, 0.8, border );
`;
		return m;

	}

	setNavLights( on ) {

		this.navOn.value = on ? 1 : 0;

	}

	dispose() {

		for ( const k of [ 'hull', 'gelcoat', 'wood', 'fittings', 'glass', 'glow', 'trap' ] ) this[ k ].dispose();

	}

}
