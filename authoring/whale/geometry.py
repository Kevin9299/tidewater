"""Humpback surfaces built from anatomy.py (numpy only, no Blender dependency).

Every part is a parametric surface, so each level of detail is a fresh, clean resampling of
the same shape (no decimation) with the same UV layout:

  body      lofted rings from the snout to the tail stock; the ring shape is a pair of
            superellipses (upper / lower half) plus head features (lower-jaw flare, lip
            crease), the dorsal fin with its hump, the splash guard and the tail-stock knuckles
  flippers  NACA-section loft along a curved span with leading-edge tubercles
  flukes    NACA-section loft across the span with the median notch and a serrated
            trailing edge
  eyes      small corneal domes seated on the head surface (finest level only)

UV0 (texture atlas, 2:1): body band V in [0, 0.70], flipper / fluke / eye band below.
UV1 is metric (metres) for tiling detail maps.
Coordinates: whale frame (x left, y up, z forward), origin at the root joint.
"""
import numpy as np
import anatomy as A
from anatomy import TL

# atlas regions (u0, v0, u1, v1)
UV_BODY = (0.004, 0.004, 0.996, 0.700)
UV_PEC = [(0.004, 0.716, 0.328, 0.996), (0.336, 0.716, 0.660, 0.996)]  # left, right
UV_FLUKE = (0.668, 0.716, 0.972, 0.996)
UV_EYE = (0.978, 0.716, 0.996, 0.752)

PART_BODY, PART_PEC_L, PART_PEC_R, PART_FLUKE, PART_EYE = 0, 1, 2, 3, 4


def root_offset():
	"""Whale-frame origin: the root joint on the body centre line (in snout frame, metres)."""
	st = A.station(A.ROOT_S)
	yc = 0.5 * (st['top'] + st['bot'])
	return np.array([0.0, yc * TL, -A.ROOT_S * TL])


ROOT = root_offset()


# ------------------------------------------------------------------ body

def s_warp(s):
	"""Body texture coordinate along the length: more texels on the head."""
	s = np.asarray(s, float)
	# density 1.45 on the head falling to 0.8 on the tail stock (normalised)
	grid = np.linspace(0, A.S_END, 2001)
	dens = 0.8 + 0.65 * (1 - A.smoothstep(0.18, 0.42, grid)) + 0.15 * (1 - A.smoothstep(0.5, 0.75, grid))
	cum = np.concatenate([[0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid))])
	cum /= cum[-1]
	return np.interp(s, grid, cum)


def body_stations(n):
	"""n station positions over [0, S_END], denser at the head, dorsal fin and tail root."""
	g = np.linspace(0, A.S_END, 4001)
	d = (1.0 + 0.9 * (1 - A.smoothstep(0.2, 0.3, g))
		+ 1.6 * np.exp(-((g - 0.645) / 0.03) ** 2)
		+ 1.3 * A.smoothstep(0.86, 0.9, g)
		+ 2.5 * np.exp(-(g / 0.012) ** 2)
		+ 1.2 * np.exp(-((g - A.BLOWHOLE_S) / 0.02) ** 2))
	c = np.concatenate([[0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(g))])
	c /= c[-1]
	return np.interp(np.linspace(0, 1, n), c, g)


def _cap_scale(s):
	"""Rounded ends: the section shrinks like an ellipsoid over the first / last few %."""
	s = np.asarray(s, float)
	nose, tail = 0.022, 0.012
	a = np.clip(s / nose, 0, 1)
	b = np.clip((A.S_END - s) / tail, 0, 1)
	return np.sqrt(np.clip(1 - (1 - a) ** 2, 0, 1)) * np.sqrt(np.clip(1 - (1 - b) ** 2, 0, 1))


def ring_points(s, a):
	"""Body surface in the snout frame (TL units) at station s and ring parameter a.

	s: (n,) or scalar; a: (m,) in [0, 2 pi] (0 = ventral midline, up the left side,
	pi = dorsal midline, down the right side). Returns x, y, z arrays of shape (n, m).
	"""
	s = np.atleast_1d(np.asarray(s, float))[:, None]
	a = np.atleast_1d(np.asarray(a, float))[None, :]
	# stations just inside the capped ends take their shape from the cap station
	s_shape = np.clip(s, 0.022, A.S_END - 0.012)
	st = {k: v for k, v in A.station(s_shape[:, 0]).items()}
	top, bot, hw, mid = st['top'][:, None], st['bot'][:, None], st['hw'][:, None], st['mid'][:, None]
	nT, nB = st['nT'][:, None], st['nB'][:, None]
	left = a <= np.pi
	b = np.where(left, a, 2 * np.pi - a)  # 0 ventral .. pi dorsal on either side
	lower = b <= np.pi / 2
	c = np.where(lower, b, b - np.pi / 2)
	cs, sn = np.abs(np.cos(c)), np.abs(np.sin(c))
	# lower quadrant: from the ventral midline (c = 0) out to the widest point (c = pi/2)
	xl = hw * sn ** (2 / nB)
	yl = mid - (mid - bot) * cs ** (2 / nB)
	# upper quadrant: from the widest point (c = 0) up to the dorsal midline (c = pi/2)
	xu = hw * cs ** (2 / nT)
	yu = mid + (top - mid) * sn ** (2 / nT)
	x = np.where(lower, xl, xu)
	y = np.where(lower, yl, yu)

	# ---- rounded nose and tail ends: scale the section about its centre
	k = _cap_scale(s)
	yc = 0.5 * (top + bot)
	x = x * k
	y = yc + (y - yc) * k

	x, y = _head_features(s, x, y, hw)
	y, dz = _dorsal_features(s, x, y)
	# lower jaw longer than the upper: the rostrum tip is set back from the chin, the chin
	# rounds down and forward
	ylip = A.lip_y(s)
	tipw = 1 - A.smoothstep(0.0, 0.045, s)
	upper = A.smoothstep(ylip - 0.002, ylip + 0.01, y)
	dz = dz - 0.0065 * tipw * upper
	chin = (1 - upper) * tipw * A.smoothstep(ylip, ylip - 0.03, y)
	y = y - 0.0025 * chin
	z = -s + dz
	x = np.where(left, x, -x)
	return x, y, np.broadcast_to(z, x.shape).copy()


def _head_features(s, x, y, hw):
	"""Lower-jaw flare below the lip line and the lip crease (lateral offsets)."""
	ylip = A.lip_y(s)
	on = (1 - A.smoothstep(A.LIP_END - 0.01, A.LIP_END + 0.012, s)) * A.smoothstep(0.004, 0.03, s)
	flare = 0.0055 * on * (0.55 + 0.45 * A.smoothstep(0.02, 0.12, s))
	r = 0.0045  # lip rounding height
	lat = np.sqrt(np.clip(x / np.maximum(hw, 1e-6), 0, 1))  # 0 at the midlines, 1 on the flank
	below = 1 - A.smoothstep(ylip - r, ylip + 0.0005, y)
	# the flare grows from the jaw's ventral edge up to the lip
	jaw = A.smoothstep(ylip - 0.05, ylip - 0.012, y)
	groove = 0.0016 * on * np.exp(-((y - ylip - 0.0004) / 0.0016) ** 2)
	x = x + (flare * below * jaw - groove) * lat
	return x, y


def _dorsal_features(s, x, y):
	"""Dorsal fin on its hump, splash guard, tail-stock knuckles. Returns (y, dz)."""
	ax = np.abs(x)
	dz = np.zeros(np.broadcast(s, x).shape)
	# hump under the dorsal fin
	hump = 0.0075 * np.exp(-((s - 0.64) / 0.04) ** 2) * np.exp(-(ax / 0.034) ** 2)
	# fin: side-view profile (gentle leading edge, steep concave trailing edge)
	s0, s1, st, H = A.DORSAL_S0, A.DORSAL_S1, A.DORSAL_TIP_S, A.DORSAL_H
	u = np.clip((s - s0) / (st - s0), 0, 1)  # leading edge 0..1 up to the tip
	w = np.clip((s1 - s) / (s1 - st), 0, 1)  # trailing edge 1..0 behind the tip
	prof = np.where(s <= st, u ** 1.6, w ** 0.55)
	prof = np.where((s < s0) | (s > s1), 0, prof)
	sigma = 0.0052 + 0.0025 * (1 - prof)  # narrower toward the tip
	h = H * prof
	fin = h * np.exp(-(ax / sigma) ** 2)
	y = y + hump + fin
	# falcate: the upper part of the fin sweeps back
	q = np.clip(fin / H, 0, 1)
	dz = dz - 0.010 * q ** 2
	# splash guard: a ridge in front of the blowholes
	sg = 0.0052 * np.exp(-((s - A.SPLASH_GUARD_S) / 0.0075) ** 2) * np.exp(-(ax / 0.02) ** 4)
	# blowholes: two slits in a shallow depression behind the guard
	bh = -0.0026 * np.exp(-((s - A.BLOWHOLE_S) / 0.009) ** 2) * np.exp(-((ax - 0.0045) / 0.0028) ** 2)
	# tail-stock knuckles along the dorsal ridge
	kn = np.zeros_like(y)
	for i, sk in enumerate(np.linspace(0.705, 0.885, 8)):
		kn = kn + (0.0016 - 0.00012 * i) * np.exp(-((s - sk) / 0.0075) ** 2)
	kn = kn * np.exp(-(ax / 0.006) ** 2)
	y = y + sg + bh + kn
	return y, dz


def _ring_param(s, K, hi=1024):
	"""Ring parameters a (K+1 values over [0, pi], left half) with feature-weighted spacing."""
	a_hi = np.linspace(0, np.pi, hi)
	x, y, z = ring_points(np.array([s]), a_hi)
	x, y, z = x[0], y[0], z[0]
	seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
	arc = np.concatenate([[0], np.cumsum(seg)])
	# weights: curvature + features
	ylip = A.lip_y(s)
	lipw = 3.0 * (1 - A.smoothstep(A.LIP_END - 0.005, A.LIP_END + 0.015, s)) * np.exp(-((y - ylip) / 0.008) ** 2) * (x > 0.3 * x.max())
	finw = 5.0 * np.exp(-((s - 0.645) / 0.035) ** 2) * np.exp(-(np.abs(x) / 0.02) ** 2)
	keel = 1.2 * A.smoothstep(0.72, 0.85, s) * (np.exp(-(np.abs(x) / 0.012) ** 2))
	top = 0.6 * np.exp(-(np.abs(x) / 0.02) ** 2) * (y > 0)
	tx, ty = np.gradient(x), np.gradient(y)
	curv = np.abs(np.gradient(np.arctan2(ty, tx + 1e-12)))
	curv = np.convolve(curv, np.ones(9) / 9, mode='same')
	w = 1.0 + lipw + finw + keel + top
	wm = 0.5 * (w[1:] + w[:-1]) * seg
	W = np.concatenate([[0], np.cumsum(wm)])
	targets = np.linspace(0, W[-1], K + 1)
	return np.interp(targets, W, a_hi), arc[-1]


def body_mesh(n_st, K):
	"""Body: n_st stations, 2K vertices per ring (+1 seam duplicate). Returns a dict."""
	ss = body_stations(n_st)
	ss[0] = 0.0
	rings_a = []
	for s in ss[1:]:
		a_half, _ = _ring_param(s, K)
		a = np.concatenate([a_half, 2 * np.pi - a_half[-2::-1]])  # 2K+1 values, 0 .. 2 pi
		rings_a.append(a)
	P, UV0, UV1, meta = [], [], [], []
	u0, v0, u1, v1 = UV_BODY
	# tip vertex
	x, y, z = ring_points(np.array([1e-5]), np.array([np.pi / 2]))
	tip = np.array([0.0, float(np.mean(ring_points(np.array([0.0003]), np.linspace(0, 2 * np.pi, 16))[1])), 0.0])
	P.append(tip * TL)
	UV0.append([u0, 0.5 * (v0 + v1)])
	UV1.append([0.0, 0.0])
	meta.append([0.0, np.pi])
	ring_start = []
	for i, s in enumerate(ss[1:]):
		a = rings_a[i]
		x, y, z = ring_points(np.array([s]), a)
		x, y, z = x[0], y[0], z[0]
		pts = np.stack([x, y, z], 1) * TL
		seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
		arc = np.concatenate([[0], np.cumsum(seg)])
		half = arc[len(arc) // 2]
		ring_start.append(len(P))
		U = u0 + s_warp(s) * (u1 - u0)
		for j in range(len(a)):
			P.append(pts[j])
			UV0.append([U, v0 + arc[j] / max(arc[-1], 1e-9) * (v1 - v0)])
			UV1.append([arc[j] - half, -s * TL])
			meta.append([s, a[j]])
	P = np.array(P) - ROOT
	UV0 = np.array(UV0)
	UV1 = np.array(UV1)
	meta = np.array(meta)
	R = 2 * K + 1
	F = []
	# tip fan
	r0 = ring_start[0]
	for j in range(R - 1):
		F.append([0, r0 + j, r0 + j + 1])
	for i in range(len(ring_start) - 1):
		a0, b0 = ring_start[i], ring_start[i + 1]
		for j in range(R - 1):
			p, q = a0 + j, a0 + j + 1
			r, t = b0 + j, b0 + j + 1
			F.append([p, r, q])
			F.append([q, r, t])
	F = np.array(F, np.int32)
	# close the tail end with a fan to the last ring's centre
	last = ring_start[-1]
	c = P[last:last + R - 1].mean(0)
	ci = len(P)
	P = np.vstack([P, c])
	UV0 = np.vstack([UV0, [u1, 0.5 * (v0 + v1)]])
	UV1 = np.vstack([UV1, [0, -A.S_END * TL]])
	meta = np.vstack([meta, [A.S_END, np.pi]])
	F = np.vstack([F, [[ci, last + j + 1, last + j] for j in range(R - 1)]])
	# welded topology for normals (seam duplicates -> first vertex of the ring)
	weld = np.arange(len(P))
	for st in ring_start:
		weld[st + R - 1] = st
	N = vertex_normals(P, F, weld)
	return dict(P=P, N=N, UV0=UV0, UV1=UV1, F=F, part=np.full(len(P), PART_BODY), s=meta[:, 0], a=meta[:, 1], stations=ss, K=K)


def vertex_normals(P, F, weld=None):
	idx = F if weld is None else weld[F]
	n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
	N = np.zeros_like(P)
	for k in range(3):
		np.add.at(N, idx[:, k], n)
	if weld is not None:
		N = N[weld]
	L = np.linalg.norm(N, axis=1, keepdims=True)
	return N / np.maximum(L, 1e-12)


# ------------------------------------------------------------------ flippers

def naca_half_thickness(xc, t):
	return 5 * t * (0.2969 * np.sqrt(np.clip(xc, 0, 1)) - 0.1260 * xc - 0.3516 * xc ** 2 + 0.2843 * xc ** 3 - 0.1036 * xc ** 4)


def pec_knobs(r):
	k = np.zeros_like(r)
	for rr, amp, wid in A.PEC_KNOBS:
		k += amp * np.exp(-((r - rr) / wid) ** 2)
	return k


def pec_planform(r):
	"""Chord (m), leading-edge and trailing-edge offsets from the fin axis (m) over span r."""
	c = A.table(A.PEC_CHORD, r) * TL
	c = np.maximum(c, 0)
	# rounded tip
	c = c * np.sqrt(np.clip((1 - r) / 0.03, 0, 1)) ** 0.3
	fle = A.table(A.PEC_LE_FRAC, r)
	knob = pec_knobs(r) * c
	le = fle * c + knob
	te = -(1 - fle) * c
	return c, le, te, knob


def pec_root(side):
	"""Root centre (whale frame, m) and the outward body normal there."""
	s = A.PEC_S
	st = A.station(np.array([s]))
	y = st['bot'][0] + A.PEC_Y_FRAC * (st['top'][0] - st['bot'][0])
	a = np.linspace(0, np.pi, 4001)
	x, yy, z = ring_points(np.array([s]), a)
	j = np.argmin(np.abs(yy[0] - y) + (x[0] < 0) * 10)
	# pick the flank crossing (lower quadrant side)
	cand = np.where(np.abs(np.diff(np.sign(yy[0] - y))) > 0)[0]
	j = cand[0] if len(cand) else j
	p = np.array([x[0][j], yy[0][j], z[0][j]]) * TL
	tx, ty = x[0][j + 1] - x[0][j - 1], yy[0][j + 1] - yy[0][j - 1]
	n = np.array([ty, -tx, 0.0])
	n /= np.linalg.norm(n)
	p = p - ROOT
	if side < 0:
		p[0] = -p[0]
		n[0] = -n[0]
	return p, n


def rot(axis, deg):
	axis = np.asarray(axis, float) / np.linalg.norm(axis)
	t = np.radians(deg)
	K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
	return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def pec_frame(side):
	"""Rest (bind) frame of the flipper: origin at the root, columns = span, up, chord-forward."""
	p, n = pec_root(side)
	# base frame: span out along +x (left) / -x (right), chord forward +z
	span = np.array([side * 1.0, 0, 0])
	up = np.array([0, 1.0, 0])
	fwd = np.array([0, 0, 1.0])
	R = np.stack([span, up, fwd], 1)
	# anhedral (tip down), sweep (tip back), twist (leading edge down) about the span
	Rd = rot([0, 0, 1], side * A.PEC_REST_DIHEDRAL)
	Rs = rot([0, 1, 0], side * A.PEC_REST_SWEEP)
	R = Rs @ Rd @ R
	Rt = rot(R[:, 0], side * A.PEC_REST_TWIST)
	R = Rt @ R
	# root sits ~12 cm inside the body
	origin = p - n * 0.12
	return origin, R


def pec_mesh(side, n_r, n_c):
	"""Flipper surface: n_r span stations, n_c points per half airfoil (upper / lower)."""
	L = A.PEC_LEN * TL
	# span stations: denser over the tubercles and the tip
	g = np.linspace(0, 1, 4001)
	d = 1.0 + 1.2 * A.smoothstep(0.2, 0.3, g) + 1.5 * A.smoothstep(0.9, 1.0, g)
	c = np.concatenate([[0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(g))])
	c /= c[-1]
	rs = np.interp(np.linspace(0, 1, n_r), c, g)
	rs[-1] = 1.0
	chord, le, te, knob = pec_planform(rs)
	thick = A.table(A.PEC_THICK, rs)
	# the tubercles are bulbous: thicker where they protrude
	tfac = 1 + 1.6 * knob / np.maximum(chord, 1e-6)
	# the fin axis curves slightly (leading edge convex): bow the span line back near the tip
	bow = -0.035 * TL * (rs ** 2) * 0.35
	origin, R = pec_frame(side)
	# chordwise parameter (cosine spacing): 0 = leading edge .. 1 = trailing edge
	xc = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_c)))
	u0, v0, u1, v1 = UV_PEC[0 if side > 0 else 1]
	P, UV0, UV1, meta = [], [], [], []
	ring = []
	for i, r in enumerate(rs[:-1]):
		ring.append(len(P))
		cc = max(chord[i], 1e-4)
		yt = naca_half_thickness(xc, thick[i] * tfac[i]) * cc
		yt = np.maximum(yt, 0.004 * (xc > 0.98))  # finite trailing edge
		zc = le[i] - xc * (le[i] - te[i])  # chord line position (m), fwd positive
		# airfoil loop: upper surface TE -> LE, then lower surface LE -> TE
		zu, yu = zc[::-1], yt[::-1]
		zl, yl = zc[1:], -yt[1:]
		zz = np.concatenate([zu, zl])
		yy = np.concatenate([yu, yl])
		# slight camber: the lower surface is flatter
		yy = yy + 0.012 * cc * np.sin(np.pi * (1 - np.concatenate([xc[::-1], xc[1:]])))
		per = np.concatenate([[0], np.cumsum(np.hypot(np.diff(zz), np.diff(yy)))])
		for j in range(len(zz)):
			local = np.array([r * L, yy[j], zz[j] + bow[i]])
			P.append(origin + R @ local)
			UV0.append([u0 + r * (u1 - u0), v0 + per[j] / per[-1] * (v1 - v0)])
			UV1.append([r * L, per[j] - per[-1] * 0.5])
			meta.append([r, per[j] / per[-1]])
	# tip vertex
	tip_local = np.array([L, 0, le[-1] * 0 + 0.5 * (le[-2] + te[-2]) + bow[-1]])
	P.append(origin + R @ tip_local)
	UV0.append([u1, 0.5 * (v0 + v1)])
	UV1.append([L, 0])
	meta.append([1.0, 0.5])
	P, UV0, UV1, meta = map(np.array, (P, UV0, UV1, meta))
	M = 2 * n_c - 1
	F = []
	for i in range(len(ring) - 1):
		a0, b0 = ring[i], ring[i + 1]
		for j in range(M - 1):
			p, q, rr, t = a0 + j, a0 + j + 1, b0 + j, b0 + j + 1
			F += [[p, q, rr], [q, t, rr]]
		# close the trailing edge (last -> first point of the loop)
		p, q, rr, t = a0 + M - 1, a0, b0 + M - 1, b0
		F += [[p, q, rr], [q, t, rr]]
	tipi = len(P) - 1
	a0 = ring[-1]
	for j in range(M):
		F.append([a0 + j, a0 + (j + 1) % M, tipi])
	# root cap (inside the body, keeps the mesh closed for the shadow pass)
	rc = len(P)
	P = np.vstack([P, P[ring[0]:ring[0] + M].mean(0)])
	UV0 = np.vstack([UV0, [u0, 0.5 * (v0 + v1)]])
	UV1 = np.vstack([UV1, [0, 0]])
	meta = np.vstack([meta, [0, 0.5]])
	for j in range(M):
		F.append([ring[0] + (j + 1) % M, ring[0] + j, rc])
	F = np.array(F, np.int32)
	if side < 0:
		F = F[:, [0, 2, 1]]  # mirrored: keep the winding outward
	N = vertex_normals(P, F)
	part = PART_PEC_L if side > 0 else PART_PEC_R
	return dict(P=P, N=N, UV0=UV0, UV1=UV1, F=F, part=np.full(len(P), part), s=meta[:, 0], a=meta[:, 1], origin=origin, R=R)


# ------------------------------------------------------------------ flukes

def _serration(q, seed=3):
	"""Trailing-edge scallops (TL): irregular spacing and depth, deeper toward the tips."""
	rng = np.random.default_rng(seed)
	# cumulative phase with random local frequency
	grid = np.linspace(0, 1, 2001)
	freq = 13 + 5 * rng.standard_normal(40).cumsum() * 0.08
	f = np.interp(grid, np.linspace(0, 1, 40), np.clip(freq, 8, 20))
	ph = np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(grid))])
	depth = np.interp(grid, np.linspace(0, 1, 40), 0.7 + 0.6 * rng.random(40))
	p = np.interp(q, grid, ph)
	dp = np.interp(q, grid, depth)
	# points between rounded scallops: 1 - |sin| gives sharp points toward the rear
	sc = (np.abs(np.sin(np.pi * p)) ** 0.8) * dp
	amp = 0.0011 + 0.0012 * A.smoothstep(0.2, 0.9, q)
	return -amp * (1 - sc) * A.smoothstep(0.03, 0.08, q) * (1 - A.smoothstep(0.96, 1.0, q))


def fluke_planform(q, side=1):
	le = A.table(A.FLUKE_LE, q)
	te = A.table(A.FLUKE_TE, q) + _serration(q, seed=3 if side > 0 else 11)
	return le, te


def fluke_mesh(n_x, n_c):
	b = A.FLUKE_HALF_SPAN
	# span stations x/b in [-1, 1], denser at the notch and near the tips
	g = np.linspace(0, 1, 4001)
	d = 1.0 + 1.5 * np.exp(-(g / 0.05) ** 2) + 0.8 * A.smoothstep(0.85, 1.0, g) + 0.6 * A.smoothstep(0.1, 0.4, g)
	c = np.concatenate([[0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(g))])
	c /= c[-1]
	half = np.interp(np.linspace(0, 1, n_x // 2 + 1), c, g)
	qs = np.concatenate([-half[::-1], half[1:]])  # -1 .. 1
	xc = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_c)))
	u0, v0, u1, v1 = UV_FLUKE
	zroot = -1.0  # notch at s = 1
	P, UV0, UV1, meta = [], [], [], []
	ring = []
	tips = []
	for i, qq in enumerate(qs):
		q = abs(qq)
		side = 1 if qq >= 0 else -1
		le, te = fluke_planform(np.array([q]), side)
		le, te = le[0], te[0]
		if q >= 0.9999:
			tips.append(len(P))
			P.append(np.array([qq * b, A.FLUKE_Y, zroot + le]) * TL)
			UV0.append([u0 + (qq * 0.5 + 0.5) * (u1 - u0), 0.5 * (v0 + v1)])
			UV1.append([qq * b * TL, 0])
			meta.append([qq, 0.5])
			ring.append(None)
			continue
		ring.append(len(P))
		cc = (le - te)
		t = A.table(A.FLUKE_THICK, np.array([q]))[0]
		yt = naca_half_thickness(xc, 1.0) / (5 * 0.2969 * 0.2) * 0 + naca_half_thickness(xc, 0.2) / 0.2 * t * 0.5 / 0.5
		# thickness t (TL) is the maximum: scale the unit profile
		prof = naca_half_thickness(xc, 1.0)
		yt = prof / prof.max() * t * 0.5
		yt = np.maximum(yt, 0.0003 * (xc > 0.97))
		zc = le - xc * cc
		# slight dihedral: flukes are flat; a hint of upward curl near the tips
		yoff = 0.004 * A.smoothstep(0.6, 1.0, q)
		zz = np.concatenate([zc[::-1], zc[1:]])
		yy = np.concatenate([yt[::-1], -yt[1:]]) + A.FLUKE_Y + yoff
		per = np.concatenate([[0], np.cumsum(np.hypot(np.diff(zz), np.diff(yy)))])
		for j in range(len(zz)):
			P.append(np.array([qq * b, yy[j], zroot + zz[j]]) * TL)
			UV0.append([u0 + (qq * 0.5 + 0.5) * (u1 - u0), v0 + per[j] / per[-1] * (v1 - v0)])
			UV1.append([qq * b * TL, (per[j] - per[-1] * 0.5) * TL])
			meta.append([qq, per[j] / per[-1]])
	P = np.array(P) - ROOT
	UV0, UV1, meta = map(np.array, (UV0, UV1, meta))
	M = 2 * n_c - 1
	F = []
	idx = [r for r in ring]
	for i in range(len(idx) - 1):
		a0, b0 = idx[i], idx[i + 1]
		if a0 is None and b0 is None:
			continue
		if a0 is None or b0 is None:
			# fan to a tip vertex
			tipi = tips[0] if a0 is None else tips[-1]
			r0 = b0 if a0 is None else a0
			for j in range(M):
				f = [r0 + j, r0 + (j + 1) % M, tipi]
				F.append(f if a0 is not None else [f[1], f[0], f[2]])
			continue
		for j in range(M - 1):
			p, qv, r, t = a0 + j, a0 + j + 1, b0 + j, b0 + j + 1
			F += [[p, r, qv], [qv, r, t]]
		p, qv, r, t = a0 + M - 1, a0, b0 + M - 1, b0
		F += [[p, r, qv], [qv, r, t]]
	F = np.array(F, np.int32)
	N = vertex_normals(P, F)
	return dict(P=P, N=N, UV0=UV0, UV1=UV1, F=F, part=np.full(len(P), PART_FLUKE), s=meta[:, 0], a=meta[:, 1])


# ------------------------------------------------------------------ eyes

def eye_mesh(side, rings=7, seg=20):
	s, y = A.EYE_S, A.EYE_Y
	a = np.linspace(0, np.pi, 4001)
	x, yy, z = ring_points(np.array([s]), a)
	cand = np.where(np.abs(np.diff(np.sign(yy[0] - y))) > 0)[0]
	j = cand[-1]
	p = np.array([x[0][j], yy[0][j], z[0][j]]) * TL - ROOT
	tx, ty = x[0][j + 1] - x[0][j - 1], yy[0][j + 1] - yy[0][j - 1]
	n = np.array([ty, -tx, 0.0])
	n /= np.linalg.norm(n)
	# tilt the eye slightly forward and down (it looks forward-sideways)
	n = rot([0, 1, 0], 18) @ n
	n /= np.linalg.norm(n)
	fwd = np.array([0, 0, 1.0])
	fwd = fwd - n * fwd.dot(n)
	fwd /= np.linalg.norm(fwd)
	up = np.cross(n, fwd)
	ra, rb, h = 0.055, 0.034, 0.016  # m: half length, half height, dome height
	P, UV0, UV1 = [], [], []
	u0, v0, u1, v1 = UV_EYE
	for i in range(rings + 1):
		t = i / rings
		rr = np.sin(t * np.pi / 2)
		for k in range(seg):
			ph = 2 * np.pi * k / seg
			cx, cy = np.cos(ph) * rr, np.sin(ph) * rr
			hh = h * np.cos(t * np.pi / 2) - 0.006  # seated 6 mm into the skin at the rim
			P.append(p + fwd * cx * ra + up * cy * rb + n * hh)
			UV0.append([0.5 * (u0 + u1) + cx * 0.5 * (u1 - u0), 0.5 * (v0 + v1) + cy * 0.5 * (v1 - v0)])
			UV1.append([cx * ra, cy * rb])
	P = np.array(P)
	if side < 0:
		P[:, 0] = -P[:, 0]
	F = []
	for i in range(rings):
		for k in range(seg):
			a0 = i * seg + k
			b0 = i * seg + (k + 1) % seg
			F += [[a0, a0 + seg, b0], [b0, a0 + seg, b0 + seg]]
	F = np.array(F, np.int32)
	# remove the degenerate apex ring duplicates: keep as is (tiny)
	if side < 0:
		F = F[:, [0, 2, 1]]
	N = vertex_normals(P, F)
	return dict(P=P, N=N, UV0=np.array(UV0), UV1=np.array(UV1), F=F, part=np.full(len(P), PART_EYE), s=np.full(len(P), s), a=np.zeros(len(P)))


# ------------------------------------------------------------------ levels of detail

LODS = {
	# body stations, half-ring segments, flipper span x chord, fluke span x chord, eyes
	'lod0': dict(body=(230, 60), pec=(120, 17), fluke=(150, 15), eyes=True),
	'lod1': dict(body=(90, 22), pec=(40, 8), fluke=(56, 7), eyes=False),
	'lod2': dict(body=(34, 9), pec=(12, 4), fluke=(18, 4), eyes=False),
}


def build_parts(lod='lod0'):
	cfg = LODS[lod]
	parts = [body_mesh(*cfg['body'])]
	parts.append(pec_mesh(1, *cfg['pec']))
	parts.append(pec_mesh(-1, *cfg['pec']))
	parts.append(fluke_mesh(*cfg['fluke']))
	if cfg['eyes']:
		parts.append(eye_mesh(1))
		parts.append(eye_mesh(-1))
	return parts


def merge(parts):
	out = {k: [] for k in ('P', 'N', 'UV0', 'UV1', 'part', 's', 'a')}
	F = []
	base = 0
	for p in parts:
		for k in out:
			out[k].append(p[k])
		F.append(p['F'] + base)
		base += len(p['P'])
	res = {k: np.concatenate(v) for k, v in out.items()}
	res['F'] = np.concatenate(F)
	return res


if __name__ == '__main__':
	for lod in LODS:
		m = merge(build_parts(lod))
		print(lod, 'verts', len(m['P']), 'tris', len(m['F']), 'bbox', m['P'].min(0).round(2), m['P'].max(0).round(2))
