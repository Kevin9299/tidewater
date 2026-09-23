"""Humpback whale (Megaptera novaeangliae) anatomy: stations, landmarks and fin planforms.

Units: TL (total length, snout tip to fluke notch) unless a name ends in _m (metres).
Whale frame (matches the runtime / glTF): x = whale's left, y = up, z = forward.
The snout tip is at z = 0 and the body runs toward -z (z = -s * TL); heights are relative
to the snout tip. geometry.py re-centres everything on the root joint (ROOT_S).

Sources: NOAA Fisheries lateral illustration (profile digitised in authoring, see README),
underwater / aerial / fluke photographs (Wikimedia Commons), and published proportions:
flippers 25-33 % TL, flukes up to ~1/3 TL, dorsal fin at ~2/3 TL, ventral grooves from the
chin to the umbilicus, tubercles on the rostrum and lower jaw.
"""
import numpy as np

TL = 14.5  # m, an adult (females average ~15 m)
ROOT_S = 0.40  # root joint (approximate centre of mass)

# ------------------------------------------------------------------ body stations
# s, top, bottom, half width, widest level, n_top, n_bottom
#   top / bottom: dorsal and ventral outline (side view), relative to the snout tip
#   widest level: height of the widest point of the section (the lip line on the head)
#   n_top / n_bottom: superellipse exponents of the upper / lower half (2 = ellipse,
#   > 2 boxier (flat rostrum), < 2 keeled (tail stock))
STATIONS = np.array([
	# s      top     bot     hw     mid     nT    nB
	[0.000, 0.000, -0.026, 0.000, -0.014, 2.4, 2.1],
	[0.006, 0.003, -0.032, 0.019, -0.014, 2.6, 2.1],
	[0.018, 0.006, -0.036, 0.031, -0.015, 2.9, 2.2],
	[0.040, 0.009, -0.039, 0.040, -0.017, 3.1, 2.3],
	[0.070, 0.013, -0.043, 0.047, -0.020, 3.1, 2.3],
	[0.100, 0.018, -0.050, 0.052, -0.024, 3.0, 2.3],
	[0.130, 0.023, -0.057, 0.057, -0.029, 2.85, 2.3],
	[0.160, 0.027, -0.064, 0.062, -0.034, 2.7, 2.3],
	[0.190, 0.030, -0.072, 0.066, -0.040, 2.55, 2.3],
	[0.215, 0.031, -0.081, 0.071, -0.046, 2.4, 2.3],
	[0.240, 0.031, -0.091, 0.076, -0.050, 2.3, 2.3],
	[0.270, 0.032, -0.103, 0.081, -0.050, 2.2, 2.3],
	[0.310, 0.033, -0.115, 0.086, -0.050, 2.15, 2.25],
	[0.360, 0.034, -0.127, 0.090, -0.052, 2.1, 2.2],
	[0.410, 0.035, -0.135, 0.092, -0.054, 2.1, 2.2],
	[0.460, 0.035, -0.140, 0.092, -0.056, 2.1, 2.15],
	[0.510, 0.034, -0.142, 0.090, -0.057, 2.1, 2.1],
	[0.560, 0.033, -0.142, 0.085, -0.058, 2.05, 2.0],
	[0.610, 0.033, -0.141, 0.077, -0.058, 2.0, 1.95],
	[0.650, 0.032, -0.139, 0.068, -0.058, 1.95, 1.85],
	[0.690, 0.025, -0.135, 0.058, -0.058, 1.85, 1.75],
	[0.730, 0.013, -0.129, 0.047, -0.060, 1.7, 1.6],
	[0.770, -0.001, -0.123, 0.036, -0.062, 1.6, 1.45],
	[0.810, -0.015, -0.117, 0.026, -0.066, 1.5, 1.35],
	[0.850, -0.030, -0.110, 0.018, -0.070, 1.42, 1.3],
	[0.885, -0.043, -0.103, 0.0135, -0.073, 1.42, 1.3],
	[0.910, -0.052, -0.098, 0.0122, -0.075, 1.55, 1.45],
	[0.930, -0.058, -0.092, 0.0135, -0.075, 1.9, 1.8],
	[0.950, -0.066, -0.084, 0.0150, -0.075, 2.3, 2.2],
	[0.965, -0.071, -0.079, 0.0110, -0.075, 2.5, 2.4],
	[0.975, -0.074, -0.076, 0.0000, -0.075, 2.5, 2.4],
])
S_END = 0.975  # the tail stock closes inside the fluke root

# ------------------------------------------------------------------ head features
# lip (mouth) line: height (relative to snout tip) along s; the gape corner at LIP_END
LIP = np.array([
	[0.000, -0.014], [0.03, -0.0145], [0.07, -0.017], [0.11, -0.021], [0.15, -0.027],
	[0.19, -0.036], [0.215, -0.044], [0.232, -0.052], [0.242, -0.058],
])
LIP_END = 0.242
EYE_S, EYE_Y = 0.252, -0.043  # eye just above and behind the gape corner
BLOWHOLE_S = 0.192  # centre of the paired blowholes (dorsal midline)
SPLASH_GUARD_S = 0.176
UMBILICUS_S = 0.50  # the ventral grooves end here
GENITAL_S, ANUS_S = 0.655, 0.705

# dorsal fin on its hump: base from DORSAL_S0 to DORSAL_S1, tip height DORSAL_H (above the back)
DORSAL_S0, DORSAL_S1, DORSAL_TIP_S, DORSAL_H = 0.618, 0.672, 0.660, 0.021

# ------------------------------------------------------------------ pectoral flipper
PEC_S = 0.318  # centre of the root (along the body)
PEC_Y_FRAC = 0.30  # root height as a fraction of the section depth (0 = belly, 1 = back)
PEC_LEN = 0.31  # span, root to tip
# chord (TL) and the leading-edge offset of the fin axis over the span r (0 root .. 1 tip)
PEC_CHORD = np.array([[0.0, 0.066], [0.08, 0.070], [0.2, 0.068], [0.4, 0.062], [0.6, 0.054],
	[0.75, 0.046], [0.87, 0.036], [0.95, 0.024], [0.985, 0.013], [1.0, 0.0]])
# fraction of the chord ahead of the fin axis (the leading edge is convex, the trailing edge straighter)
PEC_LE_FRAC = np.array([[0.0, 0.45], [0.3, 0.42], [0.7, 0.45], [1.0, 0.5]])
PEC_THICK = np.array([[0.0, 0.22], [0.3, 0.19], [0.7, 0.16], [1.0, 0.13]])  # t / c
# leading-edge tubercles: (r, amplitude as a fraction of the local chord, width in r)
PEC_KNOBS = [(0.27, 0.05, 0.030), (0.345, 0.07, 0.030), (0.42, 0.08, 0.030), (0.49, 0.085, 0.028),
	(0.555, 0.09, 0.027), (0.615, 0.095, 0.025), (0.672, 0.10, 0.024), (0.725, 0.105, 0.022),
	(0.775, 0.11, 0.021), (0.822, 0.11, 0.020), (0.866, 0.11, 0.019), (0.905, 0.10, 0.017),
	(0.94, 0.09, 0.015)]
# rest pose (the bind pose of the flipper bones): anhedral (down), sweep (back), incidence
PEC_REST_DIHEDRAL = -32.0  # deg
PEC_REST_SWEEP = 28.0  # deg
PEC_REST_TWIST = -8.0  # deg (leading edge down)

# ------------------------------------------------------------------ flukes
FLUKE_HALF_SPAN = 0.158  # TL (span ~0.32 TL)
FLUKE_ROOT_S = 0.918  # leading edge at the insertion
# leading edge z (TL, forward of the notch) over the half span q = |x| / half span
FLUKE_LE = np.array([[0.0, 0.082], [0.12, 0.075], [0.3, 0.060], [0.5, 0.041], [0.7, 0.019],
	[0.85, -0.003], [0.95, -0.024], [1.0, -0.044]])
# trailing edge z over q (the notch apex at q = 0)
FLUKE_TE = np.array([[0.0, 0.0], [0.035, -0.010], [0.1, -0.009], [0.3, -0.008], [0.5, -0.010],
	[0.7, -0.016], [0.85, -0.025], [0.95, -0.034], [1.0, -0.044]])
FLUKE_THICK = np.array([[0.0, 0.0140], [0.1, 0.0120], [0.3, 0.0080], [0.6, 0.0045], [0.9, 0.0022], [1.0, 0.0012]])  # TL
FLUKE_Y = -0.075  # height of the fluke plane (relative to the snout tip)


# ------------------------------------------------------------------ interpolation

def pchip(x, y, xi):
	"""Monotone cubic (Fritsch-Carlson) interpolation: C1, no overshoot."""
	x = np.asarray(x, float)
	y = np.asarray(y, float)
	xi = np.asarray(xi, float)
	h = np.diff(x)
	d = np.diff(y) / h
	m = np.zeros_like(y)
	if len(x) > 2:
		w1 = 2 * h[1:] + h[:-1]
		w2 = h[1:] + 2 * h[:-1]
		same = d[:-1] * d[1:] > 0
		with np.errstate(divide='ignore', invalid='ignore'):
			hm = (w1 + w2) / (w1 / d[:-1] + w2 / d[1:])
		m[1:-1] = np.where(same, hm, 0.0)
	# end slopes (three-point, shape preserving)
	def end(h0, h1, d0, d1):
		v = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
		if np.sign(v) != np.sign(d0):
			v = 0.0
		elif np.sign(d0) != np.sign(d1) and abs(v) > abs(3 * d0):
			v = 3 * d0
		return v
	if len(x) > 2:
		m[0] = end(h[0], h[1], d[0], d[1])
		m[-1] = end(h[-1], h[-2], d[-1], d[-2])
	else:
		m[:] = d[0]
	i = np.clip(np.searchsorted(x, xi) - 1, 0, len(x) - 2)
	t = (np.clip(xi, x[0], x[-1]) - x[i]) / h[i]
	t2, t3 = t * t, t * t * t
	return ((2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * h[i] * m[i]
		+ (-2 * t3 + 3 * t2) * y[i + 1] + (t3 - t2) * h[i] * m[i + 1])


def table(tab, xi, col=1):
	tab = np.asarray(tab, float)
	return pchip(tab[:, 0], tab[:, col], xi)


def station(s):
	"""Section parameters at s (array): dict of arrays in TL units."""
	s = np.asarray(s, float)
	S = STATIONS
	out = {}
	for k, name in enumerate(['top', 'bot', 'hw', 'mid', 'nT', 'nB'], start=1):
		out[name] = pchip(S[:, 0], S[:, k], s)
	# the widest level stays inside the section
	out['mid'] = np.clip(out['mid'], out['bot'] + 0.2 * (out['top'] - out['bot']), out['top'] - 0.2 * (out['top'] - out['bot']))
	return out


def lip_y(s):
	return table(LIP, np.clip(s, 0, LIP_END))


def smoothstep(e0, e1, x):
	t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
	return t * t * (3 - 2 * t)
