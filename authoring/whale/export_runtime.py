"""Write the runtime mesh (all levels of detail) as one binary + JSON manifest.

  <out>/humpback.json   counts, byte offsets, rig landmarks
  <out>/humpback.bin    per level: position f32x3, normal i16x3 (snorm), uv f32x2 (texture atlas),
                        uv1 f32x2 (shading coordinates), rig f32x4 (axial z, part, span r, 0), index u32

Run with Blender's Python (numpy):  .../python3.13 authoring/whale/export_runtime.py <out dir>
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anatomy as A
import geometry as G

out = sys.argv[1] if len(sys.argv) > 1 else '.'
os.makedirs(out, exist_ok=True)
blob = bytearray()
levels = []

def add(arr):
	global blob
	while len(blob) % 16: blob += b'\0'
	off = len(blob)
	blob += arr.tobytes()
	return off

pec_root_z = {}
for lod in ['lod0', 'lod1', 'lod2']:
	parts = G.build_parts(lod)
	m = G.merge(parts)
	P = m['P'].astype(np.float32)
	N = np.clip(np.round(m['N'] * 32767), -32767, 32767).astype(np.int16)
	part = m['part']
	# shading coordinates: body (s, a / 2pi), flippers (r, perimeter), flukes (x / half span, perimeter)
	uv = np.stack([m['s'], m['a']], 1).astype(np.float32)
	uv[part == 0, 1] /= 2 * np.pi
	rig = np.zeros((len(P), 4), np.float32)
	rig[:, 0] = P[:, 2]
	rig[:, 1] = part
	for side, pid in ((1, G.PART_PEC_L), (-1, G.PART_PEC_R)):
		o, R = G.pec_frame(side)
		sel = part == pid
		rig[sel, 0] = o[2]
		rig[sel, 2] = m['s'][sel]  # span fraction
		pec_root_z[side] = o
	rig[part == G.PART_EYE, 2] = 0
	F = m['F'].astype(np.uint32)
	atlas = m['UV0'].astype(np.float32)
	lv = dict(name=lod, vertices=int(len(P)), indices=int(F.size),
		position=add(P), normal=add(N), uv=add(atlas), uv1=add(uv), rig=add(rig), index=add(F))
	levels.append(lv)
	print(lod, lv['vertices'], 'verts', F.shape[0], 'tris')

# rest centre line of the body (y of the section centre over axial z), for the spine frames
zs = np.linspace(0, A.S_END, 200)
st = A.station(zs)
yc = 0.5 * (st['top'] + st['bot']) * A.TL - G.ROOT[1]
zz = -zs * A.TL - G.ROOT[2]
pec = {}
for side in (1, -1):
	o, R = G.pec_frame(side)
	pec['left' if side > 0 else 'right'] = dict(origin=o.round(5).tolist(), basis=R.round(6).tolist())
# blowhole position (whale frame)
bx, by, bz = G.ring_points(np.array([A.BLOWHOLE_S]), np.array([np.pi]))
blow = (np.array([0, by[0][0], bz[0][0]]) * A.TL - G.ROOT).round(4).tolist()
manifest = dict(
	version=1, length=A.TL, levels=levels,
	centerline=dict(z=zz.round(4).tolist(), y=yc.round(4).tolist()),
	snoutZ=float(-G.ROOT[2]), tailZ=float(-A.S_END * A.TL - G.ROOT[2]), flukeTipZ=float((-1 - A.FLUKE_LE[-1][1] * -1) * 0),
	pectoral=pec, blowhole=blow, snoutY=float(-G.ROOT[1]), rootS=A.ROOT_S,
	flukeRootZ=float(-A.FLUKE_ROOT_S * A.TL - G.ROOT[2]), notchZ=float(-A.TL - G.ROOT[2]),
)
open(os.path.join(out, 'humpback.bin'), 'wb').write(bytes(blob))
open(os.path.join(out, 'humpback.json'), 'w').write(json.dumps(manifest))
print('bytes', len(blob), 'blowhole', blow)
