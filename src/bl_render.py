"""Synthetic two-view training data for one rig.
blender -b <project>/rig/clean.blend --python src/bl_render.py -- PROJECT START COUNT SEED [nodraw]
Per sample: random pose inside the joint limits -> deformation metric -> 2 cameras (front-ish, side-ish)
 -> EEVEE RGB (transparent bg) + Workbench rest-position map (16-bit, Raw) + 2D/3D keypoint labels.
'nodraw' only samples poses and writes the deformation metric (for threshold tuning).
"""
import bpy, sys, os, math, json, numpy as np
from mathutils import Matrix, Euler, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qconfig as Q

argv = sys.argv[sys.argv.index("--") + 1:]
PROJECT, START, COUNT, SEED = argv[0], int(argv[1]), int(argv[2]), int(argv[3])
NODRAW = len(argv) > 4 and argv[4] == "nodraw"
cfg = Q.load(PROJECT); PATHS = Q.paths(PROJECT); OUT = PATHS["data"]
RC = cfg["render"]; RES = int(RC["resolution"])
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

sc = bpy.context.scene
arm = bpy.data.objects[cfg["armature"]]
me = bpy.data.objects[cfg["mesh"]]
names = [b.name for b in arm.data.bones]
IDX = {n: i for i, n in enumerate(names)}
B = len(names)

# reference poses (e.g. a walk cycle shipped with the rig) -> sampling prior
REF = np.zeros((1, B, 3))
if os.path.exists(PATHS["poses"]):
    rp = np.load(PATHS["poses"]); rn = [str(n) for n in rp["names"]]
    REF = np.zeros((len(rp["euler_xzy"]), B, 3))
    for j, n in enumerate(rn):
        if n in IDX: REF[:, IDX[n]] = rp["euler_xzy"][:, j]
HAS_REF = os.path.exists(PATHS["poses"])

LIM = [None if (L := Q.bone_limits_deg(cfg, n, names)) is None else np.radians(np.array(L, float)) for n in names]
HELP = [(i, IDX[s[0]], s[1]) for i, n in enumerate(names) if (s := Q.helper_source(cfg, n, names))]
CHAINS = Q.chains(cfg, names)
LTOK = (cfg.get("side_tokens") or [None])[0]
MSIGN = np.array(cfg["mirror_sign"], float)
FWD, LEFT = Vector(Q.axis(cfg["orientation"]["forward"])), Vector(Q.axis(cfg["orientation"]["left"]))
AZ_F, AZ_S = Q.view_azimuths_deg(cfg)


def sample_pose():
    e = np.zeros((B, 3))
    mode = rng.random()
    base = REF[rng.integers(len(REF))] if (HAS_REF and rng.random() < 0.8) else np.zeros((B, 3))
    for i, L in enumerate(LIM):
        if L is None: continue
        lo, hi = L[:, 0], L[:, 1]
        if mode < RC["near_ref_fraction"]:          # near a reference pose
            v = base[i] + rng.normal(0, 1, 3) * np.radians([12, 5, 6])
        else:                                       # wide random, temperature towards the limit box
            t = rng.uniform(0.35, 1.0)
            v = base[i] + (rng.uniform(lo, hi) - base[i]) * t
        e[i] = np.clip(v, lo, hi)
    for idx, amp in CHAINS:                         # tails etc.: one smooth curve along the chain
        k = np.arange(len(idx)) / (len(idx) - 1.0)
        for ax, a_deg in ((0, amp[0]), (2, amp[1])):
            a, b, ph, f = rng.normal(0, 0.5), rng.normal(0, 1), rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 1.6)
            curve = np.radians(a_deg) * (a + b * np.cos(np.pi * f * k + ph)) / 1.5
            for c, i in zip(curve, idx):
                e[i, ax] = c if LIM[i] is None else np.clip(c, LIM[i][ax, 0], LIM[i][ax, 1])
    if LTOK and rng.random() < RC["symmetric_fraction"]:   # symmetric pose: right side mirrors left
        for i, n in enumerate(names):
            m = Q.mirror_name(cfg, n)
            if LTOK in n and m in IDX:
                e[IDX[m]] = e[i] * MSIGN
    for i, s, fac in HELP:
        e[i] = fac * e[s]
    return e


def apply_pose(e):
    for i, n in enumerate(names):
        pb = arm.pose.bones[n]
        pb.rotation_mode = 'XZY'
        pb.location = (0, 0, 0)
        pb.rotation_euler = Euler((e[i, 0], e[i, 1], e[i, 2]), 'XZY')
    pitch = math.radians(rng.uniform(-RC["body_pitch_deg"], RC["body_pitch_deg"]))
    roll = math.radians(rng.uniform(-RC["body_roll_deg"], RC["body_roll_deg"]))
    arm.rotation_mode = 'XYZ'
    arm.rotation_euler = (Matrix.Rotation(roll, 3, FWD) @ Matrix.Rotation(pitch, 3, LEFT)).to_euler('XYZ')
    bpy.context.view_layer.update()


# ---------------- deformation metric (edge stretch vs rest)
m = me.data
EDGES = np.empty(len(m.edges) * 2, np.int32); m.edges.foreach_get("vertices", EDGES); EDGES = EDGES.reshape(-1, 2)
R0 = np.load(PATHS["rig"])["verts"]
L0 = np.linalg.norm(R0[EDGES[:, 0]] - R0[EDGES[:, 1]], axis=1) + 1e-9


def deformed_verts():
    dg = bpy.context.evaluated_depsgraph_get()
    ev = me.evaluated_get(dg)
    mm = ev.to_mesh()
    co = np.empty(len(mm.vertices) * 3, np.float32); mm.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    co = co.reshape(-1, 3)
    mw = np.array(me.matrix_world)
    return co @ mw[:3, :3].T + mw[:3, 3]


def deform_metric(V):
    r = np.linalg.norm(V[EDGES[:, 0]] - V[EDGES[:, 1]], axis=1) / L0
    return float(np.mean((r > 1.8) | (r < 0.45))), float(r.max()), float(r.min())


KP = [(k[0], k[1], Q.end_value(k[2])) for k in cfg["keypoints"]]


def kp3d():
    aw = arm.matrix_world
    out = []
    for _, b, end in KP:
        pb = arm.pose.bones[b]
        out.append(list(aw @ (pb.head + (pb.tail - pb.head) * end)))
    return np.array(out, np.float32)


# ---------------- cameras
cam_data = bpy.data.cameras.new("SynCam"); cam = bpy.data.objects.new("SynCam", cam_data)
sc.collection.objects.link(cam); sc.camera = cam
sc.render.resolution_x = sc.render.resolution_y = RES
sc.render.resolution_percentage = 100


def look_at(obj, eye, target):
    d = Vector(target) - Vector(eye)
    obj.location = eye
    obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def setup_camera(V, view):
    ctr = (V.min(0) + V.max(0)) / 2
    rad = float(np.linalg.norm(V - ctr, axis=1).max())
    if view == 0:
        az = math.radians(AZ_F + rng.uniform(-30, 30))
    else:
        az = math.radians(AZ_S + (0 if rng.random() < 0.7 else 180) + rng.uniform(-30, 30))
    if rng.random() < 0.15: az = rng.uniform(0, 2 * math.pi)
    el = math.radians(rng.uniform(-3, 25))
    dirv = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    ortho = rng.random() < 0.6
    cam_data.type = 'ORTHO' if ortho else 'PERSP'
    cam_data.lens = rng.uniform(40, 85); cam_data.sensor_width = 36; cam_data.sensor_fit = 'HORIZONTAL'
    dist = rad * 6 if ortho else rad / math.tan(math.atan(18 / cam_data.lens)) * 1.15
    cam_data.clip_end = max(100.0, dist * 4)
    look_at(cam, Vector(ctr + dirv * dist), Vector(ctr))
    cam_data.shift_x = cam_data.shift_y = 0
    cam_data.ortho_scale = 2 * rad
    bpy.context.view_layer.update()
    uv = project(V)                                   # frame tightly using projected vertices
    lo, hi = uv.min(0), uv.max(0)
    span = float((hi - lo).max())
    fill = rng.uniform(0.6, 0.92)
    if ortho:
        cam_data.ortho_scale *= span / fill
    else:
        cam_data.lens *= fill / span
    bpy.context.view_layer.update()
    uv = project(V); c = (uv.min(0) + uv.max(0)) / 2
    jit = rng.uniform(-0.06, 0.06, 2)
    cam_data.shift_x += (c[0] - 0.5) - jit[0]
    cam_data.shift_y += (c[1] - 0.5) - jit[1]
    bpy.context.view_layer.update()


def proj_mats():
    dg = bpy.context.evaluated_depsgraph_get()
    return np.array(cam.calc_matrix_camera(dg, x=RES, y=RES)), np.array(cam.matrix_world.inverted())


def project(X):
    P, Vw = proj_mats()
    Xh = np.c_[X, np.ones(len(X))] @ (P @ Vw).T
    return (Xh[:, :2] / Xh[:, 3:4] + 1) / 2              # 0..1, y up


def to_pixels(X):
    uv = project(X)
    return np.c_[uv[:, 0] * RES, (1 - uv[:, 1]) * RES]


# ---------------- render setups
m.color_attributes.active_color = m.color_attributes["rest_col"]
world = sc.world or bpy.data.worlds.new("W"); sc.world = world
world.use_nodes = True
bg = next(n for n in world.node_tree.nodes if n.type == 'BACKGROUND')
sun_d = bpy.data.lights.new("Sun", 'SUN'); sun = bpy.data.objects.new("Sun", sun_d); sc.collection.objects.link(sun)


def render_rgb(path):
    sc.render.engine = 'BLENDER_EEVEE'
    sc.eevee.taa_render_samples = 16
    sc.view_settings.view_transform = 'Standard'
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.render.image_settings.color_depth = '8'
    g = rng.uniform(0.3, 1.0)
    bg.inputs[0].default_value = (g, g * rng.uniform(0.9, 1.05), g * rng.uniform(0.85, 1.1), 1)
    bg.inputs[1].default_value = rng.uniform(0.4, 1.5)
    sun_d.energy = rng.uniform(1.0, 5.0)
    sun.rotation_euler = Euler((math.radians(rng.uniform(0, 70)), 0, rng.uniform(0, 2 * math.pi)))
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def render_dense(path):
    sc.render.engine = 'BLENDER_WORKBENCH'
    sh = sc.display.shading
    sh.light = 'FLAT'; sh.color_type = 'VERTEX'
    sh.show_shadows = False; sh.show_cavity = False; sh.show_object_outline = False
    sh.show_specular_highlight = False
    sc.display.render_aa = 'OFF'
    sc.view_settings.view_transform = 'Raw'
    sc.view_settings.look = 'None'
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.render.image_settings.color_depth = '16'
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


# ---------------- main loop
labels = []
for s in range(START, START + COUNT):
    e = sample_pose(); apply_pose(e)
    V = deformed_verts()
    frac, rmax, rmin = deform_metric(V)
    rec = {"id": s, "euler": e.tolist(), "arm_rot": list(arm.rotation_euler), "deform": [frac, rmax, rmin]}
    if not NODRAW:
        X = kp3d(); views = []
        for v in range(2):
            setup_camera(V, v)
            P, Vw = proj_mats()
            stem = f"{OUT}/{s:05d}_v{v}"
            render_rgb(stem + "_rgb.png")
            render_dense(stem + "_dense.png")
            views.append({"P": P.tolist(), "V": Vw.tolist(), "ortho": cam_data.type == 'ORTHO',
                          "kp2d": to_pixels(X).tolist()})
        rec["kp3d"] = X.tolist(); rec["views"] = views
    labels.append(rec)
    print(f"SAMPLE {s} deform={frac:.4f} rmax={rmax:.2f} rmin={rmin:.2f}", flush=True)

with open(f"{OUT}/labels_{START:05d}.json", "w") as f:
    json.dump({"kp_names": [k[0] for k in KP], "res": RES, "samples": labels}, f)
print("DONE", COUNT, flush=True)
