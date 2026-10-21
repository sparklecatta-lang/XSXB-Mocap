"""Import a rigged quadruped: clean it for synthetic data and export the rig for the fitter.
blender -b <project>/rig/source.blend --python src/bl_prep.py -- PROJECT
Writes rig/clean.blend, rig/rig.npz, rig/inspect.json, rig/ref_poses.npz (frames of every action that moves the rig)
and, when PROJECT/config.json does not exist yet, a draft config guessed from the bone names.
"""
import bpy, sys, os, re, json, numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qconfig as Q
from blcheck import skin_error, require

ARGS = sys.argv[sys.argv.index("--") + 1:]
PROJECT = ARGS[0]
P = Q.paths(PROJECT)
if len(ARGS) > 1:                                    # .fbx / .glb source: import into an empty scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = os.path.splitext(ARGS[1])[1].lower()
    if ext == ".fbx": bpy.ops.import_scene.fbx(filepath=ARGS[1])
    else: bpy.ops.import_scene.gltf(filepath=ARGS[1])
    print("IMPORTED", ARGS[1], flush=True)
os.makedirs(os.path.dirname(P["rig"]), exist_ok=True)
draft = not os.path.exists(P["config"])
cfg = Q.normalize({}) if draft else Q.load(PROJECT)
sc = bpy.context.scene

# --- 1. pick armature + skinned mesh(es)
arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
if not arms:
    raise SystemExit("ERROR no armature in the file")
arm = bpy.data.objects.get(cfg["armature"] or "")
if arm is None or arm.type != 'ARMATURE':
    if cfg["armature"]: print("WARNING armature from config not found, picking the largest one", flush=True)
    arm = max(arms, key=lambda o: len(o.data.bones))
want = [cfg["mesh"]] if isinstance(cfg["mesh"], str) else (cfg["mesh"] or [])
if want and all(n in bpy.data.objects for n in want):
    meshes = [bpy.data.objects[n] for n in want]
else:
    meshes = [o for o in bpy.data.objects if o.type == 'MESH' and
              any(m.type == 'ARMATURE' and m.object == arm for m in o.modifiers)]
if not meshes:
    raise SystemExit("ERROR no mesh with an Armature modifier pointing at " + arm.name)
print("ARMATURE", arm.name, "MESHES", [m.name for m in meshes], flush=True)

# --- 2. reference poses from ALL actions that move this armature (sampling prior), every frame
names = [b.name for b in arm.data.bones]
ad = arm.animation_data or arm.animation_data_create()
act0 = ad.action
eul, act_names, act_start = [], [], []
for act in ([act0] if act0 else []) + [a for a in bpy.data.actions if a != act0]:
    try:
        ad.action = act
        if getattr(ad, "action_slot", 0) is None:                     # Blender 4.4+ slotted actions
            slots = [s for s in getattr(act, "slots", []) if getattr(s, "target_id_type", "OBJECT") in ("OBJECT", "UNSPECIFIED")]
            if slots: ad.action_slot = slots[0]
    except Exception as ex:
        print("WARNING skipped action", act.name, ex, flush=True); continue
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    f0, f1 = int(act.frame_range[0]), int(act.frame_range[1])
    rows = []
    for f in range(f0, f1 + 1, max(1, (f1 - f0 + 1) // 400)):         # at most ~400 frames per action
        sc.frame_set(f)
        rows.append([list(arm.pose.bones[n].matrix_basis.to_euler('XZY')) for n in names])
    rows = np.array(rows, np.float32)
    if not len(rows) or np.abs(rows).max() < 1e-4: continue             # does not move this armature
    act_names.append(act.name); act_start.append(len(eul)); eul.extend(rows)
if eul:
    np.savez(P["poses"], names=np.array(names), euler_xzy=np.array(eul, np.float32),
             actions=np.array(act_names), action_start=np.array(act_start))
    print("REF_POSES", len(act_names), "actions", len(eul), "frames", act_names, flush=True)
elif os.path.exists(P["poses"]):
    print("REF_POSES kept existing file", flush=True)

# --- 3. strip everything else, rest pose, bake object transforms into the data
keep = {arm.name} | {m.name for m in meshes}
for o in list(bpy.data.objects):
    if o.name not in keep:
        bpy.data.objects.remove(o, do_unlink=True)
for o in [arm] + meshes:
    if o.animation_data: o.animation_data_clear()
for a in list(bpy.data.actions):
    bpy.data.actions.remove(a)
for pb in arm.pose.bones:
    pb.rotation_mode = 'XZY'
    pb.matrix_basis = Matrix.Identity(4)
for m in meshes:
    for md in list(m.modifiers):                       # subdivision etc. would change the vertex count
        if md.type != 'ARMATURE':
            print("REMOVED_MODIFIER", m.name, md.name, md.type)
            m.modifiers.remove(md)
        elif md.use_deform_preserve_volume:              # the fitter uses linear blend skinning
            md.use_deform_preserve_volume = False; print("DISABLED_PRESERVE_VOLUME", m.name, flush=True)
    if m.data.shape_keys:
        m.shape_key_clear()
sc.frame_start = sc.frame_end = 1
sc.frame_set(1)
for o in [arm] + meshes:
    if o.parent:
        mw = o.matrix_world.copy(); o.parent = None; o.matrix_world = mw
for o in [arm] + meshes:
    o.hide_set(False); o.hide_viewport = False
need_apply = any(not np.allclose(np.array(o.matrix_world), np.eye(4), atol=1e-6) for o in [arm] + meshes)
if need_apply:
    with bpy.context.temp_override(active_object=arm, object=arm, selected_objects=[arm] + meshes,
                                   selected_editable_objects=[arm] + meshes):
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    print("APPLIED_TRANSFORMS", flush=True)
if len(meshes) > 1:
    with bpy.context.temp_override(active_object=meshes[0], object=meshes[0], selected_objects=meshes,
                                   selected_editable_objects=meshes):
        bpy.ops.object.join()
    print("JOINED_MESHES", flush=True)
me = meshes[0]
# the mesh must follow the armature object (the fitter moves/rotates the object): parent it, both are identity now
me.parent = arm; me.parent_type = 'OBJECT'; me.matrix_parent_inverse = Matrix.Identity(4)
bpy.context.view_layer.update()

# --- 4. rest-position colour attribute (normalised 0..1 in the rest bbox)
m = me.data
co = np.empty(len(m.vertices) * 3, np.float32)
m.vertices.foreach_get('co', co)
co = co.reshape(-1, 3)
mw = np.array(me.matrix_world)
cw = co @ mw[:3, :3].T + mw[:3, 3]
lo, hi = cw.min(0), cw.max(0)
ctr, ext = (lo + hi) / 2, (hi - lo).max() * 1.02
nrm = (cw - ctr) / ext + 0.5
if "rest_col" in m.attributes:
    m.attributes.remove(m.attributes["rest_col"])
att = m.attributes.new("rest_col", 'FLOAT_COLOR', 'POINT')
att.data.foreach_set('color', np.concatenate([nrm, np.ones((len(nrm), 1), np.float32)], 1).astype(np.float32).ravel())

# --- 5. rig/mesh export for the fitter
m.calc_loop_triangles()
tris = np.empty(len(m.loop_triangles) * 3, np.int32)
m.loop_triangles.foreach_get('vertices', tris)
vg_names = [g.name for g in me.vertex_groups]
bidx = {n: i for i, n in enumerate(names)}
W = np.zeros((len(m.vertices), len(names)), np.float32)
for v in m.vertices:
    for g in v.groups:
        n = vg_names[g.group]
        if n in bidx:
            W[v.index, bidx[n]] = g.weight
s = W.sum(1, keepdims=True); W = np.where(s > 0, W / np.maximum(s, 1e-8), W)
aw = np.array(arm.matrix_world)
rest = np.array([np.array(aw @ b.matrix_local) for b in arm.data.bones], np.float32)
heads = np.array([list(aw @ b.head_local.to_4d())[:3] for b in arm.data.bones], np.float32)
tails = np.array([list(aw @ b.tail_local.to_4d())[:3] for b in arm.data.bones], np.float32)
parents = np.array([bidx[b.parent.name] if b.parent else -1 for b in arm.data.bones], np.int32)
np.savez(P["rig"], names=np.array(names), parents=parents, rest_mats=rest, heads=heads, tails=tails,
         verts=cw.astype(np.float32), tris=tris.reshape(-1, 3), weights=W, restcol_center=ctr,
         restcol_extent=np.float32(ext))
print("RIG verts", cw.shape[0], "tris", tris.size // 3, "bones", len(names), "unweighted", int((s[:, 0] == 0).sum()), flush=True)

info = {"armature": arm.name, "mesh": me.name, "verts": int(cw.shape[0]), "bbox_min": lo.tolist(), "bbox_max": hi.tolist(),
        "bones": [{"name": b.name, "parent": b.parent.name if b.parent else None, "deform": b.use_deform,
                   "head": heads[i].tolist(), "tail": tails[i].tolist()} for i, b in enumerate(arm.data.bones)]}
json.dump(info, open(P["inspect"], "w", encoding="utf-8"), indent=1, ensure_ascii=False)


# --- 6. draft config (only when the project has none yet)
def draft_config():
    size = float((hi - lo).max())
    pairs = [("_L_", "_R_"), (".L", ".R"), ("_L", "_R"), ("Left", "Right"), ("_l_", "_r_"), (".l", ".r"),
             ("_l", "_r"), ("L_", "R_"), ("left", "right")]
    dnames = [n for n in names if arm.data.bones[n].use_deform] or names      # helper/cloth bones must not vote
    best, tok = 0, None
    for a, b in pairs:
        k = sum(1 for n in dnames if a in n and n.replace(a, b) in bidx)
        if k > best: best, tok = k, (a, b)
    orient = {"forward": "-Y", "left": "+X"}
    if tok:                                          # infer left axis from where the left bones sit
        L = np.mean([heads[bidx[n]] for n in dnames if tok[0] in n and n.replace(*tok) in bidx], 0)
        R = np.mean([heads[bidx[n.replace(*tok)]] for n in dnames if tok[0] in n and n.replace(*tok) in bidx], 0)
        d = (L - R)[:2]; ax = int(np.argmax(np.abs(d)))
        left = ("+" if d[ax] > 0 else "-") + "XY"[ax]
        fwd = np.cross(Q.axis(left), (0, 0, 1))      # forward = left x up
        fa = int(np.argmax(np.abs(fwd)))
        orient = {"forward": ("+" if fwd[fa] > 0 else "-") + "XYZ"[fa], "left": left}

    def kp_name(bn, tail):
        side = ""
        if tok:
            if tok[0] in bn: side, bn = "l_", bn.replace(tok[0], "_")
            elif tok[1] in bn: side, bn = "r_", bn.replace(tok[1], "_")
        c = re.sub(r"[^a-z0-9]+", "_", bn.lower()).strip("_")
        return side + c + ("_end" if tail else "")

    deform = [i for i, b in enumerate(arm.data.bones) if b.use_deform] or list(range(len(names)))
    dset = set(deform)
    kps, pos = [], []
    for i in deform:
        for tail in (0, 1):
            if tail and any(c in dset for c in range(len(names)) if parents[c] == i):
                continue                              # tail only for chain ends
            p = (tails if tail else heads)[i]
            if any(np.linalg.norm(p - q) < 0.01 * size for q in pos):
                continue
            kps.append([kp_name(names[i], tail), names[i], "tail" if tail else "head"]); pos.append(p)
    # skeleton: each keypoint to the nearest keypoint on its own bone or the closest ancestor bone
    kb = {}
    for k, (_, b, _) in enumerate(kps): kb.setdefault(b, []).append(k)
    skel = []
    for k, (n, b, e) in enumerate(kps):
        cand = [j for j in kb[b] if j != k and kps[j][2] == "head" and e == "tail"]
        p = parents[bidx[b]]
        while not cand and p >= 0:
            cand = kb.get(names[p], []); p = parents[p]
        if cand:
            j = min(cand, key=lambda j: np.linalg.norm(pos[j] - pos[k]))
            skel.append([kps[j][0], n])
    zmin, zh = lo[2], hi[2] - lo[2]
    feet = [kps[k][0] for k in range(len(kps)) if pos[k][2] < zmin + 0.06 * zh]
    return {"name": os.path.basename(os.path.normpath(PROJECT)),
            "armature": arm.name, "mesh": me.name, "orientation": orient, "orientation_confirmed": False,
            "side_tokens": list(tok) if tok else None, "keypoints": kps, "skeleton": skel, "foot_keypoints": feet,
            "default_limits": [[-45, 45], [-15, 15], [-25, 25]],
            "joint_limits": [{"match": "^" + re.escape(names[i]) + "$", "limits": None}
                             for i in range(len(names)) if parents[i] < 0],
            "helper_bones": [], "smooth_chains": []}


if draft:
    d = draft_config()
    Q.save(PROJECT, d)
    print("DRAFT_CONFIG keypoints", len(d["keypoints"]), "orientation", d["orientation"], "sides", d["side_tokens"], flush=True)
elif cfg["armature"] != arm.name or cfg["mesh"] != me.name:
    raw = json.load(open(P["config"], encoding="utf-8")); raw["armature"], raw["mesh"] = arm.name, me.name
    Q.save(PROJECT, raw)

# --- 7. self-check: with a random pose AND a moved/rotated armature object, Blender's deformed mesh must equal
# linear blend skinning with the exported rest matrices + weights (what the fitter assumes)
rng = np.random.default_rng(0)
for pb in arm.pose.bones:
    pb.rotation_euler = rng.uniform(-0.3, 0.3, 3)
arm.rotation_mode = 'XYZ'; arm.rotation_euler = (0.2, -0.3, 0.5); arm.location = (0.3, -0.2, 0.1)
bpy.context.view_layer.update()
err = skin_error(arm, me, P["rig"])
for pb in arm.pose.bones:
    pb.matrix_basis = Matrix.Identity(4)
arm.rotation_euler = (0, 0, 0); arm.location = (0, 0, 0)
bpy.context.view_layer.update()
require(err, "import")

bpy.ops.wm.save_as_mainfile(filepath=P["clean"], compress=True)
print("PREP_DONE", flush=True)
