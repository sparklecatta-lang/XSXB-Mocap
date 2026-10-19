"""Apply a fit to the rig: save .blend with the action, export FBX, or render both panels per frame
with the start-frame cameras (for the side-by-side comparison).
blender -b <project>/rig/clean.blend --python src/bl_apply.py -- PROJECT FIT_DIR MODE [PANEL_W PANEL_H]
MODE: blend | fbx | render     (writes FIT_DIR/mocap.blend, FIT_DIR/mocap.fbx, FIT_DIR/render/*.jpg)
"""
import bpy, sys, json, math, os, numpy as np
from mathutils import Euler, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qconfig as Q
from blcheck import skin_error, require

argv = sys.argv[sys.argv.index("--") + 1:]
PROJECT, FD, MODE = argv[0], argv[1], argv[2]
cfg = Q.load(PROJECT); cams = json.load(open(Q.paths(PROJECT)["cams"]))
f = np.load(os.path.join(FD, "fit.npz"))
sc = bpy.context.scene
arm = bpy.data.objects[cfg["armature"]]; me = bpy.data.objects[cfg["mesh"]]
names = [str(n) for n in f["names"]]
E, R, L = f["euler"], f["obj_rot"], f["obj_loc"]
T = len(E)
fps = int(round(cfg["video"]["fps"]))

# --- action
arm.animation_data_create()
act = bpy.data.actions.new(cfg["name"] + "_mocap"); arm.animation_data.action = act
arm.rotation_mode = 'XYZ'
free = [n for n in names if Q.bone_limits_deg(cfg, n, names) is not None or Q.helper_source(cfg, n, names)]
for t in range(T):
    fr = t + 1
    arm.location = Vector(L[t]); arm.rotation_euler = Euler(R[t], 'XYZ')
    arm.keyframe_insert("location", frame=fr); arm.keyframe_insert("rotation_euler", frame=fr)
    for i, n in enumerate(names):
        pb = arm.pose.bones[n]; pb.rotation_mode = 'XZY'
        pb.rotation_euler = Euler(E[t, i], 'XZY')
        if n in free:
            pb.keyframe_insert("rotation_euler", frame=fr)
    if fr % 20 == 0: print(f"PROGRESS {fr}/{T}", flush=True)
sc.frame_start, sc.frame_end = 1, T
sc.render.fps = fps
for fr in sorted({1, (T + 1) // 2, T}):
    sc.frame_set(fr)
    require(skin_error(arm, me, Q.paths(PROJECT)["rig"]), f"frame {fr}")

if MODE == "blend":
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(FD, "mocap.blend"), compress=True, copy=True)
    print("SAVED_BLEND", flush=True); sys.exit(0)

if MODE == "fbx":
    for o in bpy.context.view_layer.objects:
        o.select_set(o in (arm, me))
    bpy.context.view_layer.objects.active = arm
    bpy.ops.export_scene.fbx(filepath=os.path.join(FD, "mocap.fbx"), use_selection=True,
                             object_types={'ARMATURE', 'MESH'}, add_leaf_bones=False, bake_anim=True,
                             bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False,
                             bake_anim_force_startend_keying=True, bake_anim_simplify_factor=0.0)
    print("SAVED_FBX", flush=True); sys.exit(0)

# --- MODE render: scene like the start frame
PW, PH = int(argv[3]), int(argv[4])
os.makedirs(os.path.join(FD, "render"), exist_ok=True)
ow = cams["panels"]["front"]["ortho_width_m"]
bg_col = (0.62, 0.62, 0.62)
bpy.ops.mesh.primitive_plane_add(size=40 * ow / 3, location=(0, 0, 0))
floor = bpy.context.active_object
fm = bpy.data.materials.new("Floor"); fm.use_nodes = True
bsdf = next(n for n in fm.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
bsdf.inputs["Base Color"].default_value = (*bg_col, 1); bsdf.inputs["Roughness"].default_value = 0.9
floor.data.materials.append(fm)
world = sc.world or bpy.data.worlds.new("W"); sc.world = world; world.use_nodes = True
bgn = next(n for n in world.node_tree.nodes if n.type == 'BACKGROUND')
bgn.inputs[0].default_value = (*bg_col, 1); bgn.inputs[1].default_value = 0.9
sun_d = bpy.data.lights.new("Key", 'SUN'); sun_d.energy = 2.2; sun_d.angle = math.radians(25)
sun = bpy.data.objects.new("Key", sun_d); sc.collection.objects.link(sun)
sun.rotation_euler = Euler((math.radians(35), 0, math.radians(cams["panels"]["side"]["az_deg"] - 30)))
sc.render.engine = 'BLENDER_EEVEE'; sc.eevee.taa_render_samples = 16
sc.view_settings.view_transform = 'Standard'
sc.render.resolution_x, sc.render.resolution_y = PW, PH
sc.render.image_settings.file_format = 'JPEG'
cam_d = bpy.data.cameras.new("Cam"); cam = bpy.data.objects.new("Cam", cam_d)
sc.collection.objects.link(cam); sc.camera = cam
cam_d.type = 'ORTHO'; cam_d.sensor_fit = 'HORIZONTAL'; cam_d.clip_end = 100 * ow / 3
n = 0
for key in ("front", "side"):
    c = cams["panels"][key]
    cam_d.ortho_scale = c["ortho_width_m"]
    cam.location = Vector(c["location"]); cam.rotation_euler = Euler(c["rotation_euler"])
    for t in range(T):
        sc.frame_set(t + 1)
        sc.render.filepath = os.path.join(FD, "render", f"{key}_{t:05d}.jpg")
        bpy.ops.render.render(write_still=True)
        n += 1
        if n % 5 == 0: print(f"PROGRESS {n}/{2 * T}", flush=True)
print("RENDER_DONE", flush=True)
