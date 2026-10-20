"""Start frame for the video model: top = front view, bottom = left-side view, same ortho scale.
blender -b <project>/rig/clean.blend --python src/bl_start_frame.py -- PROJECT
Writes start_frame/panel_front.png, panel_side.png, start_frame.png (stacked) and cameras.json
(3x4 projection matrices in full-image pixels, top-left origin).
"""
import bpy, sys, math, json, os, numpy as np
from mathutils import Vector, Euler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qconfig as Q

PROJECT = sys.argv[sys.argv.index("--") + 1:][0]
cfg = Q.load(PROJECT); P = Q.paths(PROJECT); OUT = P["start"]
sf = cfg["start_frame"]
W, H, TILT = int(sf["panel_width"]), int(sf["panel_height"]), float(sf["tilt_deg"])
LR = sf.get("layout", "tb") == "lr"                  # panels side by side instead of stacked
os.makedirs(OUT, exist_ok=True)
sc = bpy.context.scene
arm = bpy.data.objects[cfg["armature"]]; me = bpy.data.objects[cfg["mesh"]]

# feet on the floor
dg = bpy.context.evaluated_depsgraph_get()
ev = me.evaluated_get(dg); mm = ev.to_mesh()
co = np.array([me.matrix_world @ v.co for v in mm.vertices]); ev.to_mesh_clear()
zmin = co[:, 2].min(); arm.location.z -= zmin
ctr = (co.min(0) + co.max(0)) / 2; ctr[2] -= zmin
ext = co.max(0) - co.min(0)
ORTHO_W = float(sf["ortho_width_m"] or max(2.0 * max(ext[0], ext[1]),  # body length ~ half the panel width
                                           1.4 * max(ext[2], min(ext[0], ext[1])) * W / H))   # wide panels: keep height room
AIM_Z = float(sf["aim_height_m"] or 0.4865 * ext[2])
bpy.context.view_layer.update()

# floor (shadow) + backdrop colour
bg_col = (0.62, 0.62, 0.62)
bpy.ops.mesh.primitive_plane_add(size=40 * ORTHO_W / 3, location=(0, 0, 0))
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
AZ_F, AZ_S = Q.view_azimuths_deg(cfg)
sun.rotation_euler = Euler((math.radians(35), 0, math.radians(AZ_S - 30)))

sc.render.engine = 'BLENDER_EEVEE'
sc.eevee.taa_render_samples = 64
sc.view_settings.view_transform = 'Standard'
sc.render.film_transparent = False
sc.render.resolution_x, sc.render.resolution_y = W, H
sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGB'

cam_d = bpy.data.cameras.new("Cam"); cam = bpy.data.objects.new("Cam", cam_d)
sc.collection.objects.link(cam); sc.camera = cam
cam_d.type = 'ORTHO'; cam_d.ortho_scale = ORTHO_W; cam_d.sensor_fit = 'HORIZONTAL'
cam_d.clip_end = 100 * ORTHO_W / 3


def place(az_deg):
    t = math.radians(TILT); a = math.radians(az_deg)
    d = Vector((math.cos(t) * math.cos(a), math.cos(t) * math.sin(a), math.sin(t)))
    tgt = Vector((ctr[0], ctr[1], AIM_Z))
    cam.location = tgt + d * 20 * ORTHO_W / 3
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()


cams = {}
for i, (name, az) in enumerate((("front", AZ_F), ("side", AZ_S))):
    xoff, yoff = (i * W, 0) if LR else (0, i * H)
    place(az)
    Pm = np.array(cam.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=W, y=H)) @ np.array(cam.matrix_world.inverted())
    Kp = np.array([[W / 2, 0, 0, W / 2 + xoff], [0, -H / 2, 0, H / 2 + yoff], [0, 0, 0, 1]])   # NDC -> stacked-image px
    cams[name] = {"P_pixels": (Kp @ Pm).tolist(), "panel_y0": yoff, "panel_x0": xoff, "az_deg": az, "tilt_deg": TILT,
                  "ortho_width_m": ORTHO_W, "location": list(cam.location), "rotation_euler": list(cam.rotation_euler)}
    sc.render.filepath = f"{OUT}/panel_{name}.png"
    bpy.ops.render.render(write_still=True)
    print(f"PROGRESS {i + 1}/2", flush=True)

# stack: top = front, bottom = side, thin divider drawn over the seam (geometry unchanged)
imgs = [bpy.data.images.load(f"{OUT}/panel_{k}.png") for k in ("front", "side")]
px = [np.array(im.pixels[:], np.float32).reshape(H, W, 4) for im in imgs]   # bottom-up rows
if LR:
    full = np.concatenate([px[0], px[1]], 1); full[:, W - 2:W + 2, :3] = 0.25
else:
    full = np.concatenate([px[1], px[0]], 0); full[H - 2:H + 2, :, :3] = 0.25
IW, IH = (2 * W, H) if LR else (W, 2 * H)
out = bpy.data.images.new("start_frame", IW, IH, alpha=False)
out.pixels[:] = full.ravel()
out.filepath_raw = P["start_png"]; out.file_format = 'PNG'; out.save()

json.dump({"image_size": [IW, IH], "layout": "lr" if LR else "tb", "panels": cams, "armature_location": list(arm.location)},
          open(P["cams"], "w"), indent=1)
print("START_FRAME_DONE", flush=True)
