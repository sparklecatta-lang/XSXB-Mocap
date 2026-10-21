"""Render the rig from 8 camera azimuths so the user can click the view where the character faces the camera.
blender -b rig/clean.blend --python bl_orient.py -- PROJECT_DIR
Output: PROJECT_DIR/rig/orient/az_000.jpg ... az_315.jpg (azimuth = camera direction, degrees from +X towards +Y, as in qconfig.view_azimuths_deg)."""
import bpy, sys, os, math
from mathutils import Vector

d = sys.argv[sys.argv.index("--") + 1]
out = os.path.join(d, "rig", "orient"); os.makedirs(out, exist_ok=True)
sc = bpy.context.scene
meshes = [o for o in bpy.data.objects if o.type == 'MESH' and not o.hide_render]
for o in bpy.data.objects:
    if o.type != 'MESH': o.hide_render = True
bpy.context.view_layer.update()
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
ctr = (lo + hi) / 2; size = max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
sc.render.engine = 'BLENDER_WORKBENCH'
sc.render.resolution_x = sc.render.resolution_y = 256
sc.render.image_settings.file_format = 'JPEG'
sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'TEXTURE'
sc.display.shading.show_shadows = False
w = sc.world or bpy.data.worlds.new("W"); sc.world = w; w.use_nodes = True
bg = next((n for n in w.node_tree.nodes if n.type == 'BACKGROUND'), None)
if bg: bg.inputs[0].default_value = (0.82, 0.82, 0.82, 1)
cd = bpy.data.cameras.new("OC"); cd.type = 'ORTHO'; cd.ortho_scale = size * 1.12
cam = bpy.data.objects.new("OC", cd); sc.collection.objects.link(cam); sc.camera = cam
for az in range(0, 360, 45):
    a = math.radians(az); t = math.radians(12)
    v = Vector((math.cos(t) * math.cos(a), math.cos(t) * math.sin(a), math.sin(t)))
    cam.location = ctr + v * size * 3
    cam.rotation_euler = (-v).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = os.path.join(out, f"az_{az:03d}.jpg")
    bpy.ops.render.render(write_still=True)
print("ORIENT_DONE", out, flush=True)
