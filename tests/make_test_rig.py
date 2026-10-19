"""Build a blocky test quadruped that differs from the cat on purpose:
'.L/.R' bone names, faces +Y after the object rotation, object scale 0.5, two skinned meshes, a short action.
blender -b --python tests/make_test_rig.py -- OUT.blend
"""
import bpy, bmesh, sys, math
from mathutils import Vector, Matrix

OUT = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

B = {  # name: (head, tail, parent, half-thickness, deform)
    "root": ((0, 0, 0), (0.2, 0, 0), None, 0, False),
    "hips": ((-0.3, 0, 0.5), (-0.1, 0, 0.5), "root", 0.12, True),
    "spine": ((-0.1, 0, 0.5), (0.15, 0, 0.52), "hips", 0.12, True),
    "chest": ((0.15, 0, 0.52), (0.35, 0, 0.55), "spine", 0.13, True),
    "neck": ((0.35, 0, 0.55), (0.45, 0, 0.7), "chest", 0.06, True),
    "head": ((0.45, 0, 0.7), (0.65, 0, 0.7), "neck", 0.08, True),
    "tail.001": ((-0.3, 0, 0.5), (-0.42, 0, 0.55), "hips", 0.03, True),
    "tail.002": ((-0.42, 0, 0.55), (-0.54, 0, 0.6), "tail.001", 0.025, True),
    "tail.003": ((-0.54, 0, 0.6), (-0.66, 0, 0.62), "tail.002", 0.02, True),
}
for s, y in (("L", 0.1), ("R", -0.1)):
    B.update({f"upperarm.{s}": ((0.3, y, 0.5), (0.3, y, 0.3), "chest", 0.045, True),
              f"forearm.{s}": ((0.3, y, 0.3), (0.3, y, 0.08), f"upperarm.{s}", 0.035, True),
              f"hand.{s}": ((0.3, y, 0.08), (0.38, y, 0.0), f"forearm.{s}", 0.03, True),
              f"thigh.{s}": ((-0.25, y, 0.5), (-0.2, y, 0.3), "hips", 0.05, True),
              f"shin.{s}": ((-0.2, y, 0.3), (-0.28, y, 0.1), f"thigh.{s}", 0.035, True),
              f"foot.{s}": ((-0.28, y, 0.1), (-0.2, y, 0.0), f"shin.{s}", 0.03, True)})

ad = bpy.data.armatures.new("DogArm"); arm = bpy.data.objects.new("Dog_Rig", ad); sc.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='EDIT')
for n, (h, t, p, _, d) in B.items():
    eb = ad.edit_bones.new(n); eb.head, eb.tail = h, t; eb.use_deform = d
    eb.roll = 0
for n, (h, t, p, _, d) in B.items():
    if p: ad.edit_bones[n].parent = ad.edit_bones[p]
bpy.ops.object.mode_set(mode='OBJECT')


def box_mesh(name, bones, color):
    bm = bmesh.new(); groups = []
    for n in bones:
        h, t, _, r, _ = B[n]
        h, t = Vector(h), Vector(t); y = (t - h).normalized()
        x = y.cross(Vector((0, 0, 1)) if abs(y.z) < 0.9 else Vector((1, 0, 0))).normalized(); z = x.cross(y)
        M = Matrix((x * r * 2, y * (t - h).length * 1.15, z * r * 2)).transposed().to_4x4()
        M.translation = (h + t) / 2
        res = bmesh.ops.create_cube(bm, size=1.0, matrix=M)
        new = res["verts"]
        bmesh.ops.subdivide_edges(bm, edges=list({e for v in new for e in v.link_edges}), cuts=2, use_grid_fill=True)
        groups.append((n, len(bm.verts)))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob)
    start = 0
    for n, end in groups:
        vg = ob.vertex_groups.new(name=n); vg.add(list(range(start, end)), 1.0, 'REPLACE'); start = end
    mat = bpy.data.materials.new(name + "_mat"); mat.use_nodes = True
    next(nd for nd in mat.node_tree.nodes if nd.type == "BSDF_PRINCIPLED").inputs["Base Color"].default_value = color
    me.materials.append(mat)
    md = ob.modifiers.new("Armature", 'ARMATURE'); md.object = arm
    ob.parent = arm
    return ob


box_mesh("DogBody", [n for n in B if B[n][4] and n != "head"], (0.75, 0.5, 0.25, 1))
box_mesh("DogHead", ["head"], (0.6, 0.35, 0.2, 1))
arm.rotation_euler = (0, 0, math.radians(90))      # now faces +Y, its left side is -X
arm.scale = (0.5, 0.5, 0.5)

act = bpy.data.actions.new("trot"); arm.animation_data_create(); arm.animation_data.action = act
for f, a in ((1, 25), (10, -25), (20, 25)):
    for s, k in (("L", 1), ("R", -1)):
        for b, sgn in ((f"thigh.{s}", 1), (f"upperarm.{s}", -1)):
            pb = arm.pose.bones[b]; pb.rotation_mode = 'XZY'
            pb.rotation_euler = (math.radians(a * k * sgn), 0, 0); pb.keyframe_insert("rotation_euler", frame=f)
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print("TEST_RIG_SAVED", OUT)
