"""Blender-side consistency check: the deformed mesh must equal linear blend skinning with rig.npz
(rest matrices + weights) and the current bone matrices -- that is what the fitter and the labels assume."""
import bpy, numpy as np


def skin_error(arm, me, rig_npz):
    """Max vertex error relative to the body size for the current frame/pose."""
    d = np.load(rig_npz)
    names = [str(n) for n in d["names"]]; W = d["weights"]; cw = d["verts"]
    dg = bpy.context.evaluated_depsgraph_get()
    ev, ae = me.evaluated_get(dg), arm.evaluated_get(dg)
    mm = ev.to_mesh()
    got = np.empty(len(mm.vertices) * 3, np.float32); mm.vertices.foreach_get("co", got); ev.to_mesh_clear()
    mw = np.array(ev.matrix_world)
    got = got.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    Sk = np.array([np.array(ae.matrix_world @ ae.pose.bones[n].matrix) for n in names]) @ np.linalg.inv(d["rest_mats"])
    want = np.einsum("vb,bij,vj->vi", W, Sk[:, :3, :], np.c_[cw, np.ones(len(cw))])
    ok = W.sum(1) > 0                                   # unweighted vertices stay at rest in Blender
    return float(np.abs(got - want)[ok].max()) / float(np.ptp(cw, 0).max())


def require(err, where, tol=1e-3):
    print(f"SKIN_CHECK {where} max error {err:.2e} of body size", flush=True)
    if err > tol:
        raise SystemExit(f"ERROR {where}: the mesh does not follow the rig's skinning (error {err:.3f} of body size). "
                         "Re-import the rig (step 1); unsupported deformers or constraints can also cause this.")
