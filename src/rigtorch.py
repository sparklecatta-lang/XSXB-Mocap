"""Differentiable rig (torch): FK with Blender conventions + LBS on vertex subsets.
Bone rotations are XZY eulers of the pose-bone basis (Blender), the object gets an XYZ euler + location."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qconfig as Q


def euler_xzy(e):
    """Blender Euler((x,y,z),'XZY').to_matrix(): R = Ry @ Rz @ Rx.  e (...,3) -> (...,3,3)"""
    x, y, z = e.unbind(-1)
    cx, sx, cy, sy, cz, sz = x.cos(), x.sin(), y.cos(), y.sin(), z.cos(), z.sin()
    o, l = torch.zeros_like(x), torch.ones_like(x)
    Rx = torch.stack([l, o, o, o, cx, -sx, o, sx, cx], -1).reshape(*x.shape, 3, 3)
    Ry = torch.stack([cy, o, sy, o, l, o, -sy, o, cy], -1).reshape(*x.shape, 3, 3)
    Rz = torch.stack([cz, -sz, o, sz, cz, o, o, o, l], -1).reshape(*x.shape, 3, 3)
    return Ry @ Rz @ Rx


def euler_xyz(e):
    """Blender object rotation_euler 'XYZ': R = Rz @ Ry @ Rx."""
    x, y, z = e.unbind(-1)
    cx, sx, cy, sy, cz, sz = x.cos(), x.sin(), y.cos(), y.sin(), z.cos(), z.sin()
    o, l = torch.zeros_like(x), torch.ones_like(x)
    Rx = torch.stack([l, o, o, o, cx, -sx, o, sx, cx], -1).reshape(*x.shape, 3, 3)
    Ry = torch.stack([cy, o, sy, o, l, o, -sy, o, cy], -1).reshape(*x.shape, 3, 3)
    Rz = torch.stack([cz, -sz, o, sz, cz, o, o, o, l], -1).reshape(*x.shape, 3, 3)
    return Rz @ Ry @ Rx


class Rig:
    def __init__(self, project, dev="cuda", dtype=torch.float32):
        cfg = Q.load(project)
        d = np.load(Q.paths(project)["rig"])
        self.names = [str(n) for n in d["names"]]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.parents = d["parents"].tolist()
        f = lambda a: torch.tensor(a, device=dev, dtype=dtype)
        self.rest = f(d["rest_mats"])                                   # (B,4,4) armature space
        self.rest_inv = torch.linalg.inv(self.rest)
        self.length = f(np.linalg.norm(d["tails"] - d["heads"], axis=1))
        local = []
        for i, p in enumerate(self.parents):
            local.append(self.rest[i] if p < 0 else self.rest_inv[p] @ self.rest[i])
        self.local = torch.stack(local)                                 # rest of bone in parent space
        self.verts = f(d["verts"]); self.tris = torch.tensor(d["tris"], device=dev, dtype=torch.long)
        self.W = f(d["weights"])
        self.rc_center = f(d["restcol_center"]); self.rc_extent = float(d["restcol_extent"])
        KP = cfg["keypoints"]
        self.kp_bone = torch.tensor([self.idx[b] for _, b, _ in KP], device=dev)
        self.kp_end = f([Q.end_value(e) for _, _, e in KP])
        self.kp_names = [k for k, _, _ in KP]
        lo, hi, has = [], [], []
        for n in self.names:
            L = Q.bone_limits_deg(cfg, n, self.names)
            has.append(L is not None)
            L = L or ((0, 0), (0, 0), (0, 0))
            lo.append([a for a, _ in L]); hi.append([b for _, b in L])
        self.lim_lo, self.lim_hi = f(np.radians(lo)), f(np.radians(hi))
        self.free = torch.tensor(has, device=dev)                      # bones with their own rotation
        self.mid_src = [(i, self.idx[h[0]], h[1]) for i, n in enumerate(self.names)
                        if (h := Q.helper_source(cfg, n, self.names))]
        self.dev, self.dtype = dev, dtype

    def full_euler(self, e):
        """Copy half-angle helper bones from their source bones. e (...,B,3)"""
        e = e.clone()
        for i, s, fac in self.mid_src:
            e[..., i, :] = fac * e[..., s, :]
        return e

    def fk(self, e, obj_rot, obj_loc):
        """e (N,B,3) XZY eulers, obj_rot (N,3) XYZ euler, obj_loc (N,3) -> world bone matrices (N,B,4,4)"""
        N, B = e.shape[:2]
        e = self.full_euler(e)
        basis = torch.zeros(N, B, 4, 4, device=self.dev, dtype=self.dtype)
        basis[..., :3, :3] = euler_xzy(e); basis[..., 3, 3] = 1
        A = torch.zeros(N, 4, 4, device=self.dev, dtype=self.dtype)
        A[:, :3, :3] = euler_xyz(obj_rot); A[:, :3, 3] = obj_loc; A[:, 3, 3] = 1
        G = [None] * B
        for i, p in enumerate(self.parents):
            M = self.local[i] @ basis[:, i]
            G[i] = (A @ M) if p < 0 else G[p] @ M
        return torch.stack(G, 1)

    def keypoints(self, G):
        H = G[:, self.kp_bone]                                          # (N,K,4,4)
        y = H[..., :3, 1] * (self.length[self.kp_bone] * self.kp_end)[None, :, None]
        return H[..., :3, 3] + y

    def skin(self, G, vidx):
        """posed positions of rest vertices vidx (N,M) or (M,) -> (N,M,3)"""
        S = G @ self.rest_inv[None]                                     # (N,B,4,4)
        v = self.verts[vidx]; w = self.W[vidx]
        if vidx.dim() == 1:
            v, w = v[None].expand(G.shape[0], -1, -1), w[None].expand(G.shape[0], -1, -1)
        vh = torch.cat([v, torch.ones_like(v[..., :1])], -1)
        Sv = torch.einsum("nbij,nmj->nmbi", S, vh)[..., :3]             # (N,M,B,3)
        return (w[..., None] * Sv).sum(2)

    def restcol_to_xyz(self, c):
        return (c - 0.5) * self.rc_extent + self.rc_center
