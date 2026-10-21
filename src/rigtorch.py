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


def rot_between_t(a, b):
    """batched minimal rotation taking unit vectors a to b (N,3) -> (N,3,3)"""
    a = a / a.norm(dim=-1, keepdim=True); b = b / b.norm(dim=-1, keepdim=True)
    v = torch.cross(a, b, dim=-1); c = (a * b).sum(-1)
    K = torch.zeros(a.shape[0], 3, 3, device=a.device, dtype=a.dtype)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -v[:, 2], v[:, 1], v[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -v[:, 0], -v[:, 1], v[:, 0]
    R = torch.eye(3, device=a.device, dtype=a.dtype) + K + K @ K / (1 + c).clamp(min=1e-6)[:, None, None]
    anti = c < -0.9999                                              # opposite vectors: half turn about any perpendicular axis
    if anti.any():
        ex = torch.tensor([1.0, 0, 0], device=a.device, dtype=a.dtype).expand_as(a)
        ey = torch.tensor([0, 1.0, 0], device=a.device, dtype=a.dtype).expand_as(a)
        n = torch.cross(a, torch.where((a * ex).abs().sum(-1, keepdim=True) < 0.9, ex, ey), dim=-1)
        n = n / n.norm(dim=-1, keepdim=True)
        Rpi = 2 * n[:, :, None] * n[:, None, :] - torch.eye(3, device=a.device, dtype=a.dtype)
        R = torch.where(anti[:, None, None], Rpi, R)
    return R


def rotvec_to_matrix(rv):
    th = rv.norm(dim=-1, keepdim=True).clamp(min=1e-8); k = rv / th
    K = torch.zeros(rv.shape[0], 3, 3, device=rv.device, dtype=rv.dtype)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    th = th[:, :, None]
    return torch.eye(3, device=rv.device, dtype=rv.dtype) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


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
        self.pole = None
        if cfg.get("pole"):
            pc = cfg["pole"]
            self.pole = {"root": self.idx[pc["root"]], "hands": [self.idx[h] for h in pc["hands"]],
                         "g": f(np.array([np.append(g, 1.0) for g in pc["grips"]])),      # (2,4)
                         "B0": f(pc["bottom"]), "u0": f(pc["axis"]), "param_c": float(pc["param_c"]),
                         "kp": [self.kp_names.index(k) for k in pc.get("keypoints", []) if k in self.kp_names]}

    def full_euler(self, e):
        """Copy half-angle helper bones from their source bones. e (...,B,3)"""
        e = e.clone()
        for i, s, fac in self.mid_src:
            e[..., i, :] = fac * e[..., s, :]
        return e

    def obj_matrix(self, obj_rot, obj_loc):
        A = torch.zeros(obj_rot.shape[0], 4, 4, device=self.dev, dtype=self.dtype)
        A[:, :3, :3] = euler_xyz(obj_rot); A[:, :3, 3] = obj_loc; A[:, 3, 3] = 1
        return A

    def fk(self, e, obj_rot, obj_loc, pole_A=None):
        """e (N,B,3) XZY eulers, obj_rot (N,3) XYZ euler, obj_loc (N,3) -> world bone matrices (N,B,4,4)
        pole_A (N,4,4): armature-space transform of the free pole body (pose matrix of its root = pole_A @ rest)"""
        N, B = e.shape[:2]
        e = self.full_euler(e)
        basis = torch.zeros(N, B, 4, 4, device=self.dev, dtype=self.dtype)
        basis[..., :3, :3] = euler_xzy(e); basis[..., 3, 3] = 1
        A = self.obj_matrix(obj_rot, obj_loc)
        G = [None] * B
        for i, p in enumerate(self.parents):
            if pole_A is not None and self.pole and i == self.pole["root"]:
                G[i] = A @ pole_A @ self.rest[i]
                continue
            M = self.local[i] @ basis[:, i]
            G[i] = (A @ M) if p < 0 else G[p] @ M
        return torch.stack(G, 1)

    # ---- hand-held pole (see pole.py)
    def pole_matrix(self, R, bottom):
        """armature-space transform: rest pole -> rotation R (N,3,3), bottom point (N,3)"""
        A = torch.zeros(R.shape[0], 4, 4, device=self.dev, dtype=self.dtype)
        A[:, :3, :3] = R; A[:, :3, 3] = bottom - (R @ self.pole["B0"][:, None])[..., 0]; A[:, 3, 3] = 1
        return A

    def pole_grips(self, G, A):
        """armature-space grip points of the two hands, (N,2,3); G world bone matrices, A object matrix"""
        Ai = torch.linalg.inv(A)
        return torch.stack([(Ai @ G[:, h] @ self.pole["g"][k])[..., :3] for k, h in enumerate(self.pole["hands"])], 1)

    def pole_rule(self, G, A, sign):
        """pole placed by the hands: tip along (R grip - L grip) * sign, grip midpoint at param_c from the bottom. sign (N,)"""
        g = self.pole_grips(G, A)
        v = g[:, 1] - g[:, 0]
        t = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-4) * sign[:, None]
        return self.pole_matrix(rot_between_t(self.pole["u0"].expand_as(t), t), (g[:, 0] + g[:, 1]) / 2 - self.pole["param_c"] * t)

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
