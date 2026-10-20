"""Fit the rig to two-view detections (keypoints + dense surface correspondences).
python src/fit.py --project PROJECT --pred VIDEO_DIR/det/pred.npz --out VIDEO_DIR/fit [--mode ours|triangulate]
Optimises all frames jointly: per-frame bone eulers (XZY, Blender basis), object rotation/location,
plus a smooth per-frame scale + vertical shift for the front panel (video-model drift there).
"""
import argparse, json, os, sys, time, numpy as np, torch, cv2
from scipy.spatial import cKDTree
from scipy.signal import butter, filtfilt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rigtorch import Rig
from train import RES, OUT_RES
import qconfig as Q


def rho(d, delta):                     # pseudo-Huber on distances (px)
    return delta * (torch.sqrt(1 + (d / delta) ** 2) - 1)


def second_diff(x):
    return x[2:] - 2 * x[1:-1] + x[:-2]


def accel_cost(x, delta):
    """Cost of per-bone angular accelerations x (..., 3), rad/frame^2. delta 0: squared norm (plain smoothing).
    delta > 0: Charbonnier, squared below delta and linear above it, so fast real motion (galloping legs) costs little
    while small jitter is still removed."""
    n2 = x.pow(2).sum(-1)
    return n2 if delta <= 0 else 2 * delta ** 2 * (torch.sqrt(1 + n2 / delta ** 2) - 1)


def dejitter(x, fps, fc, fast):
    """Zero-phase low-pass (2nd-order Butterworth, cutoff fc Hz) of curves x (T, ...) along time, faded out where the
    low-passed curve moves faster than `fast` (per frame), so frame-to-frame jitter goes but fast motion is kept.
    Last axis of a (T, B, 3) array = one rotation, judged by its overall speed."""
    if fc <= 0 or fc >= fps / 2 or len(x) < 8:
        return x
    b, a = butter(2, fc / (fps / 2))
    lp = filtfilt(b, a, x, axis=0, padtype="odd", padlen=min(len(x) - 1, 15))
    g = np.gradient(lp, axis=0)
    v = np.linalg.norm(g, axis=-1, keepdims=True) if x.ndim == 3 else np.abs(g)
    return x + (lp - x) / (1 + (v / fast) ** 2)


def dense_targets(pred, rig, per_view=256, max_dist=0.04, seed=0):
    """-> lists per (frame, view): vertex ids (T,V,M), target px (T,V,M,2), valid (T,V,M)"""
    rng = np.random.default_rng(seed)
    tree = cKDTree(rig.verts.cpu().numpy())
    D, TF = pred["dense"].astype(np.float32), pred["tf"]
    T, V = D.shape[:2]
    vid = np.zeros((T, V, per_view), np.int64); px = np.zeros((T, V, per_view, 2), np.float32)
    ok = np.zeros((T, V, per_view), np.float32)
    cell = RES / OUT_RES
    for t in range(T):
        for v in range(V):
            m = D[t, v, 0] > 0.6
            m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            ys, xs = np.nonzero(m)
            if len(ys) == 0: continue
            sel = rng.choice(len(ys), min(per_view, len(ys)), replace=False)
            ys, xs = ys[sel], xs[sel]
            c = D[t, v, 1:, ys, xs].T if D[t, v, 1:, ys, xs].shape[0] == 3 else D[t, v, 1:, ys, xs]
            c = np.asarray(c).reshape(-1, 3)
            xyz = (c - 0.5) * rig.rc_extent + rig.rc_center.cpu().numpy()
            dist, idx = tree.query(xyz)
            x0, y0, scale, ox, oy = TF[t, v]
            sx = (xs + 0.5) * cell * scale - ox + x0
            sy = (ys + 0.5) * cell * scale - oy + y0
            n = len(sel)
            vid[t, v, :n] = idx; px[t, v, :n, 0] = sx; px[t, v, :n, 1] = sy
            ok[t, v, :n] = (dist < max_dist)
    return vid, px, ok


def triangulate(P, kp, w):
    """Linear (DLT) triangulation of every keypoint from the two panels. P (V,3,4), kp (T,V,K,2), w (T,V,K)."""
    T, V, K, _ = kp.shape
    rows = []
    for v in range(V):
        Pv = P[v]
        u, vv = kp[:, v, :, 0:1], kp[:, v, :, 1:2]                     # (T,K,1)
        r1 = u * Pv[2] - Pv[0]; r2 = vv * Pv[2] - Pv[1]                 # (T,K,4)
        ww = w[:, v, :, None]
        rows += [r1 * ww, r2 * ww]
    A = torch.stack(rows, 2)                                            # (T,K,2V,4)
    M, b = A[..., :3], -A[..., 3]
    X = torch.linalg.lstsq(M, b[..., None]).solution[..., 0]           # (T,K,3)
    return X, w.min(1).values                                          # trust = weaker view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True); ap.add_argument("--pred", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cams", default=""); ap.add_argument("--iters", type=int); ap.add_argument("--dense_iters", type=int)
    ap.add_argument("--w_dense", type=float); ap.add_argument("--w_smooth", type=float)
    ap.add_argument("--smooth_delta", type=float, help="rad/frame^2; 0 = quadratic smoothing")
    ap.add_argument("--post_filter_hz", type=float, help="de-jitter low-pass cutoff after the fit; 0 = off")
    ap.add_argument("--mode", default="ours", choices=["ours", "triangulate"],
                    help="triangulate = original video workflow: keypoints only, triangulate 3D, then fit bones to 3D")
    a = ap.parse_args()
    cfg = Q.load(a.project); fc = cfg["fit"]
    for k in ("iters", "dense_iters", "w_dense", "w_smooth", "smooth_delta", "post_filter_hz"):
        if getattr(a, k) is None: setattr(a, k, fc[k])
    a.cams = a.cams or Q.paths(a.project)["cams"]
    drift = bool(fc["front_drift"]) and a.mode != "triangulate"
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    rig = Rig(a.project, dev)
    pred = dict(np.load(a.pred))
    cams = json.load(open(a.cams))
    W_src = float(pred["src_size"][0])
    s = W_src / cams["image_size"][0]
    P = torch.tensor([np.diag([s, s, 1.0]) @ np.array(cams["panels"][k]["P_pixels"]) for k in ("front", "side")],
                     device=dev, dtype=torch.float32)                    # (V,3,4) world -> source px
    T = pred["kp"].shape[0]
    kp_det = torch.tensor(pred["kp"], device=dev)                       # (T,V,K,2)
    score = torch.tensor(pred["score"], device=dev)
    sw = (score / score.median()).clamp(max=1.0)
    print(f"T={T} score median {score.median():.3f} p10 {score.quantile(0.1):.3f}", flush=True)
    if a.mode == "triangulate":
        a.dense_iters = 0
        X_tri, w_tri = triangulate(P, kp_det, sw)
    vid, px, ok = dense_targets(pred, rig)
    vid, px, ok = (torch.tensor(x, device=dev) for x in (vid, px, ok))
    foot = torch.tensor([rig.kp_names.index(n) for n in cfg["foot_keypoints"] if n in rig.kp_names], device=dev, dtype=torch.long)
    front_c = torch.tensor([W_src / 2, float(pred["src_size"][1]) / 4], device=dev)

    B = len(rig.names)
    e = torch.zeros(T, B, 3, device=dev, requires_grad=True)
    rot = torch.zeros(T, 3, device=dev, requires_grad=True)
    loc0 = torch.tensor(cams["armature_location"], device=dev, dtype=torch.float32)
    loc = loc0.repeat(T, 1).clone().requires_grad_(True)
    la = torch.zeros(T, device=dev, requires_grad=True)                # front panel log-scale
    dy = torch.zeros(T, device=dev, requires_grad=True)                # front panel vertical shift (px)
    free = rig.free[None, :, None].float()
    opt = torch.optim.Adam([{"params": [e, rot], "lr": 0.02}, {"params": [loc], "lr": 0.01},
                            {"params": [la], "lr": 0.003}, {"params": [dy], "lr": 0.5}])

    def project(X):                                                     # X (T,N,3) -> (T,V,N,2)
        Xh = torch.cat([X, torch.ones_like(X[..., :1])], -1)
        p = torch.einsum("vij,tnj->tvni", P, Xh)
        p = p[..., :2] / p[..., 2:3]
        f = front_c + la.exp()[:, None, None] * (p[:, 0] - front_c)
        f = f + torch.stack([torch.zeros_like(dy), dy], -1)[:, None]
        return torch.stack([f, p[:, 1]], 1)

    total = a.iters + a.dense_iters
    t0 = time.time()
    for it in range(total):
        use_dense = it >= a.iters
        E = e * free
        G = rig.fk(E, rot, loc)
        K3 = rig.keypoints(G)
        kp2 = project(K3)
        d_kp = (kp2 - kp_det).norm(dim=-1)
        if a.mode == "triangulate":                                     # 3D point loss (metres -> ~px scale)
            d3 = (K3 - X_tri).norm(dim=-1) * 150
            L_kp = (rho(d3, 4.0) * w_tri).sum() / w_tri.sum()
        else:
            L_kp = (rho(d_kp, 4.0) * sw).sum() / sw.sum()
        L = L_kp
        L_dense = torch.zeros((), device=dev)
        if use_dense:
            Xv = rig.skin(G, vid.reshape(T, -1)).reshape(T, 2, -1, 3)   # vertices per view
            pv = torch.stack([project(Xv[:, v])[:, v] for v in range(2)], 1)
            d_dn = (pv - px).norm(dim=-1)
            L_dense = (rho(d_dn, 4.0) * ok).sum() / ok.sum().clamp(min=1)
            L = L + a.w_dense * L_dense
        L_lim = ((rig.lim_lo - E).clamp(min=0) ** 2 + (E - rig.lim_hi).clamp(min=0) ** 2).sum((1, 2)).mean() * 100
        L_reg = (E[..., 1:] ** 2).sum((1, 2)).mean() * 0.05 + (E ** 2).sum((1, 2)).mean() * 0.005
        L_sm = a.w_smooth * (accel_cost(second_diff(E), a.smooth_delta).sum(1).mean()
                             + accel_cost(second_diff(rot), a.smooth_delta).mean()
                             + 10 * second_diff(loc).pow(2).sum(1).mean())
        if not drift:
            la.data.zero_(); dy.data.zero_()
        L_front = 50 * second_diff(la).pow(2).mean() + 0.02 * second_diff(dy).pow(2).mean() + 2 * la.pow(2).mean()
        fz = K3[:, foot, 2]
        L_floor = 200 * (-fz).clamp(min=0).pow(2).sum(1).mean() if len(foot) else torch.zeros((), device=dev)
        L = L + L_lim + L_reg + L_sm + L_front + L_floor
        opt.zero_grad(); L.backward(); opt.step()
        if it % 50 == 0 or it == total - 1:
            print(f"PROGRESS {it + 1}/{total}", flush=True)
        if it % 250 == 0 or it == total - 1:
            print(f"it {it} kp {L_kp.item():.2f} dense {L_dense.item():.2f} lim {L_lim.item():.4f} sm {L_sm.item():.3f} "
                  f"floor {L_floor.item():.4f} kp_px_med {d_kp.median().item():.1f} scale[{la.exp().min().item():.3f},"
                  f"{la.exp().max().item():.3f}] {time.time() - t0:.0f}s", flush=True)
    with torch.no_grad():
        fps, fast = float(cfg["video"]["fps"]), float(fc["post_filter_fast"])
        f = lambda t: torch.tensor(dejitter(t.detach().cpu().double().numpy(), fps, a.post_filter_hz, fast),
                                   device=dev, dtype=torch.float32)
        e, rot, loc = f(e * free), f(rot), f(loc)
        E = rig.full_euler(e * free)
        G = rig.fk(e * free, rot, loc); kp2 = project(rig.keypoints(G))
        res = (kp2 - kp_det).norm(dim=-1)
    np.savez(f"{a.out}/fit.npz", euler=E.cpu().numpy(), obj_rot=rot.detach().cpu().numpy(), obj_loc=loc.detach().cpu().numpy(),
             front_scale=la.exp().detach().cpu().numpy(), front_dy=dy.detach().cpu().numpy(), names=np.array(rig.names),
             kp2d_fit=kp2.cpu().numpy(), kp_res=res.cpu().numpy())
    print("RES px median front %.1f side %.1f | mean front %.1f side %.1f" % (
        res[:, 0].median(), res[:, 1].median(), res[:, 0].mean(), res[:, 1].mean()))
    json.dump({"mode": a.mode, "frames": T, "kp_px_median": {"front": float(res[:, 0].median()), "side": float(res[:, 1].median())},
               "kp_px_mean": {"front": float(res[:, 0].mean()), "side": float(res[:, 1].mean())},
               "per_kp_px_mean": {n: float(res[..., i].mean()) for i, n in enumerate(rig.kp_names)}},
              open(f"{a.out}/fit_report.json", "w"), indent=1)
    worst = res.mean((0, 1)); order = worst.argsort(descending=True)[:6]
    print("WORST", [(rig.kp_names[i], round(worst[i].item(), 1)) for i in order])


if __name__ == "__main__":
    main()
