"""Synthetic ground-truth benchmark: our fit vs. the original triangulation workflow.
python src/synth_benchmark.py --project PROJECT --blender PATH/blender.exe [--frames 120 --stride 2]
       [--blur K]  (fast-motion test: each video frame = mean of K renders spread over the frame interval)
       [--name benchmark] [--ckpt head.pt]
1. take the rig's reference poses (rig/ref_poses.npz) as ground truth,
2. render them with the start-frame cameras and stack the panels into a Seedance-sized two-panel video,
3. detect -> fit with --mode ours and --mode triangulate,
4. report 3D keypoint error (eval_synth.py). Output: PROJECT/benchmark/
"""
import argparse, json, os, subprocess, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qconfig as Q
from videoio import VideoOut

SRC = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True); ap.add_argument("--blender", required=True)
    ap.add_argument("--frames", type=int, default=120); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--panel", default="834x556", help="panel size of the synthetic video")
    ap.add_argument("--blur", type=int, default=1, help="sub-frames averaged per video frame (motion blur)")
    ap.add_argument("--name", default="benchmark", help="output folder inside the project")
    ap.add_argument("--ckpt", default=None, help="detector checkpoint (default: the project's model)")
    a = ap.parse_args()
    d = os.path.abspath(a.project); P = Q.paths(d); cfg = Q.load(d)
    out = os.path.join(d, a.name); gt_dir = os.path.join(out, "gt"); os.makedirs(gt_dir, exist_ok=True)
    rig = np.load(P["rig"]); names = [str(n) for n in rig["names"]]
    rp = np.load(P["poses"]); rn = [str(n) for n in rp["names"]]
    E = np.zeros((len(rp["euler_xzy"]), len(names), 3), np.float32)
    for j, n in enumerate(rn):
        if n in names: E[:, names.index(n)] = rp["euler_xzy"][:, j]
    for i, n in enumerate(names):                      # helpers follow their source, fixed bones stay at rest
        h = Q.helper_source(cfg, n, names)
        if h: E[:, i] = h[1] * E[:, names.index(h[0])]
        elif Q.bone_limits_deg(cfg, n, names) is None: E[:, i] = 0
    K = max(1, a.blur)
    h0 = a.stride / 2 if K > 1 else 0               # K=1 keeps the original frame indices
    centers = np.arange(h0, len(E) - 1 - h0, a.stride)[:a.frames]
    T = len(centers)
    # sub-frame times spread over one frame interval around each center (K=1: the center itself)
    tq = (centers[:, None] + (np.arange(K) - (K - 1) / 2)[None] * a.stride / K).ravel()
    i0 = np.floor(tq).astype(int); w0 = (tq - i0)[:, None, None]
    Es = ((1 - w0) * E[i0] + w0 * E[np.minimum(i0 + 1, len(E) - 1)]).astype(np.float32)
    Eg = Es.reshape(T, K, *E.shape[1:])[:, (K - 1) // 2] if K % 2 else Es.reshape(T, K, *E.shape[1:]).mean(1)
    loc = np.array(json.load(open(P["cams"]))["armature_location"], np.float32)
    save = lambda path, X: np.savez(path, euler=X, obj_rot=np.zeros((len(X), 3), np.float32),
                                    obj_loc=np.repeat(loc[None], len(X), 0), names=np.array(names))
    save(os.path.join(gt_dir, "fit.npz"), Eg)
    rd = os.path.join(out, "sub"); os.makedirs(rd, exist_ok=True); save(os.path.join(rd, "fit.npz"), Es)
    pw, ph = map(int, a.panel.split("x"))
    run([a.blender, "-b", P["clean"], "--python", os.path.join(SRC, "bl_apply.py"), "--", d, rd, "render", str(pw), str(ph)])
    video = os.path.join(out, "synth.mp4")
    w = VideoOut(video, cfg["video"]["fps"], (pw, 2 * ph))
    for t in range(T):
        pan = []
        for k in ("front", "side"):
            ims = [cv2.imread(os.path.join(rd, "render", f"{k}_{t * K + j:05d}.jpg")).astype(np.float32) for j in range(K)]
            pan.append(np.mean(ims, 0).round().astype(np.uint8))
        w.write(np.vstack(pan))
    w.release()
    py = sys.executable
    det = [py, os.path.join(SRC, "predict.py"), "--project", d, "--src", video, "--out", os.path.join(out, "det"), "--no_png"]
    run(det + (["--ckpt", a.ckpt] if a.ckpt else []))
    for mode in ("ours", "triangulate"):
        run([py, os.path.join(SRC, "fit.py"), "--project", d, "--pred", os.path.join(out, "det", "pred.npz"),
             "--out", os.path.join(out, "fit_" + mode), "--mode", mode])
    run([py, os.path.join(SRC, "eval_synth.py"), "--project", d, "--gt", os.path.join(gt_dir, "fit.npz"),
         "--fit", "ours=" + os.path.join(out, "fit_ours", "fit.npz"),
         "--fit", "triangulate=" + os.path.join(out, "fit_triangulate", "fit.npz"), "--out", os.path.join(out, "results.json")])


if __name__ == "__main__":
    main()
