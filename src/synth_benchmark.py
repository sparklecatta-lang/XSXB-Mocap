"""Synthetic ground-truth benchmark: our fit vs. the original triangulation workflow.
python src/synth_benchmark.py --project PROJECT --blender PATH/blender.exe [--frames 120 --stride 2]
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
    a = ap.parse_args()
    d = os.path.abspath(a.project); P = Q.paths(d); cfg = Q.load(d)
    out = os.path.join(d, "benchmark"); gt_dir = os.path.join(out, "gt"); os.makedirs(gt_dir, exist_ok=True)
    rig = np.load(P["rig"]); names = [str(n) for n in rig["names"]]
    rp = np.load(P["poses"]); rn = [str(n) for n in rp["names"]]
    E = np.zeros((len(rp["euler_xzy"]), len(names), 3), np.float32)
    for j, n in enumerate(rn):
        if n in names: E[:, names.index(n)] = rp["euler_xzy"][:, j]
    E = E[::a.stride][:a.frames]
    for i, n in enumerate(names):                      # helpers follow their source, fixed bones stay at rest
        h = Q.helper_source(cfg, n, names)
        if h: E[:, i] = h[1] * E[:, names.index(h[0])]
        elif Q.bone_limits_deg(cfg, n, names) is None: E[:, i] = 0
    T = len(E)
    loc = np.array(json.load(open(P["cams"]))["armature_location"], np.float32)
    np.savez(os.path.join(gt_dir, "fit.npz"), euler=E, obj_rot=np.zeros((T, 3), np.float32),
             obj_loc=np.repeat(loc[None], T, 0), names=np.array(names))
    pw, ph = map(int, a.panel.split("x"))
    run([a.blender, "-b", P["clean"], "--python", os.path.join(SRC, "bl_apply.py"), "--", d, gt_dir, "render", str(pw), str(ph)])
    video = os.path.join(out, "synth.mp4")
    w = VideoOut(video, cfg["video"]["fps"], (pw, 2 * ph))
    for t in range(T):
        w.write(np.vstack([cv2.imread(os.path.join(gt_dir, "render", f"{k}_{t:05d}.jpg")) for k in ("front", "side")]))
    w.release()
    py = sys.executable
    run([py, os.path.join(SRC, "predict.py"), "--project", d, "--src", video, "--out", os.path.join(out, "det"), "--no_png"])
    for mode in ("ours", "triangulate"):
        run([py, os.path.join(SRC, "fit.py"), "--project", d, "--pred", os.path.join(out, "det", "pred.npz"),
             "--out", os.path.join(out, "fit_" + mode), "--mode", mode])
    run([py, os.path.join(SRC, "eval_synth.py"), "--project", d, "--gt", os.path.join(gt_dir, "fit.npz"),
         "--fit", "ours=" + os.path.join(out, "fit_ours", "fit.npz"),
         "--fit", "triangulate=" + os.path.join(out, "fit_triangulate", "fit.npz"), "--out", os.path.join(out, "results.json")])


if __name__ == "__main__":
    main()
