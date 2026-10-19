"""3D keypoint error of fits against a synthetic ground truth (both as fit.npz-style eulers).
python src/eval_synth.py --project PROJECT --gt gt.npz --fit ours=a/fit.npz --fit triangulate=b/fit.npz [--out report.json]
Error = mean distance (cm) per keypoint after aligning the first keypoint (pelvis) per frame.
Groups: limbs = keypoints starting with l_/r_, tail = keypoints containing 'tail'.
"""
import argparse, json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rigtorch import Rig


def kp3d(rig, f):
    t = lambda k: torch.tensor(np.asarray(f[k]), dtype=torch.float32, device=rig.dev)
    with torch.no_grad():
        return rig.keypoints(rig.fk(t("euler"), t("obj_rot"), t("obj_loc"))).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--fit", action="append", required=True); ap.add_argument("--out", default="")
    a = ap.parse_args()
    rig = Rig(a.project, "cuda" if torch.cuda.is_available() else "cpu")
    G = kp3d(rig, np.load(a.gt)); G = G - G[:, :1]
    names = rig.kp_names
    limbs = [i for i, n in enumerate(names) if n.startswith(("l_", "r_"))]
    tail = [i for i, n in enumerate(names) if "tail" in n]
    rep = {}
    for spec in a.fit:
        key, path = spec.split("=", 1)
        X = kp3d(rig, np.load(path)); X = X - X[:, :1]
        n = min(len(X), len(G))
        d = np.linalg.norm(X[:n] - G[:n], axis=-1) * 100
        rep[key] = {"all_cm": float(d[:, 1:].mean()), "limbs_cm": float(d[:, limbs].mean()),
                    "tail_cm": float(d[:, tail].mean()), "frames": n,
                    "per_kp_cm": {nm: float(d[:, i].mean()) for i, nm in enumerate(names)}}
        print(f"{key:12s} all {rep[key]['all_cm']:.2f} cm  limbs {rep[key]['limbs_cm']:.2f}  tail {rep[key]['tail_cm']:.2f}")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
