"""Run the trained head on a two-panel video (or images) and draw keypoints + dense map.
python src/predict.py --project PROJECT --src VIDEO --out OUTDIR [--ckpt head.pt]
       [--crops tb | "x0,y0,x1,y1;..."]   (tb = top/bottom halves; default from config "video.layout")
       [--every N] [--max N]
Saves pred.npz (kp[x,y] in source pixels, kp score, dense map, per frame/view), overlay.mp4 and PNGs.
"""
import argparse, glob, os, re, sys, numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Backbone, Head, decode_kp, spatial_softmax, RES, OUT_RES, MEAN, STD
import qconfig as Q
from videoio import VideoOut

COL = {"l": (40, 140, 255), "r": (255, 140, 40)}       # BGR: left orange, right blue


def frames(src, every, maxn):
    if os.path.isfile(src) and src.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
        cap = cv2.VideoCapture(src); i = 0
        while True:
            ok, f = cap.read()
            if not ok: break
            if i % every == 0: yield i, f
            i += 1
            if maxn and i >= maxn * every: break
    else:
        for i, p in enumerate(sorted(glob.glob(src))[::every][:maxn or None]):
            yield i, cv2.imread(p)


def to_square(img):
    h, w = img.shape[:2]; s = max(h, w)
    pad = np.full((s, s, 3), 180, np.uint8); oy, ox = (s - h) // 2, (s - w) // 2
    pad[oy:oy + h, ox:ox + w] = img
    return cv2.resize(pad, (RES, RES), interpolation=cv2.INTER_AREA), s / RES, ox, oy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True); ap.add_argument("--ckpt", default="")
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--crops", default=None); ap.add_argument("--every", type=int, default=1); ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--no_png", action="store_true"); ap.add_argument("--fps", type=float, default=0)
    a = ap.parse_args()
    cfg = Q.load(a.project)
    a.ckpt = a.ckpt or Q.paths(a.project)["model"]
    if a.crops is None: a.crops = cfg["video"]["layout"]
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu")
    names = ck["names"]; K = len(names); ni = {n: i for i, n in enumerate(names)}
    BONES = [(u, v) for u, v in cfg["skeleton"] if u in ni and v in ni]
    n_total = 0
    if os.path.isfile(a.src):
        cap = cv2.VideoCapture(a.src); n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        a.fps = a.fps or cap.get(cv2.CAP_PROP_FPS) or 24; cap.release()
    a.fps = a.fps or 24
    n_total = (n_total + a.every - 1) // a.every
    if a.max: n_total = min(n_total, a.max)
    bb = Backbone(ck["backbone"]).cuda().half()
    head = Head(ck["dim"], K).cuda().eval(); head.load_state_dict(ck["head"])
    rec_kp, rec_sc, rec_f, rec_dense, rec_tf = [], [], [], [], []
    writer = None
    for fi, frame in frames(a.src, a.every, a.max):
        H, W = frame.shape[:2]
        if a.crops == "tb":                                   # Seedance layout: top = front, bottom = side
            crops = [(0, 0, W, H // 2), (0, H // 2, W, H)]
        elif a.crops:
            crops = [tuple(map(int, c.split(","))) for c in a.crops.split(";")]
        else:
            crops = [(0, 0, W, H)]
        kps, scs, tiles, dens, tfs = [], [], [], [], []
        for (x0, y0, x1, y1) in crops:
            sub = frame[y0:y1, x0:x1]
            sq, scale, ox, oy = to_square(sub)
            x = (sq[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD
            t = torch.from_numpy(x.transpose(2, 0, 1).copy())[None].cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                o = head(bb(t.half()), t).float()
            hm = spatial_softmax(o[:, :K])
            p, sc = decode_kp(hm)
            p = p[0].cpu().numpy() * (RES / OUT_RES)                  # in 448 square
            src_xy = np.c_[p[:, 0] * scale - ox + x0, p[:, 1] * scale - oy + y0]
            kps.append(src_xy); scs.append(sc[0].cpu().numpy())
            mask = o[0, K].sigmoid().cpu().numpy()
            coord = o[0, K + 1:].sigmoid().cpu().numpy().transpose(1, 2, 0)
            dens.append(np.concatenate([mask[None], coord.transpose(2, 0, 1)]).astype(np.float16))
            tfs.append([x0, y0, scale, ox, oy])                       # 448-square px -> source px
            if not a.no_png:
                dvis = (coord[..., ::-1] * 255 * (mask[..., None] > 0.5)).astype(np.uint8)
                dvis = cv2.resize(dvis, (RES, RES), interpolation=cv2.INTER_NEAREST)
                vis = sq.copy()
                for u, v in BONES:
                    c = COL.get(Q.kp_side(v), (230, 230, 230))
                    cv2.line(vis, tuple(map(int, p[ni[u]])), tuple(map(int, p[ni[v]])), c, 2, cv2.LINE_AA)
                for k, n in enumerate(names):
                    c = COL.get(Q.kp_side(n), (255, 255, 255))
                    r = 3 if scs[-1][k] > 0.05 else 2
                    cv2.circle(vis, tuple(map(int, p[k])), r, c, -1, cv2.LINE_AA)
                tiles.append(np.hstack([vis, cv2.addWeighted(sq, 0.35, dvis, 0.65, 0)]))
        rec_kp.append(kps); rec_sc.append(scs); rec_f.append(fi); rec_dense.append(dens); rec_tf.append(tfs)
        if len(rec_f) % 5 == 0 or len(rec_f) == n_total:
            print(f"PROGRESS {len(rec_f)}/{n_total}", flush=True)
        if not a.no_png:
            sheet = np.vstack(tiles)
            cv2.imwrite(f"{a.out}/f{fi:05d}.png", sheet)
            if writer is None:
                writer = VideoOut(f"{a.out}/overlay.mp4", a.fps / a.every, (sheet.shape[1], sheet.shape[0]))
            writer.write(sheet)
    if writer is not None: writer.release()
    np.savez(f"{a.out}/pred.npz", kp=np.array(rec_kp, np.float32), score=np.array(rec_sc, np.float32),
             frames=np.array(rec_f), names=np.array(names), dense=np.array(rec_dense),
             tf=np.array(rec_tf, np.float32), src_size=np.array([W, H]))
    print("frames", len(rec_f), "->", a.out)
    print("RESULT", os.path.join(a.out, "overlay.mp4"))


if __name__ == "__main__":
    main()
