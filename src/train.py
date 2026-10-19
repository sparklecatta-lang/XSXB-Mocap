"""Train a rig-specific keypoint + dense-correspondence head on a frozen DINO backbone.
python src/train.py --project PROJECT [--steps 6000]   (defaults come from the project's config.json "train")
Reads PROJECT/data, writes PROJECT/model/head.pt + eval.json.
Outputs per pixel (stride 3.5 on 448 input -> 128x128):
  K keypoint heatmaps, 1 mask logit, 3 rest-position coords (0..1 in rest bbox).
"""
import argparse, glob, json, math, os, sys, time, random
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qconfig as Q

RES, OUT_RES = 448, 128
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_items(data, held_out=100):
    items, names = [], None
    for f in sorted(glob.glob(f"{data}/labels_*.json")):
        lab = json.load(open(f)); names = lab["kp_names"]
        for r in lab["samples"]:
            for v, view in enumerate(r["views"]):
                items.append({"stem": f"{data}/{r['id']:05d}_v{v}", "kp": np.array(view["kp2d"], np.float32),
                              "deform": r["deform"][0], "id": r["id"]})
    ids = sorted({it["id"] for it in items})
    test_ids = set(ids[-held_out:])
    return [i for i in items if i["id"] not in test_ids], [i for i in items if i["id"] in test_ids], names


def random_background(rng):
    h = w = RES
    kind = rng.random()
    c1, c2 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
    if kind < 0.35:                                     # vertical gradient (wall/floor-ish)
        t = np.linspace(0, 1, h)[:, None, None]
        img = c1 * (1 - t) + c2 * t
        img = np.broadcast_to(img, (h, w, 3)).copy()
    elif kind < 0.65:                                   # floor split + vignette
        y0 = int(rng.uniform(0.4, 0.85) * h)
        img = np.empty((h, w, 3)); img[:y0] = c1; img[y0:] = c2
    else:                                               # smooth noise texture
        n = rng.uniform(0, 255, (int(rng.integers(4, 24)),) * 2 + (3,)).astype(np.float32)
        img = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    img = img + rng.normal(0, rng.uniform(0, 12), img.shape)
    return np.clip(img, 0, 255).astype(np.float32)


class RigSet(Dataset):
    def __init__(self, items, train=True):
        self.items, self.train = items, train

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        rng = np.random.default_rng(None if self.train else i)
        rgba = cv2.imread(it["stem"] + "_rgb.png", cv2.IMREAD_UNCHANGED)[..., [2, 1, 0, 3]].astype(np.float32)
        dense = cv2.imread(it["stem"] + "_dense.png", cv2.IMREAD_UNCHANGED)
        dmask = (dense[..., 3] > 0).astype(np.float32)
        dcoord = dense[..., [2, 1, 0]].astype(np.float32) / 65535.0
        kp = it["kp"].copy()
        if self.train:                                  # random similarity transform
            s = rng.uniform(0.8, 1.15); a = rng.uniform(-15, 15); t = rng.uniform(-30, 30, 2)
            M = cv2.getRotationMatrix2D((RES / 2, RES / 2), a, s); M[:, 2] += t
            rgba = cv2.warpAffine(rgba, M, (RES, RES), flags=cv2.INTER_LINEAR, borderValue=0)
            dcoord = cv2.warpAffine(dcoord, M, (RES, RES), flags=cv2.INTER_NEAREST)
            dmask = cv2.warpAffine(dmask, M, (RES, RES), flags=cv2.INTER_NEAREST)
            kp = kp @ M[:, :2].T + M[:, 2]
        alpha = rgba[..., 3:4] / 255.0
        bg = random_background(rng) if self.train else np.full((RES, RES, 3), 180, np.float32)
        img = rgba[..., :3] * alpha + bg * (1 - alpha)
        if self.train:
            img = img * rng.uniform(0.6, 1.4) + rng.uniform(-30, 30)                 # brightness / contrast
            g = img.mean(2, keepdims=True); img = g + (img - g) * rng.uniform(0.5, 1.4)   # saturation
            img = img * rng.uniform(0.85, 1.15, 3)                                      # white balance
            if rng.random() < 0.4:                                                      # motion blur
                k = int(rng.integers(3, 15)); ker = np.zeros((k, k), np.float32); ker[k // 2] = 1
                ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), rng.uniform(0, 180), 1), (k, k))
                img = cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))
            if rng.random() < 0.3:
                img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 2.0))
            img = img + rng.normal(0, rng.uniform(0, 8), img.shape)
            if rng.random() < 0.5:                                                      # jpeg
                ok, enc = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 90))])
                img = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED).astype(np.float32)
        img = (np.clip(img, 0, 255) / 255.0 - MEAN) / STD
        # targets at OUT_RES
        sc = OUT_RES / RES
        m_small = cv2.resize(dmask, (OUT_RES, OUT_RES), interpolation=cv2.INTER_AREA)
        c_small = cv2.resize(dcoord, (OUT_RES, OUT_RES), interpolation=cv2.INTER_NEAREST)
        inner = cv2.erode((m_small > 0.99).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        f32 = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        return (f32(img.transpose(2, 0, 1)), f32(kp * sc), f32(m_small), f32(c_small.transpose(2, 0, 1)), f32(inner))


class Head(nn.Module):
    def __init__(self, cin, K, mid=256):
        super().__init__()
        def blk(a, b): return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(16, b), nn.GELU())
        self.proj = blk(cin, mid)
        self.up1 = blk(mid, mid); self.up2 = blk(mid, 128)
        self.rgb = nn.Sequential(blk(3, 32), blk(32, 64))                     # sharp low-level cue
        self.fuse = blk(128 + 64, 128)
        self.out = nn.Conv2d(128, K + 4, 1)

    def forward(self, feat, img):
        x = self.proj(feat)
        x = self.up1(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        x = self.up2(F.interpolate(x, size=(OUT_RES, OUT_RES), mode="bilinear", align_corners=False))
        r = self.rgb(F.interpolate(img, size=(OUT_RES, OUT_RES), mode="bilinear", align_corners=False))
        return self.out(self.fuse(torch.cat([x, r], 1)))


class Backbone(nn.Module):
    """Frozen DINO (v2 or v3 via transformers); returns concat of last N layers' patch tokens as a grid."""
    def __init__(self, path, layers=(-1, -4)):
        super().__init__()
        path = Q.tool_path(path)                                               # relative to the tool root
        self.m = AutoModel.from_pretrained(path).eval()
        for p in self.m.parameters(): p.requires_grad_(False)
        self.layers = layers
        self.patch = self.m.config.patch_size
        self.dim = self.m.config.hidden_size * len(layers)

    @torch.no_grad()
    def forward(self, x):
        hs = self.m(pixel_values=x, output_hidden_states=True).hidden_states
        g = x.shape[-1] // self.patch
        outs = []
        for l in self.layers:
            t = hs[l][:, -g * g:]                                              # drop CLS / register tokens
            outs.append(t.transpose(1, 2).reshape(x.shape[0], -1, g, g))
        return torch.cat(outs, 1).float()


def heatmaps(kp, K, sigma=2.0):
    ys, xs = torch.meshgrid(torch.arange(OUT_RES, device=kp.device), torch.arange(OUT_RES, device=kp.device), indexing="ij")
    d2 = (xs[None, None] - kp[..., 0, None, None]) ** 2 + (ys[None, None] - kp[..., 1, None, None]) ** 2
    return torch.exp(-d2 / (2 * sigma ** 2))


def spatial_softmax(logits):
    B, K, H, W = logits.shape
    return F.softmax(logits.flatten(2), -1).reshape(B, K, H, W)


def decode_kp(hm):
    B, K, H, W = hm.shape
    flat = hm.reshape(B, K, -1)
    idx = flat.argmax(-1)
    y, x = (idx // W).float(), (idx % W).float()
    # soft refinement in a 5x5 window
    out = torch.stack([x, y], -1)
    pad = F.pad(hm, (2, 2, 2, 2))
    for b in range(B):
        for k in range(K):
            xi, yi = int(x[b, k]), int(y[b, k])
            w = pad[b, k, yi:yi + 5, xi:xi + 5].clamp(min=0)
            if w.sum() > 0:
                gy, gx = torch.meshgrid(torch.arange(-2, 3, device=hm.device), torch.arange(-2, 3, device=hm.device), indexing="ij")
                out[b, k, 0] += (w * gx).sum() / w.sum(); out[b, k, 1] += (w * gy).sum() / w.sum()
    return out, flat.max(-1).values


def evaluate(bb, head, loader, names, dev):
    head.eval(); errs, cerr = [], []
    with torch.no_grad():
        for img, kp, m, c, inner in loader:
            img = img.to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                o = head(bb(img), img).float()
            p, _ = decode_kp(spatial_softmax(o[:, :len(names)]))
            errs.append((p.cpu() - kp).norm(dim=-1) * (RES / OUT_RES))
            e = (o[:, -3:].sigmoid().cpu() - c).abs().sum(1)
            cerr.append((e * inner).sum((1, 2)) / inner.sum((1, 2)).clamp(min=1))
    head.train()
    E = torch.cat(errs); C = torch.cat(cerr)
    return E, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--steps", type=int); ap.add_argument("--bs", type=int); ap.add_argument("--lr", type=float)
    ap.add_argument("--max_deform", type=float); ap.add_argument("--backbone")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    tc = Q.load(a.project)["train"]; P = Q.paths(a.project)
    for k, ck in (("steps", "steps"), ("bs", "batch"), ("lr", "lr"), ("max_deform", "max_deform"), ("backbone", "backbone")):
        if getattr(a, k) is None: setattr(a, k, tc[ck])
    a.data, a.out = P["data"], P["model_dir"]
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    tr, te, names = load_items(a.data, min(int(tc["held_out"]), max(1, len(glob.glob(f"{a.data}/*_v0_rgb.png")) // 10)))
    tr = [i for i in tr if i["deform"] <= a.max_deform]
    K = len(names)
    print(f"train {len(tr)} test {len(te)} K {K}", flush=True)
    if not tr: raise SystemExit("ERROR no training images (render data first, or raise train.max_deform)")
    a.bs = min(a.bs, len(tr))
    print(f"PROGRESS 0/{a.steps}", flush=True)
    dl = DataLoader(RigSet(tr, True), batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0)
    tl = DataLoader(RigSet(te, False), batch_size=a.bs, shuffle=False, num_workers=2)
    bb = Backbone(a.backbone).to(dev).half()
    head = Head(bb.dim, K).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    step, t0 = 0, time.time()
    while step < a.steps:
        for img, kp, m, c, inner in dl:
            img, kp, m, c, inner = (x.to(dev, non_blocking=True) for x in (img, kp, m, c, inner))
            with torch.autocast("cuda", dtype=torch.float16):
                feat = bb(img.half())
                o = head(feat, img).float()
            hm_t = heatmaps(kp, K)
            t = hm_t.flatten(2); t = t / t.sum(-1, keepdim=True).clamp(min=1e-6)       # spatial distribution target
            l_hm = -(t * F.log_softmax(o[:, :K].flatten(2), -1)).sum(-1).mean() * 0.5
            l_m = F.binary_cross_entropy_with_logits(o[:, K], m)
            l_c = ((o[:, K + 1:].sigmoid() - c).abs().sum(1) * inner).sum() / inner.sum().clamp(min=1)
            loss = l_hm + l_m + l_c * 5
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            step += 1
            if step % max(1, min(50, a.steps // 100)) == 0 or step == a.steps:
                print(f"PROGRESS {step}/{a.steps}", flush=True)
            if step % 200 == 0:
                print(f"step {step} loss {loss.item():.4f} hm {l_hm.item():.4f} mask {l_m.item():.4f} "
                      f"coord {l_c.item():.4f} {time.time() - t0:.0f}s", flush=True)
            if step >= a.steps: break
    torch.save({"head": head.state_dict(), "backbone": a.backbone, "names": names, "dim": bb.dim}, f"{a.out}/head.pt")
    E, C = evaluate(bb, head, tl, names, dev)
    rep = {"px_mean": float(E.mean()), "px_median": float(E.median()), "pck_10px": float((E < 10).float().mean()),
           "pck_20px": float((E < 20).float().mean()), "coord_l1_mean": float(C.mean()),
           "per_joint_px": {n: float(E[:, k].mean()) for k, n in enumerate(names)}}
    json.dump(rep, open(f"{a.out}/eval.json", "w"), indent=1)
    print("EVAL", json.dumps({k: v for k, v in rep.items() if k != "per_joint_px"}))
    worst = sorted(rep["per_joint_px"].items(), key=lambda x: -x[1])[:8]
    print("WORST", worst)


if __name__ == "__main__":
    main()
