"""Local web app: one button per pipeline step, jobs run as subprocesses (Blender headless / the venv Python).
python app/server.py [--port 7860]   then open http://127.0.0.1:7860
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, threading, time, uuid
from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
import qconfig as Q

SETTINGS_FILE = os.path.join(ROOT, "settings.json")
EXAMPLES = os.path.join(ROOT, "examples")


# ---------------------------------------------------------------- settings
def find_blender():
    exe = shutil.which("blender")
    if exe: return exe
    for pat in (r"C:\Program Files\Blender Foundation\Blender*\blender.exe", "/Applications/Blender.app/Contents/MacOS/Blender",
                "/usr/bin/blender", "/snap/bin/blender"):
        hits = sorted(glob.glob(pat))
        if hits: return hits[-1]
    return ""


def load_settings():
    s = {"blender": "", "workspace": os.path.join(ROOT, "workspace")}
    if os.path.exists(SETTINGS_FILE):
        s.update(json.load(open(SETTINGS_FILE, encoding="utf-8")))
    if not s["blender"]: s["blender"] = find_blender()
    os.makedirs(s["workspace"], exist_ok=True)
    return s


S = load_settings()


def ws(*a):
    return os.path.join(S["workspace"], *a)


def project_dir(p):
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", p): raise HTTPException(400, "bad project name")
    d = ws(p)
    if not os.path.isdir(d): raise HTTPException(404, "no such project")
    return d


def video_dir(p, v):
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", v): raise HTTPException(400, "bad video id")
    d = os.path.join(project_dir(p), "videos", v)
    if not os.path.isdir(d): raise HTTPException(404, "no such video")
    return d


# ---------------------------------------------------------------- jobs
class Job:
    """Sequential stages; each stage is a list of commands run in parallel. Progress from 'PROGRESS a/b' or a
    count of lines matching stage['count'] (e.g. 'SAMPLE ')."""
    def __init__(self, project, title, stages, log_path, est=None):
        self.id = uuid.uuid4().hex[:10]; self.project = project; self.title = title
        self.stages = stages; self.stage = 0; self.frac = 0.0
        self.status = "queued"; self.error = ""; self.log = deque(maxlen=4000); self.nlines = 0
        self.procs = []; self.cancelled = False; self.t0 = time.time(); self.t1 = None; self.t_run = None
        self.log_path = log_path; self.est = est          # est = expected wall time in seconds (optional)

    def eta(self):
        """Seconds left: the up-front estimate early on, then more and more the measured rate."""
        if self.status not in ("queued", "running"): return 0
        p = self.progress(); el = time.time() - (self.t_run or time.time())
        r1 = el * (1 - p) / p if p > 0.005 and el > 5 else None
        r0 = self.est * (1 - p) if self.est else None
        if r0 is None: return None if r1 is None else round(r1)
        if r1 is None: return round(r0)
        w = min(1.0, p / 0.15)
        return round(w * r1 + (1 - w) * r0)

    def add(self, line):
        self.log.append(line); self.nlines += 1
        with open(self.log_path, "a", encoding="utf-8") as f: f.write(line + "\n")

    def progress(self):
        w = [s.get("weight", 1.0) for s in self.stages]
        done = sum(w[:self.stage]) + (w[self.stage] * self.frac if self.stage < len(w) else 0)
        return 1.0 if self.status == "done" else done / sum(w)

    def run(self):
        self.status = "running"; self.t_run = time.time()
        env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8")
        try:
            for i, st in enumerate(self.stages):
                self.stage, self.frac = i, 0.0
                self.add(f"=== {st['label']} ===")
                if "func" in st:
                    st["func"](self); continue
                cmds = st["cmds"]; prog = [0.0] * len(cmds); counts = [0] * len(cmds)
                total = st.get("total")

                def reader(k, p):
                    for raw in p.stdout:
                        line = raw.rstrip()
                        m = re.match(r"PROGRESS (\d+)/(\d+)", line)
                        if m:
                            prog[k] = int(m.group(1)) / max(1, int(m.group(2)))
                        elif st.get("count") and line.startswith(st["count"]):
                            counts[k] += 1
                            prog[k] = counts[k] / max(1, st["per_cmd"][k])
                        if not m:
                            self.add(line)
                        self.frac = sum(prog) / len(prog)
                self.procs, threads = [], []
                for k, c in enumerate(cmds):
                    self.add("$ " + " ".join(f'"{x}"' if " " in x else x for x in c))
                    p = subprocess.Popen(c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                                         text=True, encoding="utf-8", errors="replace")
                    self.procs.append(p)
                    t = threading.Thread(target=reader, args=(k, p), daemon=True); t.start(); threads.append(t)
                for p, t in zip(self.procs, threads):
                    p.wait(); t.join()
                if self.cancelled: raise RuntimeError("cancelled")
                bad = [p.returncode for p in self.procs if p.returncode != 0]
                if bad: raise RuntimeError(f"step '{st['label']}' failed (exit {bad[0]}), see log")
                if st.get("check"):
                    st["check"](self)
            self.status = "done"
        except Exception as e:
            self.status = "cancelled" if self.cancelled else "error"; self.error = str(e); self.add("ERROR " + str(e))
        finally:
            self.t1 = time.time()

    def cancel(self):
        self.cancelled = True
        for p in self.procs:
            if p.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
                else:
                    p.kill()

    def info(self, since=0):
        lines = list(self.log)
        first = self.nlines - len(lines)
        new = lines[max(0, since - first):] if since else lines[-300:]
        return {"id": self.id, "project": self.project, "title": self.title, "status": self.status,
                "error": self.error, "progress": round(self.progress(), 4),
                "stage": self.stages[min(self.stage, len(self.stages) - 1)]["label"],
                "stage_index": self.stage, "stages": len(self.stages), "elapsed": round((self.t1 or time.time()) - self.t0),
                "eta": self.eta(), "est": self.est, "started": self.t0, "lines": new, "next": self.nlines}


JOBS = {}
QUEUE = deque()
LOCK = threading.Lock()


def worker():
    while True:
        with LOCK:
            job = QUEUE.popleft() if QUEUE else None
        if job is None:
            time.sleep(0.3); continue
        if not job.cancelled:
            job.run()


threading.Thread(target=worker, daemon=True).start()


def submit(project, title, stages, est=None):
    os.makedirs(ws(project, "logs"), exist_ok=True)
    job = Job(project, title, stages, ws(project, "logs", time.strftime("%Y%m%d-%H%M%S") + f"_{title}.log"), est)
    JOBS[job.id] = job
    with LOCK: QUEUE.append(job)
    return job


def blender(*args):
    if not S["blender"] or not os.path.exists(S["blender"]):
        raise HTTPException(400, "Blender not found: set its path in Settings")
    return [S["blender"], "-b", *args]


def py(script, *args):
    return [sys.executable, os.path.join(SRC, script), *args]


# ---------------------------------------------------------------- presets + start-frame options
# measured on an RTX 5060 Ti: Blender ~0.81 s per sample per worker (2 views + afterimages), training ~0.34 s/step
RATE_RENDER, RATE_TRAIN = 0.81, 0.34
QUALITY = {"fast": {"samples": 600, "steps": 3000}, "std": {"samples": 1500, "steps": 6000},
           "fine": {"samples": 3000, "steps": 12000}}


def est_build(n, w, steps, render=True):
    return (n * RATE_RENDER / max(1, w) if render else 0) + steps * RATE_TRAIN + 40


SF_VIEWS = {"fs": None,                                                        # front over side (default)
            "ft": [{"az_deg": "front"}, {"az_deg": "side", "tilt_deg": 90}],   # front over top
            "st": [{"az_deg": "side"}, {"az_deg": "side", "tilt_deg": 90}]}    # side over top
SF_RATIOS = {"3:4": (3, 4), "4:3": (4, 3), "16:9": (16, 9), "9:16": (9, 16)}


def start_options(cfg):
    """{views, layout, ratio} as the UI shows them, from the config (stored, or derived for older projects)."""
    sf = cfg["start_frame"]
    if isinstance(sf.get("options"), dict): return {**sf["options"], "adjust": sf.get("adjust") or {}}
    v = sf.get("views") or [{}, {}]
    v0, v1 = (v + [{}, {}])[:2]
    views = "st" if (v0 or {}).get("az_deg") == "side" else "ft" if float((v1 or {}).get("tilt_deg") or 0) >= 60 else "fs"
    layout = sf.get("layout") or cfg["video"]["layout"] or "tb"
    W, H = float(sf["panel_width"]), float(sf["panel_height"])
    r = (2 * W / H) if layout == "lr" else W / (2 * H)
    ratio = min(SF_RATIOS, key=lambda k: abs(SF_RATIOS[k][0] / SF_RATIOS[k][1] - r))
    return {"views": views, "layout": layout if layout in ("tb", "lr") else "tb", "ratio": ratio, "adjust": sf.get("adjust") or {}}


def set_adjust(d, adj):
    """Per-panel framing tweaks from the UI: {"front": {"zoom", "dy"}, "side": {...}}."""
    P = Q.paths(d)
    raw = json.load(open(P["config"], encoding="utf-8"))
    out = {}
    for k in ("front", "side"):
        a = (adj or {}).get(k) or {}
        z, dy = float(a.get("zoom", 1.0)), float(a.get("dy", 0.0))
        out[k] = {"zoom": round(min(4.0, max(0.25, z)), 3), "dy": round(min(0.6, max(-0.6, dy)), 3)}
    raw.setdefault("start_frame", {})["adjust"] = out
    Q.save(d, raw)


def apply_start_options(d, o):
    views, layout, ratio = o.get("views", "fs"), o.get("layout", "tb"), o.get("ratio", "3:4")
    if views not in SF_VIEWS or layout not in ("tb", "lr") or ratio not in SF_RATIOS:
        raise HTTPException(400, "views fs|ft|st, layout tb|lr, ratio 3:4|4:3|16:9|9:16")
    P = Q.paths(d)
    raw = json.load(open(P["config"], encoding="utf-8"))
    rw, rh = SF_RATIOS[ratio]
    pw, ph = (rw, rh / 2) if layout == "tb" else (rw / 2, rh)              # one panel of the whole video frame
    k = 1536 / max(pw, ph)
    sf = raw.setdefault("start_frame", {})
    sf.update(panel_width=int(round(pw * k / 16) * 16), panel_height=int(round(ph * k / 16) * 16),
              views=SF_VIEWS[views], layout=layout, options={"views": views, "layout": layout, "ratio": ratio})
    raw.setdefault("video", {})["layout"] = layout
    r = raw.setdefault("render", {})
    r["view0"] = "side" if views == "st" else "front"                     # training cameras match the panels
    r["top_fraction"] = 0.0 if views == "fs" else (float(r.get("top_fraction") or 0) or 0.5)
    Q.save(d, raw)
    return sf["options"]


# ---------------------------------------------------------------- thumbnails (cv2)
def subject_box(img):
    """Bounding box of whatever differs from the border colour (start frames and Seedance videos: plain backdrop)."""
    b = np.concatenate([img[:4].reshape(-1, 3), img[-4:].reshape(-1, 3), img[:, :4].reshape(-1, 3), img[:, -4:].reshape(-1, 3)])
    bg = np.median(b, 0)
    m = (np.abs(img.astype(np.int16) - bg).max(2) > 32).astype(np.uint8)
    m[:8] = m[-8:] = 0; m[:, :8] = m[:, -8:] = 0                     # panel divider / edge pixels
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(m)
    if len(xs) < 50: return None, bg
    x0, x1, y0, y1 = np.percentile(xs, 0.5), np.percentile(xs, 99.5), np.percentile(ys, 0.5), np.percentile(ys, 99.5)
    return (x0, y0, x1, y1), bg


def tight_crop(img, aspect, pad=0.14, out_w=800):
    h, w = img.shape[:2]
    box, bg = subject_box(img)
    if box is None or (box[2] - box[0]) * (box[3] - box[1]) > 0.85 * w * h:
        box = (0, 0, w, h); pad = 0
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
    bw, bh = max(bw, bh * aspect), max(bh, bw / aspect)
    X0, Y0 = int(round(cx - bw / 2)), int(round(cy - bh / 2)); X1, Y1 = X0 + int(round(bw)), Y0 + int(round(bh))
    P = max(0, -X0, -Y0, X1 - w, Y1 - h)
    big = cv2.copyMakeBorder(img, P, P, P, P, cv2.BORDER_CONSTANT, value=[int(c) for c in bg])
    crop = big[Y0 + P:Y1 + P, X0 + P:X1 + P]
    return cv2.resize(crop, (out_w, int(round(out_w / aspect))), interpolation=cv2.INTER_AREA)


def panel_of(img, layout, i):
    h, w = img.shape[:2]
    if layout == "lr": return img[:, :w // 2] if i == 0 else img[:, w // 2:]
    return img[:h // 2] if i == 0 else img[h // 2:]


def cover_panel(cfg):
    """Which panel shows the character best: the side view (2nd panel for 'fs', 1st for 'st'), else the front."""
    return 1 if start_options(cfg)["views"] == "fs" else 0


def cached_jpg(out, src, make):
    if not (os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src)):
        img = make()
        if img is None: raise HTTPException(404, "no image")
        cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return FileResponse(out, headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- status
FIT_VIEWS = ("both", "front", "side")


def fit_dirs(d):
    """{views: fit directory} of the finished-or-started fits of a video; a legacy 'fit' folder counts as both panels."""
    legacy = os.path.join(d, "fit")
    if os.path.isdir(legacy) and not os.path.exists(os.path.join(d, "fit_both")):
        os.rename(legacy, os.path.join(d, "fit_both"))
    return {k: os.path.join(d, "fit_" + k) for k in FIT_VIEWS if os.path.exists(os.path.join(d, "fit_" + k, "fit.npz"))}


def video_status(d):
    vd = lambda *a: os.path.join(d, *a)
    st = {"id": os.path.basename(d), "source": os.path.exists(vd("source.mp4")),
          "detected": os.path.exists(vd("det", "pred.npz")), "overlay": os.path.exists(vd("det", "overlay.mp4"))}
    fits = []
    for k, fd in fit_dirs(d).items():
        f = {"views": k, "mtime": int(os.path.getmtime(os.path.join(fd, "fit.npz"))),
             "compare": os.path.exists(os.path.join(fd, "compare.mp4")), "blend": os.path.exists(os.path.join(fd, "mocap.blend")),
             "fbx": os.path.exists(os.path.join(fd, "mocap.fbx"))}
        if f["compare"]: f["compare_mtime"] = int(os.path.getmtime(os.path.join(fd, "compare.mp4")))
        rp = os.path.join(fd, "fit_report.json")
        if os.path.exists(rp):
            f["report"] = json.load(open(rp)); f["report"].pop("per_kp_px_mean", None)
        fits.append(f)
    fits.sort(key=lambda f: -f["mtime"])
    st["fits"] = fits; st["fitted"] = bool(fits)
    done = next((f for f in fits if f["compare"] and f["fbx"]), None)                  # newest complete fit drives the card
    st["compare"] = bool(done); st["fbx"] = bool(done); st["blend"] = bool(done and done["blend"])
    if done:
        st["compare_mtime"] = done["compare_mtime"]; st["fit_views"] = done["views"]
        if "report" in done: st["report"] = done["report"]
    if st["source"]:
        cap = cv2.VideoCapture(vd("source.mp4"))
        st["size"] = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        st["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); st["fps"] = cap.get(cv2.CAP_PROP_FPS); cap.release()
        st["duration"] = round(st["frames"] / st["fps"], 1) if st["fps"] else 0
    return st


def job_brief(j):
    return {k: v for k, v in j.info(10 ** 9).items() if k != "lines"}


def project_jobs(p):
    """Active job of the project (queued/running) and the latest job per video id, newest first."""
    js = sorted((j for j in JOBS.values() if j.project == p), key=lambda j: -j.t0)
    active = next((j for j in js if j.status in ("queued", "running")), None)
    per_video = {}
    for j in js:
        m = re.match(r"(?:detect_fit|preview|export|process)_(.+)$", j.title)
        if m and m.group(1) not in per_video: per_video[m.group(1)] = j
    return active, per_video, (js[0] if js else None)


def action_count(P):
    if not os.path.exists(P["poses"]): return 0
    try:
        with np.load(P["poses"]) as z: return len(z["actions"]) if "actions" in z.files else 1
    except Exception:
        return 1


def project_status(p):
    d = project_dir(p); P = Q.paths(d)
    st = {"name": p, "source": os.path.exists(P["source"]), "prepped": os.path.exists(P["rig"]) and os.path.exists(P["clean"]),
          "config": os.path.exists(P["config"]), "start_frame": os.path.exists(P["start_png"]) and os.path.exists(P["cams"]),
          "samples": len(glob.glob(os.path.join(P["data"], "*_v0_rgb.png"))), "model": os.path.exists(P["model"]),
          "ref_poses": os.path.exists(P["poses"]),
          "orient_views": os.path.exists(os.path.join(d, "rig", "orient", "az_000.jpg"))}
    if st["config"]:
        try:
            c = Q.load(d); st["keypoints"] = len(c["keypoints"]); st["render_samples"] = c["render"]["samples"]
            st["train_steps"] = c["train"]["steps"]; st["config_name"] = c["name"]
            st["start_options"] = start_options(c); st["quality"] = c["train"].get("quality")
            st["layout"] = c["video"]["layout"]
            st["config_warnings"] = validate_config(d, json.load(open(P["config"], encoding="utf-8")))[:8]
        except Exception as e:
            st["config_error"] = str(e)
    if os.path.exists(P["inspect"]):
        i = json.load(open(P["inspect"], encoding="utf-8"))
        st["rig"] = {"armature": i["armature"], "mesh": i["mesh"], "verts": i["verts"], "bones": len(i["bones"])}
    ev = os.path.join(P["model_dir"], "eval.json")
    if os.path.exists(ev):
        e = json.load(open(ev)); st["eval"] = {k: e[k] for k in ("px_mean", "px_median", "pck_10px", "pck_20px") if k in e}
    vdir = P["videos"]
    st["videos"] = [video_status(v) for v in sorted(glob.glob(os.path.join(vdir, "*"))) if os.path.isdir(v)]
    st["busy"] = any(j.project == p and j.status in ("queued", "running") for j in JOBS.values())
    st["actions"] = action_count(P)
    if st["start_frame"]: st["start_mtime"] = int(os.path.getmtime(P["start_png"]))
    active, per_video, last = project_jobs(p)
    st["job"] = job_brief(active) if active else None
    st["last_job"] = job_brief(last) if last else None
    for v in st["videos"]:
        j = per_video.get(v["id"])
        v["job"] = job_brief(j) if j else None
    return st


# ---------------------------------------------------------------- app
app = FastAPI(title="XSXB MOCAP")


@app.get("/api/settings")
def get_settings():
    return {**S, "python": sys.executable, "examples": sorted(os.listdir(EXAMPLES)) if os.path.isdir(EXAMPLES) else []}


@app.post("/api/settings")
def set_settings(body: dict):
    if "blender" in body: S["blender"] = body["blender"].strip().strip('"')
    json.dump({"blender": S["blender"], "workspace": S["workspace"]}, open(SETTINGS_FILE, "w", encoding="utf-8"), indent=1)
    return get_settings()


@app.get("/api/presets")
def presets():
    w = int(Q.DEFAULTS["render"]["workers"])
    return {k: {**v, "minutes": int(round(est_build(v["samples"], w, v["steps"]) / 60 / 5) * 5)} for k, v in QUALITY.items()}


@app.post("/api/import")
async def import_rig(file: UploadFile = File(...), name: str = Form("")):
    """Drop a rigged model -> new project named after the file, import job started."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".blend", ".fbx", ".glb", ".gltf"): raise HTTPException(400, "upload a .blend, .fbx or .glb")
    base = re.sub(r"[^A-Za-z0-9_\-]+", "_", name or os.path.splitext(os.path.basename(file.filename))[0]).strip("_-")[:32]
    base = base if re.match(r"[A-Za-z0-9]", base or "") else "character"
    p, k = base, 1
    while os.path.exists(ws(p)):
        k += 1; p = f"{base}_{k}"
    d = ws(p); P = Q.paths(d)
    os.makedirs(os.path.dirname(P["source"]))
    dst = P["source"] if ext == ".blend" else os.path.join(os.path.dirname(P["source"]), "source" + ext)
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)
    job = submit(p, "import_rig", [prep_stage(d, dst), orient_stage(d)])
    return {"name": p, "job": job.id}


@app.get("/api/projects/{p}/start_frame_options")
def get_start_options(p: str):
    d = project_dir(p)
    if not os.path.exists(Q.paths(d)["config"]): raise HTTPException(400, "import the rig first")
    return start_options(Q.load(d))


@app.put("/api/projects/{p}/start_frame_options")
def put_start_options(p: str, body: dict):
    d = project_dir(p)
    if not os.path.exists(Q.paths(d)["config"]): raise HTTPException(400, "import the rig first")
    return apply_start_options(d, body)


@app.get("/api/projects/{p}/cover.jpg")
def cover(p: str):
    d = project_dir(p); P = Q.paths(d)
    if not os.path.exists(P["start_png"]): raise HTTPException(404, "no start frame yet")
    cfg = Q.load(d); lay = start_options(cfg)["layout"]; i = cover_panel(cfg)
    return cached_jpg(os.path.join(P["start"], "cover.jpg"), P["start_png"],
                      lambda: tight_crop(panel_of(cv2.imread(P["start_png"]), lay, i), 4 / 3))


@app.get("/api/projects/{p}/videos/{v}/thumb.jpg")
def video_thumb(p: str, v: str):
    d = project_dir(p); vd = video_dir(p, v); src = os.path.join(vd, "source.mp4")
    if not os.path.exists(src): raise HTTPException(404, "no video")
    cfg = Q.load(d) if os.path.exists(Q.paths(d)["config"]) else Q.normalize({})
    lay = cfg["video"]["layout"]; i = cover_panel(cfg)

    def make():
        cap = cv2.VideoCapture(src); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(n * 0.35))); ok, fr = cap.read(); cap.release()
        return tight_crop(panel_of(fr, lay, i), 3 / 2, pad=0.22, out_w=720) if ok else None
    return cached_jpg(os.path.join(vd, "thumb.jpg"), src, make)


@app.get("/api/projects")
def list_projects():
    return [project_status(n) for n in sorted(os.listdir(S["workspace"]))
            if os.path.isdir(ws(n)) and re.fullmatch(r"[A-Za-z0-9_\-]+", n) and not n.startswith("_")]


def prep_stage(d, source=None):
    P = Q.paths(d)
    src = source or P["source"]
    args = [src] if src.lower().endswith(".blend") else []
    return {"label": "prep", "cmds": [blender(*args, "--python", os.path.join(SRC, "bl_prep.py"), "--", d, *([] if args else [src]))]}


def orient_stage(d):
    P = Q.paths(d)
    return {"label": "facing views", "cmds": [blender(P["clean"], "--python", os.path.join(SRC, "bl_orient.py"), "--", d)]}


@app.post("/api/projects")
def create_project(body: dict):
    name = body.get("name", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]*", name): raise HTTPException(400, "name: letters, digits, _ or - only")
    d = ws(name)
    if os.path.exists(d): raise HTTPException(400, "project exists")
    ex = body.get("example")
    if ex:
        src = os.path.join(EXAMPLES, ex)
        if not os.path.isdir(src): raise HTTPException(404, "no such example")
        shutil.copytree(src, d)
        job = submit(name, "prep", [prep_stage(d)])
        return {"name": name, "job": job.id}
    os.makedirs(d)
    return {"name": name}


@app.delete("/api/projects/{p}")
def delete_project(p: str):
    d = project_dir(p)
    if project_status(p)["busy"]: raise HTTPException(400, "project busy")
    shutil.rmtree(d)
    return {"ok": True}


@app.get("/api/projects/{p}")
def get_project(p: str):
    return project_status(p)


@app.get("/api/projects/{p}/config")
def get_config(p: str):
    P = Q.paths(project_dir(p))
    if not os.path.exists(P["config"]): return JSONResponse({"text": "", "inspect": None})
    ins = json.load(open(P["inspect"], encoding="utf-8")) if os.path.exists(P["inspect"]) else None
    return {"text": open(P["config"], encoding="utf-8").read(), "bones": [b["name"] for b in ins["bones"]] if ins else []}


@app.put("/api/projects/{p}/config")
def put_config(p: str, body: dict):
    d = project_dir(p)
    try:
        cfg = json.loads(body["text"])
    except Exception as e:
        raise HTTPException(400, f"JSON error: {e}")
    problems = validate_config(d, cfg)
    Q.save(d, cfg)
    return {"ok": True, "warnings": problems}


def validate_config(d, cfg):
    out = []
    c = Q.normalize(cfg)
    P = Q.paths(d)
    if os.path.exists(P["inspect"]):
        bones = {b["name"] for b in json.load(open(P["inspect"], encoding="utf-8"))["bones"]}
        for k in c["keypoints"]:
            if k[1] not in bones: out.append(f"keypoint {k[0]}: bone '{k[1]}' not in rig")
    kn = set(Q.kp_names(c))
    if len(kn) != len(c["keypoints"]): out.append("duplicate keypoint names")
    for a, b in c["skeleton"]:
        if a not in kn or b not in kn: out.append(f"skeleton edge {a}-{b}: unknown keypoint")
    for f in c["foot_keypoints"]:
        if f not in kn: out.append(f"foot keypoint {f} unknown")
    for ax in ("forward", "left"):
        v = c["orientation"].get(ax)
        if ax == "left" and v in (None, "auto"): continue
        if not isinstance(v, (int, float)) and str(v).upper() not in Q.AXES:
            out.append(f"orientation.{ax} must be one of {list(Q.AXES)}" + (" or a yaw angle in degrees" if ax == "forward" else ' or "auto"'))
    if not cfg.get("orientation_confirmed"):
        out.append("Facing not confirmed: open model info and click the view in which the character faces the camera.")
    return out


@app.post("/api/projects/{p}/rig")
async def upload_rig(p: str, file: UploadFile = File(...), reset_config: str = Form("0")):
    d = project_dir(p); P = Q.paths(d)
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".blend", ".fbx", ".glb", ".gltf"): raise HTTPException(400, "upload a .blend, .fbx or .glb")
    os.makedirs(os.path.dirname(P["source"]), exist_ok=True)
    dst = P["source"] if ext == ".blend" else os.path.join(os.path.dirname(P["source"]), "source" + ext)
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if reset_config == "1" and os.path.exists(P["config"]):
        os.replace(P["config"], P["config"] + ".bak")
    job = submit(p, "import_rig", [prep_stage(d, dst), orient_stage(d)])
    return {"job": job.id}


@app.post("/api/projects/{p}/run/{step}")
def run_step(p: str, step: str, body: dict = None):
    body = body or {}
    d = project_dir(p); P = Q.paths(d)
    if project_status(p)["busy"]: raise HTTPException(400, "a job is already running for this project")
    if step == "prep":
        return {"job": submit(p, "prep", [prep_stage(d), orient_stage(d)]).id}
    if not os.path.exists(P["clean"]): raise HTTPException(400, "import the rig first")
    if step == "orient":
        return {"job": submit(p, "orient", [orient_stage(d)], est=15).id}
    cfg = Q.load(d)
    if not cfg["keypoints"]: raise HTTPException(400, "config has no keypoints")
    if step == "start_frame":
        if "adjust" in body: set_adjust(d, body["adjust"])
        if any(k in body for k in ("views", "layout", "ratio")):          # options from the UI / skill
            apply_start_options(d, {**start_options(cfg), **body})
        st = {"label": "start frame", "cmds": [blender(P["clean"], "--python", os.path.join(SRC, "bl_start_frame.py"), "--", d)]}
        return {"job": submit(p, "start_frame", [st], est=40).id}
    if step in ("render", "train", "build"):
        stages = []
        q = QUALITY.get(body.get("quality") or "")
        n = int(body.get("samples") or (q or {}).get("samples") or cfg["render"]["samples"])
        w = max(1, int(body.get("workers") or cfg["render"]["workers"]))
        steps = int(body.get("steps") or (q or {}).get("steps") or cfg["train"]["steps"])
        reuse = step == "train" or (step == "build" and body.get("reuse_data") and project_status(p)["samples"] > 0)
        if not reuse:
            stages.append(render_stage(d, n, w, append=bool(body.get("append"))) | {"weight": n * RATE_RENDER / w})
        if step != "render":
            # weights ~ wall time (seconds), see RATE_RENDER / RATE_TRAIN
            stages.append({"label": f"train {steps} steps", "weight": steps * RATE_TRAIN,
                           "cmds": [py("train.py", "--project", d, "--steps", str(steps))]})
            if reuse and not glob.glob(os.path.join(P["data"], "labels_*.json")):
                raise HTTPException(400, "no training data yet: untick 'reuse data'")
        if q and step == "build":                                           # remember the chosen quality for the UI
            raw = json.load(open(P["config"], encoding="utf-8"))
            raw.setdefault("train", {})["quality"] = body["quality"]; Q.save(d, raw)
        est = est_build(n, w, steps if step != "render" else 0, render=not reuse)
        return {"job": submit(p, step, stages, est=est).id}
    raise HTTPException(404, "unknown step")


def render_stage(d, n, w, append=False):
    P = Q.paths(d)
    if not append:
        shutil.rmtree(P["data"], ignore_errors=True)
    os.makedirs(P["data"], exist_ok=True)
    start = 0
    for f in glob.glob(os.path.join(P["data"], "labels_*.json")):
        lab = json.load(open(f)); start = max(start, max(r["id"] for r in lab["samples"]) + 1)
    chunk = [c for c in (n // w + (1 if k < n % w else 0) for k in range(w)) if c]
    cmds, s0 = [], start
    for c in chunk:
        cmds.append(blender(P["clean"], "--python", os.path.join(SRC, "bl_render.py"), "--", d, str(s0), str(c), str(1000 + s0)))
        s0 += c
    return {"label": f"render {n} samples x 2 views", "cmds": cmds, "count": "SAMPLE ", "per_cmd": chunk}


@app.post("/api/projects/{p}/videos")
async def upload_video(p: str, file: UploadFile = File(...)):
    d = project_dir(p)
    stem = os.path.splitext(os.path.basename(file.filename))[0]
    m = re.search(r"\d{4}-\d{2}-\d{2}-(\d+)", stem)                     # Seedance/Jimeng style names -> the id number
    vid = m.group(1) if m else (re.sub(r"[^A-Za-z0-9_\-]", "_", stem)[:24].strip("_") or "video")
    base, k = vid, 1
    while os.path.exists(os.path.join(d, "videos", vid)):
        k += 1; vid = f"{base}_{k}"
    vd = os.path.join(d, "videos", vid); os.makedirs(vd)
    with open(os.path.join(vd, "source.mp4"), "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"id": vid}


@app.delete("/api/projects/{p}/videos/{v}")
def delete_video(p: str, v: str):
    shutil.rmtree(video_dir(p, v)); return {"ok": True}


@app.post("/api/projects/{p}/videos/{v}/run/{step}")
def run_video_step(p: str, v: str, step: str, body: dict = None):
    body = body or {}
    d = project_dir(p); vd = video_dir(p, v); P = Q.paths(d)
    busy = [j for j in JOBS.values() if j.project == p and j.status in ("queued", "running")]
    if any(j.title == f"process_{v}" for j in busy): raise HTTPException(400, "this video is already being processed")
    if busy and not (step == "process" and all(j.title.startswith("process_") for j in busy)):   # several videos may queue
        raise HTTPException(400, "a job is already running for this project")
    for need, msg in ((P["clean"], "import the rig first"), (P["cams"], "make the start frame first (cameras.json)")):
        if not os.path.exists(need): raise HTTPException(400, msg)
    src = os.path.join(vd, "source.mp4")
    fit_dirs(vd)                                                             # migrate a legacy "fit" folder
    fv = body.get("views", "both")
    if fv not in FIT_VIEWS: raise HTTPException(400, "views both|front|side")
    fd = os.path.join(vd, "fit_" + fv)
    if step in ("process", "detect_fit") and os.path.exists(os.path.join(fd, "compare.mp4")) and os.path.exists(os.path.join(fd, "mocap.fbx")):
        raise HTTPException(400, "this video already has a fit with these views")
    det = os.path.join(vd, "det")
    have_det = os.path.exists(os.path.join(det, "pred.npz")) and os.path.getmtime(os.path.join(det, "pred.npz")) >= max(
        os.path.getmtime(src), os.path.getmtime(P["model"]))                         # a retrained model invalidates old detections

    def detect_fit_stages(mode, views="both"):
        return [{"label": "detect keypoints", "cmds": [] if have_det else [py("predict.py", "--project", d, "--src", src, "--out", det)]},
                {"label": "fit rig", "weight": 2.0,
                 "cmds": [py("fit.py", "--project", d, "--pred", os.path.join(vd, "det", "pred.npz"), "--out", fd, "--mode", mode,
                                                         *(["--views", views] if views in ("front", "side") else []))]}]

    def preview_stages():
        W, H = video_status(vd)["size"]
        PW, PH = (W // 2, H) if Q.load(d)["video"]["layout"] == "lr" else (W, H // 2)    # one panel
        return [{"label": "render fitted rig", "weight": 3.0,
                 "cmds": [blender(P["clean"], "--python", os.path.join(SRC, "bl_apply.py"), "--", d, fd, "render", str(PW), str(PH))]},
                {"label": "compose comparison", "weight": 0.5, "cmds": [py("compose_compare.py", src, fd)]}]

    def export_stages():
        return [{"label": f"export {m}", "weight": 0.5,
                 "cmds": [blender(P["clean"], "--python", os.path.join(SRC, "bl_apply.py"), "--", d, fd, m)]} for m in ("blend", "fbx")]

    if step in ("detect_fit", "process"):
        if not os.path.exists(P["model"]): raise HTTPException(400, "train the model first")
        if step == "process":                     # the one-click path: detect + fit (ours) + preview + export
            return {"job": submit(p, f"process_{v}", detect_fit_stages("ours", fv) + preview_stages() + export_stages()).id}
        return {"job": submit(p, f"detect_fit_{v}", detect_fit_stages(body.get("mode", "ours"))).id}
    if not os.path.exists(os.path.join(fd, "fit.npz")): raise HTTPException(400, "run detect + fit first")
    if step == "preview":
        return {"job": submit(p, f"preview_{v}", preview_stages()).id}
    if step == "export":
        return {"job": submit(p, f"export_{v}", export_stages()).id}
    raise HTTPException(404, "unknown step")


@app.get("/api/jobs")
def list_jobs():
    return [j.info(10 ** 9) | {"lines": []} for j in sorted(JOBS.values(), key=lambda j: -j.t0)[:30]]


@app.get("/api/jobs/{jid}")
def job_info(jid: str, since: int = 0):
    if jid not in JOBS: raise HTTPException(404, "no such job")
    return JOBS[jid].info(since)


@app.post("/api/jobs/{jid}/cancel")
def job_cancel(jid: str):
    if jid not in JOBS: raise HTTPException(404, "no such job")
    JOBS[jid].cancel(); return {"ok": True}


@app.get("/files/{p}/{path:path}")
def files(p: str, path: str):
    d = os.path.realpath(project_dir(p)); f = os.path.realpath(os.path.join(d, path))
    if not f.startswith(d + os.sep) or not os.path.isfile(f): raise HTTPException(404)
    return FileResponse(f, headers={"Cache-Control": "no-cache"})


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "app", "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "app", "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=7860); ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not a.no_browser:
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{a.port}")).start()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
