"""Local web app: one button per pipeline step, jobs run as subprocesses (Blender headless / the venv Python).
python app/server.py [--port 7860]   then open http://127.0.0.1:7860
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, threading, time, uuid
from collections import deque

import cv2
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
    def __init__(self, project, title, stages, log_path):
        self.id = uuid.uuid4().hex[:10]; self.project = project; self.title = title
        self.stages = stages; self.stage = 0; self.frac = 0.0
        self.status = "queued"; self.error = ""; self.log = deque(maxlen=4000); self.nlines = 0
        self.procs = []; self.cancelled = False; self.t0 = time.time(); self.t1 = None
        self.log_path = log_path

    def add(self, line):
        self.log.append(line); self.nlines += 1
        with open(self.log_path, "a", encoding="utf-8") as f: f.write(line + "\n")

    def progress(self):
        w = [s.get("weight", 1.0) for s in self.stages]
        done = sum(w[:self.stage]) + (w[self.stage] * self.frac if self.stage < len(w) else 0)
        return 1.0 if self.status == "done" else done / sum(w)

    def run(self):
        self.status = "running"
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
                "lines": new, "next": self.nlines}


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


def submit(project, title, stages):
    os.makedirs(ws(project, "logs"), exist_ok=True)
    job = Job(project, title, stages, ws(project, "logs", time.strftime("%Y%m%d-%H%M%S") + f"_{title}.log"))
    JOBS[job.id] = job
    with LOCK: QUEUE.append(job)
    return job


def blender(*args):
    if not S["blender"] or not os.path.exists(S["blender"]):
        raise HTTPException(400, "Blender not found: set its path in Settings")
    return [S["blender"], "-b", *args]


def py(script, *args):
    return [sys.executable, os.path.join(SRC, script), *args]


# ---------------------------------------------------------------- status
def video_status(d):
    vd = lambda *a: os.path.join(d, *a)
    st = {"id": os.path.basename(d), "source": os.path.exists(vd("source.mp4")),
          "detected": os.path.exists(vd("det", "pred.npz")), "overlay": os.path.exists(vd("det", "overlay.mp4")),
          "fitted": os.path.exists(vd("fit", "fit.npz")), "compare": os.path.exists(vd("fit", "compare.mp4")),
          "blend": os.path.exists(vd("fit", "mocap.blend")), "fbx": os.path.exists(vd("fit", "mocap.fbx"))}
    if os.path.exists(vd("fit", "fit_report.json")):
        st["report"] = json.load(open(vd("fit", "fit_report.json")))
        st["report"].pop("per_kp_px_mean", None)
    if st["source"]:
        cap = cv2.VideoCapture(vd("source.mp4"))
        st["size"] = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        st["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); st["fps"] = cap.get(cv2.CAP_PROP_FPS); cap.release()
    return st


def project_status(p):
    d = project_dir(p); P = Q.paths(d)
    st = {"name": p, "source": os.path.exists(P["source"]), "prepped": os.path.exists(P["rig"]) and os.path.exists(P["clean"]),
          "config": os.path.exists(P["config"]), "start_frame": os.path.exists(P["start_png"]) and os.path.exists(P["cams"]),
          "samples": len(glob.glob(os.path.join(P["data"], "*_v0_rgb.png"))), "model": os.path.exists(P["model"]),
          "ref_poses": os.path.exists(P["poses"])}
    if st["config"]:
        try:
            c = Q.load(d); st["keypoints"] = len(c["keypoints"]); st["render_samples"] = c["render"]["samples"]
            st["train_steps"] = c["train"]["steps"]; st["config_name"] = c["name"]
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
    return st


# ---------------------------------------------------------------- app
app = FastAPI(title="quadmocap")


@app.get("/api/settings")
def get_settings():
    return {**S, "python": sys.executable, "examples": sorted(os.listdir(EXAMPLES)) if os.path.isdir(EXAMPLES) else []}


@app.post("/api/settings")
def set_settings(body: dict):
    if "blender" in body: S["blender"] = body["blender"].strip().strip('"')
    json.dump({"blender": S["blender"], "workspace": S["workspace"]}, open(SETTINGS_FILE, "w", encoding="utf-8"), indent=1)
    return get_settings()


@app.get("/api/projects")
def list_projects():
    return [project_status(n) for n in sorted(os.listdir(S["workspace"]))
            if os.path.isdir(ws(n)) and re.fullmatch(r"[A-Za-z0-9_\-]+", n) and not n.startswith("_")]


def prep_stage(d, source=None):
    P = Q.paths(d)
    src = source or P["source"]
    args = [src] if src.lower().endswith(".blend") else []
    return {"label": "prep", "cmds": [blender(*args, "--python", os.path.join(SRC, "bl_prep.py"), "--", d, *([] if args else [src]))]}


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
        if c["orientation"][ax].upper() not in Q.AXES: out.append(f"orientation.{ax} must be one of {list(Q.AXES)}")
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
    job = submit(p, "import_rig", [prep_stage(d, dst)])
    return {"job": job.id}


@app.post("/api/projects/{p}/run/{step}")
def run_step(p: str, step: str, body: dict = None):
    body = body or {}
    d = project_dir(p); P = Q.paths(d)
    if project_status(p)["busy"]: raise HTTPException(400, "a job is already running for this project")
    if step == "prep":
        return {"job": submit(p, "prep", [prep_stage(d)]).id}
    if not os.path.exists(P["clean"]): raise HTTPException(400, "import the rig first")
    cfg = Q.load(d)
    if not cfg["keypoints"]: raise HTTPException(400, "config has no keypoints")
    if step == "start_frame":
        st = {"label": "start frame", "cmds": [blender(P["clean"], "--python", os.path.join(SRC, "bl_start_frame.py"), "--", d)]}
        return {"job": submit(p, "start_frame", [st]).id}
    if step in ("render", "train", "build"):
        stages = []
        n = int(body.get("samples") or cfg["render"]["samples"]); w = max(1, int(body.get("workers") or cfg["render"]["workers"]))
        steps = int(body.get("steps") or cfg["train"]["steps"])
        reuse = step == "train" or (step == "build" and body.get("reuse_data") and project_status(p)["samples"] > 0)
        if not reuse:
            stages.append(render_stage(d, n, w, append=bool(body.get("append"))))
        if step != "render":
            # weights ~ wall time: rendering ~0.5 s/sample/worker, training ~0.28 s/step on an RTX 5060 Ti
            stages.append({"label": f"train {steps} steps", "weight": max(0.1, steps * 0.28 / max(1.0, n * 0.5 / w)),
                           "cmds": [py("train.py", "--project", d, "--steps", str(steps))]})
            if reuse and not glob.glob(os.path.join(P["data"], "labels_*.json")):
                raise HTTPException(400, "no training data yet: untick 'reuse data'")
        return {"job": submit(p, step, stages).id}
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
    if project_status(p)["busy"]: raise HTTPException(400, "a job is already running for this project")
    for need, msg in ((P["clean"], "import the rig first"), (P["cams"], "make the start frame first (cameras.json)")):
        if not os.path.exists(need): raise HTTPException(400, msg)
    src = os.path.join(vd, "source.mp4"); fd = os.path.join(vd, "fit")
    if step == "detect_fit":
        if not os.path.exists(P["model"]): raise HTTPException(400, "train the model first")
        mode = body.get("mode", "ours")
        stages = [{"label": "detect keypoints", "cmds": [py("predict.py", "--project", d, "--src", src, "--out", os.path.join(vd, "det"))]},
                  {"label": "fit rig", "weight": 2.0,
                   "cmds": [py("fit.py", "--project", d, "--pred", os.path.join(vd, "det", "pred.npz"), "--out", fd, "--mode", mode)]}]
        return {"job": submit(p, f"detect_fit_{v}", stages).id}
    if not os.path.exists(os.path.join(fd, "fit.npz")): raise HTTPException(400, "run detect + fit first")
    if step == "preview":
        st = video_status(vd); W, H = st["size"]
        stages = [{"label": "render fitted rig", "weight": 3.0,
                   "cmds": [blender(P["clean"], "--python", os.path.join(SRC, "bl_apply.py"), "--", d, fd, "render", str(W), str(H // 2))]},
                  {"label": "compose comparison", "cmds": [py("compose_compare.py", src, fd)]}]
        return {"job": submit(p, f"preview_{v}", stages).id}
    if step == "export":
        stages = [{"label": f"export {m}", "cmds": [blender(P["clean"], "--python", os.path.join(SRC, "bl_apply.py"), "--", d, fd, m)]}
                  for m in ("blend", "fbx")]
        return {"job": submit(p, f"export_{v}", stages).id}
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
