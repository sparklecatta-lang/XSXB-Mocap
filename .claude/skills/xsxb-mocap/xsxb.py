"""XSXB MOCAP chat client: a thin wrapper over the QuadMocap HTTP API (same backend as the web UI).
Stdlib only. Run with the project venv:  F:/Workflow-Setup/quadmocap/.venv/Scripts/python.exe xsxb.py <command> ...

  up                                   start the server if it is not answering, print settings
  projects                             one line per character
  status NAME                          character details + full paths of its files
  import MODEL_FILE [--name NAME]      new character from a rigged .blend/.fbx/.glb -> prints name + job
  frame NAME [--views fs|ft|st] [--layout tb|lr] [--ratio 3:4|4:3|16:9|9:16]   make the start frame -> job
  presets                              training quality presets with minutes
  train NAME [--quality fast|std|fine] start training -> job (+ estimated minutes)
  prompt NAME                          the Seedance prompt for this character's layout
  video NAME VIDEO_FILE                add a generated video -> video id
  process NAME VIDEO_ID                detect + fit + compare video + FBX/.blend export -> job
  job JOB_ID                           one-line job state
  wait JOB_ID [--max SECONDS]          block until the job ends (prints progress every ~20 s), exit 1 on failure
  cancel JOB_ID
"""
import argparse, json, mimetypes, os, subprocess, sys, time, uuid, urllib.error, urllib.request

BASE = os.environ.get("XSXB_URL", "http://127.0.0.1:7860")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))   # quadmocap repo
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join(ROOT, ".venv", "bin", "python")
PROMPT = {"tb": "上下两个画面需要完全同步。拍摄角度完全固定，不要缩放，背景和地面不变。",
          "lr": "左右两个画面需要完全同步。拍摄角度完全固定，不要缩放，背景和地面不变。"}
QNAME = {"fast": "快", "std": "标准", "fine": "精细"}


def call(method, path, body=None, timeout=60):
    data, headers = None, {}
    if body is not None:
        data = json.dumps(body).encode(); headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        try: msg = json.loads(e.read()).get("detail")
        except Exception: msg = e.reason
        sys.exit(f"ERROR {e.code}: {msg}")


def upload(path, file, fields=None):
    if not os.path.isfile(file): sys.exit(f"ERROR no such file: {file}")
    b = uuid.uuid4().hex; parts = []
    for k, v in (fields or {}).items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    ctype = mimetypes.guess_type(file)[0] or "application/octet-stream"
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{os.path.basename(file)}"\r\n'
                 f'Content-Type: {ctype}\r\n\r\n'.encode())
    with open(file, "rb") as f: parts.append(f.read())
    parts.append(f"\r\n--{b}--\r\n".encode())
    req = urllib.request.Request(BASE + path, data=b"".join(parts), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r: return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"ERROR {e.code}: {e.read().decode(errors='replace')}")


def alive():
    try:
        urllib.request.urlopen(BASE + "/api/settings", timeout=3); return True
    except Exception:
        return False


def ws():
    return os.path.normpath(call("GET", "/api/settings")["workspace"])


def fpath(*a):
    return os.path.normpath(os.path.join(ws(), *a))


def job_line(j):
    eta = j.get("eta")
    eta = f" · 还剩约 {max(1, round(eta / 60))} 分钟" if eta and j["status"] in ("running", "queued") else ""
    err = f" · {j['error']}" if j.get("error") else ""
    return f"JOB {j['id']} {j['project']} {j['title']} {j['status']} {round(j['progress'] * 100)}% [{j['stage']}]{eta}{err}"


def cmd_up(a):
    if not alive():
        flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0          # DETACHED_PROCESS | NEW_PROCESS_GROUP
        subprocess.Popen([PY, os.path.join(ROOT, "app", "server.py"), "--no-browser"], cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        for _ in range(60):
            time.sleep(1)
            if alive(): break
        else:
            sys.exit("ERROR server did not start: run  " + PY + " " + os.path.join(ROOT, "app", "server.py"))
        print("STARTED", BASE)
    s = call("GET", "/api/settings")
    print("SERVER", BASE, "| web UI:", BASE)
    print("BLENDER", s["blender"] or "MISSING (ask the user for blender.exe, then POST /api/settings {\"blender\": PATH})")
    print("WORKSPACE", os.path.normpath(s["workspace"]))


def cmd_projects(a):
    for p in call("GET", "/api/projects"):
        j = p.get("job")
        state = ("训练好了" if p["model"] else "首帧图已有" if p["start_frame"] else "已导入" if p["prepped"] else "未导入")
        print(f"{p['name']}: {state} · 动作视频 {len(p['videos'])} 段 · 自带动作 {p.get('actions', 0)}"
              + (f" · {job_line(j)}" if j else ""))


def cmd_status(a):
    p = call("GET", f"/api/projects/{a.name}")
    o = p.get("start_options") or {}
    print(f"NAME {p['name']}  prepped={p['prepped']} start_frame={p['start_frame']} model={p['model']} "
          f"quality={p.get('quality')} options={o} built_in_actions={p.get('actions', 0)}")
    if p.get("config_error") or p.get("config_warnings"):
        print("CONFIG_PROBLEMS", p.get("config_error") or "", p.get("config_warnings"))
    if p["start_frame"]: print("START_FRAME", fpath(a.name, "start_frame", "start_frame.png"))
    if p.get("job"): print(job_line(p["job"]))
    elif p.get("last_job") and p["last_job"]["status"] == "error": print("LAST_FAILED", job_line(p["last_job"]))
    for v in p["videos"]:
        fd = fpath(a.name, "videos", v["id"], "fit")
        line = f"VIDEO {v['id']} {v.get('duration', 0)}s fitted={v['fitted']} fbx={v['fbx']}"
        if v.get("job") and v["job"]["status"] in ("running", "queued"): line += " | " + job_line(v["job"])
        print(line)
        if v["compare"]: print("  COMPARE", os.path.join(fd, "compare.mp4"))
        if v["fbx"]: print("  FBX", os.path.join(fd, "mocap.fbx"))
        if v["blend"]: print("  BLEND", os.path.join(fd, "mocap.blend"))


def cmd_import(a):
    r = upload("/api/import", os.path.abspath(a.file), {"name": a.name or ""})
    print("NAME", r["name"]); print("JOB", r["job"])


def cmd_frame(a):
    body = {k: getattr(a, k) for k in ("views", "layout", "ratio") if getattr(a, k)}
    r = call("POST", f"/api/projects/{a.name}/run/start_frame", body)
    print("JOB", r["job"]); print("OPTIONS", call("GET", f"/api/projects/{a.name}/start_frame_options"))
    print("START_FRAME", fpath(a.name, "start_frame", "start_frame.png"), "(ready when the job is done)")


def cmd_presets(a):
    for k, v in call("GET", "/api/presets").items():
        print(f"{k} {QNAME[k]}: ~{v['minutes']} 分钟 ({v['samples']} samples, {v['steps']} steps)")


def cmd_train(a):
    r = call("POST", f"/api/projects/{a.name}/run/build", {"quality": a.quality})
    print("JOB", r["job"]); print(job_line(call("GET", f"/api/jobs/{r['job']}")))


def cmd_prompt(a):
    o = call("GET", f"/api/projects/{a.name}/start_frame_options")
    print(PROMPT[o.get("layout", "tb")])


def cmd_video(a):
    r = upload(f"/api/projects/{a.name}/videos", os.path.abspath(a.file))
    print("VIDEO", r["id"]); print("SOURCE", fpath(a.name, "videos", r["id"], "source.mp4"))


def cmd_process(a):
    r = call("POST", f"/api/projects/{a.name}/videos/{a.video}/run/process")
    print("JOB", r["job"])
    fd = fpath(a.name, "videos", a.video, "fit")
    print("WILL_WRITE", os.path.join(fd, "compare.mp4"), os.path.join(fd, "mocap.fbx"), os.path.join(fd, "mocap.blend"))


def cmd_job(a):
    print(job_line(call("GET", f"/api/jobs/{a.job}")))


def cmd_wait(a):
    t0, last = time.time(), 0
    while True:
        j = call("GET", f"/api/jobs/{a.job}?since=999999999")
        if j["status"] not in ("running", "queued"):
            print(job_line(j))
            if j["status"] != "done":
                tail = call("GET", f"/api/jobs/{a.job}")["lines"][-15:]
                print("LOG_TAIL\n" + "\n".join(tail)); sys.exit(1)
            return
        if time.time() - last > 20:
            print(job_line(j), flush=True); last = time.time()
        if a.max and time.time() - t0 > a.max:
            print("STILL_RUNNING", job_line(j)); return
        time.sleep(3)


def cmd_cancel(a):
    call("POST", f"/api/jobs/{a.job}/cancel"); print("CANCELLED", a.job)


def main():
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("up"); sp.add_parser("projects"); sp.add_parser("presets")
    for c in ("status", "prompt"): sp.add_parser(c).add_argument("name")
    p = sp.add_parser("import"); p.add_argument("file"); p.add_argument("--name", default="")
    p = sp.add_parser("frame"); p.add_argument("name"); p.add_argument("--views", choices=["fs", "ft", "st"])
    p.add_argument("--layout", choices=["tb", "lr"]); p.add_argument("--ratio", choices=["3:4", "4:3", "16:9", "9:16"])
    p = sp.add_parser("train"); p.add_argument("name"); p.add_argument("--quality", default="std", choices=["fast", "std", "fine"])
    p = sp.add_parser("video"); p.add_argument("name"); p.add_argument("file")
    p = sp.add_parser("process"); p.add_argument("name"); p.add_argument("video")
    for c in ("job", "cancel"): sp.add_parser(c).add_argument("job")
    p = sp.add_parser("wait"); p.add_argument("job"); p.add_argument("--max", type=float, default=0)
    a = ap.parse_args()
    if a.cmd != "up" and not alive():
        sys.exit(f"ERROR server not answering at {BASE}: run  xsxb.py up  first")
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
