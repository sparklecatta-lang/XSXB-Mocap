"""Rig configuration + project folder layout, shared by the Blender scripts and the torch side.
Only stdlib + numpy, so it also imports inside Blender's bundled Python.

A project is one folder:  <project>/config.json, rig/, start_frame/, data/, model/, videos/<id>/
See examples/cat/config.json for a full rig config and README for the field reference.
"""
import json, math, os, re, copy

TOOL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "name": "rig",
    "armature": None,                      # armature object name (auto-picked on import if null)
    "mesh": None,                          # skinned mesh object name, or a list (joined on import)
    "orientation": {"forward": "-Y", "left": "+X"},   # world axes the animal faces / its left side points to
    "side_tokens": None,                   # e.g. ["_L_", "_R_"] or [".L", ".R"]; enables symmetric pose sampling
    "mirror_sign": [1, -1, -1],            # euler (x,y,z) sign flip when copying a left pose to the right bone
    "keypoints": [],                       # [name, bone, end]; end = "head" | "tail" | 0..1 along the bone
    "skeleton": [],                        # [kp_a, kp_b] pairs, only for drawing overlays
    "foot_keypoints": [],                  # kept above the floor during fitting
    "default_limits": [[-30, 30], [-10, 10], [-15, 15]],   # degrees, XZY euler: x=flex, y=twist(bone axis), z=side
    "joint_limits": [],                    # [{"match": regex, "limits": [[lo,hi]x3] | null}], first match wins; null = fixed
    "helper_bones": [],                    # [{"match": regex, "source": replacement, "factor": 0.5}] follow another bone
    "smooth_chains": [],                   # [{"match": regex, "amp_deg": [x, z]}] e.g. tails: sampled as one smooth curve
    "start_frame": {"panel_width": 1536, "panel_height": 1024, "tilt_deg": 8.0,
                    "ortho_width_m": None, "aim_height_m": None},   # null = derived from the mesh size
    "render": {"resolution": 448, "samples": 1500, "workers": 3, "near_ref_fraction": 0.45,
               "symmetric_fraction": 0.3, "body_pitch_deg": 35, "body_roll_deg": 15,
               "motion_fraction": 0.3},          # share of samples with afterimages of fast limb / tail motion
    "train": {"steps": 6000, "batch": 16, "lr": 1e-3, "max_deform": 0.02, "backbone": "weights/dinov2-base",
              "held_out": 100},
    "fit": {"iters": 1500, "dense_iters": 1500, "w_dense": 0.5, "w_smooth": 200.0, "front_drift": True},
    "video": {"layout": "tb", "fps": 24},  # tb = top panel front view, bottom panel side view
}

AXES = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0), "+Z": (0, 0, 1), "-Z": (0, 0, -1)}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def normalize(cfg):
    return _merge(DEFAULTS, cfg)


def load(project):
    with open(os.path.join(project, "config.json"), encoding="utf-8") as f:
        return normalize(json.load(f))


def save(project, cfg):
    with open(os.path.join(project, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1, ensure_ascii=False)


def paths(project):
    p = lambda *a: os.path.join(project, *a)
    return {"config": p("config.json"), "source": p("rig", "source.blend"), "clean": p("rig", "clean.blend"),
            "rig": p("rig", "rig.npz"), "poses": p("rig", "ref_poses.npz"), "inspect": p("rig", "inspect.json"),
            "start": p("start_frame"), "cams": p("start_frame", "cameras.json"),
            "start_png": p("start_frame", "start_frame.png"), "data": p("data"),
            "model_dir": p("model"), "model": p("model", "head.pt"), "videos": p("videos")}


def tool_path(rel):
    return rel if os.path.isabs(rel) else os.path.join(TOOL_ROOT, rel)


def end_value(e):
    return {"head": 0.0, "tail": 1.0}.get(e, e) if isinstance(e, str) else float(e)


def kp_names(cfg):
    return [k[0] for k in cfg["keypoints"]]


def helper_source(cfg, name, names):
    """-> (source_bone, factor) if `name` is a helper bone that copies a fraction of another bone's rotation."""
    for r in cfg["helper_bones"]:
        if re.search(r["match"], name):
            src = re.sub(r["match"], r["source"], name)
            if src in names and src != name:
                return src, float(r.get("factor", 0.5))
    return None


def bone_limits_deg(cfg, name, names):
    """[[lo,hi]]*3 in degrees, or None for bones without their own rotation (fixed or helper)."""
    if helper_source(cfg, name, names):
        return None
    for r in cfg["joint_limits"]:
        if re.search(r["match"], name):
            return r["limits"]
    return cfg["default_limits"]


def mirror_name(cfg, name):
    t = cfg.get("side_tokens")
    if not t: return None
    a, b = t
    if a in name: return name.replace(a, b)
    if b in name: return name.replace(b, a)
    return None


def chains(cfg, names):
    """-> [(bone indices in hierarchy order, amp_deg)]"""
    out = []
    for c in cfg["smooth_chains"]:
        idx = [i for i, n in enumerate(names) if re.search(c["match"], n)]
        if len(idx) >= 2:
            out.append((idx, c.get("amp_deg", [22, 28])))
    return out


def axis(s):
    return AXES[s.upper().replace(" ", "")] if s[0] in "+-" else AXES["+" + s.upper()]


def view_azimuths_deg(cfg):
    """Camera azimuths (deg, from +X towards +Y) for the front view and the left-side view."""
    f, l = axis(cfg["orientation"]["forward"]), axis(cfg["orientation"]["left"])
    return math.degrees(math.atan2(f[1], f[0])), math.degrees(math.atan2(l[1], l[0]))


def kp_side(name):
    n = name.lower()
    return "l" if n.startswith("l_") else "r" if n.startswith("r_") else ""
