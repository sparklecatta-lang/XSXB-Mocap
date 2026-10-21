"""Hand-held pole (staff, spear, ...) as a free rigid body that rests in the character's grip.
Config block (qconfig "pole"):
  {"root": "Fan_root", "hands": ["hand_L", "hand_R"], "grips": [[x,y,z],[x,y,z]]   # grip points in each hand bone's head frame
   "bottom": [x,y,z], "axis": [x,y,z]            # pole bottom point + unit direction bottom->top in the rig's rest pose (armature space)
   "param_c": 0.515, "length": 1.4,              # distance bottom -> midpoint of the two grips, pole length (m)
   "keypoints": ["pole_bottom", "pole_top"], "slide_m": 0.25, "flip_prob": 0.5, "wiggle_deg": 3,
   "free_fraction": 0.0}      # share of training samples with the pole standing / leaning / floating on its own, not in the hands
The pole root bone is placed by an armature-space transform A (4x4) so that its pose matrix is A @ rest.
During data generation A comes from the hands (random end-for-end flip and slide along the grip line), during
fitting it is free (6 DoF per frame) with a soft "both grips lie on the axis" term."""
import numpy as np


def rot_between(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if c < -0.999999:
        ax = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]); ax /= np.linalg.norm(ax)
        return 2 * np.outer(ax, ax) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def rotvec_matrix(v):
    v = np.asarray(v, float); th = np.linalg.norm(v)
    if th < 1e-9: return np.eye(3)
    k = v / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class Pole:
    def __init__(self, pcfg, names):
        self.cfg = pcfg
        self.root = names.index(pcfg["root"])
        self.hands = [names.index(h) for h in pcfg["hands"]]
        self.g = [np.append(np.asarray(g, float), 1.0) for g in pcfg["grips"]]
        self.B0 = np.asarray(pcfg["bottom"], float); self.u0 = np.asarray(pcfg["axis"], float)
        self.param_c = float(pcfg["param_c"]); self.length = float(pcfg["length"])
        self.kps = list(pcfg.get("keypoints", []))

    def grips(self, PL, PR):
        return (PL @ self.g[0])[:3], (PR @ self.g[1])[:3]

    def matrix(self, R, bottom):
        """armature-space transform that takes the rest pole to rotation R with its bottom at `bottom`"""
        A = np.eye(4); A[:3, :3] = R; A[:3, 3] = bottom - R @ self.B0
        return A

    def from_hands(self, PL, PR, sign=1.0, slide=0.0, wiggle=None):
        """PL, PR: armature-space pose matrices of the two hand bones. sign -1 = tip at the other end."""
        pL, pR = self.grips(PL, PR)
        v = pR - pL; d = np.linalg.norm(v)
        t = v / d if d > 1e-4 else self.u0
        t = t * sign
        if wiggle is not None:
            t = rotvec_matrix(wiggle) @ t
        c = (pL + pR) / 2 + slide * t
        return self.matrix(rot_between(self.u0, t), c - self.param_c * t)

    def random_state(self, rng):
        c = self.cfg
        sign = -1.0 if rng.random() < float(c.get("flip_prob", 0.5)) else 1.0
        slide = rng.uniform(-1, 1) * float(c.get("slide_m", 0.25))
        wig = rng.normal(0, 1, 3) * np.radians(float(c.get("wiggle_deg", 3)))
        return sign, slide, wig

    def random_free(self, rng):
        """pole independent of the hands (planted on the ground, leaning, floating): world offset of its bottom from the body
        base (x, y, height above the ground) and the world direction bottom->top"""
        c = self.cfg; r = rng.random()
        ang = rng.uniform(0, 2 * np.pi); rad = rng.uniform(0.1, float(c.get("free_radius_m", 0.7)))
        off = np.array([rad * np.cos(ang), rad * np.sin(ang), 0.0])
        if r < 0.55:   tilt, off[2] = np.radians(rng.uniform(0, 8)), rng.uniform(0, 0.05)       # planted, upright
        elif r < 0.8:  tilt, off[2] = np.radians(rng.uniform(8, 40)), rng.uniform(0, 0.05)      # leaning on the ground
        else:          tilt, off[2] = np.radians(rng.uniform(0, 80)), rng.uniform(0.1, 1.2)    # in the air
        az = rng.uniform(0, 2 * np.pi)
        d = np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)])
        if rng.random() < float(c.get("free_flip_prob", 0.5)): d = -d
        return off, d
