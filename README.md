# QuadMocap

Motion capture for **rigged characters** from AI-generated two-view videos (tested on quadrupeds), as a local one-click web app.
You give it a rigged animal; it renders a start frame for a video model (e.g. Seedance), trains a keypoint detector on
synthetic renders of *your* rig, and fits the rig to the generated video. Output: a Blender action, `.blend` and FBX.

[中文说明](README_zh.md)

![Example: source video (left of each pair) vs. fitted rig (right). Top = front panel, bottom = side panel.](examples/cat/videos/5456/fit/compare_sheet.jpg)

*Example cat, frames 0 / 50 / 100 / 150. In each pair: left = Seedance video, right = fitted rig. The result is **not perfect** — see [Limitations](#limitations).*

## Credit

The workflow comes from the YouTube video by **Can It Code?** —
[“AI Can Now Animate Any Monster”](https://www.youtube.com/watch?v=AS92HYsEBqo): generate a two-view video with a video
model, detect keypoints with a DINO-based model trained on synthetic renders of the rig, triangulate, and fit the skeleton.
This repository is an independent re-implementation with the changes listed below. It is not affiliated with the video author.

## What we changed

| | Original workflow (as we understood it) | This repo |
|---|---|---|
| Fitting target | Triangulate 3D points from the two views, then fit bones to the 3D points | Fit bones **directly to the 2D detections in both views** (reprojection error), no triangulation step |
| Signal | Sparse keypoints | Keypoints **+ dense surface correspondences**: the head also predicts, per pixel, the rest-pose position on the mesh; those pixels are matched to mesh vertices (KD-tree) and the skinned vertices are fitted too |
| Video drift | — | The video model's front panel drifts (zoom / moves toward the camera). The fitter solves a **smooth per-frame scale + vertical shift** for the front panel |
| Training data | — | 30 % of the synthetic samples carry **afterimages** like fast motion in generated video: faint copies of nearby limb / tail poses. Labels and mask come from the main pose only, so the detector learns to ignore the ghosts |
| Evaluation | — | **Synthetic ground-truth benchmark**: render the rig's own animation with the same cameras, run the whole pipeline, measure 3D error |
| Rig | One creature | **Any rigged character** (tested on the cat and a procedural dog) through one JSON config (bone names, keypoints, joint limits, helper bones, orientation) |

Other details: bone lengths are fixed (only joint rotations are solved), joint-limit / floor terms, a robust smoothness
term that keeps fast motion (galloping legs), a de-jitter low-pass after the fit,
whole-sequence joint optimisation, and badly deformed synthetic samples (edge stretch) are dropped from training.

### Comparison data

Same detections, same rig, only the fitting method differs (`fit.py --mode ours` vs `--mode triangulate`).
The original author's detector was not reproduced: "original" means our detector followed by the triangulation workflow.

**Synthetic walk, 120 frames, ground truth known** (mean 3D keypoint error after aligning the pelvis, cm):

| Method | All keypoints | Limbs | Tail |
|---|---|---|---|
| Triangulate, then fit (original) | 3.10 | 3.52 | 2.70 |
| **Ours** (2D + surface, drift correction) | **2.87** | **3.27** | **2.26** |

![Bar chart of the table above](examples/cat/benchmark/chart.svg)

About 7 % lower error overall, 7 % on limbs, 16 % on the tail. The pelvis itself (always 0 after alignment) is excluded.
The robust smoothing and de-jitter filter (see [Post-processing](#post-processing-suggestions)) lowered our error from
3.58 to 2.87 cm here, and from 5.76 to 3.56 cm on a fast, motion-blurred synthetic walk.
Data and commands: [`examples/cat/benchmark`](examples/cat/benchmark), see [the example README](examples/cat/README.md#benchmark);
`src/synth_benchmark.py` runs the whole procedure (render → video → detect → fit both ways → evaluate) for any project.
With the previous detector (trained without afterimages) on the same video the gap is larger: ours 2.84 / 3.28 / 2.49 cm
vs. triangulation 3.72 / 4.23 / 3.57 cm (all / limbs / tail). The new detector mostly helps triangulation; our fit stays
about the same.

**Real Seedance video (example cat, 193 frames)** — no ground truth, so this is the median 2D distance between the fitted
rig's keypoints and the detections (px, 834×556 panels):

| Method | Front panel | Side panel |
|---|---|---|
| Triangulate, then fit | 8.4 | 7.8 |
| **Ours** | **8.0** | **6.7** |

Previous detector: triangulation 9.6 / 7.1, ours 9.0 / 6.2. A second clip (0° camera tilt, 193 frames) gives
triangulation 8.7 / 7.4, ours 8.1 / 6.6 (previous detector: 9.7 / 7.5 and 9.0 / 6.5).

## Requirements

- Windows 10/11 or Linux, NVIDIA GPU with CUDA (tested: RTX 5060 Ti 16 GB). Training uses a few GB of VRAM.
- [Blender](https://www.blender.org/) 4.2 or newer (tested 5.1.2), used headless as an external program.
- Python 3.10–3.12.
- Put the repo and your project files on a path **without non-ASCII characters** (OpenCV cannot read such paths on Windows).

## Install

Windows: run `setup.bat` (creates `.venv`, installs CUDA PyTorch and the requirements, downloads DINOv2-base into `weights/`).

Manual:
```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python src/download_weights.py
```

## Use the web app

Run `start.bat` (or `python app/server.py`) and open http://127.0.0.1:7860. The app only listens on localhost.
If Blender is not found automatically, set its path under **Settings**. Each step shows progress and the log.

1. **Import rig** — upload a `.blend` / `.fbx` / `.glb` with one armature and its skinned mesh(es). The scene is cleaned
   (other objects removed, transforms applied, meshes joined) and a **draft config** is guessed from the bone names
   (left/right tokens, orientation, keypoints at bone heads and chain ends, feet). The import also checks that the mesh
   follows the exported skinning (the same check runs before every render and export). Review the config in the config editor —
   especially joint limits; the defaults are generic. If the file has an animation (e.g. a walk cycle), it is used as a
   pose prior for the synthetic data.
2. **Start frame** — renders the start image: top = front view, bottom = left side, same orthographic scale, plus
   `cameras.json`. Give the image to an image-to-video model as the first frame.
   The cameras look down 8° by default (`start_frame.tilt_deg`), which shows the far-side feet. Against a level
   camera (0°) it was better in our tests (measured before the smoothing update): 3.58 vs 3.83 cm on the synthetic walk, with the biggest gap on the tail,
   and better on a real clip as well. Keep the default unless you have a reason not to.
3. **Training data + train** (one run) — random poses inside the joint limits rendered in two views (RGB, a
   rest-position map, keypoint labels; 30 % with afterimages), then a small convolutional head is trained on a
   frozen DINOv2 backbone.
   The defaults are the recommended values: 1500 samples, 3 Blender workers, 6000 steps (≈ 40 min on an RTX 5060 Ti).
   Tick *reuse existing training data* to retrain only.
4. **Upload video → Detect + fit → Preview → Export** — detection overlay, fitted-vs-source comparison video,
   `mocap.blend` and `mocap.fbx` (baked action, for Unreal / Unity / other DCCs).

To try it without training: **New project → Copy example: cat**. It ships with a trained head, the start frame and the
test video, so you can go straight to step 4.
**Copy example: doll** is a biped with a hand-held pole (a rigid object detected and fitted with the character), with two clips fitted;
see [`examples/doll/README.md`](examples/doll/README.md).

### Making the video

Use the start frame as the first frame of an image-to-video model. Ask for both panels to stay in sync and for the
camera to stay fixed. The prompt used for the example video (Seedance, originally in Chinese):

> Top and bottom panels must stay completely in sync. The kitten starts standing, then runs happily with proud, relaxed
> steps, then stops…

The video must keep the start frame's layout (top = front, bottom = side). Any resolution with the same aspect ratio
works; the cameras are scaled by width.

## Command line

Every step is a plain script; the app only runs these.

```bash
blender -b rig.blend --python src/bl_prep.py -- PROJECT              # import / clean / draft config
blender -b PROJECT/rig/clean.blend --python src/bl_start_frame.py -- PROJECT
blender -b PROJECT/rig/clean.blend --python src/bl_render.py -- PROJECT START COUNT SEED
python src/train.py --project PROJECT
python src/predict.py --project PROJECT --src video.mp4 --out PROJECT/videos/x/det
python src/fit.py --project PROJECT --pred PROJECT/videos/x/det/pred.npz --out PROJECT/videos/x/fit [--mode triangulate]
blender -b PROJECT/rig/clean.blend --python src/bl_apply.py -- PROJECT PROJECT/videos/x/fit blend|fbx|render [W H]
python src/compose_compare.py video.mp4 PROJECT/videos/x/fit
python src/synth_benchmark.py --project PROJECT --blender path/to/blender   # synthetic ground-truth benchmark
#   options: --stride 4 --blur 5 (faster, motion-blurred video)  --ckpt other/head.pt  --name output_folder
```

## Rig config

One JSON per project (`PROJECT/config.json`); full example: [`examples/cat/config.json`](examples/cat/config.json).

| Field | Meaning |
|---|---|
| `armature`, `mesh` | Object names. `mesh` may be a list (joined on import). |
| `orientation` | `forward`: world axis the animal faces (`-Y` for the cat), `left`: axis its left side points to (`+X`). |
| `pole` | optional hand-held pole (staff, spear) tracked as a free rigid body with two keypoints (bottom, top); see `src/pole.py` for the fields (`root`, `hands`, `grips`, `bottom`, `axis`, `param_c`, `length`, `keypoints`). Training renders place it in the grip (random end-for-end flip + slide), and a share `free_fraction` (default 0) of samples with the pole standing, leaning or floating on its own (`free_radius_m`, `free_flip_prob`); the fit gives it 6 DoF per frame with a "both grips on its axis" term. |
| `side_tokens` | e.g. `["_L_", "_R_"]` or `[".L", ".R"]`; used for symmetric pose sampling. |
| `keypoints` | `[name, bone, end]`, `end` = `"head"`, `"tail"` or 0–1 along the bone. Names starting with `l_` / `r_` are drawn in left/right colours. |
| `skeleton` | Keypoint pairs, only for drawing. |
| `foot_keypoints` | Kept above the floor during fitting. |
| `joint_limits` | `[{"match": regex, "limits": [[lo,hi],[lo,hi],[lo,hi]] or null}]`, degrees, first match wins, `null` = no own rotation. Axes are the bone's XZY Euler: x = bend, y = twist around the bone, z = sideways. |
| `default_limits` | Limits for bones no rule matches. |
| `helper_bones` | `[{"match": regex, "source": replacement, "factor": 0.5}]`: bones that copy a fraction of another bone's rotation (e.g. half-angle wrist helpers). |
| `smooth_chains` | `[{"match": regex, "amp_deg": [x, z]}]`: chains (tails) sampled as one smooth curve. |
| `start_frame` | Panel size, camera tilt, ortho width and aim height (`null` = derived from the mesh size). |
| `render`, `train`, `fit`, `video` | Sample count, Blender workers, training steps / backbone, fitting weights, front-drift correction, video layout and fps. |

## Example: our cat

[`examples/cat`](examples/cat) is a complete project: the rigged cat (`rig/source.blend`, 53 bones, with a walk cycle as
pose prior), its config, the trained detector head, the start frame and cameras, the Seedance test video, the detection
overlay, the comparison video, the fitted result (`mocap.blend`, `mocap.fbx`) and the synthetic benchmark data.
The 1500-sample training set (~850 MB) is not included; step 3 regenerates it.

The cat model was generated with [Tripo](https://www.tripo3d.ai/) from photos of the author's own cat and rigged by hand.
See [`examples/cat/README.md`](examples/cat/README.md) for its attribution.

## Post-processing (suggestions)

The fitter smooths the action in two steps: a robust smoothness term (small frame-to-frame wobble is penalised, large
accelerations such as galloping legs cost little) and a de-jitter low-pass on the bone curves after the fit (fades out
where a curve moves fast). Some small shakes can remain and feet still slide. Treat `mocap.blend` / `mocap.fbx` as raw capture and clean it up before use. General approach:

1. **Low-pass the rotations, not the Euler numbers.** Per pose bone: convert the rotation keys to quaternions, keep
   them in one hemisphere (flip `q` when `dot(q[t], q[t-1]) < 0`), filter each component with a zero-phase low-pass
   (e.g. Butterworth, `scipy.signal.filtfilt`, 2nd order, cutoff ≈ 4–6 Hz at 24 fps), renormalise, convert back to the
   bone's rotation mode keeping Euler continuity. Filter the armature object's location/rotation separately with a lower
   cutoff (≈ 2–3 Hz). Blender 4.0+ also has *Graph Editor → Key → Smooth → Butterworth Smooth* for quick manual work.
2. **Lock feet during contact.** Find contact frames from the foot keypoints (low height and low speed for a few frames),
   pin each foot to its average position there with IK, blend a few frames in and out, then bake.
3. **Or smooth more in the fit itself.** Project config, then re-run *Detect + fit* and compare `fit_report.json`:
   - `fit.post_filter_hz` (default 6; 0 = off): de-jitter cutoff after the fit. Lower = calmer but softer motion.
   - `fit.post_filter_fast` (default 0.05 rad/frame): curves faster than this are left unfiltered.
   - `fit.w_smooth` (default 200) and `fit.smooth_delta` (default 0.002 rad/frame²): smoothness weight; accelerations
     above `smooth_delta` are penalised linearly instead of quadratically. Raising either smooths more.
     `smooth_delta` 0 is the old plain quadratic term, which flattens fast leg motion (legs barely move in a gallop).
4. **Check against the video.** Re-render the comparison (*Preview*), and make sure fast motions (jumps, landings)
   were not smeared — lower the cutoff only where needed.

Example prompts for a coding agent (Claude Code, Codex, …), run in this repo:

```text
Open workspace/<project>/videos/<id>/fit/mocap.blend headless with Blender. The action on the armature was fitted per
frame and jitters. Write a Blender Python script that, for every pose bone, converts the rotation keys to quaternions
(keep hemisphere continuity), applies a zero-phase 2nd-order Butterworth low-pass (scipy filtfilt, cutoff 5 Hz at the
scene fps) to each component, renormalises, converts back to the bone's rotation mode (XZY Euler, keep continuity) and
rewrites one key per frame. Smooth the armature object's location and rotation with a 2.5 Hz cutoff. Do not change
bone lengths or the rest pose. Save mocap_smooth.blend next to it, export mocap_smooth.fbx with the same FBX settings
as src/bl_apply.py, and report the largest per-bone change in degrees.
```

```text
Using workspace/<project>/videos/<id>/fit/fit.npz, src/rigtorch.py and the project's config.json, compute the world
position of every foot keypoint per frame. Mark contact frames where the foot is below 2 cm and slower than
0.15 m/s for at least 3 frames. In mocap_smooth.blend, pin each foot during its contacts with an IK constraint to a
target empty at the contact's mean position, blend the constraint influence in and out over 3 frames, bake the result
into a new action, and save mocap_clean.blend. List the contact ranges you found per foot.
```

```text
Re-render the comparison for mocap_clean.blend with the start-frame cameras (see the render mode of src/bl_apply.py
and src/compose_compare.py) and give me a before/after sheet of frames 0, 50, 100, 150 so I can check that fast
motions were not smeared.
```

## Limitations

The example result is **not perfect**:

- **Jitter**: mostly removed by the smoothing and de-jitter filter, but small shakes can remain — see
  [Post-processing](#post-processing-suggestions).
- **Tail** is the weakest part (largest residuals); in the front panel the tail sits too high.
- **Airborne frames** of the run float slightly too high.
- **No foot locking** yet, so feet can slide during contact.
- The video model's front panel drifts (≈5 % zoom, moves toward the camera); the drift correction absorbs part of it.
- The detector was trained on 1500 synthetic samples only (30 % with afterimages); held-out synthetic error is
  10.0 px mean / 4.4 px median (448 px crops, afterimage samples included), worst on tail tip and hind toes.
- The afterimage training has been checked on synthetic data and on two real clips that have few afterimages; it has
  not yet been tested on a clip with heavy afterimages. Motion blur is not simulated, and a synthetic fast,
  motion-blurred walk (`--stride 4 --blur 5`) is about 3.4–3.6 cm with either detector (5.8 cm before the smoothing update).
- Generic-rig support has been tested on the cat and on a procedural blocky test rig
  ([`tests/make_test_rig.py`](tests/make_test_rig.py): `.L/.R` names, other facing axis, scaled object, two meshes) —
  not yet on many real rigs. The draft config's joint limits are generic and usually need editing.

## Swapping the backbone (DINOv3)

The backbone is loaded with `transformers.AutoModel`, so DINOv3 works once you have access:
`python src/download_weights.py --model facebook/dinov3-vitb16-pretrain-lvd1689m` (gated: accept the license on
Hugging Face and log in first), then set `"train": {"backbone": "weights/dinov3-vitb16-pretrain-lvd1689m"}` and retrain.

## Third-party software and services

- **DINOv2** (Meta) — Apache-2.0, weights downloaded from Hugging Face, not included.
- **Blender** — GPL, called as an external program; not bundled.
- **Seedance** (ByteDance) — external video-generation service used to make the example video; not part of this repo.
- **Tripo** — used to generate the example cat model.
- PyTorch, Hugging Face Transformers, OpenCV, SciPy, FastAPI, Uvicorn, imageio-ffmpeg — see their licenses.

## License

Code: [MIT](LICENSE). The example cat assets have their own note in [`examples/cat/README.md`](examples/cat/README.md).
