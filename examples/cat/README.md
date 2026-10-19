# Example project: cat

A complete QuadMocap project. In the web app: **New project → Copy example: cat** (the rig is re-imported automatically),
then go straight to the video step, or look at the shipped results.

| Path | What |
|---|---|
| `config.json` | Rig config: 38 keypoints, joint limits, wrist/hock half-angle helper bones, tail as a smooth chain. Cat faces −Y, left side +X. |
| `rig/source.blend` | Rigged cat (`CF_Cat_Rig`, 53 bones; mesh `CF_CatForm_Original`, 22 001 vertices, with textures). |
| `rig/ref_poses.npz` | Walk cycle (241 frames) used as pose prior for the synthetic data. |
| `model/head.pt`, `model/eval.json` | Detector head trained 6000 steps on 1500 synthetic samples (DINOv2-base backbone, not included — `src/download_weights.py`). |
| `start_frame/` | Start frame given to Seedance (1536×2048: top front, bottom left side) and `cameras.json`. |
| `videos/5456/source.mp4` | Seedance test video (834×1112, 193 frames): the cat runs, then stops. |
| `videos/5456/det/overlay.mp4` | Detected keypoints (left) and predicted surface coordinates (right). |
| `videos/5456/fit/` | Fit result: `compare.mp4` (source vs fitted rig), `mocap.blend`, `mocap.fbx`, `fit.npz`, `fit_report.json`. |
| `benchmark/` | Synthetic ground-truth comparison (ours vs triangulation), see below. |

The result is **not perfect**: the tail is the weakest part and sits too high in the front view, airborne frames float a
little, and feet are not locked. See the main README.

## Benchmark

`benchmark/synth_walk.mp4` was rendered from a 120-frame walk sequence of the rig (ground truth: `gt.npz`) with the start-frame cameras, then run
through detection and both fitting modes (`fit_ours/`, `fit_triangulate/`). Reproduce the numbers:

```bash
python src/eval_synth.py --project workspace/cat --gt examples/cat/benchmark/gt.npz \
  --fit ours=examples/cat/benchmark/fit_ours/fit.npz --fit triangulate=examples/cat/benchmark/fit_triangulate/fit.npz
```

(`workspace/cat` = this example after it was imported in the app, so that `rig/rig.npz` exists.)
`rerun_results.json` is a fresh run of `src/synth_benchmark.py`, which runs the whole procedure (render → video → detect → fit both ways → evaluate) for any project.

## Cat model attribution

The 3D cat was generated with **[Tripo](https://www.tripo3d.ai/)** from photos of the author's own cat, then re-rigged by
hand (anatomical skeleton with helper bones). 3D model generated with Tripo AI.
The test video was generated with Seedance from `start_frame/start_frame.png`.
