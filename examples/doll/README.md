# Example project: doll with a pole (biped + hand-held object)

A biped with a long staff in its hands. It shows two things the cat example does not: a **biped** rig, and an **object that is
not part of the body** (the pole is its own rigid bone, `Fan_staff`) which is detected and fitted together with the character.
In the web app: **New project → Copy example: doll**, then go straight to the video step, or look at the shipped results.

| Path | What |
|---|---|
| `config.json` | Rig config: 22 keypoints (18 body + `pole_bottom`, `pole_top`), data-driven joint limits, the `pole` block (see `src/pole.py`). The doll faces yaw 270° in this start frame (`orientation`). |
| `rig/source.blend` | Rigged doll with the staff merged in (`Fan_root` → `Fan_staff`), hands with grip fingers, 20 built-in actions. |
| `rig/ref_poses.npz` | Pose prior for the synthetic data (8 idle poses + 7 attack actions + movement). |
| `model/head.pt`, `model/eval.json` | Detector head, 6000 steps on 1500 samples. 40 % of the samples have the pole planted / leaning / floating on its own (`pole.free_fraction`), the rest has it in the hands. DINOv2-base backbone is not included (`src/download_weights.py`). |
| `start_frame/` | Start frame for Seedance (top: front, bottom: side) and `cameras.json`. |
| `videos/2523/` | Spear thrust and a vertical pole at the end. Fit with both panels (`fit/`). |
| `videos/1459/` | Flying kick on a planted pole (gold end up). The two panels of the generated video face different ways, so after the pole vault the legs flail: a **limitation example**, fitted with both panels (`fit/`). Fitting with only one panel did not improve it. |

Each `fit/` has `compare.mp4` (source vs fitted rig), `mocap.blend`, `mocap.fbx`, `fit.npz`, `fit_report.json`.

## What it teaches

- **A held object is just one more rigid bone.** Declare it in `pole`; the fitter gives it 6 DoF per frame plus a soft "both hands grip the axis" term.
  Training renders put it in the hands (random end-for-end flip and slide) and, with `free_fraction`, also standing on the ground or floating —
  without those samples a pole planted in the ground is not detected, because the detector only ever saw it next to the hands.
- **Choose the facing in the web app** (picker with 8 views) instead of relying on the model's coordinate axes; start-frame zoom/shift can be adjusted per project.
- **Fit again with another option.** Fits are stored per option (`fit_both`, `fit_front`, `fit_side`); a video can be fitted once per option and the results compared.
- The result depends on the generated video: if the two panels disagree, or one is missing, the fit gets worse (a single panel is an option, but it is not a cure).

The result is **not perfect**: some frames have the lower leg or the pole end below the floor, and no foot locking. Cleanup is done in chat afterwards.
