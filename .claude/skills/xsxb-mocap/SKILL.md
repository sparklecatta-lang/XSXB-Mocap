---
name: xsxb-mocap
description: XSXB MOCAP (QuadMocap) in chat - motion-capture a rigged character from Seedance two-view videos. Import a rigged .blend/.fbx/.glb, make the start frame (views / layout / ratio), train the detector (fast / standard / fine), then drop generated videos to get a compare video plus FBX/.blend. Use when the user says XSXB MOCAP, 动捕, 动作捕捉, QuadMocap, 首帧图, 训练角色, or hands over a rigged model or a Seedance video for a character.
---

# XSXB MOCAP

Same flow as the web app (http://127.0.0.1:7860), driven from chat. The backend is the one brain:
this skill is only a thin client of its HTTP API through `xsxb.py` (stdlib only). Never call `src/` scripts
or Blender directly, and never edit project files by hand.

```
PY=F:/Workflow-Setup/quadmocap/.venv/Scripts/python.exe
X=F:/Workflow-Setup/quadmocap/.claude/skills/xsxb-mocap/xsxb.py
"$PY" "$X" <command> ...        # run "$PY" "$X" -h for the full list
```

## How to talk to the user
- Short Chinese, plain words, no technical terms (the user is a gamer, not an engineer). Make technical choices yourself.
- Always give full absolute Windows paths (`F:\Workflow-Setup\quadmocap\workspace\...`), never relative ones.
- Show results: send the start frame image and the compare video with `SendUserFile`.
- There is only ONE fit method. Never mention fit modes, keypoints, samples/steps or other internals unless asked.
- Training is ~20-80 minutes: start it only after the user picked a quality (that pick is the go-ahead).

## Flow

**0. Server.** `"$PY" "$X" up` (starts `app/server.py --no-browser` if it is down). If it prints `BLENDER MISSING`,
ask the user for the path to blender.exe and set it:
`curl -s -X POST -H "Content-Type: application/json" -d "{\"blender\":\"E:/path/blender.exe\"}" http://127.0.0.1:7860/api/settings`.

**1. Character.** `"$PY" "$X" projects`.
- User gave a rigged model file -> `"$PY" "$X" import "<file>"` -> prints `NAME` and `JOB`; then `"$PY" "$X" wait <JOB>` (about a minute).
- Otherwise, if several characters exist and it is unclear which one, ask with AskUserQuestion (character names as options).
- `"$PY" "$X" status <NAME>` shows where it stands. If it prints `CONFIG_PROBLEMS`, tell the user the model needs a look
  and point them to the web app (角色页 -> 模型信息 ›); do not edit config.json yourself.

**2. Start frame (首帧图).** Ask ONE AskUserQuestion call with three questions:
- 拍哪两个角度: `fs` 正面 + 侧面 (Recommended, 最常用) / `ft` 正面 + 俯视 / `st` 侧面 + 俯视
- 两个画面怎么摆: `tb` 上下 (Recommended) / `lr` 左右
- 视频比例: `3:4` (Recommended) / `4:3` / `16:9` / `9:16`
Skip it when the character already has a start frame and the user did not ask to change it.
Then `"$PY" "$X" frame <NAME> --views fs --layout tb --ratio 3:4`, `"$PY" "$X" wait <JOB>` (under a minute),
and send `START_FRAME` (`F:\...\start_frame\start_frame.png`) to the user with SendUserFile.
Changing views/layout/ratio of a trained character means it must be trained again; say so.

**3. Training (训练).** `"$PY" "$X" presets`, then AskUserQuestion 训练精度 with the minutes from presets:
`std` 标准 (Recommended, 大多数动作) / `fast` 快 (先看看效果) / `fine` 精细 (快速、复杂的动作).
`"$PY" "$X" train <NAME> --quality std` -> `JOB` + ETA. Tell the user how long, and that meanwhile they can take the
start frame to Seedance with this prompt (from `"$PY" "$X" prompt <NAME>`, show it in a code block so it is easy to copy):
`上下两个画面需要完全同步。拍摄角度完全固定，不要缩放，背景和地面不变。` (左右 for the lr layout).
Wait for training with `"$PY" "$X" wait <JOB>` as a background command (run_in_background) so you are notified when
it ends; `"$PY" "$X" job <JOB>` gives the current state and minutes left.

**4. Actions (动作).** For each video the user hands over:
`"$PY" "$X" video <NAME> "<file.mp4>"` -> `VIDEO <id>`, then `"$PY" "$X" process <NAME> <id>` -> `JOB`
(detect + fit + compare video + FBX/.blend in one job, a few minutes for an 8 s clip; several videos queue up).
`"$PY" "$X" wait <JOB>`, then `"$PY" "$X" status <NAME>` for the paths. Send `COMPARE` (compare.mp4: left = source,
right = fitted character) with SendUserFile and give the full paths of `mocap.fbx` and `mocap.blend`.
The character must be trained first; if not, go back to step 3.

**Failures.** `wait` exits 1 and prints `LOG_TAIL`. Explain in one plain sentence; offer to retry. A busy character
("a job is already running") just means: wait for the running job (`status` shows it).

The web app shows the same characters and jobs live; mention http://127.0.0.1:7860 when the user prefers clicking.
