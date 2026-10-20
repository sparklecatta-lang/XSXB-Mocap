"""Side-by-side video: [source panel | fitted render] for front (top) and side (bottom).
python src/compose_compare.py VIDEO FIT_DIR [sheet frames e.g. 0,60,120,180]"""
import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from videoio import VideoOut

video, fd = sys.argv[1], sys.argv[2]
sheet_frames = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 and sys.argv[3] else []
cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 24
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
writer, t, rows_for_sheet = None, 0, {}
while True:
    ok, fr = cap.read()
    if not ok: break
    H, W = fr.shape[:2]
    rf = cv2.imread(f"{fd}/render/front_{t:05d}.jpg"); rs = cv2.imread(f"{fd}/render/side_{t:05d}.jpg")
    if rf is None: break
    lr = abs(rf.shape[1] / rf.shape[0] - W / (2 * H)) < abs(rf.shape[1] / rf.shape[0] - 2 * W / H)
    if lr:                                                      # left/right video: one row per view
        rf = cv2.resize(rf, (W // 2, H)); rs = cv2.resize(rs, (W - W // 2, H))
        top = np.hstack([fr[:, :W // 2], rf]); bot = np.hstack([fr[:, W // 2:], rs])
    else:
        rf = cv2.resize(rf, (W, H // 2)); rs = cv2.resize(rs, (W, H - H // 2))
        top = np.hstack([fr[:H // 2], rf]); bot = np.hstack([fr[H // 2:], rs])
    img = np.vstack([top, bot])
    cv2.putText(img, f"{t}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)
    if writer is None:
        writer = VideoOut(f"{fd}/compare.mp4", fps, (img.shape[1], img.shape[0]))
    writer.write(img)
    if t in sheet_frames: rows_for_sheet[t] = img
    t += 1
    if t % 10 == 0: print(f"PROGRESS {t}/{n_total}", flush=True)
if writer: writer.release()
if rows_for_sheet:
    s = np.hstack([rows_for_sheet[k] for k in sheet_frames if k in rows_for_sheet])
    s = cv2.resize(s, None, fx=1600 / s.shape[1], fy=1600 / s.shape[1], interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{fd}/compare_sheet.jpg", s)
print("frames", t, "->", f"{fd}/compare.mp4")
