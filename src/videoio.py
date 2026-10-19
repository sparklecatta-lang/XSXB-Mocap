"""Write browser-playable H.264 mp4 through ffmpeg (PATH or the imageio-ffmpeg wheel); fall back to OpenCV mp4v."""
import shutil, subprocess, cv2


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe: return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class VideoOut:
    def __init__(self, path, fps, size):
        w, h = size
        self.cv = None; self.p = None
        exe = ffmpeg_exe()
        if exe:
            self.p = subprocess.Popen([exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                       "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-",
                                       "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                       "-crf", "20", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, img):
        if self.p: self.p.stdin.write(img.tobytes())
        else: self.cv.write(img)

    def release(self):
        if self.p:
            self.p.stdin.close(); self.p.wait()
        else:
            self.cv.release()
