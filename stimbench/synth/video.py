import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

_VSYNC = None


def _vsync_flags():
    # ffmpeg >= 5 spells it -fps_mode; 4.x only knows -vsync
    global _VSYNC
    if _VSYNC is None:
        try:
            h = subprocess.run(["ffmpeg", "-hide_banner", "-h", "full"],
                               capture_output=True, text=True).stdout
            _VSYNC = ["-fps_mode", "cfr"] if "-fps_mode" in h else ["-vsync", "cfr"]
        except OSError:
            _VSYNC = ["-vsync", "cfr"]
    return _VSYNC


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def retime_cfr(path: Path, factor: float, src_fps: float, crf: int = 14) -> Tuple[bool, float, str]:
    # timestamps are rewritten, never resampled: every frame kept, constant rate
    out_fps = src_fps * factor
    tmp = path.with_suffix(".slow.mp4")
    path.rename(tmp)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
           "-vf", f"setpts=N/{out_fps:.6f}/TB", "-r", f"{out_fps:.6f}",
           *_vsync_flags(), "-an", "-c:v", "libx264", "-crf", str(crf),
           "-pix_fmt", "yuv420p", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        err = (r.stderr or "").strip()
        ok = r.returncode == 0 and path.exists() and path.stat().st_size > 0
    except OSError as e:
        ok, err = False, str(e)
    if ok:
        tmp.unlink()
        return True, out_fps, ""
    if path.exists():
        path.unlink()
    tmp.rename(path)
    return False, src_fps, err.splitlines()[-1][:200] if err else "ffmpeg failed"


def probe(path: Path) -> Optional[dict]:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
           "-show_entries", "stream=nb_read_frames,avg_frame_rate,width,height",
           "-show_entries", "format=duration", "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True)
        meta = json.loads(out.stdout)
        s = meta["streams"][0]
        num, den = s["avg_frame_rate"].split("/")
        fps = float(num) / float(den) if float(den) else 0.0
        return {"out_frames": int(s.get("nb_read_frames", 0)),
                "out_fps": round(fps, 3),
                "out_duration_s": round(float(meta["format"]["duration"]), 3),
                "width": int(s["width"]), "height": int(s["height"])}
    except Exception:
        return None
