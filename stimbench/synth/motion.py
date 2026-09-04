import csv
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

LAG_BAND = (6, 40)         # frames; below 6 is noise, above 40 cannot repeat in 81 frames
MIN_PEAK = 0.4             # autocorrelation below this is not a period, just noise


def read_gray(path: Path, scale: int = 4) -> np.ndarray:
    import imageio.v3 as iio
    frames = iio.imread(str(path), plugin="pyav") if _has_pyav() else _read_ffmpeg(path)
    g = frames[..., :3].astype(np.float32).mean(axis=-1)
    return g[:, ::scale, ::scale]


def _has_pyav() -> bool:
    try:
        import av  # noqa: F401
        return True
    except ImportError:
        return False


def _read_ffmpeg(path: Path) -> np.ndarray:
    import imageio
    with imageio.get_reader(str(path), format="ffmpeg") as r:
        return np.stack([f for f in r])


def _period(signal: np.ndarray):
    x = signal - signal.mean()
    if np.abs(x).max() < 1e-6:
        return None, 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / ac[0]
    lo, hi = LAG_BAND
    hi = min(hi, len(ac) - 2)
    # an interior local maximum only: a monotonic tail would otherwise be
    # accepted at the band edge as if it were a period
    peaks = [k for k in range(lo, hi + 1) if ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]]
    if not peaks:
        return None, 0.0
    lag = max(peaks, key=lambda k: ac[k])
    return lag, float(ac[lag])


def measure(path: Path) -> dict:
    g = read_gray(path)
    diff = np.diff(g, axis=0)
    energy = np.abs(diff).mean(axis=(1, 2))
    rows = np.arange(diff.shape[1], dtype=np.float32)[None, :, None]
    mass = np.abs(diff).sum(axis=(1, 2)) + 1e-6
    # signed vertical centre of change: an up-down oscillation keeps its sign
    # per half cycle, so the period is not halved the way rectified energy is
    centroid = (diff * rows).sum(axis=(1, 2)) / mass
    per_energy, h_energy = _period(energy)
    per_centroid, h_centroid = _period(centroid)
    median = float(np.median(energy))
    return {
        "frames": int(g.shape[0]),
        "motion_energy": round(float(energy.mean()), 3),
        "freeze_fraction": round(float((energy < 0.2 * median).mean()), 3) if median > 0 else 1.0,
        "period_energy": per_energy or "",
        "peak_energy": round(h_energy, 3),
        "period_centroid": per_centroid or "",
        "peak_centroid": round(h_centroid, 3),
    }


def achieved_hz(period_frames, gen_fps: float, slow_factor: float):
    # clips are generated at gen_fps and played back slow_factor times faster
    return round(gen_fps * slow_factor / period_frames, 3) if period_frames else ""


def measure_set(root: Path, records: Iterable[dict], out_csv: Optional[Path] = None,
                log=print) -> List[dict]:
    rows = []
    records = list(records)
    for i, r in enumerate(records, 1):
        path = Path(root) / r["file"]
        try:
            m = measure(path)
        except Exception as e:
            log(f"[{i}/{len(records)}] {r['file']}: unreadable ({e})")
            continue
        gen_fps = float(r.get("gen_fps", 16))
        slow = float(r.get("slow_factor", 1.0))
        rows.append({
            "file": r["file"], "cls": r["cls"], "severity": r.get("severity", ""),
            "topography_id": r.get("topography_id", ""), "aspect": r.get("aspect", ""),
            "requested_hz": r.get("requested_hz", ""),
            "resolved": m["period_centroid"] != "" and m["peak_centroid"] >= MIN_PEAK,
            "achieved_hz": achieved_hz(m["period_centroid"], gen_fps, slow)
            if m["period_centroid"] != "" and m["peak_centroid"] >= MIN_PEAK else "",
            **m,
        })
        if i % 20 == 0 or i == len(records):
            log(f"[{i}/{len(records)}] measured")
    if out_csv and rows:
        with Path(out_csv).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def summarise(rows: List[dict]) -> str:
    lines = []
    for cls in sorted({r["cls"] for r in rows}):
        rs = [r for r in rows if r["cls"] == cls]
        hz = [r["achieved_hz"] for r in rs if r["resolved"]]
        energy = [r["motion_energy"] for r in rs]
        req = rs[0]["requested_hz"]
        hz_txt = (f"achieved Hz median {np.median(hz):.2f} (p10 {np.percentile(hz, 10):.2f}, "
                  f"p90 {np.percentile(hz, 90):.2f}) vs requested {req}") if hz else "no resolved period"
        lines.append(f"{cls:<12} n={len(rs):<4} resolved {len(hz):<4} motion {np.median(energy):.2f}  "
                     f"{hz_txt}  frozen {sum(r['freeze_fraction'] > 0.5 for r in rs)}")
    return "\n".join(lines)
