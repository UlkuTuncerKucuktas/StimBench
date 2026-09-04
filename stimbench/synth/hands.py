import csv
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

MODEL_DIR = os.environ.get("STIMBENCH_MP_MODELS", "")
HAND_PIPS = ((5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19))
POSE_IDX = {"left": (11, 13, 15), "right": (12, 14, 16)}      # shoulder, elbow, wrist


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    d = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / d, -1, 1))))


def _landmarkers():
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    d = Path(MODEL_DIR or Path.home() / ".cache" / "stimbench_mp")
    cpu = mpp.BaseOptions.Delegate.CPU      # the GPU delegate is unavailable on many hosts
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(d / "hand_landmarker.task"), delegate=cpu),
        num_hands=2, running_mode=vision.RunningMode.VIDEO))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(d / "pose_landmarker_lite.task"), delegate=cpu),
        running_mode=vision.RunningMode.VIDEO))
    return mp, hands, pose


def _lag_frames(a: np.ndarray, b: np.ndarray, max_lag: int = 8):
    # lag at which b best follows a; positive means b trails a
    a = a - a.mean(); b = b - b.mean()
    if np.abs(a).max() < 1e-6 or np.abs(b).max() < 1e-6:
        return None
    best, best_r = 0, -2
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[:-lag], b[lag:]) if lag > 0 else (a[-lag:] if lag < 0 else a, b[:len(b) + lag] if lag < 0 else b)
        if len(x) < 10:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if r > best_r:
            best, best_r = lag, r
    return best, best_r


def measure(path: Path, fps: float = 16.0) -> dict:
    import cv2
    mp, hands, pose = _landmarkers()
    cap = cv2.VideoCapture(str(path))
    n = 0
    curls, wrist_flex, elbow_ang, palm_down, wrist_y = [], [], [], [], []
    hands_seen = 0
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hr = hands.detect_for_video(img, t)
        pr = pose.detect_for_video(img, t)
        t += int(1000 / fps)
        frame_curl, frame_flex, frame_elbow, frame_palm, frame_wy = [], [], [], [], []
        pose_pts = None
        if pr.pose_landmarks:
            pose_pts = np.array([[l.x, l.y, l.z] for l in pr.pose_landmarks[0]])
        for h, handed in zip(hr.hand_landmarks, hr.handedness):
            p = np.array([[l.x, l.y, l.z] for l in h])
            frame_curl.append(np.mean([_angle(p[a], p[b], p[c]) for a, b, c in HAND_PIPS]))
            normal = np.cross(p[5] - p[0], p[17] - p[0])
            frame_palm.append(float(normal[1] > 0))
            frame_wy.append(float(p[0][1]))
            if pose_pts is not None:
                # mediapipe's "Left" label is the image-mirrored hand; match by wrist distance instead
                side = min(POSE_IDX, key=lambda s: np.linalg.norm(pose_pts[POSE_IDX[s][2], :2] - p[0][:2]))
                sh, el, wr = (pose_pts[i] for i in POSE_IDX[side])
                frame_flex.append(_angle(el[:2], p[0][:2], p[9][:2]))
                frame_elbow.append(_angle(sh[:2], el[:2], wr[:2]))
        if frame_curl:
            hands_seen += 1
            curls.append(np.mean(frame_curl)); palm_down.append(np.mean(frame_palm)); wrist_y.append(np.mean(frame_wy))
            wrist_flex.append(np.mean(frame_flex) if frame_flex else np.nan)
            elbow_ang.append(np.mean(frame_elbow) if frame_elbow else np.nan)
    cap.release()
    hands.close(); pose.close()
    out = {"frames": n, "hand_detect_rate": round(hands_seen / max(n, 1), 3)}
    if hands_seen < 10:
        return out
    curls = np.array(curls); wf = np.array(wrist_flex); ea = np.array(elbow_ang); wy = np.array(wrist_y)
    out.update({
        "finger_curl_mean": round(float(curls.mean()), 1),        # ~180 straight, ~90 fist
        "finger_curl_sd": round(float(curls.std()), 1),
        "palm_down_fraction": round(float(np.mean(palm_down)), 2),
    })
    ok = ~np.isnan(wf) & ~np.isnan(ea)
    if ok.sum() >= 10:
        out["wrist_flex_amp"] = round(float(np.percentile(wf[ok], 95) - np.percentile(wf[ok], 5)), 1)
        out["wrist_flex_sd"] = round(float(wf[ok].std()), 1)
        out["elbow_amp"] = round(float(np.percentile(ea[ok], 95) - np.percentile(ea[ok], 5)), 1)
        lag = _lag_frames(np.diff(ea[ok]), np.diff(wf[ok]))
        if lag:
            out["wrist_lag_frames"], out["wrist_lag_corr"] = lag[0], round(lag[1], 2)
        out["hand_to_elbow_ratio"] = round(out["wrist_flex_amp"] / max(out["elbow_amp"], 1e-3), 2)
    spec = np.abs(np.fft.rfft(wy - wy.mean()))
    freqs = np.fft.rfftfreq(len(wy), d=1 / fps)
    band = (freqs >= 0.5) & (freqs <= 6)
    if band.any():
        out["wrist_y_peak_hz"] = round(float(freqs[band][np.argmax(spec[band])]), 2)
    return out


def measure_set(root: Path, files: List[str], out_csv: Optional[Path] = None, fps: float = 16.0,
                log=print) -> List[dict]:
    rows = []
    for i, f in enumerate(files, 1):
        try:
            m = measure(Path(root) / f, fps)
        except Exception as e:
            log(f"[{i}/{len(files)}] {f}: failed ({e})")
            continue
        rows.append({"file": f, **m})
        log(f"[{i}/{len(files)}] {f} detect={m.get('hand_detect_rate')} curl={m.get('finger_curl_mean')} "
            f"flex_amp={m.get('wrist_flex_amp')} lag={m.get('wrist_lag_frames')} ratio={m.get('hand_to_elbow_ratio')}")
    if out_csv and rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "file", k))
        with Path(out_csv).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
    return rows
