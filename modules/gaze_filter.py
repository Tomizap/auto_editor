# modules/gaze_filter.py
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import statistics
import math


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class GazeFilterConfig:
    sample_fps: float = 10.0

    max_invalid_gap_s: float = 0.40
    min_segment_duration_s: float = 0.45
    merge_gap_s: float = 0.35

    # 🔑 Reading detection (2D gaze vector)
    reading_window_s: float = 1.0
    min_samples_in_window: int = 8

    dx_min: float = 0.030           # latéral min
    angle_max_deg: float = 60.0     # au-delà = trop vertical
    var_max: float = 0.0030         # stabilité

    smooth_alpha: float = 0.55

    debug: bool = True


# =============================================================================
# MAIN
# =============================================================================

def refine_segments_by_gaze(video_path, segments, cfg):
    if not segments:
        return []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(fps / cfg.sample_fps), 1)

    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path="models/face_landmarker.task"),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
    )

    refined = []
    win_len = max(int(cfg.reading_window_s * cfg.sample_fps), cfg.min_samples_in_window)

    dbg_total = dbg_reading = 0
    dbg_angles = []

    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        cap.set(cv2.CAP_PROP_POS_MSEC, seg_start * 1000)

        t = seg_start
        active_start = last_keep_t = None

        buf = []
        tbuf = []
        ema = None

        while t <= seg_end:
            for _ in range(step - 1):
                cap.grab()
            ret, frame = cap.read()
            if not ret:
                break

            t += step / fps
            dbg_total += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.face_landmarks:
                if active_start is None:
                    active_start = t
                last_keep_t = t
                continue

            gx, gy = _gaze_xy(res.face_landmarks[0])
            if gx is None:
                last_keep_t = t
                continue

            sig = (gx, gy)
            if ema is None:
                ema = sig
            else:
                ema = (
                    cfg.smooth_alpha * sig[0] + (1 - cfg.smooth_alpha) * ema[0],
                    cfg.smooth_alpha * sig[1] + (1 - cfg.smooth_alpha) * ema[1],
                )

            buf.append(ema)
            tbuf.append(t)
            if len(buf) > win_len:
                buf.pop(0)
                tbuf.pop(0)

            reading, angle = _is_reading_2d(buf, cfg)
            if angle is not None:
                dbg_angles.append(angle)

            if reading:
                dbg_reading += 1
                if active_start and last_keep_t and (t - last_keep_t) > cfg.max_invalid_gap_s:
                    if last_keep_t - active_start >= cfg.min_segment_duration_s:
                        refined.append((active_start, last_keep_t))
                    active_start = last_keep_t = None
            else:
                last_keep_t = t
                if active_start is None:
                    active_start = t

        if active_start and last_keep_t:
            if last_keep_t - active_start >= cfg.min_segment_duration_s:
                refined.append((active_start, last_keep_t))

    cap.release()
    out = merge_close_segments(refined, cfg.merge_gap_s)

    if cfg.debug:
        print("GAZE DEBUG")
        print(f"  frames: {dbg_total}")
        print(f"  reading_frames: {dbg_reading}")
        if dbg_angles:
            print(f"  angle p50={sorted(dbg_angles)[len(dbg_angles)//2]:.1f} "
                  f"p90={sorted(dbg_angles)[int(0.9*len(dbg_angles))]:.1f}")

    return out


# =============================================================================
# READING DETECTION (2D)
# =============================================================================

def _is_reading_2d(buf, cfg):
    if len(buf) < cfg.min_samples_in_window:
        return False, None

    xs = [p[0] - 0.5 for p in buf]
    ys = [p[1] - 0.5 for p in buf]

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    if abs(mean_x) < cfg.dx_min:
        return False, None

    angle = abs(math.degrees(math.atan2(abs(mean_y), abs(mean_x))))
    if angle > cfg.angle_max_deg:
        return False, angle

    if statistics.variance(xs) + statistics.variance(ys) > cfg.var_max:
        return False, angle

    return True, angle


# =============================================================================
# GAZE
# =============================================================================

def _gaze_xy(landmarks):
    if len(landmarks) < 478:
        return None, None

    def avg(ids):
        return sum(landmarks[i].x for i in ids) / len(ids), \
               sum(landmarks[i].y for i in ids) / len(ids)

    r_ids = [469, 470, 471, 472]
    l_ids = [474, 475, 476, 477]

    rx, ry = avg(r_ids)
    lx, ly = avg(l_ids)

    gx = (rx + lx) * 0.5
    gy = (ry + ly) * 0.5
    return gx, gy


# =============================================================================
# HELPERS
# =============================================================================

def merge_close_segments(segs, gap):
    if not segs:
        return []
    out = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = out[-1]
        if s - pe <= gap:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out
