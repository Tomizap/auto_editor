from dataclasses import dataclass
from typing import List, Tuple
import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass
class GazeFilterConfig:
    sample_fps: float = 4.0

    yaw_max_deg: float = 40.0
    pitch_min_deg: float = -30.0
    pitch_max_deg: float = 22.0

    max_yaw_speed_deg_s: float = 180.0
    max_pitch_speed_deg_s: float = 160.0

    min_valid_duration_s: float = 0.35
    max_invalid_gap_s: float = 0.45
    min_segment_duration_s: float = 0.6

    entry_grace_s: float = 0.30
    exit_grace_s: float = 0.40

    merge_gap_s: float = 0.40


def refine_segments_by_gaze(
    video_path: str,
    segments: List[Tuple[float, float]],
    cfg: GazeFilterConfig,
) -> List[Tuple[float, float]]:

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(fps / cfg.sample_fps), 1)

    base_options = python.BaseOptions(
        model_asset_path="models/face_landmarker.task"
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    refined = []

    for seg_start, seg_end in segments:
        cap.set(cv2.CAP_PROP_POS_MSEC, seg_start * 1000)

        t = seg_start
        last_yaw = last_pitch = last_t = None

        active_start = None
        last_valid_t = None

        while t <= seg_end:
            for _ in range(step - 1):
                cap.grab()

            ret, frame = cap.read()
            if not ret:
                break

            t += step / fps

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            res = landmarker.detect(mp_image)
            if not res.face_landmarks:
                continue

            yaw, pitch = estimate_head_pose(res.face_landmarks[0])

            yaw_speed = pitch_speed = 0.0
            if last_yaw is not None:
                dt = max(t - last_t, 1e-6)
                yaw_speed = abs(yaw - last_yaw) / dt
                pitch_speed = abs(pitch - last_pitch) / dt

            # --- YAW INERTIA ---
            yaw_inertia = 3
            if last_yaw is not None and yaw_speed > 30.0:
                direction = math.copysign(1.0, yaw - last_yaw)
                yaw_inertia += direction

            yaw_inertia = max(min(yaw_inertia, 5.0), -5.0)

            reading_like = abs(yaw_inertia) < 1.5

            valid = is_valid_gaze(
                yaw, pitch, yaw_speed, pitch_speed, cfg
            )

            valid = valid and not reading_like

            last_yaw, last_pitch, last_t = yaw, pitch, t

            if valid:
                last_valid_t = t
                if active_start is None:
                    active_start = t
            else:
                if active_start and last_valid_t:
                    if t - last_valid_t > cfg.max_invalid_gap_s:
                        if last_valid_t - active_start >= cfg.min_segment_duration_s:
                            refined.append((active_start, last_valid_t))
                        active_start = None
                        last_valid_t = None

        if active_start and last_valid_t:
            if last_valid_t - active_start >= cfg.min_segment_duration_s:
                refined.append((active_start, last_valid_t))

    cap.release()
    return merge_close_segments(refined, cfg.merge_gap_s)


def is_valid_gaze(yaw, pitch, yaw_speed, pitch_speed, cfg: GazeFilterConfig):
    if abs(yaw) > cfg.yaw_max_deg:
        return False
    if not (cfg.pitch_min_deg <= pitch <= cfg.pitch_max_deg):
        return False
    if yaw_speed > cfg.max_yaw_speed_deg_s:
        return False
    if pitch_speed > cfg.max_pitch_speed_deg_s:
        return False
    return True


def estimate_head_pose(landmarks):
    nose = landmarks[1]
    left = landmarks[33]
    right = landmarks[263]

    dx = right.x - left.x
    dy = nose.y - (left.y + right.y) / 2

    yaw = math.degrees(math.atan2(dx, 1.0))
    pitch = -math.degrees(math.atan2(dy, 1.0))
    return yaw, pitch


def merge_close_segments(segs, gap):
    if not segs:
        return []
    out = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = out[-1]
        if s - pe <= gap:
            out[-1] = (ps, e)
        else:
            out.append((s, e))
    return out
