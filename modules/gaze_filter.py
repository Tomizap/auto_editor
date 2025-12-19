from dataclasses import dataclass
from typing import List, Tuple, Literal
import math
import cv2
import mediapipe as mp


Preset = Literal["soft", "normal", "strict"]


@dataclass
class GazeFilterConfig:
    preset: Preset = "normal"
    sample_fps: float = 6.0

    min_stable_s: float = 0.40
    gap_merge_s: float = 0.40

    # ENTRY GUARD
    entry_guard_s: float = 0.35
    entry_score_min: float = 0.60
    entry_max_bad_frames: int = 2

    # 🔑 NEW: cooldown après fuite de regard
    entry_cooldown_s: float = 0.35

    # EXIT GUARD
    exit_guard_s: float = 0.45
    exit_score_min: float = 0.70
    exit_max_bad_frames: int = 2


PRESETS = {
    "soft": {
        "score_min": 0.45,
        "yaw_max": 42,
        "pitch_min": -38,
        "pitch_max": 26,
        "yaw_speed_max": 200,
        "pitch_speed_max": 180,
        "speed_penalty": 0.45,
    },
    "normal": {
        "score_min": 0.55,
        "yaw_max": 32,
        "pitch_min": -26,
        "pitch_max": 18,
        "yaw_speed_max": 150,
        "pitch_speed_max": 130,
        "speed_penalty": 0.30,
    },
    "strict": {
        "score_min": 0.60,
        "yaw_max": 30,
        "pitch_min": -24,
        "pitch_max": 16,
        "yaw_speed_max": 110,
        "pitch_speed_max": 120,
        "speed_penalty": 0.28,
    },
}


def refine_segments_by_gaze(video_path: str,
                            segments: List[Tuple[float, float]],
                            cfg: GazeFilterConfig) -> List[Tuple[float, float]]:

    params = PRESETS[cfg.preset]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(fps / cfg.sample_fps), 1)

    refined = []

    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False) as face_mesh:

        for seg_start, seg_end in segments:
            cap.set(cv2.CAP_PROP_POS_MSEC, seg_start * 1000)

            t = seg_start
            start_t = seg_start
            end_t = seg_end

            last_yaw = last_pitch = last_t = None
            stable_start = None

            entry_bad = 0
            exit_bad = 0

            # 🔑 NEW
            cooldown_until = seg_start

            while t <= end_t:
                for _ in range(step - 1):
                    cap.grab()

                ret, frame = cap.read()
                if not ret:
                    break

                t += step / fps

                res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not res.multi_face_landmarks:
                    continue

                yaw, pitch = estimate_head_pose(res.multi_face_landmarks[0])

                score = gaze_score(
                    yaw, pitch,
                    params["yaw_max"],
                    params["pitch_min"],
                    params["pitch_max"],
                )

                # vitesse
                if last_yaw is not None:
                    dt = max(t - last_t, 1e-6)
                    yaw_speed = abs(yaw - last_yaw) / dt
                    pitch_speed = abs(pitch - last_pitch) / dt

                    if yaw_speed > params["yaw_speed_max"] or pitch_speed > params["pitch_speed_max"]:
                        score *= params["speed_penalty"]

                        # 🔥 fuite rapide → cooldown
                        cooldown_until = t + cfg.entry_cooldown_s
                        stable_start = None

                last_yaw, last_pitch, last_t = yaw, pitch, t

                # ENTRY WINDOW
                if t < cooldown_until:
                    start_t = t
                    continue

                if (t - seg_start) <= cfg.entry_guard_s:
                    if score < cfg.entry_score_min:
                        entry_bad += 1
                        if entry_bad >= cfg.entry_max_bad_frames:
                            start_t = t
                            stable_start = None
                    continue

                # EXIT WINDOW
                if (seg_end - t) <= cfg.exit_guard_s:
                    if score < cfg.exit_score_min:
                        exit_bad += 1
                        if exit_bad > cfg.exit_max_bad_frames:
                            end_t = t
                            break
                    else:
                        exit_bad = 0
                    continue

                # STABILITÉ
                if score >= params["score_min"]:
                    if stable_start is None:
                        stable_start = t
                else:
                    stable_start = None

                if stable_start and (t - stable_start) >= cfg.min_stable_s:
                    if end_t - start_t >= 0.6:
                        refined.append((start_t, end_t))
                    break

    cap.release()
    return merge_close_segments(refined, cfg.gap_merge_s)


def gaze_score(yaw, pitch, yaw_max, pitch_min, pitch_max):
    yaw_score = max(0.0, 1.0 - abs(yaw) / yaw_max)
    pitch_score = 1.0 if pitch_min <= pitch <= pitch_max else 0.0
    return 0.6 * yaw_score + 0.4 * pitch_score


def estimate_head_pose(landmarks):
    nose = landmarks.landmark[1]
    left = landmarks.landmark[33]
    right = landmarks.landmark[263]

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
