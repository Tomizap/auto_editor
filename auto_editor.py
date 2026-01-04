import shutil
import time
import subprocess
from pathlib import Path
from typing import List, Tuple

from modules.input_concat import concat_folder_videos
from modules.video_normalize import normalize_video
from modules.audio_vad import extract_wav_mono16k, vad_segments
from modules.ffmpeg_utils import cut_segment, concat_segments_single_pass
from modules.generate_karaoke_ass import generate_karaoke_ass_tiktok_punchy
from modules.stt import transcribe_with_words
from modules.gaze_filter import refine_segments_by_gaze, GazeFilterConfig
from modules.restart_filter import detect_restarts
from modules.filler_filter import detect_fillers


# =============================================================================
# PATHS
# =============================================================================

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
TMP_DIR = Path("tmp")

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)


# =============================================================================
# GLOBAL CONFIG
# =============================================================================

DEVICE = "cuda"
MODEL_NAME = "medium"

TARGET_W = 1080
TARGET_H = 1920

GAZE_CFG = GazeFilterConfig(
    # Sampling
    sample_fps=4.0,

    # Geometry — permissive (retours caméra fréquents)
    yaw_max_deg=40.0,
    pitch_min_deg=-38.0,
    pitch_max_deg=28.0,

    # Motion — clé pour détecter la lecture
    max_yaw_speed_deg_s=120.0,
    max_pitch_speed_deg_s=135.0,

    # Temporal logic — très important
    min_valid_duration_s=0.15,     # on accepte des regards caméra courts
    max_invalid_gap_s=0.55,         # on tolère la lecture brève
    min_segment_duration_s=0.50,    # sécurité anti micro-cuts

    # Guards — soft
    entry_grace_s=0.35,
    exit_grace_s=0.45,

    # Merge
    merge_gap_s=0.45,
)


# =============================================================================
# UTILS
# =============================================================================

def _now() -> float:
    return time.time()


def log_step(label: str, t0: float) -> None:
    dt = time.time() - t0
    print(f"✓ {label:<45} {dt:7.2f}s")


def log_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def count_words(aligned: dict) -> int:
    return sum(len(s.get("words", [])) for s in aligned.get("segments", []))


def is_valid_video(path: Path) -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


# =============================================================================
# PROCESS PROJECT
# =============================================================================

def process_project(
    video_path: Path,
    project_name: str,
) -> None:

    log_header(f"🎬 PROCESSING PROJECT: {project_name}")

    work = TMP_DIR / project_name
    work.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1) NORMALIZE
    # -------------------------------------------------------------------------
    t0 = _now()
    normalized = work / "normalized.mp4"

    if normalized.exists() and is_valid_video(normalized):
        print("▶ Already Cache Normalize video")
    else:
        print("▶ Normalize video")
        normalize_video(str(video_path), str(normalized))

    log_step("Normalize video", t0)
    video_path = normalized

    # -------------------------------------------------------------------------
    # 2) EXTRACT AUDIO (VAD)
    # -------------------------------------------------------------------------
    t0 = _now()
    wav = work / "audio.wav"
    print("▶ Extract audio (mono / 16kHz)")
    extract_wav_mono16k(str(video_path), str(wav))
    log_step("Extract audio", t0)

    # -------------------------------------------------------------------------
    # 3) VAD
    # -------------------------------------------------------------------------
    t0 = _now()
    print("▶ Voice Activity Detection")

    segments = vad_segments(
        wav_path=str(wav),
        aggressiveness=3,
        frame_ms=20,
        padding_ms=80,
        min_segment_ms=300,
        merge_gap_ms=250,
    )

    segments = [(max(0.0, s - 0.09), e) for s, e in segments]

    print(f"  • {len(segments)} segment(s)")
    log_step("VAD", t0)

    if not segments:
        print("⚠ No speech detected → skip")
        return

    # -------------------------------------------------------------------------
    # 4) GAZE FILTER
    # -------------------------------------------------------------------------
    t0 = _now()
    print("▶ Gaze filtering")

    segments = refine_segments_by_gaze(
        video_path=str(video_path),
        segments=segments,
        cfg=GAZE_CFG,
    )

    print(f"  • {len(segments)} segment(s)")
    log_step("Gaze filter", t0)

    if not segments:
        return

    # -------------------------------------------------------------------------
    # 5) STT #1
    # -------------------------------------------------------------------------
    t0 = _now()
    print("▶ STT #1 (cut refinement)")

    stt = transcribe_with_words(
        audio_path=str(wav),
        device=DEVICE,
        model_name=MODEL_NAME,
    )

    all_words = [
        w for seg in stt.get("segments", [])
        for w in seg.get("words", [])
    ]

    log_step("STT #1", t0)

    # -------------------------------------------------------------------------
    # 6) RESTART + FILLER FILTER
    # -------------------------------------------------------------------------
    t0 = _now()
    print("▶ Restart & filler filtering")

    clean_segments: List[Tuple[float, float]] = []

    for (start, end) in segments:
        seg_words = [w for w in all_words if start <= w["start"] <= end]

        cuts = detect_restarts(seg_words) + detect_fillers(seg_words)

        if cuts:
            last = max(c[1] for c in cuts)
            if last < end:
                clean_segments.append((last, end))
        else:
            clean_segments.append((start, end))

    segments = clean_segments
    log_step("Restart & filler filter", t0)

    if not segments:
        return

    # -------------------------------------------------------------------------
    # 7) ONE-PASS CONCAT
    # -------------------------------------------------------------------------
    t0 = _now()
    print("▶ One-pass trim+concat")

    concat_mp4 = work / "concat.mp4"

    concat_segments_single_pass(
        src=str(video_path),
        segments=segments,
        out=str(concat_mp4),
        target_w=TARGET_W,
        target_h=TARGET_H,
        prefer_nvenc=True,
    )

    log_step("One-pass concat", t0)

    # -------------------------------------------------------------------------
    # 8) STT #2 (SUBTITLES)
    # -------------------------------------------------------------------------
    t0 = _now()
    wav_concat = work / "audio_concat.wav"
    extract_wav_mono16k(str(concat_mp4), str(wav_concat))

    stt_concat = transcribe_with_words(
        audio_path=str(wav_concat),
        device=DEVICE,
        model_name=MODEL_NAME,
    )

    log_step("STT #2", t0)

    # -------------------------------------------------------------------------
    # 9) SUBTITLES
    # -------------------------------------------------------------------------
    t0 = _now()
    ass_path = work / "subs.ass"

    generate_karaoke_ass_tiktok_punchy(
        aligned=stt_concat,
        out_ass_path=ass_path,
    )

    log_step("Generate subtitles", t0)

    # -------------------------------------------------------------------------
    # 10) FINAL RENDER
    # -------------------------------------------------------------------------
    t0 = _now()
    print("▶ Final render")

    project_out = OUTPUT_DIR / project_name
    project_out.mkdir(parents=True, exist_ok=True)

    final_out = project_out / f"{project_name}_final.mp4"

    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(concat_mp4),
        "-vf", f"subtitles={ass_path}:fontsdir=assets/fonts",
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-cq", "19",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "160k",
        str(final_out),
    ], check=True)

    log_step("Final render", t0)
    print(f"\n✅ DONE → {final_out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n🚀 AUTO_EDITOR STARTED\n")

    projects = [
        p for p in INPUT_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ]

    if not projects:
        print("⚠ No valid project folders found in input/")
        return

    print(f"📁 Found {len(projects)} project(s)")

    for project in projects:
        try:
            t0 = _now()

            input_video = concat_folder_videos(
                folder=project,
                out_dir=TMP_DIR / "_input_concat"
            )

            log_step("Input concat", t0)

            process_project(
                video_path=input_video,
                project_name=project.name,
            )

        except RuntimeError:
            print(f"\n❌ ERROR on project {project.name}")
            continue

        except Exception:
            print(f"\n❌ ERROR on project {project.name}")
            raise


if __name__ == "__main__":
    main()
