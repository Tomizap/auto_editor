import shutil
import time
import subprocess
from pathlib import Path
from typing import List, Tuple

from modules.video_normalize import normalize_video
from modules.audio_vad import extract_wav_mono16k, vad_segments
from modules.ffmpeg_utils import (
    cut_segment, 
    concat_segments_single_pass
) 
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

MAX_OUTPUT_SECONDS = 180.0

TARGET_W = 1080
TARGET_H = 1920

GAZE_CFG = GazeFilterConfig(
    preset="strict",
    sample_fps=4.0,
    min_stable_s=0.30,
    gap_merge_s=0.40,
    entry_guard_s=0.30,
    entry_score_min=0.65,
    entry_max_bad_frames=2,
    entry_cooldown_s=0.32,
    exit_guard_s=0.45,
    exit_score_min=0.70,
    exit_max_bad_frames=2,
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


# =============================================================================
# PROCESS ONE VIDEO
# =============================================================================

def process_one(video_path: Path) -> None:
    original_stem = video_path.stem

    log_header(f"🎬 PROCESSING VIDEO: {video_path.name}")

    work = TMP_DIR / original_stem
    if work.exists():
        # print("🧹 Cleaning temp directory")
        # shutil.rmtree(work)
        pass
    else:
        work.mkdir(parents=True)

    # -------------------------------------------------------------------------
    # 0) NORMALIZE VIDEO
    # -------------------------------------------------------------------------
    t0 = _now()
    normalized = work / "normalized.mp4"
    if normalized.exists():
        print("▶ Already Cache Normalize video")
    else:
        print("▶ Normalize video")
        normalize_video(str(video_path), str(normalized))
    log_step("Normalize video", t0)

    video_path = normalized

    # -------------------------------------------------------------------------
    # 1) EXTRACT AUDIO (FOR VAD + FIRST STT)
    # -------------------------------------------------------------------------
    t0 = _now()
    wav = work / "audio.wav"
    print("▶ Extract audio (mono / 16kHz) [for VAD + cuts]")
    extract_wav_mono16k(str(video_path), str(wav))
    log_step("Extract audio", t0)

    # -------------------------------------------------------------------------
    # 2) VAD
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

    # Compensation audio → video
    VAD_TIME_BIAS_S = 0.09
    segments = [(max(0.0, s - VAD_TIME_BIAS_S), e) for s, e in segments]

    print(f"  • {len(segments)} segment(s) detected after VAD")
    log_step("VAD", t0)

    if not segments:
        print("⚠ No speech detected → skip")
        return

    # -------------------------------------------------------------------------
    # 3) GAZE FILTER
    # -------------------------------------------------------------------------
    t0 = _now()
    print(f"▶ Gaze filtering (preset={GAZE_CFG.preset})")
    segments = refine_segments_by_gaze(
        video_path=str(video_path),
        segments=segments,
        cfg=GAZE_CFG,
    )
    print(f"  • {len(segments)} segment(s) after gaze filter")
    log_step("Gaze filter", t0)

    if not segments:
        print("⚠ All segments removed by gaze filter → skip")
        return

    # -------------------------------------------------------------------------
    # 4) STT #1 (ONCE) — used ONLY to refine cuts (restarts/fillers)
    # -------------------------------------------------------------------------
    t0 = _now()
    print("▶ STT #1 (word-level) [for cut refinement only]")
    stt = transcribe_with_words(
        audio_path=str(wav),
        device=DEVICE,
        model_name=MODEL_NAME,
    )
    all_words = [w for seg in stt.get("segments", []) for w in seg.get("words", [])]
    print(f"  • stt words: {len(all_words)}")
    log_step("STT #1", t0)

    # -------------------------------------------------------------------------
    # 5) RESTART + FILLER FILTER
    # -------------------------------------------------------------------------
    t0 = _now()
    print("▶ Filtering restarts & fillers")

    clean_segments: List[Tuple[float, float]] = []

    for (start, end) in segments:
        seg_words = [w for w in all_words if start <= w["start"] <= end]

        restart_cuts = detect_restarts(seg_words)
        filler_cuts = detect_fillers(seg_words)
        cuts = restart_cuts + filler_cuts

        if cuts:
            last_cut_end = max(c[1] for c in cuts)
            if last_cut_end < end:
                clean_segments.append((last_cut_end, end))
        else:
            clean_segments.append((start, end))

    segments = clean_segments
    print(f"  • {len(segments)} segment(s) after cleanup")
    log_step("Restart & filler filtering", t0)

    if not segments:
        print("⚠ No segments after cleanup → skip")
        return

    # -------------------------------------------------------------------------
    # 6) CONCAT DIRECT (ONE PASS) ✅
    # -------------------------------------------------------------------------
    t0 = _now()
    print("▶ One-pass trim+concat (fast)")
    
    concat_mp4 = work / "concat.mp4"
    concat_segments_single_pass(
        src=str(video_path),
        segments=segments,
        out=str(concat_mp4),
        target_w=TARGET_W,
        target_h=TARGET_H,
        prefer_nvenc=True,
    )
    
    # Recompute total from segments (accurate)
    total = sum((e - s) for s, e in segments)
    print(f"  • total≈{total:.2f}s")
    log_step("One-pass concat", t0)

    # -------------------------------------------------------------------------
    # 8) AUDIO FROM CONCAT (SOURCE OF TRUTH FOR SUBTITLES)
    # -------------------------------------------------------------------------
    t0 = _now()
    print("▶ Extract audio from CONCAT (mono / 16kHz) [SUBTITLES SOURCE OF TRUTH]")
    wav_concat = work / "audio_concat.wav"
    extract_wav_mono16k(str(concat_mp4), str(wav_concat))
    log_step("Extract concat audio", t0)

    # -------------------------------------------------------------------------
    # 9) STT #2 ON CONCAT (THIS DRIVES SUBTITLES)
    # -------------------------------------------------------------------------
    t0 = _now()
    print("▶ STT #2 on CONCAT (word-level) [SUBTITLES SOURCE OF TRUTH]")
    stt_concat = transcribe_with_words(
        audio_path=str(wav_concat),
        device=DEVICE,
        model_name=MODEL_NAME,  # mets "small" si tu veux accélérer
    )
    print(f"  • concat stt words: {count_words(stt_concat)}")
    log_step("STT #2 (concat)", t0)

    # -------------------------------------------------------------------------
    # 8) Generate Subtitles
    # -------------------------------------------------------------------------
    t0 = _now()
    print("▶ Generate Subtitles")
    
    ass_path = work / "subs.ass" 
    generate_karaoke_ass_tiktok_punchy(
        aligned=stt_concat,
        out_ass_path=ass_path,
    )
    log_step("Generate Subtitles", t0)
    
    # -------------------------------------------------------------------------
    # 8) FINAL RENDER — DRAW TEXT (FAST, GPU)
    # -------------------------------------------------------------------------
    
    t0 = _now()
    print("▶ Final Output")
    
    final_out = OUTPUT_DIR / f"{original_stem}_final.mp4"
    
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

    videos = [
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv"}
    ]

    if not videos:
        print("⚠ No videos found in input/")
        return

    print(f"📁 Found {len(videos)} video(s)")

    for v in videos:
        try:
            process_one(v)
        except Exception:
            print(f"\n❌ ERROR on {v.name}")
            raise


if __name__ == "__main__":
    main()
