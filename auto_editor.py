import time
import subprocess
from pathlib import Path

from modules.input_concat import concat_folder_videos
from modules.video_normalize import normalize_video
from modules.audio_vad import extract_wav_mono16k
from modules.ffmpeg_utils import concat_segments_single_pass, nvenc_available
from modules.generate_karaoke_ass import generate_karaoke_ass_tiktok_punchy
from modules.stt import transcribe_with_words

from modules.audio_silence_cuts import (
    detect_silences_ffmpeg,
    build_segments_from_silences,
)

from modules.audio_repetition_filter import apply_repetition_filter


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
# CONFIG
# =============================================================================

DEVICE = "cuda"
MODEL_NAME = "medium"

TARGET_W = 1080
TARGET_H = 1920


# =============================================================================
# AUDIO CUT CONFIG (SILENCE-DRIVEN)
# =============================================================================

SILENCE_NOISE_DB = -28.0
SILENCE_MIN_DUR_S = 0.08

CUT_SILENCE_OVER_S = 0.25
MERGE_GAP_UNDER_S = 0.18

PRE_PAD_S = 0.08
POST_PAD_S = 0.10

MIN_SEGMENT_S = 0.45
DROP_SEGMENT_UNDER_S = 0.16


# =============================================================================
# UTILS
# =============================================================================

def log_step(label: str, t0: float):
    print(f"✓ {label:<40} {time.time() - t0:6.2f}s")


def ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ])
    return float(out.decode().strip())


def flatten_words(stt):
    words = []
    for seg in stt.get("segments", []):
        for w in seg.get("words", []):
            if {"start", "end", "word"} <= w.keys():
                words.append(w)
    return words


def text_for_segments(words, segments, margin_s=0.25):
    texts = []
    for s, e in segments:
        s_ext = s - margin_s
        e_ext = e + margin_s
        seg_words = [
            w["word"]
            for w in words
            if w["end"] > s_ext and w["start"] < e_ext
        ]
        texts.append(" ".join(seg_words))
    return texts


def is_hard_punct(token: str) -> bool:
    t = token.strip()
    return len(t) > 0 and t[-1] in {".", "!", "?"}

def is_soft_punct(token: str) -> bool:
    t = token.strip()
    return len(t) > 0 and t[-1] in {",", ";", ":"}

def build_punct_segments_from_words(
    words,
    max_duration_s: float = 4.0,
    min_duration_s: float = 0.6,
    max_gap_s: float = 0.9,
    allow_soft_punct_split: bool = True,
):
    """
    Build text-aligned segments from word timestamps, cut on punctuation.
    Returns: (segments, texts) where segments are (start,end).
    """
    if not words:
        return [], []

    segments = []
    texts = []

    seg_start = None
    seg_end = None
    seg_tokens = []
    last_end = None

    def flush():
        nonlocal seg_start, seg_end, seg_tokens
        if seg_start is None or seg_end is None:
            seg_start, seg_end, seg_tokens = None, None, []
            return
        dur = seg_end - seg_start
        if dur >= min_duration_s and seg_tokens:
            segments.append((seg_start, seg_end))
            texts.append(" ".join(seg_tokens).strip())
        seg_start, seg_end, seg_tokens = None, None, []

    for w in words:
        if not {"start", "end", "word"} <= w.keys():
            continue

        ws, we = float(w["start"]), float(w["end"])
        tok = str(w["word"])

        if seg_start is None:
            seg_start = ws
            seg_end = we
            seg_tokens = [tok]
            last_end = we
            continue

        # If there's a big gap, close the current segment (natural boundary)
        if last_end is not None and (ws - last_end) > max_gap_s:
            flush()
            seg_start = ws
            seg_end = we
            seg_tokens = [tok]
            last_end = we
            continue

        # Append token
        seg_tokens.append(tok)
        seg_end = max(seg_end, we)
        last_end = we

        dur = seg_end - seg_start
        hard = is_hard_punct(tok)
        soft = allow_soft_punct_split and is_soft_punct(tok)

        # Cut rules:
        # 1) always cut on hard punctuation
        # 2) optionally cut on soft punctuation if segment is getting long
        # 3) cut if max_duration exceeded (even without punctuation)
        if hard:
            flush()
        elif soft and dur >= (0.65 * max_duration_s):
            flush()
        elif dur >= max_duration_s:
            flush()

    flush()
    return segments, texts


import numpy as np
import soundfile as sf


def segment_rms(wav_path: str, start: float, end: float) -> float:
    audio, sr = sf.read(wav_path)
    s = int(start * sr)
    e = int(end * sr)
    if e <= s:
        return 0.0
    seg = audio[s:e]
    return float(np.sqrt(np.mean(seg ** 2)))


def remove_micro_stutter_segments(
    segments,
    wav_path: str,
    max_seg_dur_s=0.35,
    max_gap_s=0.15,
    rms_similarity=0.15,
):
    cleaned = []
    prev = None

    for seg in segments:
        if prev is None:
            cleaned.append(seg)
            prev = seg
            continue

        s0, e0 = prev
        s1, e1 = seg

        gap = s1 - e0
        dur0 = e0 - s0
        dur1 = e1 - s1

        if (
            dur0 <= max_seg_dur_s
            and dur1 <= max_seg_dur_s
            and gap <= max_gap_s
        ):
            rms0 = segment_rms(wav_path, s0, e0)
            rms1 = segment_rms(wav_path, s1, e1)

            if abs(rms0 - rms1) / max(rms0, rms1, 1e-6) < rms_similarity:
                # Drop the first micro-stutter segment
                cleaned[-1] = seg
                prev = seg
                continue

        cleaned.append(seg)
        prev = seg

    return cleaned



# =============================================================================
# PROCESS PROJECT
# =============================================================================

def process_project(video_path: Path, project_name: str):

    print("\n" + "=" * 60)
    print(f"🎬 PROCESSING {project_name}")
    print("=" * 60)

    work = TMP_DIR / project_name
    work.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1) NORMALIZE
    # -------------------------------------------------------------------------
    t0 = time.time()
    normalized = work / "normalized.mp4"
    normalize_video(str(video_path), str(normalized))
    log_step("Normalize video", t0)

    # -------------------------------------------------------------------------
    # 2) EXTRACT AUDIO
    # -------------------------------------------------------------------------
    t0 = time.time()
    wav = work / "audio.wav"
    extract_wav_mono16k(str(normalized), str(wav))
    log_step("Extract audio", t0)

    # -------------------------------------------------------------------------
    # 3) AUDIO CUT (SILENCE-BASED)
    # -------------------------------------------------------------------------
    t0 = time.time()
    dur = ffprobe_duration(normalized)

    silences = detect_silences_ffmpeg(
        wav_path=str(wav),
        noise_db=SILENCE_NOISE_DB,
        min_silence_dur_s=SILENCE_MIN_DUR_S,
    )

    segments_src = build_segments_from_silences(
        duration_s=dur,
        silences=silences,
        cut_silence_over_s=CUT_SILENCE_OVER_S,
        merge_gap_under_s=MERGE_GAP_UNDER_S,
        pre_pad_s=PRE_PAD_S,
        post_pad_s=POST_PAD_S,
        min_segment_s=MIN_SEGMENT_S,
        drop_segment_under_s=DROP_SEGMENT_UNDER_S,
    )

    segments_src = remove_micro_stutter_segments(
        segments_src,
        wav_path=str(wav),
    )

    log_step("Silence cut → segments", t0)
    print(f"  • segments (src): {len(segments_src)}")

    if not segments_src:
        print("⚠ No segments after silence cut")
        return

    # -------------------------------------------------------------------------
    # 4) CONCAT ROUGH
    # -------------------------------------------------------------------------
    t0 = time.time()
    concat_rough = work / "concat.mp4"

    concat_segments_single_pass(
        src=str(normalized),
        segments=segments_src,
        out=str(concat_rough),
        target_w=TARGET_W,
        target_h=TARGET_H,
        prefer_nvenc=True,
    )

    log_step("Concat rough", t0)

    # -------------------------------------------------------------------------
    # 5) STT UNIQUE FOR REPETITIONS
    # -------------------------------------------------------------------------
    t0 = time.time()
    wav_concat = work / "concat.wav"
    extract_wav_mono16k(str(concat_rough), str(wav_concat))

    stt = transcribe_with_words(
        audio_path=str(wav_concat),
        device=DEVICE,
        model_name=MODEL_NAME,
    )

    log_step("STT concat (unique)", t0)

    # -------------------------------------------------------------------------
    # 6) REPETITION FILTER (ON CONCAT TIMELINE)
    # -------------------------------------------------------------------------
    t0 = time.time()

    words = flatten_words(stt)

    # Rebuild segments based on punctuation (no margin needed)
    segments_concat, texts = build_punct_segments_from_words(
        words,
        max_duration_s=4.0,
        min_duration_s=0.6,
        max_gap_s=0.9,
        allow_soft_punct_split=True,
    )

    print("▶ Textual segments (punctuation-based)")
    for i, (seg, txt) in enumerate(zip(segments_concat, texts)):
        print(f"{i:02d} {(seg[1]-seg[0]):.2f}s | {txt}")


    segments_clean, _ = apply_repetition_filter(
        segments=segments_concat,
        texts=texts,
        lookback=3,
        min_sim=0.78,
        min_keep_s=MIN_SEGMENT_S,
    )

    log_step("Repetition filter", t0)
    print(f"  • segments (clean): {len(segments_clean)}")

    if not segments_clean:
        print("⚠ All segments removed by repetition filter")
        return

    # -------------------------------------------------------------------------
    # CONCAT CLEAN (AFTER REPETITION CUT)
    # -------------------------------------------------------------------------
    t0 = time.time()
    concat_clean = work / "concat_clean.mp4"

    concat_segments_single_pass(
        src=str(concat_rough),
        segments=segments_clean,
        out=str(concat_clean),
        target_w=TARGET_W,
        target_h=TARGET_H,
        prefer_nvenc=True,
    )

    log_step("Concat clean", t0)

    # -------------------------------------------------------------------------
    # STT FINAL (ONLY SOURCE OF TRUTH)
    # -------------------------------------------------------------------------
    t0 = time.time()

    wav_clean = work / "concat_clean.wav"
    extract_wav_mono16k(str(concat_clean), str(wav_clean))

    stt_final = transcribe_with_words(
        audio_path=str(wav_clean),
        device=DEVICE,
        model_name=MODEL_NAME,
    )

    log_step("STT final (clean)", t0)

    # -------------------------------------------------------------------------
    # 8) SUBTITLES (ASS)
    # -------------------------------------------------------------------------
    t0 = time.time()
    ass = work / "subs.ass"

    generate_karaoke_ass_tiktok_punchy(
        aligned=stt_final,
        out_ass_path=ass,
    )

    log_step("Generate subtitles", t0)

    # -------------------------------------------------------------------------
    # 9) FINAL RENDER
    # -------------------------------------------------------------------------
    t0 = time.time()

    out_dir = OUTPUT_DIR / project_name
    out_dir.mkdir(exist_ok=True)
    final_out = out_dir / f"{project_name}_final.mp4"

    use_nvenc = nvenc_available()

    cmd = [
        "ffmpeg", "-y",
        "-i", str(concat_clean),
        "-vf", f"subtitles={ass}:fontsdir=assets/fonts",
        "-c:a", "aac", "-b:a", "160k",
    ]

    if use_nvenc:
        cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19", "-b:v", "0"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19"]

    cmd.append(str(final_out))
    subprocess.run(cmd, check=True)

    log_step("Final render", t0)
    print(f"✅ DONE → {final_out}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n🚀 AUTO_EDITOR STARTED\n")

    projects = [p for p in INPUT_DIR.iterdir() if p.is_dir()]

    for project in projects:
        try:
            input_video = concat_folder_videos(
                folder=project,
                out_dir=TMP_DIR / "_input_concat"
            )
            process_project(input_video, project.name)
        except RuntimeError:
            print(f"❌ ERROR on project {project.name}")
            continue
        except Exception:
            print(f"❌ ERROR on project {project.name}")
            raise


if __name__ == "__main__":
    main()
