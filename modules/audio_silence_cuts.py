# modules/audio_silence_cuts.py
import re
import subprocess
from pathlib import Path
from typing import List, Tuple


_SILENCE_START_RE = re.compile(r"silence_start:\s*(?P<t>\d+(\.\d+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(?P<t>\d+(\.\d+)?).*silence_duration:\s*(?P<d>\d+(\.\d+)?)")


def detect_silences_ffmpeg(
    wav_path: str,
    noise_db: float = -35.0,
    min_silence_dur_s: float = 0.12,
) -> List[Tuple[float, float]]:
    """
    Detect silences using ffmpeg's silencedetect filter.
    Returns a list of (start_s, end_s).
    """
    wav = Path(wav_path)
    if not wav.exists():
        raise FileNotFoundError(f"Audio file not found: {wav}")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(wav),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence_dur_s}",
        "-f",
        "null",
        "-",
    ]

    # silencedetect prints to stderr
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr = p.stderr.splitlines()

    silences: List[Tuple[float, float]] = []
    cur_start = None

    for line in stderr:
        m1 = _SILENCE_START_RE.search(line)
        if m1:
            cur_start = float(m1.group("t"))
            continue

        m2 = _SILENCE_END_RE.search(line)
        if m2 and cur_start is not None:
            end_t = float(m2.group("t"))
            # duration = float(m2.group("d"))  # available if you want debug
            if end_t > cur_start:
                silences.append((cur_start, end_t))
            cur_start = None

    # If ffmpeg ended while still "in silence", we cannot reliably close it without duration.
    # In practice, this is rare and acceptable for our use case.

    return silences


def build_segments_from_silences(
    duration_s: float,
    silences: List[Tuple[float, float]],
    # Core behavior
    cut_silence_over_s: float = 0.35,   # if a silence is >= this, we split segments (cut it)
    merge_gap_under_s: float = 0.18,    # if a silence is < this, we treat it as natural pause and merge
    # Keep natural rhythm
    pre_pad_s: float = 0.12,            # keep a bit BEFORE speech starts (into silence)
    post_pad_s: float = 0.14,           # keep a bit AFTER speech ends (into silence)
    # Safety
    min_segment_s: float = 0.45,        # drop too short segments to avoid micro-cuts
    drop_segment_under_s: float = 0.16, # hard drop ultra tiny segments
) -> List[Tuple[float, float]]:
    """
    Convert silences into "keep" segments (speech + a bit of surrounding pause).
    Strategy:
      - Build raw speech intervals as gaps between silences
      - Merge across very short silences
      - Split (cut) silences >= cut_silence_over_s
      - Add pads at boundaries for smoother cuts
      - Drop segments that are too short
    """
    if duration_s <= 0:
        return []

    silences = sorted((max(0.0, s), min(duration_s, e)) for s, e in silences if e > s)
    # Merge overlapping/adjacent silences
    merged: List[Tuple[float, float]] = []
    for s, e in silences:
        if not merged:
            merged.append((s, e))
            continue
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    silences = merged

    # Build raw speech chunks from silence gaps: [0..silence0_start], [silence0_end..silence1_start], ...
    speech: List[Tuple[float, float]] = []
    cur = 0.0
    for s, e in silences:
        if s > cur:
            speech.append((cur, s))
        cur = max(cur, e)
    if cur < duration_s:
        speech.append((cur, duration_s))

    if not speech:
        return []

    # Now decide merge/split based on the silence gaps between speech segments
    segments: List[Tuple[float, float]] = []
    seg_s, seg_e = speech[0]

    for i in range(len(speech) - 1):
        a_s, a_e = speech[i]
        b_s, b_e = speech[i + 1]
        gap_s = b_s - a_e  # gap is the silence between them (>=0)

        # If very small silence: keep as a single segment (no cut)
        if gap_s < merge_gap_under_s:
            seg_e = b_e
            continue

        # If medium silence: still merge (keeps cadence), unless you want to cut more aggressively
        if gap_s < cut_silence_over_s:
            seg_e = b_e
            continue

        # Long enough silence: split here (cut it), but keep a tiny natural pause via pads
        # Close current segment with post_pad into the silence
        cut_end = min(a_e + post_pad_s, b_s)  # don't cross into next speech
        # Start next segment with pre_pad into the silence
        next_start = max(b_s - pre_pad_s, a_e)  # don't cross backwards into previous speech

        segments.append((seg_s, cut_end))
        seg_s, seg_e = next_start, b_e

    segments.append((seg_s, seg_e))

    # Clamp + cleanup
    cleaned: List[Tuple[float, float]] = []
    for s, e in segments:
        s = max(0.0, min(duration_s, s))
        e = max(0.0, min(duration_s, e))
        if e <= s:
            continue
        d = e - s
        if d < drop_segment_under_s:
            continue
        if d < min_segment_s:
            continue
        cleaned.append((s, e))

    # Final merge if overlap due to pads
    final: List[Tuple[float, float]] = []
    for s, e in sorted(cleaned):
        if not final:
            final.append((s, e))
            continue
        ps, pe = final[-1]
        if s <= pe:
            final[-1] = (ps, max(pe, e))
        else:
            final.append((s, e))

    return final
