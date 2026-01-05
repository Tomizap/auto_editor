# modules/audio_repetition_filter.py
import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Tuple, Set


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_FILLERS = {
    "euh", "heu", "bah", "ben", "du coup", "en fait", "genre", "quoi",
    "ok", "okay", "donc", "alors"
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = _strip_accents(s)
    s = _PUNCT_RE.sub(" ", s)

    # remove fillers
    for f in _FILLERS:
        s = re.sub(rf"\b{re.escape(f)}\b", " ", s)

    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def token_set(text: str) -> set:
    return {t for t in text.split() if len(t) > 2}


def is_trivial_segment(text: str, dur_s: float) -> bool:
    words = [w for w in text.split() if len(w) > 2]
    return dur_s < 1.0 or len(words) < 4


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Core repetition detection (UPDATED WITH CORRECTION GUARD)
# ---------------------------------------------------------------------------

def detect_repetition_segments(
    segments: List[Tuple[float, float]],
    texts: List[str],
    lookback: int = 3,
    min_sim: float = 0.78,
    min_keep_s: float = 0.45,
    correction_guard_sim: float = 0.55,
) -> Set[int]:
    """
    Identify indices of segments to DROP due to repetition.

    Rules:
    - Compare only with previous segments (windowed lookback).
    - Use normalized text similarity.
    - DROP only if repetition is obvious AND safe.
    - Correction guard:
        If current segment is followed by a clearly different one,
        we assume human correction -> keep everything.
    - Drop the weaker segment:
        * shorter duration loses
        * if tie, drop the earlier one (keep improved take)
    - Never drop if remaining segment would be < min_keep_s.
    """
    assert len(segments) == len(texts), "segments/texts length mismatch"

    norm_texts = [normalize_text(t) for t in texts]
    drop: Set[int] = set()

    n = len(norm_texts)

    for i in range(n):

        # ------------------------------------------------------------
        # Containment rule:
        # If current segment is mostly contained in the next one,
        # drop the shorter (preview-like) segment.
        # ------------------------------------------------------------
        if i + 1 < n:
            a = norm_texts[i]
            b = norm_texts[i + 1]

            ta = token_set(a)
            tb = token_set(b)

            if ta and tb:
                overlap = len(ta & tb) / len(ta)
                # if most of A is included in B, and A is shorter
                if overlap >= 0.6:
                    si, ei = segments[i]
                    sj, ej = segments[i + 1]
                    if (ei - si) < (ej - sj):
                        drop.add(i)
                        continue


        if i in drop:
            continue

        ti = norm_texts[i]
        if not ti:
            continue

        # ------------------------------------------------------------
        # Priority rule:
        # If two segments are consecutive and highly similar -> DROP
        # ------------------------------------------------------------
        if i > 0:
            prev_txt = norm_texts[i - 1]
            if similarity(ti, prev_txt) >= min_sim:
                # Consecutive repetition (A -> A')
                pass
            else:
                # ------------------------------------------------------------
                # A -> B -> A' pattern handling
                # ------------------------------------------------------------
                if i >= 2:
                    prev_txt = norm_texts[i - 2]
                    mid_txt = norm_texts[i - 1]

                    if similarity(ti, prev_txt) >= min_sim:
                        # Check if middle segment is trivial
                        s_mid, e_mid = segments[i - 1]
                        mid_dur = e_mid - s_mid

                        if is_trivial_segment(mid_txt, mid_dur):
                            # A -> (trivial B) -> A'  => drop A'
                            drop.add(i)
                            continue
                        else:
                            # True correction pattern => keep
                            continue



        for j in range(max(0, i - lookback), i):
            if j in drop:
                continue

            tj = norm_texts[j]
            if not tj:
                continue

            sim = similarity(ti, tj)
            if sim < min_sim:
                continue

            si, ei = segments[i]
            sj, ej = segments[j]
            dur_i = ei - si
            dur_j = ej - sj

            # Decide which one to drop
            if dur_i < dur_j:
                if dur_j >= min_keep_s:
                    drop.add(i)
            elif dur_j < dur_i:
                if dur_i >= min_keep_s:
                    drop.add(j)
            else:
                # same duration -> drop earlier, keep later improved take
                if dur_i >= min_keep_s:
                    drop.add(j)

            break

    return drop


def apply_repetition_filter(
    segments: List[Tuple[float, float]],
    texts: List[str],
    **kwargs
) -> Tuple[List[Tuple[float, float]], List[str]]:
    """
    Apply repetition detection and return filtered segments + texts.
    """
    drop = detect_repetition_segments(segments, texts, **kwargs)

    kept_segments = []
    kept_texts = []

    for idx, (seg, txt) in enumerate(zip(segments, texts)):
        if idx not in drop:
            kept_segments.append(seg)
            kept_texts.append(txt)

    return kept_segments, kept_texts
