import re
from typing import List, Tuple

# Patterns de fillers vocaux (beaucoup plus robustes)
FILLER_REGEX = re.compile(
    r"^(e+u+h+|h+e+u+|h+u+m+|m+m+|h+m+|u+h+)$"
)

# Mots souvent répétés inutilement
COMMON_REPEATABLE = {
    "je", "tu", "il", "elle", "on", "nous", "vous",
    "donc", "alors", "bah", "ben", "du", "de", "des",
}

def detect_fillers(
    words: List[dict],
    max_filler_s: float = 0.6,
    max_noise_s: float = 0.9,
    repeat_window_s: float = 0.5,
    short_word_s: float = 0.35,
) -> List[Tuple[float, float]]:
    """
    Returns list of (start, end) intervals to REMOVE
    """
    cuts: List[Tuple[float, float]] = []

    for i, w in enumerate(words):
        text = w["word"].lower().strip()
        start = w["start"]
        end = w["end"]
        dur = end - start

        # --------------------------------------------------
        # A) Fillers vocaux (regex)
        # --------------------------------------------------
        if dur <= max_filler_s and FILLER_REGEX.match(text):
            cuts.append((start, end))
            continue

        # --------------------------------------------------
        # B) Bruits non verbaux (STT)
        # --------------------------------------------------
        if text.startswith("[") and dur <= max_noise_s:
            cuts.append((start, end))
            continue

        # --------------------------------------------------
        # C) Répétitions immédiates
        # --------------------------------------------------
        if i > 0:
            prev = words[i - 1]
            prev_text = prev["word"].lower().strip()

            if (
                text == prev_text
                and text in COMMON_REPEATABLE
                and (start - prev["end"]) <= repeat_window_s
            ):
                # On coupe la PREMIÈRE occurrence
                cuts.append((prev["start"], prev["end"]))
                continue

        # --------------------------------------------------
        # D) Faux départs très courts
        # --------------------------------------------------
        if dur <= short_word_s and i + 1 < len(words):
            next_w = words[i + 1]
            gap = next_w["start"] - end

            # court + pause + reprise → très souvent inutile
            if gap > 0.25:
                cuts.append((start, end))
                continue

    return merge_intervals(cuts)


# --------------------------------------------------
# Merge utilitaire (important)
# --------------------------------------------------

def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []

    intervals.sort()
    merged = [intervals[0]]

    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe + 0.05:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    return merged
