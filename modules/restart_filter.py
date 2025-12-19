from typing import List, Tuple
from difflib import SequenceMatcher

Word = dict  # {"word": str, "start": float, "end": float}


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def detect_restarts(
    words: List[Word],
    min_words: int = 4,
    sim_threshold: float = 0.9,
    max_gap_s: float = 1.5,
):
    """
    Detect restart zones.
    Returns list of (cut_start, cut_end) to REMOVE.
    """

    cuts = []

    i = 0
    while i < len(words) - min_words:
        # phrase A
        a_words = words[i:i+min_words]
        a_text = " ".join(w["word"].lower() for w in a_words)
        a_end = a_words[-1]["end"]

        # look ahead
        for j in range(i+min_words, len(words) - min_words):
            gap = words[j]["start"] - a_end
            if gap > max_gap_s:
                break

            b_words = words[j:j+min_words]
            b_text = " ".join(w["word"].lower() for w in b_words)

            if similarity(a_text, b_text) >= sim_threshold:
                # CUT the first attempt
                cuts.append((a_words[0]["start"], b_words[0]["start"]))
                i = j
                break

        i += 1

    return cuts
