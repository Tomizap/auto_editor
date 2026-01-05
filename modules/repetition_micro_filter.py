from typing import List, Dict, Tuple
from difflib import SequenceMatcher


def _norm(s: str) -> str:
    return (
        s.lower()
         .replace("’", "'")
         .replace(",", " ")
         .replace(".", " ")
         .replace("  ", " ")
         .strip()
    )


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def detect_micro_repetition_cuts(
    segments: List[Dict],
    *,
    min_sim: float = 0.72,
    max_gap_s: float = 0.9,
    lookback: int = 4,
    debug: bool = False,
) -> List[Tuple[float, float]]:
    """
    Détecte les répétitions de début de phrase entre segments Whisper consécutifs.

    Signature STRICTEMENT compatible avec auto_editor.py
    """

    cuts: List[Tuple[float, float]] = []

    if debug:
        print("▶ Repetition micro-filter")
        print(f"  • segments: {len(segments)}")
        print(f"  • min_sim={min_sim} | max_gap_s={max_gap_s} | lookback={lookback}")

    for j in range(1, len(segments)):
        cur = segments[j]
        cur_words = cur.get("words", [])
        if not cur_words:
            continue

        cur_text = " ".join(w["word"] for w in cur_words[:6])

        for i in range(max(0, j - lookback), j):
            prev = segments[i]
            prev_words = prev.get("words", [])
            if not prev_words:
                continue

            gap = cur["start"] - prev["end"]
            if gap > max_gap_s:
                continue

            prev_text = " ".join(w["word"] for w in prev_words[:6])
            score = _sim(prev_text, cur_text)

            if debug:
                print(f"  compare '{prev_text}' → '{cur_text}' = {score:.2f}")

            if score >= min_sim:
                cuts.append((prev["start"], cur["start"]))
                if debug:
                    print(f"  ✂ CUT {prev['start']:.2f}s → {cur['start']:.2f}s")
                break

    return cuts
