from typing import Dict, List, Tuple

PUNCT = {".", ",", "!", "?", ";", ":"}


def rebuild_segments_from_words(
    stt: Dict,
    *,
    max_duration_s: float = 4.5,
    max_gap_s: float = 0.6,
    min_duration_s: float = 0.4,
    debug: bool = False,
):
    """
    Construit des segments vidéo propres à partir du texte Whisper.
    IMPORTANT :
    - aucune détection de répétition ici
    - aucune logique métier
    - purement structurel
    """

    segments: List[Tuple[float, float]] = []

    for seg in stt.get("segments", []):
        words = seg.get("words", [])
        if not words:
            continue

        cur_start = words[0]["start"]
        last_end = words[0]["end"]

        for w in words[1:]:
            gap = w["start"] - last_end
            dur = w["end"] - cur_start
            is_punct = any(w["word"].endswith(p) for p in PUNCT)

            if gap > max_gap_s or dur > max_duration_s or is_punct:
                if last_end - cur_start >= min_duration_s:
                    segments.append((cur_start, last_end))
                cur_start = w["start"]

            last_end = w["end"]

        if last_end - cur_start >= min_duration_s:
            segments.append((cur_start, last_end))

    return segments
