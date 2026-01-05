def trim_leading_word_repetition(
    stt: dict,
    *,
    max_words: int = 5,
    max_gap_s: float = 1.0,
    debug: bool = True,
    indent: str = "  ",
):
    """
    Supprime les répétitions verbales en début de segment Whisper
    ex: "je je vais" → garde à partir du 2e "je"
    """
    for seg in stt.get("segments", []):
        words = seg.get("words", [])
        if len(words) < 2:
            continue

        base = words[0]["word"].lower()
        cut_idx = None

        for i in range(1, min(len(words), max_words)):
            w = words[i]
            gap = w["start"] - words[0]["end"]

            if w["word"].lower() == base and gap <= max_gap_s:
                cut_idx = i
            else:
                break

        if cut_idx:
            if debug:
                txt = " ".join(w["word"] for w in words[:cut_idx + 1])
                print(f"{indent}✂ repetition trimmed: \"{txt}\"")

            seg["words"] = words[cut_idx:]
            seg["start"] = seg["words"][0]["start"]
            seg["text"] = " ".join(w["word"] for w in seg["words"])
