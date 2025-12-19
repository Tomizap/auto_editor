from typing import Dict, Any, List

def transcribe_with_words(audio_path: str, device: str = "cuda", model_name: str = "medium") -> Dict[str, Any]:
    """
    Returns dict with:
      {"segments":[{"start","end","text","words":[{"word","start","end"}...]}...]}
    Strategy:
      1) Try whisperx (if installed) for alignment
      2) Else fallback to faster-whisper word_timestamps=True
    """
    # 1) Try whisperx
    try:
        import whisperx  # type: ignore
        # whisperx expects audio array; simplest: use whisperx.load_audio on video/audio path
        audio = whisperx.load_audio(audio_path)
        model = whisperx.load_model(model_name, device=device, compute_type="float16" if device == "cuda" else "int8")
        result = model.transcribe(audio, batch_size=16)

        # Align to get word-level
        # language may be None if auto; whisperx requires language code
        lang = result.get("language", "en")
        align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
        aligned = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)

        # normalize format
        segments = []
        for seg in aligned["segments"]:
            words = seg.get("words", []) or []
            segments.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg.get("text", "").strip(),
                "words": [
                    {
                        "word": (w.get("word") or "").strip(),
                        "start": float(w["start"]) if w.get("start") is not None else float(seg["start"]),
                        "end": float(w["end"]) if w.get("end") is not None else float(seg["end"]),
                    }
                    for w in words
                    if (w.get("word") or "").strip()
                ],
            })
        return {"segments": segments}

    except Exception:
        pass

    # 2) faster-whisper fallback
    from faster_whisper import WhisperModel  # type: ignore

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    segments_out: List[Dict[str, Any]] = []
    segments, _info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=False
    )

    for seg in segments:
        words = []
        if seg.words:
            for w in seg.words:
                if (w.word or "").strip():
                    words.append({
                        "word": w.word.strip(),
                        "start": float(w.start),
                        "end": float(w.end),
                    })
        segments_out.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": (seg.text or "").strip(),
            "words": words
        })

    return {"segments": segments_out}
