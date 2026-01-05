from typing import Dict, Any, List, Optional

STT_PROMPT = (
    "Vidéo de conseil sur l'alternance et le recrutement. "
    "On parle de LinkedIn, entreprise, recruteur, candidat, "
    "motivation, entretien, expérience professionnelle, tuteur."
)


def resolve_whisper_device(requested: str) -> tuple[str, str]:
    if requested != "cuda":
        return "cpu", "int8"

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass

    print("  ⚠ CUDA unavailable → fallback to CPU")
    return "cpu", "int8"


def transcribe_with_words(
    audio_path: str,
    device: str = "cuda",
    model_name: str = "medium",
    initial_prompt: str = STT_PROMPT,
) -> Dict[str, Any]:
    """
    Whisper VERBATIM :
    - aucune suppression d'hésitation
    - word-level timestamps
    - SAFE CPU / CUDA
    - optional initial prompt for context biasing
    """

    from faster_whisper import WhisperModel

    device, compute_type = resolve_whisper_device(device)

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
    )

    segments_out: List[Dict[str, Any]] = []

    # ------------------------------------------------------------
    # Transcription with optional context prompt
    # ------------------------------------------------------------
    segments, _ = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=False,        # CRITIQUE : on ne touche pas à l'audio
        initial_prompt=initial_prompt,
    )

    for seg in segments:
        words = []
        if seg.words:
            for w in seg.words:
                if (w.word or "").strip():
                    words.append({
                        "word": w.word,
                        "start": float(w.start),
                        "end": float(w.end),
                    })

        segments_out.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text or "",
            "words": words,
        })

    return {"segments": segments_out}
