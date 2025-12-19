import subprocess
from pathlib import Path
from typing import Optional, Tuple, List


# ASS_FONT_SIZE = 64          # DOIT matcher ton ASS
# CHAR_WIDTH_FACTOR = 0.55
# EMOJI_PADDING_X = 12

# SUBTITLE_BASE_Y = 0.80     # 80% de la hauteur écran (TikTok classique)
# EMOJI_Y_OFFSET = -6  

# # ------------

# ASS_MARGIN_V = 200          # DOIT matcher Style MarginV dans ton ASS
# ASS_FONT_SIZE = 86          # DOIT matcher Style Fontsize dans ton ASS
# EMOJI_PADDING_X = 16        # petit espace vers la droite
# EMOJI_Y_OFFSET = -6         # micro-ajustement vertical

# =========================
# Core runner
# =========================

def _run(cmd: List[str]) -> None:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{p.stderr}"
        )


# =========================
# ffprobe helpers
# =========================

def ffprobe_json(path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    import json
    return json.loads(p.stdout)


def get_duration(path: str) -> float:
    """
    Durée la plus fiable possible :
    - préfère la durée du stream vidéo si dispo
    - sinon durée container
    """
    try:
        j = ffprobe_json(path)
        streams = j.get("streams", [])
        v = next((s for s in streams if s.get("codec_type") == "video"), None)
        if v and v.get("duration"):
            return float(v["duration"])
        return float(j.get("format", {}).get("duration", 0.0))
    except Exception:
        return 0.0


def get_video_stream_info(path: str) -> dict:
    j = ffprobe_json(path)
    streams = j.get("streams", [])
    return next((s for s in streams if s.get("codec_type") == "video"), {})


# =========================
# HDR detection
# =========================

def is_hdr_like(path: str) -> bool:
    v = get_video_stream_info(path)

    pix = (v.get("pix_fmt") or "").lower()
    cs  = (v.get("color_space") or "").lower()
    tr  = (v.get("color_transfer") or "").lower()
    pr  = (v.get("color_primaries") or "").lower()

    if "10" in pix:
        return True
    if "bt2020" in cs or "bt2020" in pr:
        return True
    if "arib-std-b67" in tr or "smpte2084" in tr:
        return True
    if "dovi" in str(v).lower():
        return True

    return False


# =========================
# NVENC availability
# =========================

def nvenc_available() -> bool:
    try:
        p = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-f", "lavfi",
                "-i", "testsrc=duration=0.2:size=256x256",  # ⬅️ TAILLE SAFE
                "-c:v", "h264_nvenc",
                "-f", "null", "-"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return p.returncode == 0
    except Exception:
        return False


# =========================
# HDR → SDR FILTER (GOLD)
# =========================

def build_hdr_to_sdr_filter() -> str:
    return (
        "zscale=transfer=linear:primaries=bt2020:matrix=bt2020nc,"
        "tonemap=tonemap=hable:desat=0,"
        "zscale=transfer=bt709:primaries=bt709:matrix=bt709,"
        "format=yuv420p"
    )


# =========================
# Segment cutting (FIX DRIFT)
# =========================

def cut_segment(
    src: str,
    start: float,
    end: float,
    out: str,
    prefer_nvenc: bool = True,
    force_sdr: Optional[bool] = None
) -> None:
    """
    Drift-safe cut (GPU-accelerated when possible)
    - Exact timing
    - Stable audio clock
    - HDR-safe fallback
    """
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    dur = max(0.01, end - start)
    hdr = is_hdr_like(src) if force_sdr is None else force_sdr

    # --------------------------------------------------
    # FAST PATH — GPU (SDR only)
    # --------------------------------------------------
    if not hdr and prefer_nvenc and nvenc_available():
        try:
            _run([
                "ffmpeg", "-y",
                "-hide_banner", "-loglevel", "error",

                # decode + seek
                "-ss", f"{start:.3f}",
                "-i", src,
                "-t", f"{dur:.3f}",

                # reset clocks
                "-fflags", "+genpts",
                "-avoid_negative_ts", "make_zero",

                # video
                "-vf", "setpts=PTS-STARTPTS",
                "-c:v", "h264_nvenc",
                "-preset", "p5",
                "-cq", "19",
                "-b:v", "0",
                "-profile:v", "high",
                "-pix_fmt", "yuv420p",

                # audio
                "-af", "aresample=48000:first_pts=0,asetpts=PTS-STARTPTS",
                "-c:a", "aac",
                "-b:a", "160k",
                "-ar", "48000",

                # metadata
                "-map_metadata", "-1",
                "-map_metadata:s:v", "-1",
                "-movflags", "+faststart",

                out
            ])
            return
        except Exception:
            pass  # ⬅️ fallback CPU

    # --------------------------------------------------
    # SAFE PATH — CPU (HDR or fallback)
    # --------------------------------------------------
    vf_core = build_hdr_to_sdr_filter() if hdr else "format=yuv420p"
    vf = f"{vf_core},setpts=PTS-STARTPTS"
    af = "aresample=48000:first_pts=0,asetpts=PTS-STARTPTS"

    _run([
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",

        "-i", src,
        "-ss", f"{start:.3f}",
        "-t", f"{dur:.3f}",

        "-vf", vf,
        "-af", af,

        "-map", "0:v:0",
        "-map", "0:a:0?",

        "-fflags", "+genpts",
        "-avoid_negative_ts", "make_zero",

        "-map_metadata", "-1",
        "-map_metadata:s:v", "-1",

        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",

        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",

        "-c:a", "aac",
        "-b:a", "160k",
        "-ar", "48000",

        "-movflags", "+faststart",
        out
    ])



# =========================
# Concatenation with crossfade (FIX DRIFT)
# =========================

def concat_fast(
    clips: List[str],
    out: str,
    prefer_nvenc: bool = True
) -> None:
    """
    ULTRA FAST concat:
    - No xfade
    - No acrossfade
    - Concat demuxer
    - Stream-copy friendly
    """

    if not clips:
        raise ValueError("Empty clips list")

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    list_file = Path(out).with_suffix(".txt")
    list_file.write_text(
        "\n".join(f"file '{Path(c).resolve()}'" for c in clips),
        encoding="utf-8"
    )

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-map", "0:v",
        "-map", "0:a?",
        "-movflags", "+faststart",
    ]

    if prefer_nvenc and nvenc_available():
        cmd += [
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-cq", "19",
            "-b:v", "0",
            "-pix_fmt", "yuv420p",
        ]
    else:
        cmd += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
        ]

    cmd += [
        "-c:a", "aac",
        "-b:a", "192k",
        out
    ]

    _run(cmd)



# =========================
# Final vertical render (ASS or RGBA overlay)
# =========================

TWEMOJI_DIR = Path("assets/twemoji")

WORD_TO_EMOJI_KEY = {
    "important": "warning",
    "attention": "warning",
    "argent": "money",
    "business": "briefcase",
    "rapide": "bolt",
    "simple": "check",
    "efficace": "fire",
}

EMOJI_UNICODE = {
    "warning": "⚠️",
    "money": "💰",
    "briefcase": "💼",
    "bolt": "⚡",
    "check": "✅",
    "fire": "🔥",
}

# DOIT matcher ton ASS (Style)
ASS_MARGIN_V = 200
ASS_FONT_SIZE = 86

# Réglages visuels
EMOJI_PADDING_X = 16
EMOJI_Y_OFFSET = -6


def emoji_to_twemoji_filename(emoji: str) -> str:
    cps = []
    for ch in emoji:
        if ord(ch) == 0xFE0F:  # strip VS16
            continue
        cps.append(f"{ord(ch):x}")
    return "-".join(cps) + ".png"


def burn_ass_with_emojis(
    src: str,
    out: str,
    ass_path: str,
    aligned: dict,
    crop=None,
    target_w=1080,
    target_h=1920,
    emoji_size=96,
):
    from subprocess import run

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    ff_inputs = []
    filters = []

    input_idx = 1      # 0 = video
    last_label = "v0"  # label courant

    # --------------------------------------------------
    # 1) Base vidéo (reframe)
    # --------------------------------------------------
    if crop:
        x, y, w, h = crop
        filters.append(
            f"[0:v]crop={w}:{h}:{x}:{y},scale={target_w}:{target_h}[v0]"
        )
    else:
        filters.append(
            f"[0:v]scale={target_w}:{target_h}[v0]"
        )

    # --------------------------------------------------
    # 2) Position emoji : stable, alignée sur la ligne ASS
    # --------------------------------------------------
    # Texte ASS aligné en bas avec MarginV => baseline ≈ H - ASS_MARGIN_V
    # On place l’emoji à peu près au niveau du texte (ajustable)
    emoji_x = f"(W/2)+220+{EMOJI_PADDING_X}"
    emoji_y = f"(H-{ASS_MARGIN_V})-({emoji_size})+{EMOJI_Y_OFFSET}"

    # --------------------------------------------------
    # 3) Emojis (overlay PNG direct, pas de calcul mot)
    # --------------------------------------------------
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            word = (w.get("word") or "").lower()
            if not word:
                continue

            # keyword -> emoji key
            match_key = None
            for k in WORD_TO_EMOJI_KEY:
                if k in word:
                    match_key = k
                    break
            if not match_key:
                continue

            emoji_char = EMOJI_UNICODE.get(WORD_TO_EMOJI_KEY[match_key])
            if not emoji_char:
                continue

            png = TWEMOJI_DIR / emoji_to_twemoji_filename(emoji_char)
            if not png.exists():
                continue

            start = float(w["start"])
            end = float(w["end"])
            if end <= start:
                continue

            ff_inputs += ["-i", str(png)]

            filters.append(
                f"[{last_label}][{input_idx}:v]"
                f"overlay={emoji_x}:{emoji_y}:"
                f"enable='between(t,{start},{end})'"
                f"[v{input_idx}]"
            )

            last_label = f"v{input_idx}"
            input_idx += 1

    # --------------------------------------------------
    # 4) ASS LAST — ALWAYS
    # --------------------------------------------------
    filters.append(f"[{last_label}]ass='{ass_path}'[vout]")
    filter_complex = ";".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", src,
        *ff_inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a?",
        "-shortest",
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-cq", "19",
        "-b:v", "0",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "160k",
        out
    ]

    run(cmd, check=True)


def overlay_rgba_on_video(src: str, overlay: str, out: str, target_w: int = 1080, target_h: int = 1920) -> None:
    """
    Robust RGBA overlay compositing (works across SDR/HDR, yuvj420p, timebase quirks).
    Forces PTS alignment, formats, and scaling to avoid 'invisible overlay' on some inputs.
    """
    from subprocess import run
    from pathlib import Path

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    filter_complex = (
        f"[0:v]scale={target_w}:{target_h},format=yuv420p,setpts=PTS-STARTPTS[base];"
        f"[1:v]scale={target_w}:{target_h},format=rgba,setpts=PTS-STARTPTS[ov];"
        f"[base][ov]overlay=0:0:format=auto:eof_action=pass[vout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-fflags", "+genpts",
        "-i", src,
        "-i", overlay,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a?",
        "-shortest",
        "-movflags", "+faststart",
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-cq", "19",
        "-b:v", "0",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "160k",
        out
    ]

    run(cmd, check=True)


def concat_segments_single_pass(
    src: str,
    segments: list[tuple[float, float]],
    out: str,
    target_w: int = 1080,
    target_h: int = 1920,
    prefer_nvenc: bool = True,
):
    """
    One-pass trim + concat (SAFE VERSION).
    Uses pair-wise concat to avoid ffmpeg media-type mismatch.
    """
    from pathlib import Path
    from subprocess import run

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    if not segments:
        raise ValueError("segments is empty")

    fc = []
    v_labels = []
    a_labels = []

    # 1️⃣ Trim all segments
    for i, (s, e) in enumerate(segments):
        s = float(s)
        e = float(e)

        fc.append(
            f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[v{i}]"
        )
        fc.append(
            f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}]"
        )

        v_labels.append(f"[v{i}]")
        a_labels.append(f"[a{i}]")

    # 2️⃣ Pairwise concat (v+a order is CRITICAL)
    concat_inputs = []
    for i in range(len(segments)):
        concat_inputs.append(v_labels[i])
        concat_inputs.append(a_labels[i])

    fc.append(
        f"{''.join(concat_inputs)}"
        f"concat=n={len(segments)}:v=1:a=1[vcat][acat]"
    )

    # 3️⃣ Final scale AFTER concat
    fc.append(
        f"[vcat]scale={target_w}:{target_h},format=yuv420p[vout]"
    )

    filter_complex = ";".join(fc)

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[acat]",
        "-movflags", "+faststart",
    ]

    if prefer_nvenc:
        cmd += [
            "-c:v", "h264_nvenc",
            "-preset", "p6",
            "-cq", "21",
            "-b:v", "0",
        ]
    else:
        cmd += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
        ]

    cmd += [
        "-c:a", "aac",
        "-b:a", "160k",
        out
    ]

    run(cmd, check=True)

