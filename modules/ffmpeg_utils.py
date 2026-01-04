import subprocess
from pathlib import Path
from typing import Optional, List


# ============================================================
# Core runner
# ============================================================

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


# ============================================================
# ffprobe helpers
# ============================================================

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


def get_video_stream_info(path: str) -> dict:
    j = ffprobe_json(path)
    streams = j.get("streams", [])
    return next((s for s in streams if s.get("codec_type") == "video"), {})


# ============================================================
# HDR detection
# ============================================================

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


# ============================================================
# NVENC REAL availability (ONLY reliable test)
# ============================================================

def nvenc_available() -> bool:
    """
    REAL NVENC test.
    If this returns False, NVENC MUST NOT be used.
    """
    try:
        p = subprocess.run(
            [
                "ffmpeg", "-y",
                "-hide_banner", "-loglevel", "error",
                "-f", "lavfi",
                "-i", "testsrc=size=128x128:rate=1",
                "-t", "0.1",
                "-c:v", "h264_nvenc",
                "-f", "null", "-"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return p.returncode == 0
    except Exception:
        return False


# ============================================================
# HDR → SDR filter
# ============================================================

def build_hdr_to_sdr_filter() -> str:
    return (
        "zscale=transfer=linear:primaries=bt2020:matrix=bt2020nc,"
        "tonemap=tonemap=hable:desat=0,"
        "zscale=transfer=bt709:primaries=bt709:matrix=bt709,"
        "format=yuv420p"
    )


# ============================================================
# Segment cut (SAFE)
# ============================================================

def cut_segment(
    src: str,
    start: float,
    end: float,
    out: str,
    prefer_nvenc: bool = True,
    force_sdr: Optional[bool] = None
) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    dur = max(0.01, end - start)
    hdr = is_hdr_like(src) if force_sdr is None else force_sdr

    use_nvenc = (
        prefer_nvenc
        and not hdr
        and nvenc_available()
    )

    if use_nvenc:
        try:
            _run([
                "ffmpeg", "-y",
                "-hide_banner", "-loglevel", "error",
                "-ss", f"{start:.3f}",
                "-i", src,
                "-t", f"{dur:.3f}",
                "-vf", "setpts=PTS-STARTPTS",
                "-c:v", "h264_nvenc",
                "-preset", "p5",
                "-cq", "19",
                "-b:v", "0",
                "-pix_fmt", "yuv420p",
                "-af", "aresample=48000:first_pts=0,asetpts=PTS-STARTPTS",
                "-c:a", "aac",
                "-b:a", "160k",
                "-movflags", "+faststart",
                out
            ])
            return
        except Exception:
            pass  # HARD fallback CPU

    # CPU SAFE PATH
    vf_core = build_hdr_to_sdr_filter() if hdr else "format=yuv420p"

    _run([
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-ss", f"{start:.3f}",
        "-t", f"{dur:.3f}",
        "-vf", f"{vf_core},setpts=PTS-STARTPTS",
        "-af", "aresample=48000:first_pts=0,asetpts=PTS-STARTPTS",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "160k",
        "-movflags", "+faststart",
        out
    ])


# ============================================================
# One-pass concat (SAFE – NEVER CRASH)
# ============================================================

def concat_segments_single_pass(
    src: str,
    segments: list[tuple[float, float]],
    out: str,
    target_w: int = 1080,
    target_h: int = 1920,
    prefer_nvenc: bool = True,
):
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    if not segments:
        raise ValueError("segments is empty")

    fc = []
    v_labels = []
    a_labels = []

    for i, (s, e) in enumerate(segments):
        fc.append(f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[v{i}]")
        fc.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}]")
        v_labels.append(f"[v{i}]")
        a_labels.append(f"[a{i}]")

    concat_inputs = []
    for i in range(len(segments)):
        concat_inputs += [v_labels[i], a_labels[i]]

    fc.append(
        f"{''.join(concat_inputs)}concat=n={len(segments)}:v=1:a=1[vcat][acat]"
    )
    fc.append(
        f"[vcat]scale={target_w}:{target_h},format=yuv420p[vout]"
    )

    filter_complex = ";".join(fc)

    use_nvenc = prefer_nvenc and nvenc_available()

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", src,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[acat]",
        "-movflags", "+faststart",
    ]

    if use_nvenc:
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

    _run(cmd)
