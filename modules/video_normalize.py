import subprocess
from pathlib import Path
from modules.ffmpeg_utils import is_hdr_like, nvenc_available


def _run(cmd):
    subprocess.run(cmd, check=True)


def normalize_video(src: str, out: str) -> None:
    """
    FAST & SAFE normalize (max fast-paths):
    - Full copy when strictly safe
    - Partial copy (audio/video) when possible
    - Re-encode only when a transform is mandatory
    """

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print("\n▶▶ NORMALIZE START")
    print(f"  • Source: {src}")

    # --------------------------------------------------
    # PROBE (single ffprobe pass)
    # --------------------------------------------------
    probe = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=pix_fmt,r_frame_rate,avg_frame_rate,"
            "color_transfer,color_primaries,colorspace",
            "-of", "json",
            src
        ],
        text=True
    )

    import json
    stream = json.loads(probe)["streams"][0]

    r_fps = stream.get("r_frame_rate", "")
    avg_fps = stream.get("avg_frame_rate", "")
    pix_fmt = stream.get("pix_fmt", "")
    trc = stream.get("color_transfer", "")
    prim = stream.get("color_primaries", "")
    cs = stream.get("colorspace", "")

    is_cfr = r_fps == avg_fps and r_fps != ""

    def fps_to_float(v):
        try:
            n, d = v.split("/")
            return float(n) / float(d)
        except Exception:
            return None

    fps = fps_to_float(r_fps)

    # --------------------------------------------------
    # HDR heuristic (stricter)
    # --------------------------------------------------
    hdr = (
        trc in {"smpte2084", "arib-std-b67"}
        or prim == "bt2020"
        or cs == "bt2020nc"
    )

    gpu = nvenc_available()

    print(f"  • HDR detected       : {'YES' if hdr else 'NO'}")
    print(f"  • CFR detected       : {'YES' if is_cfr else 'NO'}")
    print(f"  • FPS                : {fps}")
    print(f"  • Pix fmt            : {pix_fmt}")
    print(f"  • NVENC available    : {'YES' if gpu else 'NO'}")

    # ==================================================
    # FAST PATH 1 — PERFECT COPY
    # ==================================================
    if (
        not hdr
        and is_cfr
        and fps == 30
        and pix_fmt == "yuv420p"
    ):
        print("▶ Decision: PERFECT SDR → FULL STREAM COPY")

        _run([
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-i", src,
            "-map_metadata", "-1",
            "-map_metadata:s:v", "-1",
            "-c", "copy",
            "-movflags", "+faststart",
            out
        ])

        print("✔ Normalize done (perfect copy)\n")
        return

    # ==================================================
    # FAST PATH 2 — SDR CFR (FPS divisible → no fps filter)
    # ==================================================
    if (
        not hdr
        and is_cfr
        and fps is not None
        and fps % 30 == 0
    ):
        print("▶ Decision: SDR CFR → COPY (FPS passthrough)")

        _run([
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-i", src,
            "-map", "0",
            "-vsync", "passthrough",
            "-map_metadata", "-1",
            "-map_metadata:s:v", "-1",
            "-c", "copy",
            "-movflags", "+faststart",
            out
        ])

        print("✔ Normalize done (fps passthrough)\n")
        return

    print("▶ Decision: re-encode REQUIRED")

    # ==================================================
    # FILTERS
    # ==================================================
    if hdr:
        print("▶ Filters: HDR → SDR tonemap")
        vf = (
            "zscale=transfer=linear:primaries=bt2020:matrix=bt2020nc,"
            "tonemap=hable:desat=0,"
            "zscale=transfer=bt709:primaries=bt709:matrix=bt709,"
            "format=yuv420p,"
            "fps=30,setpts=PTS-STARTPTS"
        )
    else:
        print("▶ Filters: SDR normalize")
        vf = "format=yuv420p,fps=30,setpts=PTS-STARTPTS"

    af = "aresample=48000:first_pts=0,asetpts=PTS-STARTPTS"

    # ==================================================
    # VIDEO ENCODER
    # ==================================================
    if gpu:
        print("▶ Encoder: h264_nvenc")
        vcodec = [
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-cq", "19",
            "-b:v", "0",
            "-profile:v", "high",
        ]
    else:
        print("▶ Encoder: libx264")
        vcodec = [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-profile:v", "high",
        ]

    print("▶ Launching ffmpeg…")

    _run([
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", src,

        "-vf", vf,
        "-af", af,
        "-vsync", "cfr",
        "-r", "30",

        "-map", "0:v:0",
        "-map", "0:a:0?",

        "-map_metadata", "-1",
        "-map_metadata:s:v", "-1",

        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",

        *vcodec,

        "-c:a", "aac",
        "-b:a", "160k",
        "-ar", "48000",

        "-movflags", "+faststart",
        out
    ])

    print("✔ Normalize done (re-encoded)\n")

