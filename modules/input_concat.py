import subprocess
from pathlib import Path
from typing import List


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}



def concat_folder_videos(
    folder: Path,
    out_dir: Path,
) -> Path:
    """
    STEP 0 – CONCAT INPUT FOLDER

    - folder : input/PROJECT_X/
    - returns: Path to a SINGLE video
        - 1 video  → passthrough
        - N videos → ffmpeg concat (alphabetical)
    """

    if folder.name.startswith("."):
        raise RuntimeError(f"Hidden folder skipped: {folder.name}")

    videos: List[Path] = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )

    if not videos:
        raise RuntimeError(f"No video found in {folder}")

    print(f"\n▶ INPUT CONCAT: {folder.name}")
    print(f"  • {len(videos)} video(s) detected")

    # --------------------------------------------------
    # SINGLE VIDEO → PASSTHROUGH
    # --------------------------------------------------
    if len(videos) == 1:
        print("  • Single video → passthrough")
        return videos[0]

    # --------------------------------------------------
    # MULTI VIDEO → CONCAT
    # --------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{folder.name}_input.mp4"
    concat_txt = out_dir / f"{folder.name}_concat.txt"

    with concat_txt.open("w", encoding="utf-8") as f:
        for v in videos:
            f.write(f"file '{v.resolve()}'\n")

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_txt),
        "-c", "copy",
        str(out_path),
    ], check=True)

    concat_txt.unlink()

    print(f"  • Concatenated → {out_path.name}")
    return out_path
