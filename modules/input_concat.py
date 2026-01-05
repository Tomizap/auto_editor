import subprocess
from pathlib import Path
from typing import List

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}


def _sanitize_filename(p: Path) -> Path:
    """
    Remove spaces and unsafe characters from filename.
    """
    safe_name = p.name.replace(" ", "")
    safe_name = safe_name.replace("(", "").replace(")", "")
    safe_name = safe_name.replace("[", "").replace("]", "")
    safe_name = safe_name.replace("'", "").replace('"', "")

    if safe_name != p.name:
        new_path = p.with_name(safe_name)
        if not new_path.exists():
            p.rename(new_path)
            return new_path

    return p


def concat_folder_videos(
    folder: Path,
    out_dir: Path,
) -> Path:
    """
    STEP 0 – CONCAT INPUT FOLDER

    - folder : input/PROJECT_X/
    - returns: Path to a SINGLE video
        - 1 video  → passthrough
        - N videos → ffmpeg concat (alphabetical, sanitized)
    """

    if folder.name.startswith("."):
        raise RuntimeError(f"Hidden folder skipped: {folder.name}")

    # --------------------------------------------------
    # SANITIZE FILENAMES (REMOVE SPACES ETC.)
    # --------------------------------------------------
    sanitized: List[Path] = []

    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            sanitized.append(_sanitize_filename(p))

    videos = sorted(sanitized, key=lambda p: p.name.lower())

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
            # path already sanitized → no ambiguity
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
