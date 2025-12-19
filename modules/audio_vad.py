import collections
import wave
from pathlib import Path
from typing import List, Tuple
import subprocess
import webrtcvad

def extract_wav_mono16k(src_video: str, wav_out: str) -> None:
    Path(wav_out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y",
        "-hide_banner",
        "-loglevel", "error",

        "-i", src_video,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-af", "aresample=16000:first_pts=0,asetpts=PTS-STARTPTS",
        "-c:a", "pcm_s16le",

        wav_out
    ], check=True)


def read_wav(path: str) -> Tuple[bytes, int]:
    with wave.open(path, "rb") as wf:
        num_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        if num_channels != 1 or sample_rate != 16000 or sample_width != 2:
            raise ValueError("WAV must be mono 16kHz 16-bit PCM")
        pcm = wf.readframes(wf.getnframes())
    return pcm, sample_rate

def frame_generator(frame_ms: int, audio: bytes, sample_rate: int):
    n = int(sample_rate * (frame_ms / 1000.0) * 2)  # 16-bit => 2 bytes
    offset = 0
    timestamp = 0.0
    duration = (float(n) / 2) / sample_rate
    while offset + n <= len(audio):
        yield timestamp, audio[offset:offset+n]
        timestamp += duration
        offset += n

def vad_segments(
    wav_path: str,
    aggressiveness: int = 2,
    frame_ms: int = 20,
    padding_ms: int = 300,
    min_segment_ms: int = 350,
    merge_gap_ms: int = 200
) -> List[Tuple[float, float]]:
    pcm, sr = read_wav(wav_path)
    vad = webrtcvad.Vad(aggressiveness)

    frames = list(frame_generator(frame_ms, pcm, sr))
    if not frames:
        return []

    num_padding = int(padding_ms / frame_ms)
    ring = collections.deque(maxlen=num_padding)

    triggered = False
    start_t = 0.0
    voiced = []

    for t, frame in frames:
        is_speech = vad.is_speech(frame, sr)

        if not triggered:
            ring.append((t, is_speech))
            num_voiced = sum(1 for _, s in ring if s)
            if num_voiced > 0.9 * ring.maxlen:
                triggered = True
                start_t = ring[0][0]
                ring.clear()
        else:
            ring.append((t, is_speech))
            num_unvoiced = sum(1 for _, s in ring if not s)
            if num_unvoiced > 0.9 * ring.maxlen:
                end_t = t + (frame_ms / 1000.0)
                voiced.append((start_t, end_t))
                triggered = False
                ring.clear()

    if triggered:
        end_t = frames[-1][0] + (frame_ms / 1000.0)
        voiced.append((start_t, end_t))

    # post-process: remove short, merge close
    min_len = min_segment_ms / 1000.0
    gap = merge_gap_ms / 1000.0

    voiced = [(s, e) for s, e in voiced if (e - s) >= min_len]
    if not voiced:
        return []

    merged = [voiced[0]]
    for s, e in voiced[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    return merged
