from pathlib import Path
import os
import shutil
import subprocess
import wave
import struct


# Extensions worth naming, used when scanning a folder for recordings. This is
# not a gate on what can be translated: ffmpeg decides that by inspecting the
# file itself, so formats missing from this set still work, and a recording with
# the wrong extension or none at all works too.
AUDIO_EXTENSIONS = {
    ".3ga", ".aa", ".aac", ".ac3", ".adts", ".aif", ".aifc", ".aiff", ".amr",
    ".ape", ".au", ".awb", ".caf", ".dts", ".dsf", ".eac3", ".flac", ".gsm",
    ".it", ".m4a", ".m4b", ".m4r", ".mka", ".mlp", ".mp2", ".mp3", ".mpc",
    ".mpga", ".oga", ".ogg", ".opus", ".qcp", ".ra", ".ram", ".shn", ".snd",
    ".spx", ".tta", ".voc", ".vox", ".w64", ".wav", ".wma", ".wv",
}
VIDEO_EXTENSIONS = {
    ".3g2", ".3gp", ".asf", ".avi", ".dv", ".f4v", ".flv", ".m2ts", ".m2v",
    ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".mts", ".mxf", ".ogv",
    ".rm", ".rmvb", ".ts", ".vob", ".webm", ".wmv",
}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

# ffmpeg is pointed at local recordings only. Restricting the protocol list
# stops a crafted playlist-style file from making it open network URLs.
FFMPEG_PROTOCOLS = "file,pipe,crypto,subfile"


def safe_suffix(source: Path) -> str:
    """A short, alphanumeric extension safe to build a stored filename from.

    Uploads arrive with arbitrary names, so anything unusual becomes ".bin"
    rather than being trusted into a path.
    """
    suffix = source.suffix.casefold()
    if len(suffix) > 1 and len(suffix) <= 6 and suffix[1:].isalnum():
        return suffix
    return ".bin"


def ffmpeg_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    return env


def media_duration(source: Path) -> float:
    """Length in seconds of any media file, without decoding all of it."""
    if source.suffix.casefold() == ".wav":
        seconds = wav_duration(source)
        if seconds > 0:
            return seconds
    if not shutil.which("ffprobe"):
        raise RuntimeError(
            f"{source.name}: ffprobe is needed to measure this file. "
            "It ships with ffmpeg (apt install ffmpeg / pkg install ffmpeg)."
        )
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        capture_output=True,
        text=True,
        env=ffmpeg_env(),
    )
    value = (completed.stdout or "").strip()
    if completed.returncode or not value:
        raise ValueError(f"{source.name}: could not read the length of this file")
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{source.name}: could not read the length of this file") from None


def extract_clip(source: Path, target: Path, start: float, end: float) -> float:
    """Write one section of a recording as 16 kHz mono PCM WAV."""
    if end <= start:
        raise ValueError(f"clip end ({end}) must be after its start ({start})")
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"{source.name}: ffmpeg is needed to cut clips")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", FFMPEG_PROTOCOLS,
        # Seeking before -i is fast, and stays accurate because we re-encode.
        "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
        "-y", "-i", str(source),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, env=ffmpeg_env())
    if completed.returncode or not target.is_file() or target.stat().st_size == 0:
        detail = (completed.stderr or "").strip().splitlines()
        raise ValueError(f"{source.name}: could not cut {start:.1f}s-{end:.1f}s ({detail[-1] if detail else 'ffmpeg failed'})")
    return wav_duration(target)


def normalize_audio(source: Path, target: Path) -> float:
    """Convert any media ffmpeg can decode to mono, 16 kHz PCM WAV.

    The file extension is a hint, never a gate. Phone recorders label the same
    Opus stream .opus, .ogg, .webm or .bin depending on the app, so the content
    is what decides.
    """
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size == 0:
        raise ValueError(f"{source.name}: the file is empty")

    # A plain PCM WAV is handled in-process so the common case needs no ffmpeg.
    # WAV is only a container, so a WAV holding compressed audio falls through.
    if source.suffix.casefold() == ".wav":
        try:
            return normalize_wav(source, target)
        except (wave.Error, OSError, RuntimeError, struct.error):
            pass

    return normalize_with_ffmpeg(source, target)


def normalize_with_ffmpeg(source: Path, target: Path) -> float:
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            f"{source.name}: ffmpeg is needed to read this recording. "
            "Install it (apt install ffmpeg / pkg install ffmpeg), or upload a plain PCM WAV."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", FFMPEG_PROTOCOLS,
        "-y", "-i", str(source),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav",
        str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, env=ffmpeg_env())
    if completed.returncode or not target.is_file() or target.stat().st_size == 0:
        detail = (completed.stderr or "").strip().splitlines()
        reason = detail[-1] if detail else "ffmpeg could not decode it"
        raise ValueError(f"{source.name}: this file has no audio track that can be read ({reason})")

    duration = wav_duration(target)
    if duration <= 0.0:
        raise ValueError(f"{source.name}: the recording contains no audio")
    return duration


def normalize_wav(source: Path, target: Path) -> float:
    with wave.open(str(source), "rb") as input_file:
        channels = input_file.getnchannels()
        width = input_file.getsampwidth()
        rate = input_file.getframerate()
        frames = input_file.readframes(input_file.getnframes())

    samples = _decode_pcm_samples(frames, width)
    if channels > 1:
        samples = tuple(sum(samples[i:i + channels]) // channels for i in range(0, len(samples), channels))
    if rate != 16000 and samples:
        output_count = max(1, round(len(samples) * 16000 / rate))
        samples = tuple(samples[min(len(samples) - 1, round(i * rate / 16000))] for i in range(output_count))
    frames = struct.pack("<" + "h" * len(samples), *samples)

    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(frames)
    return len(samples) / 16000.0


def _decode_pcm_samples(frames: bytes, width: int) -> tuple[int, ...]:
    if width == 1:
        return tuple((sample - 128) << 8 for sample in frames)
    if width == 2:
        return struct.unpack("<" + "h" * (len(frames) // 2), frames)
    if width == 3:
        samples = []
        for index in range(0, len(frames) - 2, 3):
            value = int.from_bytes(frames[index:index + 3], "little", signed=False)
            if value & 0x800000:
                value -= 0x1000000
            samples.append(max(-32768, min(32767, value >> 8)))
        return tuple(samples)
    if width == 4:
        values = struct.unpack("<" + "i" * (len(frames) // 4), frames)
        return tuple(max(-32768, min(32767, value >> 16)) for value in values)
    raise RuntimeError(f"Unsupported WAV sample width: {width} bytes")


def read_wav_array(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if channels != 1 or width != 2:
        raise RuntimeError(f"Expected mono PCM16 WAV, got {path}")
    samples = struct.unpack("<" + "h" * (len(frames) // 2), frames)
    import numpy as np
    return {"array": np.asarray(samples, dtype=np.float32) / 32768.0, "sampling_rate": rate}


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate() or 1)
    except (wave.Error, OSError):
        return 0.0
