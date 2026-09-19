"""Turn whatever the phone recorded into 16 kHz mono samples.

Android hands out recordings in whatever the recorder felt like: m4a from the
voice recorder, opus in an ogg from WhatsApp, 3gp from older phones, sometimes
a video. ffmpeg reads all of them, so when it is installed nothing is rejected.

When it is not installed, plain WAV still works through the standard library
alone. That matters: it keeps the offline tier usable on a phone where only
Python was installed.
"""

from __future__ import annotations

import array
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

TARGET_RATE = 16_000
#: ffmpeg is only ever pointed at local files here. Narrowing the protocol list
#: stops a crafted playlist-shaped upload from making it open network URLs.
FFMPEG_PROTOCOLS = "file,pipe,crypto,subfile"


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def ffmpeg_env() -> dict[str, str]:
    env = os.environ.copy()
    # Termux and desktop Linux disagree about LD_LIBRARY_PATH; letting ours leak
    # into ffmpeg is a reliable way to make it fail to start.
    env.pop("LD_LIBRARY_PATH", None)
    return env


def to_wav(source: Path, target: Path) -> Path:
    """A 16 kHz mono WAV of the recording, converting only when needed."""
    if not have_ffmpeg():
        if source.suffix.casefold() != ".wav":
            raise RuntimeError(
                f"{source.name}: ffmpeg is needed to read this format. "
                "Install it with: pkg install ffmpeg"
            )
        return source
    completed = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-protocol_whitelist", FFMPEG_PROTOCOLS,
            "-y", "-i", str(source),
            "-vn", "-ac", "1", "-ar", str(TARGET_RATE), "-f", "wav",
            str(target),
        ],
        capture_output=True,
        env=ffmpeg_env(),
    )
    if completed.returncode or not target.is_file():
        detail = (completed.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(
            f"{source.name}: could not read this recording"
            + (f" — {detail[-1]}" if detail else "")
        )
    return target


def _decode_pcm(frames: bytes, width: int) -> list[int]:
    """Signed integer samples from raw PCM of 8, 16, 24 or 32 bits.

    Written by hand because the audioop module that used to do this was removed
    from the standard library in Python 3.13, and the whole point of this tier
    is that it runs on a phone where nothing has been pip-installed.
    """
    if width == 2:
        values = array.array("h")
        values.frombytes(frames[: len(frames) // 2 * 2])
        if sys.byteorder == "big":
            values.byteswap()  # WAV is little-endian whatever the CPU is
        return list(values)
    if width == 1:
        # 8-bit WAV is unsigned, centred on 128, and scaled up to 16-bit range.
        return [(byte - 128) << 8 for byte in frames]
    if width == 4:
        values = array.array("i")
        values.frombytes(frames[: len(frames) // 4 * 4])
        if sys.byteorder == "big":
            values.byteswap()
        return [value >> 16 for value in values]
    if width == 3:
        out = []
        for offset in range(0, len(frames) - 2, 3):
            value = frames[offset] | (frames[offset + 1] << 8) | (frames[offset + 2] << 16)
            if value & 0x800000:
                value -= 0x1000000
            out.append(value >> 8)
        return out
    raise RuntimeError(f"unsupported WAV sample width: {width * 8}-bit")


def _to_mono(values: list[int], channels: int) -> list[int]:
    if channels <= 1:
        return values
    usable = len(values) - len(values) % channels
    return [
        sum(values[index:index + channels]) // channels
        for index in range(0, usable, channels)
    ]


def _resample(values: list[float], source_rate: int, target_rate: int) -> list[float]:
    """Linear interpolation to the target rate.

    Not as clean as a windowed filter, but speech recognition is unbothered by
    the difference and this costs no dependency.
    """
    if source_rate == target_rate or not values:
        return values
    ratio = source_rate / target_rate
    count = int(len(values) / ratio)
    out = []
    for index in range(count):
        position = index * ratio
        left = int(position)
        right = min(left + 1, len(values) - 1)
        weight = position - left
        out.append(values[left] * (1.0 - weight) + values[right] * weight)
    return out


def read_samples(path: Path) -> tuple[list[float], int]:
    """Mono float samples in -1..1, plus the sample rate.

    Uses only the wave module, so the offline tier keeps working on a phone
    where nothing but Python itself was installed.
    """
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    values = _to_mono(_decode_pcm(frames, width), channels)
    samples = [value / 32768.0 for value in values]
    if rate != TARGET_RATE:
        samples = _resample(samples, rate, TARGET_RATE)
        rate = TARGET_RATE
    return samples, rate


def duration(path: Path) -> float:
    """Length in seconds of a WAV, without decoding the audio."""
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else 0.0
    except (wave.Error, OSError, EOFError):
        return 0.0


def rms(samples) -> float:
    if not len(samples):
        return 0.0
    total = sum(value * value for value in samples)
    return (total / len(samples)) ** 0.5


# Kept so a caller can name what it expects to see; ffmpeg decides what actually
# works by inspecting the file, so a recording with an odd extension is fine.
AUDIO_EXTENSIONS = {
    ".3ga", ".aac", ".amr", ".aiff", ".au", ".caf", ".flac", ".m4a", ".m4b",
    ".mka", ".mp2", ".mp3", ".oga", ".ogg", ".opus", ".ra", ".spx", ".wav",
    ".wma", ".wv",
}
VIDEO_EXTENSIONS = {
    ".3gp", ".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg",
    ".ogv", ".ts", ".webm", ".wmv",
}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def safe_suffix(name: str) -> str:
    """A short alphanumeric extension safe to build a filename from."""
    suffix = Path(name).suffix.casefold()
    if 1 < len(suffix) <= 6 and suffix[1:].isalnum():
        return suffix
    return ".bin"
