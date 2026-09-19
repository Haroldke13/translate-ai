"""The pipeline identifies recordings by content, not by file extension."""

import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from kikuyu_ai.audio import (
    AUDIO_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
    normalize_audio,
    safe_suffix,
)
from scripts.ingest_voice_corpus import voice_files

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is not installed")


@pytest.fixture
def source_wav(tmp_path: Path) -> Path:
    """A short mono 16 kHz PCM WAV to transcode from."""
    path = tmp_path / "source.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        samples = [int(6000 * ((index // 40) % 2 * 2 - 1)) for index in range(16000)]
        handle.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return path


def encode(source: Path, target: Path, *args: str) -> bool:
    completed = subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", str(source), *args, str(target)],
        capture_output=True,
    )
    return completed.returncode == 0 and target.is_file() and target.stat().st_size > 0


def assert_normalizes(source: Path, target: Path) -> None:
    duration = normalize_audio(source, target)

    assert duration > 0
    with wave.open(str(target), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000


def test_plain_pcm_wav_needs_no_ffmpeg(source_wav: Path, tmp_path: Path):
    assert_normalizes(source_wav, tmp_path / "out.wav")


@needs_ffmpeg
@pytest.mark.parametrize(
    "name, args",
    [
        ("out.mp3", ("-c:a", "libmp3lame")),
        ("out.m4a", ("-c:a", "aac")),
        ("out.ogg", ("-c:a", "libvorbis")),
        ("out.opus", ("-c:a", "libopus")),
        ("out.webm", ("-c:a", "libopus")),
        ("out.flac", ("-c:a", "flac")),
        ("out.aiff", ("-c:a", "pcm_s16be")),
        ("out.caf", ("-c:a", "pcm_s16le")),
        ("out.mp4", ("-c:a", "aac")),
        ("out.mkv", ("-c:a", "libopus")),
    ],
)
def test_common_formats_are_accepted(source_wav: Path, tmp_path: Path, name, args):
    encoded = tmp_path / name
    if not encode(source_wav, encoded, *args):
        pytest.skip(f"this ffmpeg build cannot encode {name}")

    assert_normalizes(encoded, tmp_path / "clean.wav")


@needs_ffmpeg
def test_recording_with_no_extension_is_accepted(source_wav: Path, tmp_path: Path):
    """Phone file managers hand over names like "recording" with no suffix."""
    encoded = tmp_path / "opus_copy.opus"
    if not encode(source_wav, encoded, "-c:a", "libopus"):
        pytest.skip("this ffmpeg build cannot encode opus")
    no_extension = tmp_path / "recording"
    no_extension.write_bytes(encoded.read_bytes())

    assert_normalizes(no_extension, tmp_path / "clean.wav")


@needs_ffmpeg
def test_compressed_audio_wearing_a_wav_name_is_accepted(source_wav: Path, tmp_path: Path):
    """WAV is only a container, so the fast path must fall through to ffmpeg."""
    encoded = tmp_path / "real.m4a"
    if not encode(source_wav, encoded, "-c:a", "aac"):
        pytest.skip("this ffmpeg build cannot encode aac")
    mislabelled = tmp_path / "voice memo.wav"
    mislabelled.write_bytes(encoded.read_bytes())

    assert_normalizes(mislabelled, tmp_path / "clean.wav")


@needs_ffmpeg
def test_video_files_are_reduced_to_their_audio_track(source_wav: Path, tmp_path: Path):
    encoded = tmp_path / "clip.mp4"
    if not encode(source_wav, encoded, "-c:a", "aac"):
        pytest.skip("this ffmpeg build cannot encode mp4")

    assert_normalizes(encoded, tmp_path / "clean.wav")


def test_empty_file_is_rejected_with_a_clear_reason(tmp_path: Path):
    empty = tmp_path / "empty.wav"
    empty.touch()

    with pytest.raises(ValueError, match="empty"):
        normalize_audio(empty, tmp_path / "clean.wav")


def test_a_file_that_is_not_audio_is_rejected(tmp_path: Path):
    not_audio = tmp_path / "notes.mp3"
    not_audio.write_text("this is not audio at all", encoding="utf-8")

    with pytest.raises(ValueError):
        normalize_audio(not_audio, tmp_path / "clean.wav")


def test_missing_file_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        normalize_audio(tmp_path / "absent.mp3", tmp_path / "clean.wav")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("clip.mp3", ".mp3"),
        ("clip.WEBM", ".webm"),
        ("recording", ".bin"),
        ("clip.verylongextension", ".bin"),
        ("clip.p n g", ".bin"),
    ],
)
def test_safe_suffix_only_keeps_short_alphanumeric_extensions(name, expected):
    assert safe_suffix(Path(name)) == expected


def test_named_extension_sets_cover_the_formats_phones_produce():
    for extension in (".m4a", ".amr", ".3gp", ".opus", ".webm", ".wma", ".aac"):
        assert extension in SUPPORTED_EXTENSIONS
    assert AUDIO_EXTENSIONS.isdisjoint(VIDEO_EXTENSIONS)


def test_voice_files_offers_unknown_extensions_and_skips_sidecars(tmp_path: Path):
    """The scan must not pre-filter formats that ffmpeg could still decode."""
    for name in ("clip.mp3", "clip.weirdformat", "recording", "clip.kikuyu.txt", ".hidden.mp3"):
        (tmp_path / name).write_bytes(b"data")

    found = {path.name for path in voice_files(tmp_path)}

    assert found == {"clip.mp3", "clip.weirdformat", "recording"}
