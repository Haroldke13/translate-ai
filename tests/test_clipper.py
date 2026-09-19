"""Clipping a long recording, tracking progress, and resuming after a stop."""

import json
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from kikuyu_ai.clipper import (
    ClipSession,
    DONE,
    FAILED,
    PENDING,
    format_time,
    parse_ranges,
    parse_time,
    plan_fixed,
)

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is not installed")


@pytest.fixture
def long_wav(tmp_path: Path) -> Path:
    """A 30-second mono 16 kHz recording to cut up."""
    path = tmp_path / "long.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        count = 16000 * 30
        handle.writeframes(struct.pack("<" + "h" * count, *([1200] * count)))
    return path


@pytest.mark.parametrize(
    "value, expected",
    [("90", 90.0), ("1:30", 90.0), ("01:02:03", 3723.0), ("0:02.5", 2.5), ("00:00:10", 10.0)],
)
def test_parse_time_reads_the_usual_notations(value, expected):
    assert parse_time(value) == expected


@pytest.mark.parametrize("value", ["", "abc", "1:2:3:4", "-5"])
def test_parse_time_rejects_nonsense(value):
    with pytest.raises(ValueError):
        parse_time(value)


def test_parse_ranges_reads_a_comma_separated_list():
    assert parse_ranges("2:30-5:00, 17:40-19:05") == [(150.0, 300.0), (1060.0, 1145.0)]


def test_parse_ranges_rejects_a_range_that_ends_before_it_starts():
    with pytest.raises(ValueError, match="ends before"):
        parse_ranges("5:00-2:30")


def test_parse_ranges_rejects_a_missing_separator():
    with pytest.raises(ValueError, match="START-END"):
        parse_ranges("2:30")


def test_plan_fixed_covers_the_whole_recording():
    spans = plan_fixed(duration=250.0, clip_seconds=100.0)

    assert spans == [(0.0, 100.0), (100.0, 200.0), (200.0, 250.0)]
    assert spans[-1][1] == 250.0


def test_plan_fixed_folds_away_a_trailing_sliver():
    """A 0.2 s tail is not worth a clip of its own."""
    spans = plan_fixed(duration=100.2, clip_seconds=50.0)

    assert spans == [(0.0, 50.0), (50.0, 100.0)]


def test_plan_fixed_supports_overlap():
    spans = plan_fixed(duration=100.0, clip_seconds=40.0, overlap=10.0)

    assert spans[0] == (0.0, 40.0)
    assert spans[1] == (30.0, 70.0)


def test_plan_fixed_rejects_impossible_settings():
    with pytest.raises(ValueError):
        plan_fixed(100.0, clip_seconds=0)
    with pytest.raises(ValueError, match="overlap"):
        plan_fixed(100.0, clip_seconds=10, overlap=10)


@needs_ffmpeg
def test_session_cuts_every_clip_and_records_progress(long_wav: Path, tmp_path: Path):
    output = tmp_path / "clips"
    session = ClipSession.open(long_wav, output, clip_seconds=10.0)

    list(session.run())

    assert [clip.status for clip in session.clips] == [DONE, DONE, DONE]
    assert session.progress() == {
        "total": 3, "done": 3, "failed": 0, "remaining": 0,
        "seconds_done": 30.0, "seconds_total": 30.0,
    }
    for clip in session.clips:
        assert Path(clip.audio).is_file()
        assert clip.duration == pytest.approx(10.0, abs=0.3)
    assert session.resume_point() is None


@needs_ffmpeg
def test_progress_survives_a_stop_and_the_next_run_continues(long_wav: Path, tmp_path: Path):
    output = tmp_path / "clips"
    session = ClipSession.open(long_wav, output, clip_seconds=10.0)

    # Stop after the first clip, the way a killed process would.
    for _ in session.run():
        break

    reopened = ClipSession.open(long_wav, output, clip_seconds=10.0)
    assert len(reopened.done()) == 1
    assert reopened.resume_point() == 10.0

    finished = [clip.index for clip in reopened.run()]
    # Only the unfinished clips are redone.
    assert finished == [2, 3]
    assert len(reopened.done()) == 3


@needs_ffmpeg
def test_restart_discards_earlier_progress(long_wav: Path, tmp_path: Path):
    output = tmp_path / "clips"
    session = ClipSession.open(long_wav, output, clip_seconds=10.0)
    list(session.run())

    restarted = ClipSession.open(long_wav, output, clip_seconds=15.0, restart=True)

    assert [clip.status for clip in restarted.clips] == [PENDING, PENDING]
    assert restarted.resume_point() == 0.0


@needs_ffmpeg
def test_a_different_recording_cannot_resume_someone_elses_progress(long_wav: Path, tmp_path: Path):
    output = tmp_path / "clips"
    ClipSession.open(long_wav, output, clip_seconds=10.0)
    other = tmp_path / "other.wav"
    other.write_bytes(long_wav.read_bytes()[: len(long_wav.read_bytes()) // 2])

    with pytest.raises(ValueError, match="different recording"):
        ClipSession.open(other, output, clip_seconds=10.0)


@needs_ffmpeg
def test_explicit_ranges_are_cut_instead_of_the_whole_file(long_wav: Path, tmp_path: Path):
    session = ClipSession.open(long_wav, tmp_path / "clips", ranges=[(5.0, 8.0), (20.0, 24.0)])

    list(session.run())

    assert [(clip.start, clip.end) for clip in session.clips] == [(5.0, 8.0), (20.0, 24.0)]
    assert all(clip.status == DONE for clip in session.clips)


@needs_ffmpeg
def test_translation_results_are_stored_per_clip_and_joined_into_a_transcript(long_wav: Path, tmp_path: Path):
    session = ClipSession.open(long_wav, tmp_path / "clips", clip_seconds=15.0)

    def translate(clip_path: Path) -> tuple[str, str]:
        return "Ngai nĩ mwega", "God is good"

    list(session.run(translate=translate))
    transcript = session.write_transcript()

    assert [clip.english for clip in session.clips] == ["God is good", "God is good"]
    body = transcript.read_text(encoding="utf-8")
    assert "God is good" in body
    assert "00:00:00 - 00:00:15" in body


@needs_ffmpeg
def test_a_failing_clip_is_recorded_and_does_not_stop_the_rest(long_wav: Path, tmp_path: Path):
    session = ClipSession.open(long_wav, tmp_path / "clips", clip_seconds=10.0)
    calls: list[Path] = []

    def translate(clip_path: Path) -> tuple[str, str]:
        calls.append(clip_path)
        if len(calls) == 2:
            raise RuntimeError("model ran out of memory")
        return "Ngai", "God"

    list(session.run(translate=translate))

    assert [clip.status for clip in session.clips] == [DONE, FAILED, DONE]
    assert "out of memory" in session.clips[1].error
    # The failed clip is the one a later run picks up.
    assert session.resume_point() == 10.0


@needs_ffmpeg
def test_state_file_is_valid_json_after_every_clip(long_wav: Path, tmp_path: Path):
    output = tmp_path / "clips"
    session = ClipSession.open(long_wav, output, clip_seconds=10.0)

    for _ in session.run():
        data = json.loads((output / "clips.json").read_text(encoding="utf-8"))
        assert data["clips"]


def test_loading_a_folder_with_no_progress_returns_nothing(tmp_path: Path):
    assert ClipSession.load(tmp_path) is None


def test_format_time_is_readable():
    assert format_time(0) == "00:00:00"
    assert format_time(3723) == "01:02:03"
