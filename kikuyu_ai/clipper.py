"""Cut a long recording into clips, convert them, and resume where you stopped.

A three-hour sermon is not something to transcribe in one pass: it takes hours,
and any interruption loses the lot. This splits the recording into clips, records
what finished in a state file next to the output, and picks up from the last
completed clip on the next run.

    # split into 60-second clips and convert them
    python -m kikuyu_ai.clipper sermon.mp4 --output data/clips/sermon --seconds 60

    # only the sections you care about
    python -m kikuyu_ai.clipper sermon.mp4 --output data/clips/sermon \\
        --ranges 2:30-5:00,17:40-19:05

    # convert and translate, stopping and resuming as often as you like
    python -m kikuyu_ai.clipper sermon.mp4 --output data/clips/sermon --translate
    python -m kikuyu_ai.clipper sermon.mp4 --output data/clips/sermon --translate --resume
    python -m kikuyu_ai.clipper sermon.mp4 --output data/clips/sermon --restart
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .audio import extract_clip, media_duration

STATE_NAME = "clips.json"
PENDING, DONE, FAILED = "pending", "done", "failed"
TIME_RE = re.compile(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)$")


def parse_time(value: str) -> float:
    """Read 90, 1:30, or 01:02:03.5 as seconds."""
    text = value.strip()
    match = TIME_RE.match(text)
    if not match:
        raise ValueError(f"{value!r} is not a time; use seconds, MM:SS, or HH:MM:SS")
    first, second, last = match.groups()
    parts = [part for part in (first, second) if part is not None]
    seconds = float(last)
    if len(parts) == 1:
        seconds += int(parts[0]) * 60
    elif len(parts) == 2:
        seconds += int(parts[0]) * 3600 + int(parts[1]) * 60
    return seconds


def parse_ranges(value: str) -> list[tuple[float, float]]:
    """Read "2:30-5:00,17:40-19:05" as a list of (start, end) pairs."""
    ranges: list[tuple[float, float]] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise ValueError(f"{chunk!r} is not a range; write it as START-END")
        start_text, _, end_text = chunk.partition("-")
        start, end = parse_time(start_text), parse_time(end_text)
        if end <= start:
            raise ValueError(f"{chunk!r} ends before it starts")
        ranges.append((start, end))
    if not ranges:
        raise ValueError("no ranges given")
    return ranges


def plan_fixed(duration: float, clip_seconds: float, overlap: float = 0.0) -> list[tuple[float, float]]:
    """Split a duration into fixed-length clips, optionally overlapping."""
    if clip_seconds <= 0:
        raise ValueError("clip length must be greater than zero")
    if overlap < 0 or overlap >= clip_seconds:
        raise ValueError("overlap must be at least zero and shorter than the clip length")
    step = clip_seconds - overlap
    ranges: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + clip_seconds, duration)
        # A trailing sliver is folded into the previous clip rather than kept.
        if ranges and end - start < 1.0:
            break
        ranges.append((round(start, 3), round(end, 3)))
        if end >= duration:
            break
        start += step
    return ranges


@dataclass
class Clip:
    index: int
    start: float
    end: float
    status: str = PENDING
    audio: str | None = None
    duration: float | None = None
    kikuyu: str | None = None
    english: str | None = None
    error: str | None = None

    @property
    def label(self) -> str:
        return f"clip_{self.index:04d}"

    def as_row(self) -> dict:
        return asdict(self)


@dataclass
class ClipSession:
    """Clip plan plus its progress, persisted so a run can be resumed."""

    source: Path
    output: Path
    clips: list[Clip] = field(default_factory=list)
    signature: str = ""

    @property
    def state_path(self) -> Path:
        return self.output / STATE_NAME

    # -- creating and loading -------------------------------------------------

    @staticmethod
    def signature_for(source: Path) -> str:
        """Identify the source so a different file cannot resume this plan."""
        stat = source.stat()
        return f"{source.name}:{stat.st_size}"

    @classmethod
    def create(
        cls,
        source: Path,
        output: Path,
        clip_seconds: float = 60.0,
        ranges: Iterable[tuple[float, float]] | None = None,
        overlap: float = 0.0,
    ) -> "ClipSession":
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if ranges is None:
            duration = media_duration(source)
            spans = plan_fixed(duration, clip_seconds, overlap)
        else:
            spans = [(float(start), float(end)) for start, end in ranges]
            if not spans:
                raise ValueError("no clip ranges to work with")
        session = cls(
            source=source,
            output=output,
            clips=[Clip(index, start, end) for index, (start, end) in enumerate(spans, 1)],
            signature=cls.signature_for(source),
        )
        session.save()
        return session

    @classmethod
    def load(cls, output: Path) -> "ClipSession | None":
        state_path = output / STATE_NAME
        if not state_path.is_file():
            return None
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return cls(
            source=Path(data["source"]),
            output=output,
            clips=[Clip(**row) for row in data.get("clips", [])],
            signature=data.get("signature", ""),
        )

    @classmethod
    def open(
        cls,
        source: Path,
        output: Path,
        clip_seconds: float = 60.0,
        ranges: Iterable[tuple[float, float]] | None = None,
        overlap: float = 0.0,
        restart: bool = False,
    ) -> "ClipSession":
        """Resume the plan in `output`, or start a new one."""
        source = source.expanduser().resolve()
        existing = None if restart else cls.load(output)
        if existing is not None:
            if existing.signature and existing.signature != cls.signature_for(source):
                raise ValueError(
                    f"{output} holds progress for a different recording "
                    f"({existing.source.name}). Use --restart to replace it."
                )
            existing.source = source
            return existing
        return cls.create(source, output, clip_seconds, ranges, overlap)

    def save(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": str(self.source),
            "signature": self.signature,
            "clips": [clip.as_row() for clip in self.clips],
        }
        # Written to a temporary file first so an interrupted save cannot leave
        # the progress file truncated.
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    # -- progress -------------------------------------------------------------

    def pending(self) -> list[Clip]:
        return [clip for clip in self.clips if clip.status != DONE]

    def done(self) -> list[Clip]:
        return [clip for clip in self.clips if clip.status == DONE]

    def failed(self) -> list[Clip]:
        return [clip for clip in self.clips if clip.status == FAILED]

    def progress(self) -> dict:
        done = len(self.done())
        return {
            "total": len(self.clips),
            "done": done,
            "failed": len(self.failed()),
            "remaining": len(self.clips) - done,
            "seconds_done": round(sum(clip.end - clip.start for clip in self.done()), 2),
            "seconds_total": round(sum(clip.end - clip.start for clip in self.clips), 2),
        }

    def resume_point(self) -> float | None:
        """Where the next run will start, or None when everything is finished."""
        remaining = self.pending()
        return remaining[0].start if remaining else None

    # -- running --------------------------------------------------------------

    def run(
        self,
        translate: Callable[[Path], tuple[str, str]] | None = None,
        on_progress: Callable[[Clip, dict], None] | None = None,
        retry_failed: bool = True,
    ) -> Iterator[Clip]:
        """Convert each unfinished clip, saving progress after every one."""
        for clip in self.clips:
            if clip.status == DONE:
                continue
            if clip.status == FAILED and not retry_failed:
                continue
            target = self.output / f"{clip.label}.wav"
            try:
                clip.duration = extract_clip(self.source, target, clip.start, clip.end)
                clip.audio = str(target)
                if translate is not None:
                    clip.kikuyu, clip.english = translate(target)
                clip.status = DONE
                clip.error = None
            except Exception as exc:  # noqa: BLE001 - recorded, then move on
                clip.status = FAILED
                clip.error = str(exc) or exc.__class__.__name__
            # Saved per clip so killing the process loses at most one clip.
            self.save()
            if on_progress is not None:
                on_progress(clip, self.progress())
            yield clip

    def write_transcript(self) -> Path | None:
        """Join the finished clips into one transcript file, in time order."""
        rows = [clip for clip in self.clips if clip.status == DONE and (clip.kikuyu or clip.english)]
        if not rows:
            return None
        target = self.output / "transcript.txt"
        lines: list[str] = []
        for clip in rows:
            lines.append(f"[{format_time(clip.start)} - {format_time(clip.end)}]")
            if clip.kikuyu:
                lines.append(f"  kikuyu : {clip.kikuyu}")
            if clip.english:
                lines.append(f"  english: {clip.english}")
            lines.append("")
        target.write_text("\n".join(lines), encoding="utf-8")
        return target


def format_time(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def build_translator(settings=None):
    """A translate callback backed by the normal pipeline."""
    from .config import Settings
    from .pipeline import Pipeline

    pipeline = Pipeline(settings or Settings.from_env())

    def translate(clip_path: Path) -> tuple[str, str]:
        result = pipeline.run(clip_path)
        return result.kikuyu, result.english

    return translate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="the long recording or video")
    parser.add_argument("--output", type=Path, required=True, help="folder for the clips and the progress file")
    parser.add_argument("--seconds", type=float, default=60.0, help="clip length (default: %(default)s)")
    parser.add_argument("--overlap", type=float, default=0.0, help="seconds of overlap between clips")
    parser.add_argument("--ranges", help='only these sections, e.g. "2:30-5:00,17:40-19:05"')
    parser.add_argument("--translate", action="store_true", help="also transcribe and translate each clip")
    parser.add_argument("--restart", action="store_true", help="discard existing progress and start again")
    parser.add_argument("--resume", action="store_true", help="continue from the last finished clip (the default)")
    parser.add_argument("--status", action="store_true", help="print progress and exit without converting")
    parser.add_argument("--no-retry-failed", action="store_true", help="skip clips that failed before")
    args = parser.parse_args()

    if args.restart and args.resume:
        raise SystemExit("--restart and --resume contradict each other; choose one")

    try:
        ranges = parse_ranges(args.ranges) if args.ranges else None
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    if args.status:
        session = ClipSession.load(args.output)
        if session is None:
            raise SystemExit(f"{args.output}: no clip progress found")
        report(session)
        return

    try:
        session = ClipSession.open(
            args.source, args.output, args.seconds, ranges, args.overlap, restart=args.restart
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None

    progress = session.progress()
    if progress["done"]:
        print(f"resuming: {progress['done']}/{progress['total']} clips already done, "
              f"starting at {format_time(session.resume_point() or 0)}")
    else:
        print(f"{progress['total']} clip(s) to convert from {session.source.name}")

    translate = build_translator() if args.translate else None

    def on_progress(clip: Clip, state: dict) -> None:
        mark = "ok  " if clip.status == DONE else "FAIL"
        line = f"  [{state['done']}/{state['total']}] {mark} {clip.label} {format_time(clip.start)}-{format_time(clip.end)}"
        if clip.english:
            line += f"  {clip.english[:60]}"
        if clip.error:
            line += f"  {clip.error[:80]}"
        print(line, flush=True)

    try:
        for _ in session.run(translate=translate, on_progress=on_progress, retry_failed=not args.no_retry_failed):
            pass
    except KeyboardInterrupt:
        state = session.progress()
        print(f"\nstopped at {state['done']}/{state['total']} clips. "
              f"Re-run the same command to continue from {format_time(session.resume_point() or 0)}.")
        raise SystemExit(130) from None

    transcript = session.write_transcript()
    report(session)
    if transcript:
        print(f"transcript: {transcript}")


def report(session: ClipSession) -> None:
    state = session.progress()
    print(f"\n{state['done']}/{state['total']} clip(s) done, "
          f"{state['seconds_done']:.0f}s of {state['seconds_total']:.0f}s")
    if state["failed"]:
        print(f"{state['failed']} clip(s) failed:")
        for clip in session.failed()[:5]:
            print(f"  {clip.label} {format_time(clip.start)}-{format_time(clip.end)}: {clip.error}")
    resume = session.resume_point()
    if resume is not None:
        print(f"next run continues at {format_time(resume)}")
    print(f"clips in {session.output}")


if __name__ == "__main__":
    main()
