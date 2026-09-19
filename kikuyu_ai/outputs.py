import json
from pathlib import Path
from .models import Segment


def srt_timestamp(seconds: float) -> str:
    millis = int(max(0, seconds) * 1000)
    hours, millis = divmod(millis, 3600000)
    minutes, millis = divmod(millis, 60000)
    seconds, millis = divmod(millis, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def write_srt(segments: list[Segment], path: Path) -> None:
    path.write_text("\n\n".join(f"{i}\n{srt_timestamp(s.start)} --> {srt_timestamp(max(s.end, s.start + 1))}\n{s.text}" for i, s in enumerate(segments, 1)), encoding="utf-8")


def write_metadata(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

