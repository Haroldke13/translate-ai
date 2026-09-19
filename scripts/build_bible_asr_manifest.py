#!/usr/bin/env python3
"""Build an ASR manifest by pairing Bible audio files with chapter text."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm"}
READALOUD_NAME_RE = re.compile(r"^[^_]+_\d+_([1-3]?[A-Z0-9]{2,3})_(\d+)_read\.txt$", re.IGNORECASE)


def normalize_text(value: str) -> str:
    lines = [line.strip() for line in value.replace("\ufeff", "").splitlines() if line.strip()]
    return " ".join(lines)


def read_readaloud_zip(path: Path) -> dict[tuple[str, int], str]:
    transcripts: dict[tuple[str, int], str] = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            match = READALOUD_NAME_RE.match(Path(name).name)
            if not match:
                continue
            book, chapter = match.groups()
            if book == "000":
                continue
            with archive.open(name) as handle:
                text = normalize_text(handle.read().decode("utf-8-sig"))
            if text:
                transcripts[(book.upper(), int(chapter))] = text
    if not transcripts:
        raise SystemExit(f"{path}: no *_read.txt chapter transcripts found")
    return transcripts


def ref_from_audio_path(path: Path) -> tuple[str, int] | None:
    tokens = re.split(r"[_\-. ]+", path.stem.upper())
    for index, token in enumerate(tokens):
        if not re.fullmatch(r"[1-3]?[A-Z]{2,3}", token):
            continue
        if index + 1 < len(tokens) and tokens[index + 1].isdigit():
            return token, int(tokens[index + 1])
    match = re.search(r"([1-3]?[A-Z]{2,3})[_-]?0*(\d{1,3})", path.stem.upper())
    if match:
        return match.group(1), int(match.group(2))
    return None


def audio_rows_from_dir(audio_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(audio_dir.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in AUDIO_EXTENSIONS:
            continue
        ref = ref_from_audio_path(path)
        if ref:
            rows.append({"audio": path, "book": ref[0], "chapter": ref[1]})
    return rows


def audio_rows_from_manifest(path: Path) -> list[dict]:
    base = path.resolve().parent
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not row.get("audio"):
            raise SystemExit(f"{path}:{line_number}: row needs audio")
        audio = Path(str(row["audio"])).expanduser()
        if not audio.is_absolute():
            audio = base / audio
        row["audio"] = audio
        if not row.get("book") or not row.get("chapter"):
            ref = ref_from_audio_path(audio)
            if ref:
                row.setdefault("book", ref[0])
                row.setdefault("chapter", ref[1])
        rows.append(row)
    return rows


def build_rows(
    transcripts: dict[tuple[str, int], str],
    audio_rows: list[dict],
    split: str,
) -> list[dict]:
    result = []
    for row in audio_rows:
        audio = Path(row["audio"]).expanduser()
        if not audio.is_file():
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            book = str(row.get("book", "")).upper()
            chapter = int(row.get("chapter", 0) or 0)
            text = transcripts.get((book, chapter), "")
        if not text:
            continue
        result.append(
            {
                "audio": str(audio.resolve()),
                "text": normalize_text(text),
                "split": str(row.get("split") or split),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ASR manifest from Bible audio and readaloud chapter text")
    parser.add_argument("--readaloud-zip", type=Path, default=Path("data/bible/raw/kik_readaloud.zip"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-dir", type=Path)
    source.add_argument("--audio-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/asr/bible_audio_manifest.jsonl"))
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    transcripts = read_readaloud_zip(args.readaloud_zip)
    audio_rows = audio_rows_from_manifest(args.audio_manifest) if args.audio_manifest else audio_rows_from_dir(args.audio_dir)
    rows = build_rows(transcripts, audio_rows, args.split)
    if not rows:
        raise SystemExit("no ASR rows matched audio files to chapter transcripts")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} ASR rows to {args.output}")


if __name__ == "__main__":
    main()
