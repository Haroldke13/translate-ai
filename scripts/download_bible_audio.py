#!/usr/bin/env python3
"""Download Bible audio files from an explicit URL manifest.

Input can be either:
- plain text with one URL per line
- JSONL rows with at least ``url`` and optional book/chapter/filename/text/split

This script intentionally downloads only URLs supplied by the operator. It does
not scrape app streams or infer hidden provider file names.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".webm"}


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if path.suffix.casefold() == ".jsonl":
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not str(row.get("url", "")).strip():
                raise SystemExit(f"{path}:{line_number}: row needs a non-empty url")
            rows.append(row)
        else:
            rows.append({"url": line})
    if not rows:
        raise SystemExit(f"{path}: no audio URLs found")
    return rows


def extension_from_url(url: str, content_type: str | None = None) -> str:
    suffix = Path(unquote(urlparse(url).path)).suffix.casefold()
    if suffix in AUDIO_EXTENSIONS:
        return suffix
    guessed = mimetypes.guess_extension(content_type or "")
    if guessed and guessed.casefold() in AUDIO_EXTENSIONS:
        return guessed.casefold()
    return ".mp3"


def safe_part(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._") or "audio"


def destination_for(row: dict, output: Path, content_type: str | None = None) -> Path:
    url = str(row["url"])
    if row.get("filename"):
        return output / safe_part(row["filename"])
    suffix = extension_from_url(url, content_type)
    if row.get("book") and row.get("chapter"):
        return output / f"{safe_part(row['book']).upper()}_{int(row['chapter']):03d}{suffix}"
    name = Path(unquote(urlparse(url).path)).name.split("?")[0]
    if name:
        return output / safe_part(name)
    return output / f"audio_{abs(hash(url))}{suffix}"


def download(row: dict, output: Path, overwrite: bool) -> Path:
    request = Request(str(row["url"]), headers={"User-Agent": "kikuyu-ai-data-builder/0.1"})
    with urlopen(request, timeout=120) as response:
        destination = destination_for(row, output, response.headers.get_content_type())
        if destination.exists() and destination.stat().st_size > 0 and not overwrite:
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Bible audio from explicit licensed URLs")
    parser.add_argument("source", type=Path, help="Plain URL list or JSONL URL manifest")
    parser.add_argument("--output", type=Path, default=Path("data/bible/audio/kik"))
    parser.add_argument("--manifest-output", type=Path, default=Path("data/bible/audio/audio_manifest.jsonl"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    downloaded = []
    for row in read_rows(args.source):
        path = download(row, args.output, args.overwrite)
        manifest_row = {key: value for key, value in row.items() if key != "url"}
        manifest_row["audio"] = str(path)
        manifest_row.setdefault("split", args.split)
        downloaded.append(manifest_row)
        print(path)

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in downloaded),
        encoding="utf-8",
    )
    print(f"wrote audio manifest rows to {args.manifest_output}")


if __name__ == "__main__":
    main()
