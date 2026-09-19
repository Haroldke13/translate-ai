#!/usr/bin/env python3
"""Build an explicit chapter-audio URL manifest from YouVersion public pages."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

AUDIO_URL_RE = re.compile(r"https?:\\?/\\?/audio-bible-cdn\.youversionapi\.com/[^\"'<>\s]+?\.mp3\?version_id=\d+")


def canonical_chapters(version: dict) -> list[dict]:
    rows = []
    for book in version.get("books", []):
        book_usfm = str(book.get("usfm", "")).strip().upper()
        if not book_usfm:
            continue
        for chapter in book.get("chapters", []):
            has_audio = chapter.get("audio", book.get("audio", False))
            if not chapter.get("canonical") or not has_audio:
                continue
            usfm = str(chapter.get("usfm", "")).strip().upper()
            match = re.fullmatch(r"([1-3]?[A-Z]{2,3})\.(\d+)", usfm)
            if not match:
                continue
            rows.append(
                {
                    "book": match.group(1),
                    "chapter": int(match.group(2)),
                    "usfm": usfm,
                    "book_title": book.get("human") or book_usfm,
                    "chapter_title": chapter.get("human") or match.group(2),
                }
            )
    return rows


def public_chapter_url(base_url: str, version_id: int | str, usfm: str) -> str:
    base = base_url.rstrip("/") + "/"
    return urljoin(base, f"{version_id}/{usfm}")


def normalize_audio_url(value: str) -> str:
    return html.unescape(value).replace("\\/", "/")


def extract_audio_url(page_html: str, version_id: int | str) -> str | None:
    for match in re.finditer(r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>", page_html, re.I | re.S):
        raw = html.unescape(match.group(1)).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            url = item.get("contentUrl")
            if isinstance(url, str) and "audio-bible-cdn.youversionapi.com" in url and ".mp3" in url:
                return normalize_audio_url(url)

    match = AUDIO_URL_RE.search(page_html)
    if match:
        return normalize_audio_url(match.group(0))

    escaped_host = "audio-bible-cdn\\.youversionapi\\.com"
    fallback = re.search(rf"https?:\\?/\\?/{escaped_host}/[^\"'<>\s]+?\.mp3\?version_id={re.escape(str(version_id))}", page_html)
    if fallback:
        return normalize_audio_url(fallback.group(0))
    return None


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "kikuyu-ai-data-builder/0.1"})
    with urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8", errors="replace")


def read_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        usfm = str(row.get("usfm", "")).upper()
        if usfm:
            rows[usfm] = row
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a JSONL manifest of public YouVersion/Bible.com chapter MP3 URLs"
    )
    parser.add_argument("--version-json", type=Path, default=Path("data/bible/raw/youversion_1622_version.json"))
    parser.add_argument("--version-id", type=int, default=1622)
    parser.add_argument("--base-url", default="https://www.bible.com/audio-bible")
    parser.add_argument("--output", type=Path, default=Path("data/bible/audio_urls.jsonl"))
    parser.add_argument("--limit", type=int, help="Fetch at most this many missing chapters")
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay between chapter page requests")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild instead of resuming existing rows")
    args = parser.parse_args()

    version = json.loads(args.version_json.read_text(encoding="utf-8"))
    chapters = canonical_chapters(version)
    if not chapters:
        raise SystemExit(f"{args.version_json}: no canonical audio chapters found")

    existing = {} if args.overwrite else read_existing(args.output)
    rows = [existing[chapter["usfm"]] for chapter in chapters if chapter["usfm"] in existing]
    fetched = 0
    failures = []

    for chapter in chapters:
        if chapter["usfm"] in existing:
            continue
        if args.limit is not None and fetched >= args.limit:
            break

        page_url = public_chapter_url(args.base_url, args.version_id, chapter["usfm"])
        try:
            page = fetch_text(page_url)
            audio_url = extract_audio_url(page, args.version_id)
        except Exception as exc:  # pragma: no cover - network failure details matter in CLI output
            failures.append(f"{chapter['usfm']}: {type(exc).__name__}: {exc}")
            continue

        if not audio_url:
            failures.append(f"{chapter['usfm']}: no mp3 contentUrl on {page_url}")
            continue

        row = {
            "url": audio_url,
            "book": chapter["book"],
            "chapter": chapter["chapter"],
            "usfm": chapter["usfm"],
            "filename": f"{chapter['book']}_{chapter['chapter']:03d}.mp3",
            "source_page": page_url,
            "source": "youversion-public-audio-page",
            "split": "train",
        }
        rows.append(row)
        fetched += 1
        print(f"{len(rows)}/{len(chapters)} {chapter['usfm']} {audio_url}")
        write_manifest(args.output, rows)
        if args.sleep > 0:
            time.sleep(args.sleep)

    if not rows:
        raise SystemExit("no audio URLs found")
    write_manifest(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    if failures:
        print(f"{len(failures)} failures", file=sys.stderr)
        for failure in failures[:20]:
            print(failure, file=sys.stderr)
        if len(failures) > 20:
            print(f"... {len(failures) - 20} more", file=sys.stderr)


if __name__ == "__main__":
    main()
