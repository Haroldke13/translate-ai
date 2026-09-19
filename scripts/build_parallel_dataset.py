#!/usr/bin/env python3
"""Build aligned Kikuyu-English Bible JSONL from verse files.

Supported inputs:
- JSON/JSONL rows with book, chapter, verse and kikuyu/english/text fields.
- eBible VPL text: ``GEN 1:1 Verse text``.
- ZIP files containing one ``*_vpl.txt`` file.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Iterable

VERSE_RE = re.compile(r"^([1-3]?[A-Z]{2,3})\s+(\d+):(\d+)\s+(.+)$")


def read_text(path: Path) -> str:
    if path.suffix.casefold() != ".zip":
        return path.read_text(encoding="utf-8-sig")
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith("_vpl.txt")]
        if not names:
            raise SystemExit(f"{path}: no *_vpl.txt found")
        with archive.open(names[0]) as handle:
            return handle.read().decode("utf-8-sig")


def json_rows(path: Path) -> Iterable[dict]:
    text = read_text(path)
    if path.suffix.casefold() == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)
        return
    data = json.loads(text)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        verses = data.get("verses")
        if isinstance(verses, list):
            yield from verses
        else:
            yield from (value for value in data.values() if isinstance(value, dict))


def vpl_rows(path: Path, language_key: str) -> Iterable[dict]:
    for line_number, line in enumerate(read_text(path).splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        match = VERSE_RE.match(line)
        if not match:
            continue
        book, chapter, verse, text = match.groups()
        yield {
            "book": book,
            "chapter": int(chapter),
            "verse": int(verse),
            language_key: text.strip(),
            "_line": line_number,
        }


def rows(path: Path, language_key: str) -> list[dict]:
    suffix = path.suffix.casefold()
    if suffix in {".json", ".jsonl"}:
        return list(json_rows(path))
    return list(vpl_rows(path, language_key))


def key(row: dict) -> tuple[str, int, int]:
    return (str(row["book"]).casefold(), int(row["chapter"]), int(row["verse"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aligned Kikuyu-English Bible JSONL")
    parser.add_argument("kikuyu", type=Path)
    parser.add_argument("english", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--translation-output", type=Path, help="Also write JSONL for training/train_translation.py")
    args = parser.parse_args()

    english_by_ref = {key(row): row for row in rows(args.english, "english")}
    result = []
    for row in rows(args.kikuyu, "kikuyu"):
        ref = key(row)
        english = english_by_ref.get(ref)
        if not english:
            continue
        kikuyu_text = str(row.get("kikuyu", row.get("text", ""))).strip()
        english_text = str(english.get("english", english.get("text", ""))).strip()
        if kikuyu_text and english_text:
            result.append(
                {
                    "book": row["book"],
                    "chapter": ref[1],
                    "verse": ref[2],
                    "kikuyu": kikuyu_text,
                    "english": english_text,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result),
        encoding="utf-8",
    )

    if args.translation_output:
        args.translation_output.parent.mkdir(parents=True, exist_ok=True)
        args.translation_output.write_text(
            "".join(
                json.dumps(
                    {"kikuyu": row["kikuyu"], "english": row["english"], "split": "train"},
                    ensure_ascii=False,
                )
                + "\n"
                for row in result
            ),
            encoding="utf-8",
        )

    print(f"wrote {len(result)} aligned verses to {args.output}")
    if args.translation_output:
        print(f"wrote translation training rows to {args.translation_output}")


if __name__ == "__main__":
    main()
