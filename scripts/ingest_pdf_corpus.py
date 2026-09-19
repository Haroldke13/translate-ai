#!/usr/bin/env python3
"""Turn PDF translations into Kikuyu-English training rows, fully offline.

Two layouts are supported:

  Paired documents - the same book as two PDFs, one per language:
      python scripts/ingest_pdf_corpus.py paired kikuyu.pdf english.pdf \
          --output data/translation/pdf.jsonl

  Bilingual document - one PDF holding both languages (facing columns,
  alternating paragraphs, or interleaved lines):
      python scripts/ingest_pdf_corpus.py bilingual book.pdf \
          --output data/translation/pdf.jsonl

Text is extracted with pypdf when installed and with poppler's `pdftotext`
otherwise. Paired documents are aligned with the Gale-Church length model, so no
dictionary, no model, and no network call is involved. Scanned PDFs with no text
layer are reported rather than silently producing empty output; run OCR on them
first (for example `ocrmypdf in.pdf out.pdf`) and re-run this script.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kikuyu_ai.corpus import (
    Pair,
    align_blocks,
    dedupe,
    guess_language,
    is_noise_line,
    normalize_text,
    split_sentences,
    strip_verse_number,
    usable_pair,
    write_jsonl,
)


def extract_pages(pdf: Path) -> list[str]:
    """Return one text string per page using whichever local extractor exists."""
    if not pdf.is_file():
        raise SystemExit(f"{pdf}: not a file")
    try:
        from pypdf import PdfReader
    except ImportError:
        return _extract_pages_poppler(pdf)
    try:
        reader = PdfReader(str(pdf))
        return [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - fall back rather than abort
        print(f"{pdf}: pypdf failed ({type(exc).__name__}: {exc}); trying pdftotext", file=sys.stderr)
        return _extract_pages_poppler(pdf)


def _extract_pages_poppler(pdf: Path) -> list[str]:
    if not shutil.which("pdftotext"):
        raise SystemExit(
            f"{pdf}: no PDF text extractor available.\n"
            "Install one offline-capable option:\n"
            "  python -m pip install -e '.[pdf]'   # pypdf\n"
            "  sudo apt install poppler-utils      # pdftotext"
        )
    completed = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), "-"],
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise SystemExit(f"{pdf}: pdftotext failed: {completed.stderr.strip()[:500]}")
    return completed.stdout.split("\f")


def page_lines(pages: list[str], skip_pages: int, last_page: int | None) -> list[str]:
    selected = pages[skip_pages : last_page if last_page else None]
    lines: list[str] = []
    for page in selected:
        for raw in page.splitlines():
            line = normalize_text(raw)
            if line and not is_noise_line(line):
                lines.append(line)
    return lines


def drop_repeated_headers(lines: list[str], threshold: int) -> list[str]:
    """Remove running headers and footers that repeat across many pages."""
    counts: dict[str, int] = {}
    for line in lines:
        if len(line) <= 80:
            counts[line.casefold()] = counts.get(line.casefold(), 0) + 1
    repeated = {text for text, count in counts.items() if count >= threshold}
    return [line for line in lines if line.casefold() not in repeated]


def sentences_from_pdf(pdf: Path, args) -> list[str]:
    lines = page_lines(extract_pages(pdf), args.skip_pages, args.last_page)
    if not lines:
        raise SystemExit(
            f"{pdf}: no extractable text. This is usually a scanned PDF with no text layer; "
            "run OCR on it first (for example `ocrmypdf in.pdf out.pdf`) and re-run."
        )
    lines = drop_repeated_headers(lines, args.header_repeats)
    sentences = split_sentences(" ".join(lines))
    return [strip_verse_number(sentence) for sentence in sentences]


def pairs_from_paired(args) -> list[Pair]:
    kikuyu = sentences_from_pdf(args.kikuyu_pdf, args)
    english = sentences_from_pdf(args.english_pdf, args)
    print(f"{args.kikuyu_pdf.name}: {len(kikuyu)} sentences", file=sys.stderr)
    print(f"{args.english_pdf.name}: {len(english)} sentences", file=sys.stderr)
    source = f"pdf:{args.kikuyu_pdf.name}|{args.english_pdf.name}"
    return [
        Pair(left, right, source)
        for left, right in align_blocks(kikuyu, english)
    ]


def language_runs(sentences: list[str]) -> list[tuple[str, list[str]]]:
    """Group consecutive sentences that are in the same language."""
    runs: list[tuple[str, list[str]]] = []
    for sentence in sentences:
        language = guess_language(sentence)
        if language == "unknown":
            # Short or ambiguous lines belong to whichever run they follow.
            if runs:
                runs[-1][1].append(sentence)
            continue
        if runs and runs[-1][0] == language:
            runs[-1][1].append(sentence)
        else:
            runs.append((language, [sentence]))
    return runs


def pair_language_runs(runs: list[tuple[str, list[str]]], source: str = "") -> list[Pair]:
    """Pair each language run with its translation run.

    Interleaved documents repeat source-then-target, so runs are consumed two at
    a time. Sliding by one instead would also emit every target paired with the
    *next* source, which is a wrong translation pair.
    """

    def pairs_at(offset: int) -> list[Pair]:
        collected: list[Pair] = []
        for index in range(offset, len(runs) - 1, 2):
            language, block = runs[index]
            next_language, next_block = runs[index + 1]
            if language == next_language:
                continue
            left, right = (block, next_block) if language == "kikuyu" else (next_block, block)
            # Same-size runs align one to one; otherwise fall back to length alignment.
            if len(left) == len(right):
                collected.extend(Pair(a, b, source) for a, b in zip(left, right))
            else:
                collected.extend(Pair(a, b, source) for a, b in align_blocks(left, right))
        return collected

    # A stray leading run shifts the whole interleaving, so try both phases.
    # Pair count alone cannot separate them: with strict alternation both phases
    # produce the same number of pairs, and the wrong one simply pairs each
    # target with the previous source. A translation and its source have similar
    # lengths, so score the phases on how balanced their pairs are instead.
    return max((pairs_at(0), pairs_at(1)), key=phase_score)


def phase_score(pairs: list[Pair]) -> float:
    """Total length balance across pairs; higher means a better interleaving."""
    total = 0.0
    for pair in pairs:
        longest = max(len(pair.kikuyu), len(pair.english))
        if longest:
            total += min(len(pair.kikuyu), len(pair.english)) / longest
    return total


def pairs_from_bilingual(args) -> list[Pair]:
    """Pair adjacent same-language runs inside one document."""
    sentences = sentences_from_pdf(args.pdf, args)
    print(f"{args.pdf.name}: {len(sentences)} sentences", file=sys.stderr)
    return pair_language_runs(language_runs(sentences), f"pdf:{args.pdf.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("data/translation/pdf.jsonl"))
    parser.add_argument("--skip-pages", type=int, default=0, help="drop this many leading pages (covers, prefaces)")
    parser.add_argument("--last-page", type=int, default=None, help="stop after this page number")
    parser.add_argument("--header-repeats", type=int, default=4, help="drop short lines repeating on this many pages")
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--max-length-ratio", type=float, default=3.0)
    parser.add_argument("--append", action="store_true", help="add to an existing output file instead of replacing it")

    modes = parser.add_subparsers(dest="mode", required=True)
    paired = modes.add_parser("paired", help="two PDFs of the same document, one per language")
    paired.add_argument("kikuyu_pdf", type=Path)
    paired.add_argument("english_pdf", type=Path)
    bilingual = modes.add_parser("bilingual", help="one PDF containing both languages")
    bilingual.add_argument("pdf", type=Path)

    args = parser.parse_args()
    pairs = pairs_from_paired(args) if args.mode == "paired" else pairs_from_bilingual(args)

    kept = dedupe(
        pair
        for pair in pairs
        if usable_pair(pair.kikuyu, pair.english, args.min_chars, args.max_chars, args.max_length_ratio)
    )
    rows = [pair.row() for pair in kept]
    if args.append and args.output.exists():
        from kikuyu_ai.corpus import read_jsonl

        rows = list(read_jsonl(args.output)) + rows

    written = write_jsonl(args.output, rows)
    dropped = len(pairs) - len(kept)
    print(f"wrote {written} rows to {args.output} ({len(pairs)} candidate pairs, {dropped} filtered out)")
    if kept:
        print("Check a few rows before training; misaligned PDFs teach the model wrong pairs:")
        for pair in kept[: min(3, len(kept))]:
            print(f"  kikuyu: {pair.kikuyu[:70]}")
            print(f"  english: {pair.english[:70]}")


if __name__ == "__main__":
    main()
