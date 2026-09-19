#!/usr/bin/env python3
"""Merge every corpus into one deduplicated train/eval file for the translator.

    python scripts/build_training_corpus.py \
        data/translation/bible.jsonl \
        data/translation/pdf.jsonl \
        data/translation/voice.jsonl \
        --output data/translation/parallel.jsonl

With no inputs it picks up whichever of those default files exist. Later inputs
win on duplicates, so list the Bible first and hand-checked voice rows last: the
human-corrected version of a sentence should replace the bulk-imported one.

Voice and PDF corpora are usually tiny next to the ~31k Bible verses, which
biases the model toward scripture register. Use --repeat to upsample them:

    --repeat data/translation/voice.jsonl=8

Everything runs locally with no API keys and no network access.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kikuyu_ai.corpus import normalize_text, pair_key, read_jsonl, usable_pair, write_jsonl

DEFAULT_INPUTS = (
    Path("data/translation/bible.jsonl"),
    Path("data/translation/pdf.jsonl"),
    Path("data/translation/voice.jsonl"),
)


def parse_repeats(values: list[str]) -> dict[str, int]:
    repeats: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--repeat expects PATH=COUNT, got {value!r}")
        path, _, count = value.partition("=")
        try:
            repeats[str(Path(path))] = max(1, int(count))
        except ValueError:
            raise SystemExit(f"--repeat count must be a whole number, got {count!r}") from None
    return repeats


def load(path: Path, args) -> list[dict]:
    rows: list[dict] = []
    skipped = 0
    for row in read_jsonl(path):
        kikuyu = normalize_text(str(row.get("kikuyu", "")))
        english = normalize_text(str(row.get("english", "")))
        if not usable_pair(kikuyu, english, args.min_chars, args.max_chars, args.max_length_ratio):
            skipped += 1
            continue
        clean = {"kikuyu": kikuyu, "english": english}
        if row.get("source"):
            clean["source"] = str(row["source"])
        else:
            clean["source"] = path.stem
        if "verified" in row:
            clean["verified"] = bool(row["verified"])
        rows.append(clean)
    print(f"{path}: {len(rows)} usable row(s), {skipped} filtered out")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", type=Path, help=f"default: {', '.join(str(p) for p in DEFAULT_INPUTS)}")
    parser.add_argument("--output", type=Path, default=Path("data/translation/parallel.jsonl"))
    parser.add_argument("--repeat", action="append", default=[], metavar="PATH=COUNT",
                        help="upsample a small but high-value corpus")
    parser.add_argument("--eval-fraction", type=float, default=0.02, help="held-out share, 0 to skip (default: 0.02)")
    parser.add_argument("--max-eval", type=int, default=1000, help="cap on held-out rows")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--max-length-ratio", type=float, default=3.0)
    args = parser.parse_args()

    if not 0.0 <= args.eval_fraction < 1.0:
        raise SystemExit("--eval-fraction must be at least 0 and less than 1")

    inputs = args.inputs or [path for path in DEFAULT_INPUTS if path.is_file()]
    if not inputs:
        raise SystemExit(
            "No corpus files found. Build at least one first:\n"
            "  python scripts/fetch_bible_corpus.py && python scripts/build_parallel_dataset.py \\\n"
            "      data/bible/raw/kik_vpl.zip data/bible/raw/engwebp_vpl.zip \\\n"
            "      data/bible/parallel.jsonl --translation-output data/translation/bible.jsonl\n"
            "  python scripts/ingest_pdf_corpus.py paired KIKUYU.pdf ENGLISH.pdf\n"
            "  python scripts/ingest_voice_corpus.py"
        )
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing input file(s): {', '.join(str(path) for path in missing)}")

    repeats = parse_repeats(args.repeat)
    unknown = [path for path in repeats if path not in {str(item) for item in inputs}]
    if unknown:
        raise SystemExit(f"--repeat names a file that is not an input: {', '.join(unknown)}")

    # Later inputs overwrite earlier ones so hand-corrected rows win over bulk imports.
    merged: dict[tuple[str, str], dict] = {}
    weights: dict[tuple[str, str], int] = {}
    for path in inputs:
        weight = repeats.get(str(path), 1)
        for row in load(path, args):
            key = pair_key(row["kikuyu"], row["english"])
            merged[key] = row
            weights[key] = weight

    unique = list(merged.values())
    if not unique:
        raise SystemExit("No usable rows survived filtering; check the input files")

    rng = random.Random(args.seed)
    rng.shuffle(unique)

    eval_count = 0
    if args.eval_fraction > 0:
        eval_count = min(args.max_eval, int(len(unique) * args.eval_fraction))
        eval_count = min(eval_count, len(unique) - 1)
        eval_count = max(0, eval_count)

    # Held-out rows are never upsampled, so eval stays one row per real sentence.
    eval_rows = [{**row, "split": "eval"} for row in unique[:eval_count]]
    train_rows: list[dict] = []
    for row in unique[eval_count:]:
        copies = weights.get(pair_key(row["kikuyu"], row["english"]), 1)
        train_rows.extend({**row, "split": "train"} for _ in range(copies))
    rng.shuffle(train_rows)

    written = write_jsonl(args.output, train_rows + eval_rows)
    sources = Counter(row.get("source", "?").split(":")[0] for row in unique)
    verified = sum(1 for row in unique if row.get("verified"))

    print(f"\nwrote {written} row(s) to {args.output}")
    print(f"  train {len(train_rows)}  eval {len(eval_rows)}  unique sentences {len(unique)}")
    print("  by source: " + ", ".join(f"{name}={count}" for name, count in sources.most_common()))
    if verified:
        print(f"  human-verified rows: {verified}")
    print(f"\nTrain with:\n  python training/train_translation.py {args.output} --output models/translation/kikuyu-english")


if __name__ == "__main__":
    main()
