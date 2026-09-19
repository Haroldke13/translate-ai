#!/usr/bin/env python3
"""Turn voice recordings and human corrections into training data, offline.

Four sources are read, all of them local:

  corrections   data/corrections.db rows saved from the app's "fix this"
                button. The corrected English is a human label.
  voice pairs   a folder of recordings with transcript sidecars:
                    clip01.wav
                    clip01.kikuyu.txt      what was said
                    clip01.english.txt     what it means (optional)
                These feed both the translator and the speech recogniser.
  sessions      finished session folders that a person has reviewed, marked by
                writing kikuyu.corrected.txt / english.corrected.txt into them.
  machine       raw uncorrected app output, only with --include-machine-output.

    python scripts/ingest_voice_corpus.py \
        --output data/translation/voice.jsonl \
        --asr-manifest data/asr/manifest.jsonl

Machine output is excluded by default on purpose. Training on the model's own
unreviewed guesses reinforces its existing hallucinations instead of fixing them.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kikuyu_ai.audio import wav_duration
from kikuyu_ai.config import Settings
from kikuyu_ai.corpus import Pair, dedupe, normalize_text, read_jsonl, usable_pair, write_jsonl

TRANSCRIPT_SUFFIXES = (".kikuyu.txt", ".kik.txt", ".txt")
TRANSLATION_SUFFIXES = (".english.txt", ".eng.txt", ".en.txt")


def read_sidecar(audio: Path, suffixes: tuple[str, ...]) -> str:
    stem = audio.with_suffix("")
    for suffix in suffixes:
        candidate = stem.with_name(stem.name + suffix)
        if candidate.is_file():
            return normalize_text(candidate.read_text(encoding="utf-8"))
    return ""


def from_corrections(db_path: Path) -> list[Pair]:
    """Human-corrected English for a Kikuyu source is ground truth."""
    if not db_path.is_file():
        return []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
        try:
            rows = db.execute("SELECT source, corrected FROM corrections").fetchall()
        except sqlite3.DatabaseError as exc:
            raise SystemExit(f"{db_path}: cannot read corrections table: {exc}") from None
    return [
        Pair(normalize_text(source), normalize_text(corrected), "corrections", {"verified": True})
        for source, corrected in rows
        if source and corrected
    ]


# Sidecars and notes live beside the recordings and must not be mistaken for them.
NON_AUDIO_SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".csv", ".srt", ".vtt", ".pdf", ".db"}


def voice_files(folder: Path) -> list[Path]:
    """Every recording in the folder, whatever it is named.

    ffmpeg decides what it can decode, so a file with an unusual extension or no
    extension at all is still offered to the pipeline rather than skipped here.
    """
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.casefold() not in NON_AUDIO_SUFFIXES
    )


def from_voice_pairs(folder: Path) -> tuple[list[Pair], list[dict], list[str]]:
    """Read recordings plus their transcript and translation sidecars."""
    pairs: list[Pair] = []
    manifest: list[dict] = []
    missing: list[str] = []
    for audio in voice_files(folder):
        kikuyu = read_sidecar(audio, TRANSCRIPT_SUFFIXES)
        if not kikuyu:
            missing.append(str(audio))
            continue
        manifest.append({"audio": str(audio), "text": kikuyu})
        english = read_sidecar(audio, TRANSLATION_SUFFIXES)
        if english:
            pairs.append(Pair(kikuyu, english, f"voice:{audio.name}", {"verified": True}))
    return pairs, manifest, missing


def session_text(folder: Path, *names: str) -> str:
    for name in names:
        candidate = folder / name
        if candidate.is_file():
            text = normalize_text(candidate.read_text(encoding="utf-8"))
            if text:
                return text
    return ""


def segment_rows(path: Path) -> dict[tuple[float, float], str]:
    if not path.is_file():
        return {}
    rows: dict[tuple[float, float], str] = {}
    for row in read_jsonl(path):
        text = normalize_text(str(row.get("text", "")))
        if text:
            rows[(float(row.get("start", 0.0)), float(row.get("end", 0.0)))] = text
    return rows


def from_sessions(sessions: Path, include_machine_output: bool) -> tuple[list[Pair], list[dict]]:
    """Read reviewed sessions, and raw output only when explicitly requested."""
    pairs: list[Pair] = []
    manifest: list[dict] = []
    if not sessions.is_dir():
        return pairs, manifest

    for folder in sorted(sessions.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        kikuyu = session_text(folder, "kikuyu.corrected.txt")
        english = session_text(folder, "english.corrected.txt")
        verified = bool(kikuyu or english)
        kikuyu = kikuyu or session_text(folder, "kikuyu.txt")
        english = english or session_text(folder, "english.txt")

        audio = folder / "audio_clean.wav"
        if kikuyu and verified and audio.is_file() and wav_duration(audio) > 0:
            manifest.append({"audio": str(audio), "text": kikuyu})

        if not (kikuyu and english):
            continue
        if not verified and not include_machine_output:
            continue

        # Per-segment rows are shorter and train better than one long blob, but
        # they only exist for machine output, so keep them behind the same gate.
        source_segments = segment_rows(folder / "asr_segments.jsonl")
        target_segments = segment_rows(folder / "english_segments.jsonl")
        shared = sorted(set(source_segments) & set(target_segments))
        if verified or not shared:
            pairs.append(Pair(kikuyu, english, f"session:{folder.name}", {"verified": verified}))
            continue
        for span in shared:
            pairs.append(Pair(source_segments[span], target_segments[span], f"session:{folder.name}", {"verified": False}))
    return pairs, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("data/translation/voice.jsonl"))
    parser.add_argument("--asr-manifest", type=Path, default=None, help="also write an ASR training manifest")
    parser.add_argument("--voice-dir", type=Path, default=Path("data/voice"), help="recordings with transcript sidecars")
    parser.add_argument("--sessions", type=Path, default=None, help="session folder (default: the configured one)")
    parser.add_argument("--corrections-db", type=Path, default=None, help="corrections database (default: the configured one)")
    parser.add_argument(
        "--include-machine-output",
        action="store_true",
        help="also emit unreviewed app output; this usually reinforces existing errors",
    )
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--max-length-ratio", type=float, default=3.0)
    args = parser.parse_args()

    settings = Settings.from_env(ROOT)
    sessions = args.sessions or settings.sessions
    corrections_db = args.corrections_db or settings.corrections_db

    correction_pairs = from_corrections(corrections_db)
    voice_pairs, voice_manifest, missing = from_voice_pairs(args.voice_dir)
    session_pairs, session_manifest = from_sessions(sessions, args.include_machine_output)

    all_pairs = correction_pairs + voice_pairs + session_pairs
    kept = dedupe(
        pair
        for pair in all_pairs
        if usable_pair(pair.kikuyu, pair.english, args.min_chars, args.max_chars, args.max_length_ratio)
    )
    written = write_jsonl(args.output, [pair.row() for pair in kept])

    print(f"corrections: {len(correction_pairs)} pair(s) from {corrections_db}")
    print(f"voice pairs: {len(voice_pairs)} pair(s) from {args.voice_dir}")
    print(f"sessions:    {len(session_pairs)} pair(s) from {sessions}")
    print(f"wrote {written} rows to {args.output} ({len(all_pairs) - len(kept)} filtered out)")

    verified = sum(1 for pair in kept if pair.meta.get("verified"))
    if written and verified < written:
        print(f"warning: {written - verified} of {written} rows are unreviewed machine output")

    if args.asr_manifest:
        manifest = voice_manifest + session_manifest
        seen: set[str] = set()
        unique = [row for row in manifest if not (row["audio"] in seen or seen.add(row["audio"]))]
        count = write_jsonl(args.asr_manifest, unique)
        print(f"wrote {count} ASR manifest row(s) to {args.asr_manifest}")

    if missing:
        print(f"\n{len(missing)} recording(s) have no transcript sidecar and were skipped:")
        for path in missing[:5]:
            print(f"  {path}  (add {Path(path).stem}.kikuyu.txt)")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")

    if not written:
        print(
            "\nNo usable rows yet. Record audio into data/voice/, write what was said into\n"
            "<name>.kikuyu.txt and what it means into <name>.english.txt, then re-run."
        )


if __name__ == "__main__":
    main()
