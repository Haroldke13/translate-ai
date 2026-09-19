"""Build the phone package on the computer, so the phone never has to.

Everything expensive happens here: reading 31,000 verses per language, learning
the word alignment table, working out the accented spellings. The phone gets
compact files it only ever reads.

Run it from the project root:

    .venv/bin/python MOBILE_VERSION/tools/pack_mobile.py --data
    .venv/bin/python MOBILE_VERSION/tools/pack_mobile.py --translation
    .venv/bin/python MOBILE_VERSION/tools/pack_mobile.py --speech /path/to/model.int8.onnx

Then copy the whole MOBILE_VERSION folder to the phone.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
MOBILE = HERE.parent
PROJECT = MOBILE.parent
sys.path.insert(0, str(MOBILE))

from mobile import gloss, languages, spelling  # noqa: E402
from mobile.neural import RUNTIME_NAME  # noqa: E402

#: Where each language's parallel verses live in the desktop project. Kikuyu
#: sits at the top level for historical reasons; the phone build normalises
#: every language to data/<code>/ instead.
SOURCE_CORPORA = {
    "kikuyu": PROJECT / "data" / "bible" / "parallel.jsonl",
    "oromo": PROJECT / "data" / "bible" / "oromo" / "parallel.jsonl",
    "somali": PROJECT / "data" / "bible" / "somali" / "parallel.jsonl",
}


def read_pairs(path: Path) -> list[tuple[str, str]]:
    """Source/English pairs from a desktop corpus file.

    The desktop format names the source column "kikuyu" whatever the language,
    so that key is read for every language rather than renamed per file.
    """
    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = (row.get("kikuyu") or "").strip()
            english = (row.get("english") or "").strip()
            if source and english:
                pairs.append((source, english))
    return pairs


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = (row.get("kikuyu") or "").strip()
            english = (row.get("english") or "").strip()
            if not source or not english:
                continue
            rows.append(
                {
                    "b": row.get("book", ""),
                    "c": int(row.get("chapter", 0) or 0),
                    "v": int(row.get("verse", 0) or 0),
                    "s": source,
                    "e": english,
                }
            )
    return rows


def write_gzip_json(path: Path, payload: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return path.stat().st_size


def write_gzip_jsonl(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path.stat().st_size


def pack_language(code: str, iterations: int, keep: int) -> dict:
    """Build one language's verse file, spelling table and gloss table."""
    language = languages.get(code)
    source = SOURCE_CORPORA.get(code)
    target = language.data_dir(MOBILE)
    target.mkdir(parents=True, exist_ok=True)

    if not language.corpus or not source or not source.is_file():
        # Recorded rather than skipped silently: the interface tells the user
        # which features this language does not have, and needs to know why.
        meta = {
            "language": code,
            "corpus": False,
            "reason": "No openly licensed parallel Bible text is available for this language.",
            "verses": 0,
        }
        (target / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return meta

    started = time.monotonic()
    rows = read_rows(source)
    pairs = [(row["s"], row["e"]) for row in rows]

    verses_bytes = write_gzip_jsonl(target / "verses.jsonl.gz", rows)

    table = spelling.build_table(row["s"] for row in rows)
    spelling_bytes = write_gzip_json(
        target / spelling.SPELLING_NAME,
        {"language": code, "words": table},
    )

    words = gloss.train(pairs, iterations=iterations, keep=keep)
    gloss_bytes = write_gzip_json(
        target / gloss.GLOSS_NAME,
        {"language": code, "iterations": iterations, "words": words},
    )

    meta = {
        "language": code,
        "corpus": True,
        "source": str(source.relative_to(PROJECT)),
        "verses": len(rows),
        "accented_spellings": len(table),
        "gloss_entries": len(words),
        "bytes": {
            "verses": verses_bytes,
            "spelling": spelling_bytes,
            "gloss": gloss_bytes,
        },
        "seconds": round(time.monotonic() - started, 1),
    }
    (target / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def pack_data(iterations: int, keep: int, jobs: int) -> list[dict]:
    codes = [language.code for language in languages.ordered()]
    print(f"packing {len(codes)} languages: {', '.join(codes)}", flush=True)
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_pack_one, [(code, iterations, keep) for code in codes]))
    else:
        results = [pack_language(code, iterations, keep) for code in codes]
    for meta in results:
        if meta.get("corpus"):
            total = sum(meta["bytes"].values()) / 1_048_576
            print(
                f"  {meta['language']:8} {meta['verses']:>6} verses  "
                f"{meta['gloss_entries']:>6} glosses  {total:5.1f} MB  {meta['seconds']}s",
                flush=True,
            )
        else:
            print(f"  {meta['language']:8} no corpus — {meta['reason']}", flush=True)
    return results


def _pack_one(job: tuple[str, int, int]) -> dict:
    return pack_language(*job)


NLLB_SOURCE = PROJECT / "models" / "downloads" / "huggingface" / "nickdee96--nllb-200-600m-kikuyu-english"


def pack_translation(quantization: str = "int8") -> dict:
    """Convert NLLB to CTranslate2, which is the only runtime a phone can get.

    Termux has no prebuilt safetensors, so installing transformers there means
    compiling Rust. CTranslate2 is packaged at 5.5 MB, and int8 halves the model
    again — 1.2 GB of fp16 weights become about 620 MB.
    """
    import subprocess

    target = MOBILE / "models" / "translation"
    if not (NLLB_SOURCE / "config.json").is_file():
        raise SystemExit(f"NLLB model not found at {NLLB_SOURCE}")

    print(f"converting NLLB to CTranslate2 ({quantization})…", flush=True)
    if target.exists():
        shutil.rmtree(target)
    completed = subprocess.run(
        [
            sys.executable.replace("python", "ct2-transformers-converter")
            if (Path(sys.executable).parent / "ct2-transformers-converter").exists()
            else str(Path(sys.executable).parent / "ct2-transformers-converter"),
            "--model", str(NLLB_SOURCE),
            "--output_dir", str(target),
            "--quantization", quantization,
            "--force",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise SystemExit(f"conversion failed:\n{completed.stderr[-2000:]}")

    # The tokenizer travels with the model: tokenizer.json is all the phone
    # needs, which is what makes sentencepiece unnecessary there.
    shutil.copyfile(NLLB_SOURCE / "tokenizer.json", target / "tokenizer.json")

    total = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    meta = {"backend": "ctranslate2", "quantization": quantization, "bytes": total}
    print(f"  translation model: {total / 1_048_576:.0f} MB", flush=True)
    update_runtime({
        "translation_backend": "ctranslate2",
        "translation_compute_type": quantization,
        "translation_beam_size": 4,
        "translation_max_new_tokens": 128,
    })
    return meta


def pack_speech(model_path: Path, tokens_path: Path | None = None) -> dict:
    """Copy an ONNX CTC speech model into the phone package.

    ONNX rather than PyTorch because torch cannot be installed on Termux
    without a Rust toolchain, and a single multilingual CTC model rather than
    MMS because MMS bakes its adapter into the weights — four languages would
    mean four separate exports of about a gigabyte each.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise SystemExit(f"speech model not found: {model_path}")
    tokens_path = Path(tokens_path) if tokens_path else model_path.parent / "tokens.txt"
    if not tokens_path.is_file():
        raise SystemExit(f"tokens.txt not found next to the model: {tokens_path}")

    target = MOBILE / "models" / "speech"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_path, target / model_path.name)
    shutil.copyfile(tokens_path, target / "tokens.txt")
    total = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    print(f"  speech model: {total / 1_048_576:.0f} MB", flush=True)
    update_runtime({"speech_file": model_path.name, "speech_backend": "onnxruntime"})
    return {"backend": "onnxruntime", "bytes": total}


def update_runtime(changes: dict) -> None:
    """Record loading choices next to the models rather than in the code.

    The packer knows what it built; the phone should not have to guess and
    discover it was wrong halfway through reading a 600 MB file.
    """
    path = MOBILE / "models" / RUNTIME_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
    current.update(changes)
    current["packed_by"] = "tools/pack_mobile.py"
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", action="store_true", help="build the offline corpus tier")
    parser.add_argument("--translation", action="store_true", help="convert NLLB to CTranslate2 int8")
    parser.add_argument("--quantization", default="int8", help="int8 (default), int8_float16, float16")
    parser.add_argument("--speech", metavar="MODEL.onnx", help="copy an ONNX CTC speech model in")
    parser.add_argument("--tokens", metavar="tokens.txt", help="token list for the speech model")
    parser.add_argument("--iterations", type=int, default=5, help="IBM Model 1 EM passes (default 5)")
    parser.add_argument("--keep", type=int, default=3, help="English options kept per source word")
    parser.add_argument("--jobs", type=int, default=3, help="languages to pack in parallel")
    args = parser.parse_args(argv)

    if not (args.data or args.translation or args.speech):
        parser.print_help()
        return 0

    if args.data:
        pack_data(args.iterations, args.keep, max(1, args.jobs))
    if args.translation:
        pack_translation(args.quantization)
    if args.speech:
        pack_speech(Path(args.speech), Path(args.tokens) if args.tokens else None)

    report()
    return 0


def report() -> None:
    """What the phone package now weighs, broken down by what it buys."""
    parts = [
        ("offline data (verses, spelling, word lists)", MOBILE / "data"),
        ("translation model", MOBILE / "models" / "translation"),
        ("speech model", MOBILE / "models" / "speech"),
        ("app itself", MOBILE / "mobile"),
    ]
    print("\npackage size:")
    total = 0
    for label, path in parts:
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0
        total += size
        print(f"  {label:44} {size / 1_048_576:8.1f} MB" + ("" if size else "   (not installed)"))
    print(f"  {'TOTAL':44} {total / 1_048_576:8.1f} MB")


if __name__ == "__main__":
    raise SystemExit(main())
