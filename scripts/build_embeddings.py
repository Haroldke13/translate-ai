#!/usr/bin/env python3
"""Optional semantic index builder for parallel.jsonl."""
import argparse, json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("Install sentence-transformers and numpy for embeddings") from exc
    records = [json.loads(x) for x in args.input.read_text(encoding="utf-8").splitlines() if x.strip()]
    matrix = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").encode([r["kikuyu"] for r in records], normalize_embeddings=True)
    np.savez_compressed(args.output, embeddings=matrix, records=np.array(records, dtype=object))


if __name__ == "__main__":
    main()

