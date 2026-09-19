"""Shared text handling for building offline Kikuyu-English training corpora.

Everything here is pure standard library so corpus building works on a phone
with no network access, no API keys, and no heavyweight NLP dependencies.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

# Kikuyu is written with i/u tilde vowels and a small closed set of very common
# grammatical words. English is detected from its own stopword set. Neither list
# needs to be exhaustive; they only have to separate the two languages.
KIKUYU_MARKERS = frozenset(
    """
    na ni wa ya cia kia gia mu ria thi uria ati no ta ku ma ha we hindi ngai
    mwathani andu mundu iria ciake wake wao akia nigetha tondu niguo mwena
    thutha nginya ngoma roho ithe maithe muoyo utuku muthenya kuria uge oiga
    """.split()
)
ENGLISH_MARKERS = frozenset(
    """
    the and of to in that he for is was with they be his you not it as will
    from have their but all this which said them then were when there who
    """.split()
)

SENTENCE_END = re.compile(r"(?<=[.!?…])[\s ]+(?=[\"'“‘(\[]?[A-ZĨŨ])")
PAGE_NUMBER = re.compile(r"^\s*(?:page\s+)?[-–—\[(]?\s*\d{1,4}\s*[-–—\])]?\s*$", re.IGNORECASE)
VERSE_PREFIX = re.compile(r"^\s*\d{1,3}[.:)]?\s+")
HYPHEN_BREAK = re.compile(r"(\w)[-‐‑]\n(\w)")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(value: str) -> str:
    """Collapse PDF/OCR whitespace damage without touching Kikuyu diacritics."""
    value = unicodedata.normalize("NFC", value or "")
    value = CONTROL.sub(" ", value)
    value = HYPHEN_BREAK.sub(r"\1\2", value)
    value = value.replace("­", "")
    value = re.sub(r"[   ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def strip_verse_number(value: str) -> str:
    """Drop a leading verse or list number that survived text extraction."""
    return VERSE_PREFIX.sub("", value, count=1).strip()


def is_noise_line(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return True
    if PAGE_NUMBER.match(stripped):
        return True
    letters = sum(1 for char in stripped if char.isalpha())
    return letters < max(2, len(stripped) // 4)


def split_sentences(text: str) -> list[str]:
    """Split normalized text into sentence-like units."""
    text = normalize_text(text)
    if not text:
        return []
    parts = [part.strip() for part in SENTENCE_END.split(text)]
    return [part for part in parts if part]


def _words(value: str) -> list[str]:
    folded = unicodedata.normalize("NFD", value.casefold())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return re.findall(r"[a-z']+", folded)


def language_scores(value: str) -> tuple[float, float]:
    """Return (kikuyu_score, english_score) in 0..1 for a line of text."""
    words = _words(value)
    if not words:
        return 0.0, 0.0
    kikuyu_hits = sum(1 for word in words if word in KIKUYU_MARKERS)
    english_hits = sum(1 for word in words if word in ENGLISH_MARKERS)
    # Tilde vowels appear in Kikuyu only, so treat them as strong evidence.
    tilde = sum(1 for char in value if char in "ĩũĨŨ")
    kikuyu = (kikuyu_hits + min(tilde, len(words))) / len(words)
    return min(1.0, kikuyu), min(1.0, english_hits / len(words))


def guess_language(value: str) -> str:
    """Return 'kikuyu', 'english', or 'unknown' for a line of text."""
    kikuyu, english = language_scores(value)
    if kikuyu >= english + 0.06 and kikuyu > 0.04:
        return "kikuyu"
    if english >= kikuyu + 0.06 and english > 0.04:
        return "english"
    return "unknown"


@dataclass(frozen=True)
class Pair:
    kikuyu: str
    english: str
    source: str = ""
    meta: dict = field(default_factory=dict)

    def row(self) -> dict:
        row = {"kikuyu": self.kikuyu, "english": self.english}
        if self.source:
            row["source"] = self.source
        row.update(self.meta)
        return row


def pair_key(kikuyu: str, english: str) -> tuple[str, str]:
    return (" ".join(_words(kikuyu)), " ".join(_words(english)))


def usable_pair(
    kikuyu: str,
    english: str,
    min_chars: int = 8,
    max_chars: int = 1200,
    max_length_ratio: float = 3.0,
) -> bool:
    """Reject pairs that would teach the model noise rather than translation."""
    kikuyu, english = kikuyu.strip(), english.strip()
    if not kikuyu or not english:
        return False
    if len(kikuyu) < min_chars or len(english) < min_chars:
        return False
    if len(kikuyu) > max_chars or len(english) > max_chars:
        return False
    ratio = len(kikuyu) / max(1, len(english))
    if ratio > max_length_ratio or ratio < 1.0 / max_length_ratio:
        return False
    # An identical source and target is an extraction failure, not a translation.
    return pair_key(kikuyu, english)[0] != pair_key(kikuyu, english)[1]


def dedupe(pairs: Iterable[Pair]) -> list[Pair]:
    seen: set[tuple[str, str]] = set()
    unique: list[Pair] = []
    for pair in pairs:
        key = pair_key(pair.kikuyu, pair.english)
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)
    return unique


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from None


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


# --- Gale & Church length-based sentence alignment -------------------------
# Standard offline algorithm for aligning a translated document to its source
# using only sentence lengths, so it needs no dictionary and no network.

_BEADS = ((1, 1, 0.0), (1, 0, 6.8), (0, 1, 6.8), (2, 1, 5.5), (1, 2, 5.5), (2, 2, 8.0))
_VARIANCE = 6.8
_MEAN_RATIO = 1.0


def _bead_cost(source_length: int, target_length: int, mean_ratio: float) -> float:
    if source_length == 0 and target_length == 0:
        return 0.0
    mean = (source_length + target_length / mean_ratio) / 2.0
    if mean <= 0:
        return 0.0
    delta = (target_length - source_length * mean_ratio) / math.sqrt(_VARIANCE * mean)
    # -2 * log(probability of this delta under a normal distribution).
    return abs(delta) * 2.0


def align_blocks(
    source: Sequence[str],
    target: Sequence[str],
    mean_ratio: float | None = None,
) -> list[tuple[str, str]]:
    """Align two lists of sentences into 1-1, 1-2, 2-1 and 2-2 pairs.

    Returns only the pairs that have text on both sides; insertions and
    deletions are dropped because they cannot be used as training rows.
    """
    if not source or not target:
        return []
    if mean_ratio is None:
        source_chars = sum(len(item) for item in source) or 1
        target_chars = sum(len(item) for item in target) or 1
        mean_ratio = max(0.25, min(4.0, target_chars / source_chars))

    rows, columns = len(source), len(target)
    best = [[math.inf] * (columns + 1) for _ in range(rows + 1)]
    back: list[list[tuple[int, int]]] = [[(0, 0)] * (columns + 1) for _ in range(rows + 1)]
    best[0][0] = 0.0

    for i in range(rows + 1):
        for j in range(columns + 1):
            if best[i][j] == math.inf:
                continue
            for di, dj, penalty in _BEADS:
                ni, nj = i + di, j + dj
                if ni > rows or nj > columns:
                    continue
                source_length = sum(len(source[k]) for k in range(i, ni))
                target_length = sum(len(target[k]) for k in range(j, nj))
                cost = best[i][j] + penalty + _bead_cost(source_length, target_length, mean_ratio)
                if cost < best[ni][nj]:
                    best[ni][nj] = cost
                    back[ni][nj] = (di, dj)

    pairs: list[tuple[str, str]] = []
    i, j = rows, columns
    while i > 0 or j > 0:
        di, dj = back[i][j]
        if di == 0 and dj == 0:
            break
        if di and dj:
            pairs.append((" ".join(source[i - di:i]), " ".join(target[j - dj:j])))
        i, j = i - di, j - dj
    pairs.reverse()
    return pairs
