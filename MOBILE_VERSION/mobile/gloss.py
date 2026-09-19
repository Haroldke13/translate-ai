"""Word-by-word English for text no model and no verse can handle.

This is the floor, not the goal. When the neural translator is not installed and
the sentence is not scripture, a gloss still tells you roughly what was said:
which words carry meaning and what each one is. It gets the words right and the
grammar wrong, and the interface says so rather than passing it off as a
translation.

The table comes from IBM Model 1 alignment over the parallel Bible corpus,
computed on the computer by tools/pack_mobile.py. On the phone this file only
does lookups, so it needs nothing but the standard library.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path

GLOSS_NAME = "gloss.json.gz"
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
#: Below this the alignment is noise and a wrong gloss is worse than none.
MIN_PROBABILITY = 0.12


@dataclass(frozen=True)
class Glossed:
    text: str
    #: Fraction of the words that had an entry, which is the honest confidence.
    coverage: float
    unknown: list[str]


class GlossTable:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.words: dict[str, list[str]] = {}
        #: Unaccented spelling -> accented key, so "muno" finds "mũno".
        self.folded: dict[str, str] = {}
        self._loaded = False

    @property
    def available(self) -> bool:
        return (self.data_dir / GLOSS_NAME).is_file()

    def load(self) -> "GlossTable":
        if self._loaded:
            return self
        self._loaded = True
        if not self.available:
            return self
        try:
            with gzip.open(self.data_dir / GLOSS_NAME, "rt", encoding="utf-8") as handle:
                self.words = json.load(handle).get("words", {})
        except (OSError, json.JSONDecodeError, AttributeError):
            self.words = {}  # a damaged table degrades to "no gloss", not a crash
        self._build_folded()
        return self

    def _build_folded(self) -> None:
        """Index the table a second time without accents.

        The corpus is correctly accented, so its keys are too, but people type
        "muno" for "mũno". Spelling repair fixes most of that before the gloss
        is reached; this catches the words the spelling table did not know.
        Built here rather than packed, because it derives entirely from the keys
        and would only make the shipped file bigger.
        """
        from .spelling import fold

        folded: dict[str, str] = {}
        for key in self.words:
            plain = fold(key)
            # First writer wins, matching the table's own most-frequent-first order.
            if plain != key and plain not in folded:
                folded[plain] = key
        self.folded = folded

    def _lookup(self, word: str) -> list[str]:
        from .spelling import fold

        # Loaded here rather than only in gloss(), so a direct best() call on a
        # fresh table answers from the file instead of from an empty dict.
        self.load()
        key = word.casefold()
        options = self.words.get(key)
        if options:
            return options
        accented = self.folded.get(fold(key))
        return self.words.get(accented, []) if accented else []

    def best(self, word: str) -> str | None:
        options = self._lookup(word)
        return options[0] if options else None

    def alternatives(self, word: str) -> list[str]:
        return list(self._lookup(word))

    def gloss(self, text: str) -> Glossed | None:
        """A word-for-word rendering, or None when too little is known.

        A gloss that recognises a third of its words is not informative, it is
        misleading, so it is withheld rather than shown.
        """
        self.load()
        if not self.words or not text.strip():
            return None

        found = 0
        total = 0
        unknown: list[str] = []
        pieces: list[str] = []
        position = 0
        for match in WORD_RE.finditer(text):
            pieces.append(text[position:match.start()])
            position = match.end()
            word = match.group(0)
            total += 1
            english = self.best(word)
            if english:
                found += 1
                pieces.append(english)
            else:
                unknown.append(word)
                # An unknown word is shown as it was written, so the reader can
                # see exactly which part was not understood.
                pieces.append(f"[{word}]")
        pieces.append(text[position:])

        if not total:
            return None
        coverage = found / total
        if coverage < 0.5:
            return None
        return Glossed(text="".join(pieces).strip(), coverage=round(coverage, 3), unknown=unknown)


def train(pairs, iterations: int = 5, keep: int = 3, min_probability: float = MIN_PROBABILITY):
    """IBM Model 1 by expectation maximisation, in plain Python.

    Runs on the computer at pack time, never on the phone. Model 1 ignores word
    order, which is exactly why it suits a gloss: it learns what a word means,
    and makes no claim about where it belongs in the sentence.
    """
    from collections import defaultdict

    source_sentences = [[w.casefold() for w in WORD_RE.findall(s)] for s, _ in pairs]
    target_sentences = [[w.casefold() for w in WORD_RE.findall(e)] for _, e in pairs]

    # Uniform start: every English word is equally likely for every source word.
    candidates: dict[str, set[str]] = defaultdict(set)
    for source, target in zip(source_sentences, target_sentences):
        for word in source:
            candidates[word].update(target)
    probability = {
        word: dict.fromkeys(options, 1.0 / len(options))
        for word, options in candidates.items()
        if options
    }

    for _ in range(iterations):
        counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        totals: dict[str, float] = defaultdict(float)
        for source, target in zip(source_sentences, target_sentences):
            if not source or not target:
                continue
            for english in target:
                denominator = sum(probability[w].get(english, 0.0) for w in source if w in probability)
                if denominator <= 0.0:
                    continue
                for word in source:
                    table = probability.get(word)
                    if not table:
                        continue
                    share = table.get(english, 0.0) / denominator
                    if share:
                        counts[word][english] += share
                        totals[word] += share
        probability = {
            word: {english: value / totals[word] for english, value in table.items()}
            for word, table in counts.items()
            if totals.get(word)
        }

    return {
        word: [english for english, _ in sorted(table.items(), key=lambda item: -item[1])[:keep]]
        for word, table in (
            (word, {e: p for e, p in table.items() if p >= min_probability})
            for word, table in probability.items()
        )
        if table
    }
