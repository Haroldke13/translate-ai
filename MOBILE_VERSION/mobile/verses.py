"""Exact Bible verse lookup, offline, with no model involved.

When somebody speaks or types a verse, the best possible English is not a
machine translation — it is the published English of that verse. Matching it is
a search problem, not a translation problem, and search runs comfortably on a
phone.

The index is an inverted list from word to verse, which turns "compare against
31,000 verses" into "look at the few hundred verses that share a word with the
query". Scores are Jaccard overlap, so a long verse cannot win just by being
long. Everything is stdlib: this tier works on a phone with nothing installed
but Python.
"""

from __future__ import annotations

import gzip
import json
import pickle
import re
import tempfile
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

VERSES_NAME = "verses.jsonl.gz"
CACHE_NAME = "verses.index"
#: Below this the "match" is a coincidence of common words, not the same verse.
DEFAULT_THRESHOLD = 0.82
PUNCTUATION = ".,!?;:()[]{}\"'“”‘’—–-«»…"


@dataclass(frozen=True)
class Verse:
    book: str
    chapter: int
    verse: int
    source: str
    english: str
    score: float

    @property
    def reference(self) -> str:
        return f"{self.book} {self.chapter}:{self.verse}"


def tokens(value: str) -> list[str]:
    """Words worth indexing: lower-cased, unpunctuated, longer than a letter.

    Single letters carry almost no signal and appear nearly everywhere, so
    dropping them shortens the posting lists without costing accuracy.
    """
    out = []
    for word in value.split():
        word = word.casefold().strip(PUNCTUATION)
        if len(word) > 1:
            out.append(word)
    return out


class VerseIndex:
    """Searchable parallel verses for one language.

    Built lazily and cached, because a language the user never opens should not
    cost anything, and a language they use every day should not be rebuilt every
    time the server starts.
    """

    def __init__(self, data_dir: Path, threshold: float = DEFAULT_THRESHOLD):
        self.data_dir = Path(data_dir)
        self.threshold = threshold
        self.rows: list[dict] = []
        self.postings: dict[str, array] = {}
        #: Distinct indexed tokens per verse, the denominator half of Jaccard.
        self.lengths: array = array("i")
        self._loaded = False

    @property
    def available(self) -> bool:
        return (self.data_dir / VERSES_NAME).is_file()

    def __len__(self) -> int:
        return len(self.rows)

    def load(self) -> "VerseIndex":
        if self._loaded:
            return self
        if not self.available:
            self._loaded = True
            return self
        if not self._load_cache():
            self._read_verses()
            self._build()
            self._save_cache()
        self._loaded = True
        return self

    def _read_verses(self) -> None:
        rows: list[dict] = []
        with gzip.open(self.data_dir / VERSES_NAME, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # one bad line should not lose the other 31,000
        self.rows = rows

    def _build(self) -> None:
        postings: dict[str, array] = defaultdict(lambda: array("i"))
        lengths = array("i")
        for index, row in enumerate(self.rows):
            unique = set(tokens(row.get("s", "")))
            lengths.append(len(unique))
            for token in unique:
                postings[token].append(index)
        self.postings = dict(postings)
        self.lengths = lengths

    # -- caching ---------------------------------------------------------
    # Parsing 31,000 JSON lines and re-indexing them takes seconds on a phone.
    # Doing it once and pickling the result makes every later start instant.

    def _cache_path(self) -> Path:
        writable = self.data_dir / CACHE_NAME
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            probe = self.data_dir / ".write-probe"
            probe.touch()
            probe.unlink()
            return writable
        except OSError:
            # Installed somewhere read-only, e.g. straight off an SD card.
            return Path(tempfile.gettempdir()) / f"kikuyu-{self.data_dir.name}-{CACHE_NAME}"

    def _signature(self) -> str:
        """Identify the corpus by size alone, deliberately not by timestamp.

        Copying a folder to a phone over USB or MTP rarely preserves mtime, and
        a signature that included it would throw away a perfectly good prebuilt
        index on the phone's very first run.
        """
        return str((self.data_dir / VERSES_NAME).stat().st_size)

    def _load_cache(self) -> bool:
        path = self._cache_path()
        if not path.is_file():
            return False
        try:
            with path.open("rb") as handle:
                data = pickle.load(handle)
            if data.get("signature") != self._signature():
                return False
            self.rows = data["rows"]
            self.postings = data["postings"]
            self.lengths = data["lengths"]
            return True
        except (OSError, pickle.UnpicklingError, KeyError, AttributeError, EOFError, ValueError):
            return False  # a stale or truncated cache is rebuilt, never fatal

    def _save_cache(self) -> None:
        path = self._cache_path()
        payload = {
            "signature": self._signature(),
            "rows": self.rows,
            "postings": self.postings,
            "lengths": self.lengths,
        }
        try:
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)
        except OSError:
            pass  # a read-only install just pays the build cost each start

    def search(self, text: str, threshold: float | None = None) -> Verse | None:
        """The published verse this text is, or None if it is not one.

        Pass a lower threshold to get the closest verse rather than a confident
        one — useful for showing "did you mean" context, never for claiming a
        translation.
        """
        self.load()
        limit = self.threshold if threshold is None else threshold
        if not self.rows:
            return None
        query = set(tokens(text))
        if not query:
            return None

        hits: Counter[int] = Counter()
        for token in query:
            posting = self.postings.get(token)
            if posting:
                hits.update(posting)
        if not hits:
            return None

        best_index, best_score = -1, 0.0
        size = len(query)
        for index, shared in hits.items():
            union = size + self.lengths[index] - shared
            score = shared / union if union else 0.0
            if score > best_score:
                best_index, best_score = index, score

        if best_index < 0 or best_score < limit:
            return None
        row = self.rows[best_index]
        return Verse(
            book=str(row.get("b", "")),
            chapter=int(row.get("c", 0)),
            verse=int(row.get("v", 0)),
            source=str(row.get("s", "")),
            english=str(row.get("e", "")),
            score=round(best_score, 4),
        )
