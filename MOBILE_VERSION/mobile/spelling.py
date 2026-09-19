"""Put back the accented letters a phone keyboard cannot type.

Kikuyu and Kamba are written with ĩ and ũ, but almost nobody has those on a
phone keyboard, so people type "ruciu" for "rũciũ". To a translation model those
are two different words, and the unaccented one usually comes back untranslated.

The fix is a lookup table: for every folded spelling, the accented spelling the
Bible corpus uses most often. The table is built once on the computer by
tools/pack_mobile.py and shipped as a small JSON file, so the phone never has to
read 31,000 verses to answer one question.

Nothing here needs a model or a network. Languages written without diacritics
ship an empty table and pass their text through untouched.
"""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
from pathlib import Path

SPELLING_NAME = "spelling.json.gz"
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def fold(value: str) -> str:
    """Lower-case and drop the diacritics, so ruciu and rũciũ share one key."""
    decomposed = unicodedata.normalize("NFD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def is_accented(word: str) -> bool:
    """Whether the writer already typed the accents themselves.

    Asking whether folding changes the word covers any language's diacritics,
    rather than hard-coding the two vowels Kikuyu happens to use.
    """
    return fold(word) != word.casefold()


def build_table(texts) -> dict[str, str]:
    """Map each folded word to the accented spelling used most often.

    Only spellings that actually gain an accent are stored; a word that is
    already its own folded form would just bloat the table.
    """
    from collections import Counter, defaultdict

    forms: dict[str, Counter] = defaultdict(Counter)
    for text in texts:
        for word in WORD_RE.findall(text):
            forms[fold(word)][word.casefold()] += 1

    table: dict[str, str] = {}
    for key, counter in forms.items():
        best, _count = counter.most_common(1)[0]
        if best != key:
            table[key] = best
    return table


def load_table(data_dir: Path) -> dict[str, str]:
    """The packed table for one language, or an empty one if it has none."""
    path = data_dir / SPELLING_NAME
    if not path.is_file():
        return {}
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle).get("words", {})
    except (OSError, json.JSONDecodeError, KeyError, AttributeError):
        return {}  # a damaged table costs accents, not the whole translation


def match_case(original: str, replacement: str) -> str:
    """Give the replacement the capitalisation the writer used."""
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def restore(text: str, table: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    """Add the missing accents, returning the text and what changed.

    A word the writer already accented is trusted as typed, and a word the
    corpus has never seen is left exactly as it is.
    """
    if not text or not table:
        return text, []

    changes: list[tuple[str, str]] = []

    def replace(match: re.Match) -> str:
        word = match.group(0)
        if is_accented(word):
            return word
        accented = table.get(fold(word))
        if not accented:
            return word
        fixed = match_case(word, accented)
        if fixed != word:
            changes.append((word, fixed))
        return fixed

    return WORD_RE.sub(replace, text), changes


def prepare(text: str, table: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    """The spelling to hand the translator, plus the corrections made.

    Text shouted in capitals is lower-cased first: the corpus is ordinary prose,
    so an all-caps word looks like an unknown token and tends to be copied
    through untranslated.
    """
    source = text.strip()
    if not source:
        return source, []
    letters = [char for char in source if char.isalpha()]
    if letters and all(char.isupper() for char in letters):
        source = source.casefold()
    return restore(source, table)
