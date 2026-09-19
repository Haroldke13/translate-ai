"""Restore Kikuyu tilde vowels on text that was typed without them.

Kikuyu is written with ĩ and ũ, but phone keyboards rarely have them, so people
type "ruciu" for "rũciũ". The translation model treats those as different words
and often just echoes the unaccented one back untranslated.

The aligned Bible corpus is 31k verses of correctly accented Kikuyu, so it can
say what the accented spelling of a folded word normally is. Nothing here needs a
network connection or a model: it is a lookup table built from local data.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

LEXICON_NAME = "orthography.json"
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
TILDE_VOWELS = "ĩũĨŨ"


def fold(value: str) -> str:
    """Lower-case and strip the diacritics, so ruciu and rũciũ share a key."""
    decomposed = unicodedata.normalize("NFD", value.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def has_tilde(value: str) -> bool:
    return any(char in TILDE_VOWELS for char in value)


def build_lexicon(texts) -> dict[str, str]:
    """Map each folded word to the accented spelling used most often."""
    forms: dict[str, Counter] = defaultdict(Counter)
    for text in texts:
        for word in WORD_RE.findall(text):
            forms[fold(word)][word.casefold()] += 1

    lexicon: dict[str, str] = {}
    for key, counter in forms.items():
        best, _count = counter.most_common(1)[0]
        # Only worth storing when it actually adds accents.
        if best != key:
            lexicon[key] = best
    return lexicon


def kikuyu_texts(bible_dir: Path):
    path = bible_dir / "parallel.jsonl"
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("kikuyu")
            if text:
                yield str(text)


def load_lexicon(bible_dir: Path, rebuild: bool = False) -> dict[str, str]:
    """Load the cached lexicon, building it from the corpus when needed."""
    cache = bible_dir / LEXICON_NAME
    source = bible_dir / "parallel.jsonl"
    if not source.is_file():
        return {}
    signature = str(source.stat().st_size)

    if cache.is_file() and not rebuild:
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("signature") == signature:
                return data.get("words", {})
        except (json.JSONDecodeError, OSError):
            pass  # a damaged cache is rebuilt rather than fatal

    lexicon = build_lexicon(kikuyu_texts(bible_dir))
    try:
        cache.write_text(
            json.dumps({"signature": signature, "words": lexicon}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass  # a read-only data folder is not a reason to fail
    return lexicon


def match_case(original: str, replacement: str) -> str:
    """Give the replacement the capitalisation the writer used."""
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def restore(text: str, lexicon: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    """Add the missing tilde vowels, returning the text and what changed.

    A word the writer already accented is left alone, and so is a word the corpus
    has never seen.
    """
    if not text or not lexicon:
        return text, []

    changes: list[tuple[str, str]] = []

    def replace(match: re.Match) -> str:
        word = match.group(0)
        if has_tilde(word):
            return word
        accented = lexicon.get(fold(word))
        if not accented:
            return word
        fixed = match_case(word, accented)
        if fixed != word:
            changes.append((word, fixed))
        return fixed

    return WORD_RE.sub(replace, text), changes


def prepare_for_translation(text: str, lexicon: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    """The spelling to hand the model, plus the corrections that were made.

    Text shouted in capitals is lower-cased first. The corpus is ordinary prose,
    so an all-caps word looks like an unknown token to the model and tends to be
    copied through untranslated.
    """
    source = text.strip()
    if not source:
        return source, []
    letters = [char for char in source if char.isalpha()]
    if letters and all(char.isupper() for char in letters):
        source = source.casefold()
    return restore(source, lexicon)
