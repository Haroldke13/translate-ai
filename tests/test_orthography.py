"""Restoring Kikuyu tilde vowels on text typed without them."""

import json
from pathlib import Path

import pytest

from kikuyu_ai.orthography import (
    LEXICON_NAME,
    build_lexicon,
    fold,
    has_tilde,
    load_lexicon,
    match_case,
    prepare_for_translation,
    restore,
)

CORPUS = [
    "Rũciũ nĩ mũthenya mwega",
    "Rũciũ tũgaathiĩ",
    "Mũndũ ũcio nĩ mwega",
    "Ngai nĩ mwega hĩndĩ ciothe",
    "Rũciinĩ tũkaarĩa",
]


@pytest.fixture
def lexicon() -> dict[str, str]:
    return build_lexicon(CORPUS)


def test_fold_strips_diacritics_and_case():
    assert fold("Rũciũ") == "ruciu"
    assert fold("rũciũ") == fold("RUCIU") == "ruciu"


def test_has_tilde_detects_the_kikuyu_vowels():
    assert has_tilde("rũciũ")
    assert not has_tilde("ruciu")


def test_lexicon_only_stores_words_that_gain_accents(lexicon):
    assert lexicon["ruciu"] == "rũciũ"
    assert lexicon["mundu"] == "mũndũ"
    # "ngai" and "mwega" carry no tilde vowels, so there is nothing to record.
    assert "ngai" not in lexicon
    assert "mwega" not in lexicon


def test_lexicon_prefers_the_spelling_used_most_often():
    lexicon = build_lexicon(["thiĩ thiĩ thiĩ", "thii"])
    assert lexicon["thii"] == "thiĩ"


def test_restore_fills_in_missing_tilde_vowels(lexicon):
    fixed, changes = restore("Mundu ucio ni mwega", lexicon)

    assert fixed == "Mũndũ ũcio nĩ mwega"
    assert ("Mundu", "Mũndũ") in changes


def test_restore_leaves_text_the_writer_already_accented(lexicon):
    fixed, changes = restore("rũciũ", lexicon)

    assert fixed == "rũciũ"
    assert changes == []


def test_restore_leaves_unknown_and_english_words_alone(lexicon):
    fixed, changes = restore("hello world", lexicon)

    assert fixed == "hello world"
    assert changes == []


def test_restore_keeps_punctuation_and_numbers(lexicon):
    fixed, _changes = restore("Ruciu, ni mwega 2026!", lexicon)

    assert fixed == "Rũciũ, nĩ mwega 2026!"


@pytest.mark.parametrize(
    "original, replacement, expected",
    [("Mundu", "mũndũ", "Mũndũ"), ("mundu", "mũndũ", "mũndũ"), ("MUNDU", "mũndũ", "MŨNDŨ")],
)
def test_match_case_follows_the_writer(original, replacement, expected):
    assert match_case(original, replacement) == expected


def test_shouted_text_is_lowercased_before_translation(lexicon):
    """All-caps looks like an unknown token and gets echoed back untranslated."""
    prepared, changes = prepare_for_translation("RUCIU", lexicon)

    assert prepared == "rũciũ"
    assert changes


def test_a_normal_sentence_keeps_its_capitalisation(lexicon):
    prepared, _changes = prepare_for_translation("Mundu ucio ni mwega", lexicon)

    assert prepared == "Mũndũ ũcio nĩ mwega"


def test_empty_input_is_returned_unchanged(lexicon):
    assert prepare_for_translation("   ", lexicon) == ("", [])


def test_no_lexicon_means_the_text_passes_through():
    assert restore("Mundu ucio", {}) == ("Mundu ucio", [])


def _write_corpus(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "parallel.jsonl").write_text(
        "".join(json.dumps({"kikuyu": text, "english": "x"}, ensure_ascii=False) + "\n" for text in CORPUS),
        encoding="utf-8",
    )


def test_load_lexicon_builds_from_the_corpus_and_caches_it(tmp_path: Path):
    _write_corpus(tmp_path)

    lexicon = load_lexicon(tmp_path)

    assert lexicon["ruciu"] == "rũciũ"
    assert (tmp_path / LEXICON_NAME).is_file()


def test_a_stale_cache_is_rebuilt_when_the_corpus_changes(tmp_path: Path):
    _write_corpus(tmp_path)
    load_lexicon(tmp_path)
    (tmp_path / "parallel.jsonl").write_text(
        json.dumps({"kikuyu": "Kĩrĩra kĩa andũ", "english": "x"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lexicon = load_lexicon(tmp_path)

    assert lexicon["kirira"] == "kĩrĩra"
    assert "ruciu" not in lexicon


def test_a_damaged_cache_is_rebuilt_rather_than_raising(tmp_path: Path):
    _write_corpus(tmp_path)
    (tmp_path / LEXICON_NAME).write_text("{not json", encoding="utf-8")

    assert load_lexicon(tmp_path)["ruciu"] == "rũciũ"


def test_a_missing_corpus_gives_an_empty_lexicon(tmp_path: Path):
    assert load_lexicon(tmp_path) == {}
