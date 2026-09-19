"""Tests for the phone build.

Two things are being defended here. One is behaviour: the tier order, the
honesty of what each tier claims, the parsers. The other is the phone
environment itself — Termux ships Python 3.14, where audioop and cgi no longer
exist, so anything that quietly depends on them has to fail here rather than on
someone's phone.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import struct
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mobile import gloss, languages, multipart, spelling  # noqa: E402
from mobile.audio import _decode_pcm, _resample, _to_mono, read_samples, safe_suffix  # noqa: E402
from mobile.engine import Engine  # noqa: E402
from mobile.verses import VerseIndex, tokens  # noqa: E402


# -- the phone's Python ----------------------------------------------------

def test_no_removed_stdlib_modules():
    """Nothing may import audioop or cgi: both are gone in Python 3.14."""
    offenders = []
    for path in (ROOT / "mobile").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for banned in ("import audioop", "import cgi\n", "from cgi ", "from audioop "):
            if banned in text:
                offenders.append(f"{path.name}: {banned.strip()}")
    assert not offenders, offenders


def test_runtime_uses_only_the_standard_library():
    """The offline tier must import with nothing installed."""
    allowed = {
        "__future__", "annotations", "argparse", "array", "collections", "dataclasses", "gzip",
        "http", "importlib", "json", "mimetypes", "os", "pathlib", "pickle", "re",
        "shutil", "socket", "struct", "subprocess", "sys", "tempfile", "threading",
        "time", "traceback", "unicodedata", "uuid", "wave",
    }
    optional = {"ctranslate2", "tokenizers", "onnxruntime", "numpy", "torch", "transformers"}
    import ast

    for name in ("languages", "spelling", "verses", "gloss", "engine", "audio", "multipart", "server"):
        tree = ast.parse((ROOT / "mobile" / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [(node.module or "").split(".")[0]] if node.level == 0 else []
            else:
                continue
            for root in roots:
                if root and root not in allowed and root not in optional and root != "mobile":
                    pytest.fail(f"{name}.py imports {root}, which a bare phone will not have")


# -- languages -------------------------------------------------------------

def test_four_languages_default_to_kikuyu():
    assert [language.code for language in languages.ordered()] == ["kikuyu", "kamba", "oromo", "somali"]
    assert languages.get(None).code == "kikuyu"
    assert languages.get("nonsense").code == "kikuyu"
    assert languages.get("SOMALI").code == "somali"


def test_kamba_is_marked_as_having_no_corpus():
    assert languages.get("kamba").corpus is False
    assert all(languages.get(code).corpus for code in ("kikuyu", "oromo", "somali"))


def test_each_language_has_a_distinct_nllb_and_mms_tag():
    tags = [language.nllb_code for language in languages.ordered()]
    codes = [language.mms_code for language in languages.ordered()]
    assert len(set(tags)) == len(tags)
    assert len(set(codes)) == len(codes)


# -- spelling --------------------------------------------------------------

def test_fold_strips_tilde_vowels():
    assert spelling.fold("rũciũ") == "ruciu"
    assert spelling.fold("RŨCIŨ") == "ruciu"
    assert spelling.fold("mũndũ") == spelling.fold("mundu")


def test_restore_adds_accents_and_reports_changes():
    table = {"ruciu": "rũciũ", "mundu": "mũndũ"}
    text, changes = spelling.restore("ruciu na mundu", table)
    assert text == "rũciũ na mũndũ"
    assert changes == [("ruciu", "rũciũ"), ("mundu", "mũndũ")]


def test_restore_leaves_words_the_writer_accented():
    table = {"ruciu": "rũciũ"}
    text, changes = spelling.restore("rũciũ", table)
    assert text == "rũciũ" and changes == []


def test_restore_keeps_capitalisation():
    table = {"ruciu": "rũciũ"}
    assert spelling.restore("Ruciu", table)[0] == "Rũciũ"
    assert spelling.restore("RUCIU", table)[0] == "RŨCIŨ"


def test_prepare_lowercases_shouting_so_the_model_recognises_it():
    table = {"ruciu": "rũciũ"}
    assert spelling.prepare("RUCIU", table)[0] == "rũciũ"


def test_unknown_words_are_left_alone():
    assert spelling.restore("zzz", {"ruciu": "rũciũ"}) == ("zzz", [])


def test_build_table_prefers_the_commonest_spelling():
    table = spelling.build_table(["rũciũ rũciũ", "ruciu"])
    assert table["ruciu"] == "rũciũ"


# -- verses ----------------------------------------------------------------

@pytest.fixture
def verse_dir(tmp_path):
    rows = [
        {"b": "GEN", "c": 1, "v": 1, "s": "Kĩambĩrĩria-inĩ Ngai nĩombire igũrũ na thĩ.",
         "e": "In the beginning, God created the heavens and the earth."},
        {"b": "JHN", "c": 11, "v": 35, "s": "Nake Jesũ akĩrĩra.", "e": "Jesus wept."},
        {"b": "PSA", "c": 23, "v": 1, "s": "Jehova nĩwe mũrĩithi wakwa.", "e": "Yahweh is my shepherd."},
    ]
    with gzip.open(tmp_path / "verses.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return tmp_path


def test_tokens_drop_punctuation_and_single_letters():
    assert tokens("Ngai, nĩ a mwega!") == ["ngai", "nĩ", "mwega"]


def test_exact_verse_matches(verse_dir):
    match = VerseIndex(verse_dir).search("Nake Jesũ akĩrĩra.")
    assert match and match.reference == "JHN 11:35"
    assert match.english == "Jesus wept."
    assert match.score == 1.0


def test_unrelated_text_does_not_match(verse_dir):
    assert VerseIndex(verse_dir).search("ndagũthiĩ nyũmba ya thukuru") is None


def test_lower_threshold_returns_the_closest_verse(verse_dir):
    index = VerseIndex(verse_dir)
    partial = "Kĩambĩrĩria-inĩ Ngai nĩombire"
    assert index.search(partial) is None
    assert index.search(partial, threshold=0.4).reference == "GEN 1:1"


def test_missing_corpus_is_not_an_error(tmp_path):
    index = VerseIndex(tmp_path)
    assert index.available is False
    assert index.search("anything") is None


def test_index_cache_is_reused_and_survives_a_changed_timestamp(verse_dir):
    VerseIndex(verse_dir).search("Jesũ")
    cache = verse_dir / "verses.index"
    assert cache.is_file()
    # Copying to a phone rarely preserves mtime; the cache must still be valid.
    cache.touch()
    (verse_dir / "verses.jsonl.gz").touch()
    fresh = VerseIndex(verse_dir)
    assert fresh._load_cache() is True


def test_damaged_cache_is_rebuilt_rather_than_fatal(verse_dir):
    VerseIndex(verse_dir).search("Jesũ")
    (verse_dir / "verses.index").write_bytes(b"not a pickle")
    assert VerseIndex(verse_dir).search("Nake Jesũ akĩrĩra.").reference == "JHN 11:35"


# -- gloss -----------------------------------------------------------------

def test_ibm1_learns_word_meanings():
    pairs = [("ngai nĩ mwega", "god is good"), ("ngai nĩ mũnene", "god is great"),
             ("mũndũ nĩ mwega", "man is good"), ("mũndũ nĩ mũnene", "man is great")] * 12
    table = gloss.train(pairs, iterations=6)
    assert table["ngai"][0] == "god"
    assert table["mwega"][0] == "good"


def test_gloss_is_withheld_when_too_little_is_known(tmp_path):
    with gzip.open(tmp_path / gloss.GLOSS_NAME, "wt", encoding="utf-8") as handle:
        json.dump({"words": {"ngai": ["god"]}}, handle)
    table = gloss.GlossTable(tmp_path)
    assert table.gloss("ngai") is not None
    assert table.gloss("ngai aaa bbb ccc ddd") is None


def test_gloss_finds_accented_entries_from_unaccented_input(tmp_path):
    with gzip.open(tmp_path / gloss.GLOSS_NAME, "wt", encoding="utf-8") as handle:
        json.dump({"words": {"mũno": ["very"]}}, handle)
    assert gloss.GlossTable(tmp_path).best("muno") == "very"


def test_unknown_words_are_bracketed_so_gaps_are_visible(tmp_path):
    with gzip.open(tmp_path / gloss.GLOSS_NAME, "wt", encoding="utf-8") as handle:
        json.dump({"words": {"ngai": ["god"], "nĩ": ["is"], "mwega": ["good"]}}, handle)
    result = gloss.GlossTable(tmp_path).gloss("ngai nĩ mwega zzz")
    assert "[zzz]" in result.text and result.unknown == ["zzz"]


# -- audio -----------------------------------------------------------------

@pytest.mark.parametrize("channels,width,rate", [(1, 2, 16000), (2, 2, 44100), (1, 1, 8000), (2, 4, 48000), (1, 3, 22050)])
def test_every_wav_flavour_decodes_without_audioop(tmp_path, channels, width, rate):
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        frames = bytearray()
        for index in range(rate):
            value = math.sin(2 * math.pi * 440 * index / rate)
            for _ in range(channels):
                if width == 1:
                    frames.append(int(value * 127) + 128)
                elif width == 2:
                    frames += struct.pack("<h", int(value * 32767))
                elif width == 3:
                    frames += struct.pack("<i", int(value * 8388607))[:3]
                else:
                    frames += struct.pack("<i", int(value * 2147483647))
        handle.writeframes(bytes(frames))

    samples, out_rate = read_samples(path)
    assert out_rate == 16000
    assert abs(len(samples) - 16000) <= 2
    energy = math.sqrt(sum(value * value for value in samples) / len(samples))
    assert 0.6 < energy < 0.8  # a full-scale sine sits at 1/sqrt(2)


def test_mono_mixdown_averages_channels():
    assert _to_mono([10, 20, 30, 40], 2) == [15, 35]


def test_resample_halves_the_sample_count():
    assert len(_resample([0.0] * 100, 32000, 16000)) == 50


def test_eight_bit_pcm_is_treated_as_unsigned():
    assert _decode_pcm(bytes([128, 255, 0]), 1) == [0, 32512, -32768]


def test_safe_suffix_refuses_odd_names():
    assert safe_suffix("note.ogg") == ".ogg"
    assert safe_suffix("no-extension") == ".bin"
    assert safe_suffix("x.verylongextension") == ".bin"


# -- multipart -------------------------------------------------------------

def build_body(boundary: str, blob: bytes) -> bytes:
    return (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"lang\"\r\n\r\nsomali\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"r.ogg\"\r\n\r\n"
    ).encode() + blob + f"\r\n--{boundary}--\r\n".encode()


@pytest.mark.parametrize("blob", [b"", bytes(range(256)) * 8, b"\r\n" * 200, "ũhoro".encode()])
def test_upload_survives_binary_content_byte_for_byte(tmp_path, blob):
    body = build_body("X9", blob)
    fields, files = multipart.parse(io.BytesIO(body), "multipart/form-data; boundary=X9", len(body), tmp_path)
    assert fields["lang"] == "somali"
    assert files["audio"].path.read_bytes() == blob
    assert files["audio"].size == len(blob)


def test_oversized_upload_is_refused(tmp_path):
    with pytest.raises(multipart.UploadTooLarge):
        multipart.parse(io.BytesIO(b"x" * 100), "multipart/form-data; boundary=X", 100, tmp_path, max_bytes=10)


def test_missing_boundary_is_refused():
    with pytest.raises(multipart.MalformedUpload):
        multipart.boundary_from("multipart/form-data")


def test_request_body_is_not_left_behind(tmp_path):
    body = build_body("X9", b"abc")
    _fields, files = multipart.parse(io.BytesIO(body), "multipart/form-data; boundary=X9", len(body), tmp_path)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".body")]
    assert leftovers == []
    assert files["audio"].path.is_file()


# -- engine ----------------------------------------------------------------

@pytest.fixture
def engine(tmp_path, verse_dir):
    (tmp_path / "data" / "kikuyu").mkdir(parents=True)
    for name in ("verses.jsonl.gz",):
        (tmp_path / "data" / "kikuyu" / name).write_bytes((verse_dir / name).read_bytes())
    with gzip.open(tmp_path / "data" / "kikuyu" / gloss.GLOSS_NAME, "wt", encoding="utf-8") as handle:
        json.dump({"words": {"ngai": ["god"], "nĩ": ["is"], "mwega": ["good"], "mũno": ["very"]}}, handle)
    with gzip.open(tmp_path / "data" / "kikuyu" / spelling.SPELLING_NAME, "wt", encoding="utf-8") as handle:
        json.dump({"words": {"muno": "mũno", "jesu": "jesũ", "akirira": "akĩrĩra"}}, handle)
    return Engine(tmp_path)


def test_scripture_returns_the_published_english(engine):
    result = engine.translate_text("Nake Jesũ akĩrĩra.", languages.get("kikuyu"))
    assert result.engine == "verse"
    assert result.english == "Jesus wept."
    assert result.verse.reference == "JHN 11:35"


def test_accents_are_filled_in_before_the_verse_is_looked_up(engine):
    result = engine.translate_text("Nake jesu akirira.", languages.get("kikuyu"))
    assert result.engine == "verse"
    assert result.read_as is not None


def test_ordinary_text_falls_back_to_a_labelled_gloss(engine):
    result = engine.translate_text("ngai nĩ mwega mũno", languages.get("kikuyu"))
    assert result.engine == "gloss"
    assert result.english == "god is good very"
    assert "word-by-word" in result.note.casefold()
    assert result.confidence < 0.6  # a gloss must never look confident


def test_a_language_with_no_data_says_so_instead_of_guessing(engine):
    result = engine.translate_text("Ngai nĩ mũseo", languages.get("kamba"))
    assert result.engine == "none"
    assert result.english == ""
    assert "kamba" in result.note.casefold()


def test_near_verse_is_offered_without_being_claimed(engine):
    result = engine.translate_text("Kĩambĩrĩria-inĩ Ngai nĩombire", languages.get("kikuyu"))
    assert result.engine != "verse"
    assert result.near_verse is not None and result.near_verse.reference == "GEN 1:1"


def test_capabilities_report_what_is_missing(engine):
    caps = engine.capabilities()
    assert caps["neural_translation"] is False
    assert caps["languages"]["kikuyu"]["verses"] is True
    assert caps["languages"]["kamba"]["verses"] is False


def test_mis_heard_transcript_is_flagged(engine):
    kikuyu = languages.get("kikuyu")
    assert engine.heard_well("oholowaku", kikuyu) == 0.0
    assert engine.heard_well("ngai nĩ mwega", kikuyu) == 1.0


def test_heard_well_is_none_when_there_is_nothing_to_check_against(engine):
    assert engine.heard_well("anything at all", languages.get("kamba")) is None


def test_result_serialises_for_the_browser(engine):
    payload = engine.translate_text("Nake Jesũ akĩrĩra.", languages.get("kikuyu")).as_dict()
    assert json.loads(json.dumps(payload))["verse"]["reference"] == "JHN 11:35"
    assert set(payload) >= {"english", "engine", "warning", "read_as", "near_verse"}
