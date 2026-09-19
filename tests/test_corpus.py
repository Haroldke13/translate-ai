import json
import sqlite3
from pathlib import Path

import pytest

from kikuyu_ai.corpus import (
    align_blocks,
    dedupe,
    guess_language,
    is_noise_line,
    normalize_text,
    pair_key,
    Pair,
    split_sentences,
    strip_verse_number,
    usable_pair,
    write_jsonl,
)
from scripts.build_training_corpus import parse_repeats
from scripts.ingest_pdf_corpus import drop_repeated_headers, language_runs, pair_language_runs, phase_score
from scripts.ingest_voice_corpus import from_corrections, from_sessions, from_voice_pairs


def test_normalize_text_rejoins_hyphenated_line_breaks():
    assert normalize_text("mwa-\nndĩki  wa\tNgai") == "mwandĩki wa Ngai"


def test_normalize_text_keeps_kikuyu_tilde_vowels():
    assert normalize_text("Kĩambĩrĩria ũrĩa") == "Kĩambĩrĩria ũrĩa"


def test_split_sentences_breaks_on_terminators_only():
    assert split_sentences("Nĩ wega. Ũrĩ atĩa? Twende!") == ["Nĩ wega.", "Ũrĩ atĩa?", "Twende!"]


def test_strip_verse_number_removes_leading_reference():
    assert strip_verse_number("16 Nĩgũkorwo Ngai") == "Nĩgũkorwo Ngai"


def test_is_noise_line_flags_page_numbers_but_keeps_prose():
    assert is_noise_line("— 42 —")
    assert is_noise_line("")
    assert not is_noise_line("Ngai nĩ mwega")


def test_guess_language_separates_the_two_languages():
    assert guess_language("Nake Ngai akiuga atĩrĩ, nĩ wega mũno") == "kikuyu"
    assert guess_language("And God said that it was very good") == "english"


def test_align_blocks_matches_sentences_one_to_one():
    kikuyu = ["Mũndũ wa mbere.", "Mũndũ wa kerĩ nĩwe mũnene.", "Thutha ũcio agĩũka."]
    english = ["The first person.", "The second person is the great one.", "After that he came."]
    assert align_blocks(kikuyu, english) == list(zip(kikuyu, english))


def test_align_blocks_recovers_when_one_side_splits_a_sentence():
    kikuyu = ["Mũndũ ũcio nĩ mwega.", "Nĩ mwega mũno na nĩ mũtaare wa andũ othe a bũrũri."]
    english = ["That person is good.", "He is very good.", "He is a counsellor of all the people of the land."]
    pairs = align_blocks(kikuyu, english)
    assert pairs[0] == ("Mũndũ ũcio nĩ mwega.", "That person is good.")
    assert pairs[1][0] == kikuyu[1]
    assert pairs[1][1] == "He is very good. He is a counsellor of all the people of the land."


def test_align_blocks_returns_nothing_for_an_empty_side():
    assert align_blocks([], ["anything"]) == []


def test_usable_pair_rejects_untranslated_and_lopsided_rows():
    assert usable_pair("Ngai nĩ mwega", "God is good")
    assert not usable_pair("Ngai nĩ mwega", "Ngai nĩ mwega")  # extraction failure
    assert not usable_pair("Ngai", "God")  # too short
    assert not usable_pair("Ngai nĩ mwega", "God is good " * 40)  # lopsided


def test_pair_key_ignores_case_and_diacritics_so_dedupe_catches_variants():
    assert pair_key("Ngai nĩ mwega", "God is good") == pair_key("ngai ni MWEGA", "god is good")


def test_dedupe_keeps_first_occurrence_only():
    pairs = [Pair("Ngai nĩ mwega", "God is good"), Pair("ngai ni mwega", "god is good")]
    assert len(dedupe(pairs)) == 1


def test_drop_repeated_headers_removes_running_page_furniture():
    lines = ["Chapter One", "Ngai nĩ mwega", "Chapter One", "Andũ nĩ mokire", "Chapter One"]
    assert drop_repeated_headers(lines, 3) == ["Ngai nĩ mwega", "Andũ nĩ mokire"]


def test_pair_language_runs_does_not_pair_english_with_the_next_kikuyu():
    """Interleaved text must be consumed two runs at a time, not slid by one."""
    runs = [
        ("kikuyu", ["Mũndũ ũcio nĩ mwega."]),
        ("english", ["That person is good."]),
        ("kikuyu", ["Andũ othe nĩ mokire."]),
        ("english", ["All the people came."]),
    ]
    assert [(pair.kikuyu, pair.english) for pair in pair_language_runs(runs)] == [
        ("Mũndũ ũcio nĩ mwega.", "That person is good."),
        ("Andũ othe nĩ mokire.", "All the people came."),
    ]


def test_pair_language_runs_recovers_when_a_stray_run_leads():
    runs = [
        ("english", ["A title page in English."]),
        ("kikuyu", ["Mũndũ ũcio nĩ mwega."]),
        ("english", ["That person is good."]),
        ("kikuyu", ["Andũ othe nĩ mokire."]),
        ("english", ["All the people came."]),
    ]
    pairs = [(pair.kikuyu, pair.english) for pair in pair_language_runs(runs)]
    assert pairs == [
        ("Mũndũ ũcio nĩ mwega.", "That person is good."),
        ("Andũ othe nĩ mokire.", "All the people came."),
    ]


def test_language_runs_attaches_ambiguous_lines_to_the_previous_run():
    runs = language_runs(["Nake Ngai akiuga atĩrĩ nĩ wega", "1997", "And God said it was good"])
    assert [language for language, _ in runs] == ["kikuyu", "english"]
    assert runs[0][1] == ["Nake Ngai akiuga atĩrĩ nĩ wega", "1997"]


def _corrections_db(path: Path) -> Path:
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE corrections (id INTEGER PRIMARY KEY, source TEXT NOT NULL, "
            "machine TEXT NOT NULL, corrected TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        db.execute(
            "INSERT INTO corrections(source, machine, corrected) VALUES (?, ?, ?)",
            ("Ngai nĩ mwega hĩndĩ ciothe", "God good all time", "God is good at all times"),
        )
    return path


def test_from_corrections_reads_human_labels(tmp_path: Path):
    pairs = from_corrections(_corrections_db(tmp_path / "corrections.db"))
    assert [(pair.kikuyu, pair.english) for pair in pairs] == [
        ("Ngai nĩ mwega hĩndĩ ciothe", "God is good at all times")
    ]
    assert pairs[0].meta["verified"] is True


def test_from_corrections_tolerates_a_missing_database(tmp_path: Path):
    assert from_corrections(tmp_path / "absent.db") == []


def _wav(path: Path, seconds: float = 0.5) -> None:
    import struct
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        count = int(16000 * seconds)
        handle.writeframes(struct.pack("<" + "h" * count, *([1000] * count)))


def test_from_voice_pairs_reads_sidecars_and_reports_missing_transcripts(tmp_path: Path):
    _wav(tmp_path / "clip01.wav")
    (tmp_path / "clip01.kikuyu.txt").write_text("Nĩĩ ndĩ mwana wa Ngai", encoding="utf-8")
    (tmp_path / "clip01.english.txt").write_text("I am a child of God", encoding="utf-8")
    _wav(tmp_path / "clip02.wav")

    pairs, manifest, missing = from_voice_pairs(tmp_path)

    assert [(pair.kikuyu, pair.english) for pair in pairs] == [("Nĩĩ ndĩ mwana wa Ngai", "I am a child of God")]
    assert manifest == [{"audio": str(tmp_path / "clip01.wav"), "text": "Nĩĩ ndĩ mwana wa Ngai"}]
    assert missing == [str(tmp_path / "clip02.wav")]


def _session(root: Path, name: str, *, corrected: bool) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    _wav(folder / "audio_clean.wav")
    (folder / "kikuyu.txt").write_text("mũndũ ũcio nĩ mwega", encoding="utf-8")
    (folder / "english.txt").write_text("that person good", encoding="utf-8")
    if corrected:
        (folder / "english.corrected.txt").write_text("That person is good", encoding="utf-8")
    return folder


def test_from_sessions_excludes_unreviewed_output_by_default(tmp_path: Path):
    _session(tmp_path, "reviewed", corrected=True)
    _session(tmp_path, "raw", corrected=False)

    pairs, manifest = from_sessions(tmp_path, include_machine_output=False)

    assert [pair.english for pair in pairs] == ["That person is good"]
    assert all(pair.meta["verified"] for pair in pairs)
    # Only the reviewed session's audio is trustworthy as an ASR label.
    assert [row["text"] for row in manifest] == ["mũndũ ũcio nĩ mwega"]


def test_from_sessions_includes_machine_output_when_asked(tmp_path: Path):
    _session(tmp_path, "raw", corrected=False)

    pairs, _ = from_sessions(tmp_path, include_machine_output=True)

    assert [pair.english for pair in pairs] == ["that person good"]
    assert pairs[0].meta["verified"] is False


def test_from_sessions_prefers_per_segment_rows_for_machine_output(tmp_path: Path):
    folder = _session(tmp_path, "raw", corrected=False)
    (folder / "asr_segments.jsonl").write_text(
        json.dumps({"start": 0.0, "end": 2.0, "text": "kĩrĩra kĩa andũ"}) + "\n", encoding="utf-8"
    )
    (folder / "english_segments.jsonl").write_text(
        json.dumps({"start": 0.0, "end": 2.0, "text": "the wisdom of people"}) + "\n", encoding="utf-8"
    )

    pairs, _ = from_sessions(tmp_path, include_machine_output=True)

    assert [(pair.kikuyu, pair.english) for pair in pairs] == [("kĩrĩra kĩa andũ", "the wisdom of people")]


def test_from_sessions_ignores_partial_session_folders(tmp_path: Path):
    (tmp_path / ".half-written-abc123").mkdir()
    assert from_sessions(tmp_path, include_machine_output=True) == ([], [])


def test_parse_repeats_reads_path_equals_count():
    assert parse_repeats(["data/translation/voice.jsonl=8"]) == {"data/translation/voice.jsonl": 8}


def test_parse_repeats_rejects_malformed_values():
    with pytest.raises(SystemExit):
        parse_repeats(["voice.jsonl"])
    with pytest.raises(SystemExit):
        parse_repeats(["voice.jsonl=many"])


def test_write_jsonl_round_trips_kikuyu_characters(tmp_path: Path):
    target = tmp_path / "rows.jsonl"
    assert write_jsonl(target, [{"kikuyu": "Kĩambĩrĩria", "english": "Beginning"}]) == 1
    assert json.loads(target.read_text(encoding="utf-8"))["kikuyu"] == "Kĩambĩrĩria"
