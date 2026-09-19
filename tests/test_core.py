import json
import wave
from pathlib import Path
from kikuyu_ai.audio import normalize_audio
from kikuyu_ai.asr import ASR
from kikuyu_ai.config import Settings
from kikuyu_ai.outputs import srt_timestamp
from kikuyu_ai.pipeline import Pipeline
from kikuyu_ai import remote
from kikuyu_ai.remote import absolutize_session_urls, check_remote_api
from kikuyu_ai.translator import BibleAligner, save_correction
from scripts.build_bible_asr_manifest import build_rows, read_readaloud_zip
from scripts.build_youversion_audio_manifest import canonical_chapters, extract_audio_url
from scripts.download_bible_audio import destination_for
from training.train_asr import read_manifest, split_examples


def test_timestamp():
    assert srt_timestamp(1.25) == "00:00:01,250"


def test_bible_match(tmp_path: Path):
    bible = tmp_path / "bible"
    bible.mkdir()
    (bible / "parallel.jsonl").write_text(json.dumps({"book":"John", "chapter":3, "verse":16, "kikuyu":"Ngai nĩwendete kĩrĩndĩ", "english":"For God so loved the world"}) + "\n", encoding="utf-8")
    match = BibleAligner(bible, threshold=.4).match("Ngai nĩwendete kĩrĩndĩ")
    assert match and match.book == "John" and match.verse == 16


def test_correction(tmp_path: Path):
    ident = save_correction(tmp_path / "corrections.db", "a", "b", "c")
    assert ident == 1


def test_settings_loads_model_env_file(tmp_path: Path, monkeypatch):
    for key in (
        "KIKUYU_API_URL",
        "KIKUYU_API_HOST",
        "KIKUYU_API_PORT",
        "KIKUYU_ASR_BACKEND",
        "KIKUYU_ASR_MODEL",
        "KIKUYU_ASR_LANGUAGE",
        "KIKUYU_TRANSLATION_MODEL",
        "KIKUYU_TRANSLATION_SRC_LANG",
        "KIKUYU_TRANSLATION_TGT_LANG",
        "KIKUYU_ASR_CHUNK_SECONDS",
        "KIKUYU_ASR_RETRY_CHUNK_SECONDS",
        "KIKUYU_ASR_SILENCE_RMS",
        "KIKUYU_ASR_MAX_NEW_TOKENS",
        "KIKUYU_ASR_NO_REPEAT_NGRAM_SIZE",
        "KIKUYU_ASR_REPETITION_PENALTY",
        "KIKUYU_TRANSLATION_MAX_NEW_TOKENS",
        "KIKUYU_KEEP_ASR_LOADED",
        "KIKUYU_KEEP_TRANSLATION_LOADED",
    ):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "KIKUYU_API_URL=https://api.example.test",
                "KIKUYU_API_HOST=0.0.0.0",
                "KIKUYU_API_PORT=9001",
                "KIKUYU_ASR_BACKEND=transformers",
                "KIKUYU_ASR_MODEL=example/asr",
                "KIKUYU_TRANSLATION_MODEL=example/translation",
                "KIKUYU_TRANSLATION_SRC_LANG=kik_Latn",
                "KIKUYU_TRANSLATION_TGT_LANG=eng_Latn",
                "KIKUYU_ASR_CHUNK_SECONDS=7.5",
                "KIKUYU_ASR_RETRY_CHUNK_SECONDS=3.5",
                "KIKUYU_ASR_SILENCE_RMS=0.002",
                "KIKUYU_ASR_MAX_NEW_TOKENS=77",
                "KIKUYU_ASR_NO_REPEAT_NGRAM_SIZE=4",
                "KIKUYU_ASR_REPETITION_PENALTY=1.2",
                "KIKUYU_TRANSLATION_MAX_NEW_TOKENS=99",
                "KIKUYU_KEEP_ASR_LOADED=true",
            ]
        ),
        encoding="utf-8",
    )
    settings = Settings.from_env(tmp_path)
    assert settings.api_url == "https://api.example.test"
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 9001
    assert settings.asr_backend == "transformers"
    assert settings.asr_model == "example/asr"
    assert settings.translation_model == "example/translation"
    assert settings.translation_src_lang == "kik_Latn"
    assert settings.translation_tgt_lang == "eng_Latn"
    assert settings.asr_chunk_seconds == 7.5
    assert settings.asr_retry_chunk_seconds == 3.5
    assert settings.asr_silence_rms == 0.002
    assert settings.asr_max_new_tokens == 77
    assert settings.asr_no_repeat_ngram_size == 4
    assert settings.asr_repetition_penalty == 1.2
    assert settings.translation_max_new_tokens == 99
    assert settings.keep_asr_loaded is True
    assert settings.keep_translation_loaded is False


def test_wav_normalization_writes_pcm16(tmp_path: Path):
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(16000)
        handle.writeframes(bytes([0, 64, 128, 192, 255]))

    duration = normalize_audio(source, target)

    with wave.open(str(target), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000
        assert handle.getnframes() == 5
    assert duration == 5 / 16000


def test_pipeline_writes_complete_session_atomically(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KIKUYU_ASR_MODEL", "")
    monkeypatch.setenv("KIKUYU_TRANSLATION_MODEL", "")
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(16000)
        handle.writeframes(bytes([128, 128, 128, 128]))

    settings = Settings.from_env(tmp_path)
    result = Pipeline(settings).run(source, "atomic-session")
    folder = settings.sessions / "atomic-session"

    assert folder.exists()
    assert result.files["metadata.json"] == str(folder / "metadata.json")
    assert any((folder / name).exists() for name in ("audio_original.mp3", "audio_original.wav"))
    for name in ("audio_clean.wav", "asr_segments.jsonl", "english_segments.jsonl", "kikuyu.txt", "english.txt", "subtitles.srt", "metadata.json"):
        assert (folder / name).exists()
    assert not list(settings.sessions.glob(".atomic-session.*"))


def test_pipeline_rejects_unsafe_session_id(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KIKUYU_ASR_MODEL", "")
    monkeypatch.setenv("KIKUYU_TRANSLATION_MODEL", "")
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(16000)
        handle.writeframes(bytes([128, 128, 128, 128]))

    settings = Settings.from_env(tmp_path)
    try:
        Pipeline(settings).run(source, "../escape")
    except ValueError as exc:
        assert "session_id" in str(exc)
    else:
        raise AssertionError("unsafe session id was accepted")


def test_asr_repetition_detection_and_trimming():
    repeated = "intro words kwa murata wa kwa murata wa kwa murata wa kwa murata wa kwa murata wa kwa murata wa"
    assert ASR._looks_repetitive(repeated)
    assert not ASR._looks_repetitive("ngai ni mwathani wa tha na wendo")
    trimmed = ASR._trim_repetitive_tail(repeated)
    assert trimmed == "intro words"


def test_transformers_asr_skips_silent_audio_without_loading_model(tmp_path: Path):
    source = tmp_path / "silent.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)

    asr = ASR("missing-local-model", backend="transformers", silence_rms=0.001)
    segments, confidence = asr.transcribe(source)

    assert [segment.text for segment in segments] == [""]
    assert confidence == 0.0
    assert asr._model is None


def test_asr_manifest_resolves_paths_and_explicit_splits(tmp_path: Path):
    train_audio = tmp_path / "train.wav"
    eval_audio = tmp_path / "eval.wav"
    train_audio.write_bytes(b"placeholder")
    eval_audio.write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"audio": "train.wav", "text": "train text", "split": "train"}),
                json.dumps({"audio": "eval.wav", "text": "eval text", "split": "eval"}),
            ]
        ),
        encoding="utf-8",
    )

    rows = read_manifest(manifest)
    train_rows, eval_rows = split_examples(rows, None, 0.1, 42)

    assert rows[0].audio == train_audio.resolve()
    assert [row.text for row in train_rows] == ["train text"]
    assert [row.text for row in eval_rows] == ["eval text"]


def test_bible_asr_manifest_matches_chapter_audio_to_readaloud_zip(tmp_path: Path):
    import zipfile

    readaloud = tmp_path / "kik_readaloud.zip"
    with zipfile.ZipFile(readaloud, "w") as archive:
        archive.writestr("kik_002_GEN_01_read.txt", "Kĩambĩrĩria.\n1.\nNgai nĩombire igũrũ na thĩ.")
    audio = tmp_path / "GEN_001.mp3"
    audio.write_bytes(b"audio")

    transcripts = read_readaloud_zip(readaloud)
    rows = build_rows(transcripts, [{"audio": audio, "book": "GEN", "chapter": 1}], "train")

    assert rows == [
        {
            "audio": str(audio.resolve()),
            "text": "Kĩambĩrĩria. 1. Ngai nĩombire igũrũ na thĩ.",
            "split": "train",
        }
    ]


def test_audio_downloader_names_book_chapter_files(tmp_path: Path):
    target = destination_for(
        {"url": "https://example.invalid/audio?id=1", "book": "gen", "chapter": 1},
        tmp_path,
        "audio/mpeg",
    )

    assert target == tmp_path / "GEN_001.mp3"


def test_remote_health_check_uses_health_endpoint(monkeypatch):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b'{"status":"ok"}'

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return Response()

    monkeypatch.setattr(remote, "urlopen", fake_urlopen)

    payload = check_remote_api("http://server.example/api", timeout=1.5)

    assert payload == {"status": "ok"}
    assert calls[0][0].full_url == "http://server.example/api/health"
    assert calls[0][1] == 1.5


def test_remote_client_absolutizes_session_urls():
    payload = {
        "audio": "/sessions/abc/translation.mp3",
        "subtitles": "/sessions/abc/subtitles.srt",
        "file_urls": {"kikuyu.txt": "/sessions/abc/kikuyu.txt", "external": "https://cdn.example/x"},
    }

    result = absolutize_session_urls("https://server.example/api", payload)

    assert result["audio"] == "https://server.example/api/sessions/abc/translation.mp3"
    assert result["subtitles"] == "https://server.example/api/sessions/abc/subtitles.srt"
    assert result["file_urls"]["kikuyu.txt"] == "https://server.example/api/sessions/abc/kikuyu.txt"
    assert result["file_urls"]["external"] == "https://cdn.example/x"


def test_youversion_audio_manifest_helpers_extract_canonical_chapter_audio():
    version = {
        "books": [
            {
                "usfm": "GEN",
                "human": "Genesis",
                "chapters": [
                    {"usfm": "GEN.INTRO1", "canonical": False, "audio": True},
                    {"usfm": "GEN.1", "canonical": True, "audio": True, "human": "1"},
                    {"usfm": "GEN.2", "canonical": True, "audio": False, "human": "2"},
                ],
            }
        ]
    }
    page = (
        '<script type="application/ld+json">'
        '{"@type":"AudioObject","contentUrl":"https://audio-bible-cdn.youversionapi.com/327/32k/GEN/1-file.mp3?version_id=1622"}'
        "</script>"
    )

    assert canonical_chapters(version) == [
        {"book": "GEN", "chapter": 1, "usfm": "GEN.1", "book_title": "Genesis", "chapter_title": "1"}
    ]
    assert extract_audio_url(page, 1622) == (
        "https://audio-bible-cdn.youversionapi.com/327/32k/GEN/1-file.mp3?version_id=1622"
    )
