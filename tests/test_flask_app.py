"""The Flask upload page: uploading a recording and reading the result back."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("flask")

from kikuyu_ai.config import Settings
from kikuyu_ai.models import BibleMatch, TranslationResult
from kikuyu_ai.web import flask_app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    base = Settings.from_env()
    # Make the shared speech model look installed, so the upload form is offered.
    # MMS needs the encoder folder *and* each language's adapter beside it.
    from kikuyu_ai.languages import LANGUAGES, MMS_MODEL

    folder = tmp_path / MMS_MODEL
    folder.mkdir(parents=True, exist_ok=True)
    for language in LANGUAGES.values():
        if language.asr_language:
            (folder / f"adapter.{language.asr_language}.safetensors").write_bytes(b"weights")
    return replace(
        base,
        root=tmp_path,
        sessions=tmp_path / "sessions",
        bible=tmp_path / "bible",
        corrections_db=tmp_path / "corrections.db",
        api_url=None,
        max_upload_mb=1,
    )


class StubPipeline:
    """Stands in for the real models, which are far too slow for a test."""

    def __init__(self, sessions: Path):
        self.sessions = sessions
        self.calls: list[bytes] = []

    def run(self, source: Path, session_id: str | None = None) -> TranslationResult:
        self.calls.append(source.read_bytes())
        folder = self.sessions / "session-1"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "subtitles.srt").write_text("1\n", encoding="utf-8")
        return TranslationResult(
            session_id="session-1",
            kikuyu="Ngai nĩ mwega",
            english="God is good",
            segments=[],
            bible_match=BibleMatch("GEN", 1, 1, "Kĩambĩrĩria", "In the beginning", 0.9),
            transcription_confidence=0.7,
            translation_confidence=0.7,
            duration=3.75,
            files={"subtitles.srt": str(folder / "subtitles.srt")},
        )


@pytest.fixture
def client(settings: Settings, monkeypatch):
    pipeline = StubPipeline(settings.sessions)
    monkeypatch.setattr(flask_app, "Pipeline", lambda _settings: pipeline)
    app = flask_app.create_app(settings)
    app.config.update(TESTING=True)
    test_client = app.test_client()
    test_client.pipeline = pipeline
    return test_client


def wait_for(client, job_id: str, tries: int = 200) -> dict:
    """Poll the job endpoint the way the result page does."""
    import time

    for _ in range(tries):
        job = client.get(f"/status/{job_id}").get_json()
        if job["state"] in {"done", "failed"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")


def upload(client, data: bytes, filename: str):
    return client.post(
        "/translate",
        data={"audio": (__import__("io").BytesIO(data), filename)},
        content_type="multipart/form-data",
    )


def test_upload_page_offers_both_a_recording_and_a_text_form():
    settings = replace(Settings.from_env(), api_url=None)
    app = flask_app.create_app(settings)
    body = app.test_client().get("/").get_data(as_text=True)

    assert 'name="audio"' in body
    assert "Translate recording" in body
    assert 'name="text"' in body
    assert "Translate text" in body


class StubTranslator:
    def __init__(self, english: str = "God is good"):
        self.english = english
        self.seen: list[str] = []

    def translate(self, text: str):
        self.seen.append(text)
        return self.english, 0.7

    def unload(self) -> None:
        pass


class TextOnlyPipeline:
    """Only the part of Pipeline that written-text translation touches.

    The verse index is not stubbed: it is built from the language's own corpus
    folder, so a test that wants a match writes a corpus file instead.
    """

    def __init__(self):
        self.translator = StubTranslator()


def submit_text(client, text: str):
    return client.post("/translate-text", data={"text": text})


def test_written_text_is_translated(settings: Settings, monkeypatch):
    pipeline = TextOnlyPipeline()
    monkeypatch.setattr(flask_app, "Pipeline", lambda _settings: pipeline)
    client = flask_app.create_app(settings).test_client()

    response = submit_text(client, "Ngai nĩ mwega")
    job_id = response.headers["Location"].rstrip("/").rsplit("/", 1)[-1]
    job = wait_for(client, job_id)

    assert job["state"] == "done", job["error"]
    assert job["kind"] == "text"
    assert job["result"]["english"] == "God is good"
    assert pipeline.translator.seen == ["Ngai nĩ mwega"]
    assert "God is good" in client.get(f"/result/{job_id}").get_data(as_text=True)


def test_a_known_verse_is_returned_as_published_not_machine_translated(settings: Settings, monkeypatch):
    settings.bible.mkdir(parents=True, exist_ok=True)
    (settings.bible / "parallel.jsonl").write_text(
        json.dumps(
            {"book": "GEN", "chapter": 1, "verse": 1,
             "kikuyu": "Kĩambĩrĩria", "english": "In the beginning"},
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    pipeline = TextOnlyPipeline()
    monkeypatch.setattr(flask_app, "Pipeline", lambda _settings: pipeline)
    client = flask_app.create_app(settings).test_client()

    response = submit_text(client, "Kĩambĩrĩria")
    job = wait_for(client, response.headers["Location"].rstrip("/").rsplit("/", 1)[-1])

    assert job["result"]["english"] == "In the beginning"
    assert job["result"]["bible_match"]["book"] == "GEN"
    assert job["result"]["source_text"] == "Kĩambĩrĩria"
    # The translation model must not be asked when the verse is already known.
    assert pipeline.translator.seen == []


def test_empty_text_is_rejected(client):
    response = submit_text(client, "   ")

    assert response.status_code == 400
    assert "Type some Kikuyu text first." in response.get_data(as_text=True)


def test_very_long_text_is_refused_rather_than_silently_truncated(client):
    response = submit_text(client, "a" * (flask_app.MAX_TEXT_CHARS + 1))

    assert response.status_code == 400
    assert "longer than" in response.get_data(as_text=True)


def test_uploading_a_recording_runs_the_pipeline_and_shows_english(client):
    response = upload(client, b"fake-opus-bytes", "WhatsApp Ptt 2026-09-03 at 13.58.20.ogg")

    assert response.status_code == 302
    job_id = response.headers["Location"].rstrip("/").rsplit("/", 1)[-1]

    job = wait_for(client, job_id)
    assert job["state"] == "done", job["error"]
    assert job["result"]["english"] == "God is good"
    assert job["result"]["source_text"] == "Ngai nĩ mwega"
    # The bytes must reach the pipeline unchanged.
    assert client.pipeline.calls == [b"fake-opus-bytes"]

    page = client.get(f"/result/{job_id}").get_data(as_text=True)
    assert "God is good" in page
    assert "GEN 1:1" in page


def test_a_filename_with_no_extension_is_still_accepted(client):
    """Phone file managers hand over names like "recording"."""
    response = upload(client, b"fake-bytes", "recording")
    job_id = response.headers["Location"].rstrip("/").rsplit("/", 1)[-1]

    assert wait_for(client, job_id)["state"] == "done"


def test_upload_with_no_file_is_rejected(client):
    response = client.post("/translate", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert "Choose a recording first." in response.get_data(as_text=True)


def test_empty_file_is_rejected_before_starting_a_job(client):
    response = upload(client, b"", "empty.ogg")

    assert response.status_code == 400
    assert "empty" in response.get_data(as_text=True)
    assert client.pipeline.calls == []


def test_oversized_upload_reports_the_limit(client):
    response = upload(client, b"x" * (2 * 1024 * 1024), "big.wav")

    assert response.status_code == 413
    assert "larger than" in response.get_data(as_text=True)


def test_a_failing_pipeline_is_reported_not_swallowed(settings: Settings, monkeypatch):
    class Broken:
        def run(self, source, session_id=None):
            raise ValueError("this file has no audio track that can be read")

    monkeypatch.setattr(flask_app, "Pipeline", lambda _settings: Broken())
    client = flask_app.create_app(settings).test_client()

    response = upload(client, b"not-audio", "notes.mp3")
    job_id = response.headers["Location"].rstrip("/").rsplit("/", 1)[-1]
    job = wait_for(client, job_id)

    assert job["state"] == "failed"
    assert "no audio track" in job["error"]
    assert "no audio track" in client.get(f"/result/{job_id}").get_data(as_text=True)


def test_corrections_are_saved_for_later_training(client, settings: Settings):
    import sqlite3

    response = client.post(
        "/corrections",
        data={"source": "Ngai nĩ mwega", "machine": "God good", "corrected": "God is good"},
    )

    assert response.status_code == 302
    rows = sqlite3.connect(settings.corrections_db).execute(
        "SELECT source, corrected FROM corrections"
    ).fetchall()
    assert rows == [("Ngai nĩ mwega", "God is good")]


def test_session_downloads_cannot_escape_the_sessions_folder(client, settings: Settings):
    (settings.root / "secret.txt").write_text("private", encoding="utf-8")
    folder = settings.sessions / "session-1"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "subtitles.srt").write_text("1\n", encoding="utf-8")

    assert client.get("/sessions/session-1/subtitles.srt").status_code == 200
    assert client.get("/sessions/session-1/../../secret.txt").status_code == 404
    assert client.get("/sessions/missing/subtitles.srt").status_code == 404


def test_unknown_job_is_not_found(client):
    assert client.get("/status/deadbeef").status_code == 404
    assert client.get("/result/deadbeef").status_code == 404
