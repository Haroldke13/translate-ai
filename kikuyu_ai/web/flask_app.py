"""A Flask upload page for translating Kikuyu recordings into English.

    pip install -e '.[flask,ml]'
    python -m kikuyu_ai.web.flask_app --lan

Transcription runs for far longer than a browser will wait on a form post, so an
upload starts a background job and the result page polls until it is done. Only
one job decodes at a time: the models are large and running two at once would
load several gigabytes twice over.

Set KIKUYU_API_URL to send uploads to a FastAPI server instead of loading the
models in this process.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import threading
import uuid
from dataclasses import asdict
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from ..audio import safe_suffix
from ..config import Settings
from ..languages import (
    DEFAULT_LANGUAGE,
    Language,
    get as get_language,
    ordered as ordered_languages,
    settings_for,
    speech_available,
)
from ..orthography import load_lexicon, prepare_for_translation
from ..translator import BibleAligner
from ..pipeline import Pipeline
from ..remote import RemoteAPIError, translate_audio_remote, translate_text_remote
from ..translator import save_correction

# Long inputs are translated as a single sequence, so cap them rather than
# letting the model silently truncate a pasted page.
MAX_TEXT_CHARS = 5000

STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


class JobStore:
    """In-memory job table guarded by a lock, with a single worker slot."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        # Inference is memory-hungry, so jobs queue instead of running together.
        self._worker = threading.Semaphore(1)

    def create(self, label: str, kind: str = "audio") -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "state": "queued",
                "label": label,
                "kind": kind,
                "result": None,
                "error": None,
            }
        return job_id

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def run(self, job_id: str, work) -> None:
        """Run `work` in a worker thread, recording success or failure."""

        def target() -> None:
            with self._worker:
                self.update(job_id, state="working")
                try:
                    self.update(job_id, state="done", result=work())
                except Exception as exc:  # noqa: BLE001 - surfaced to the page
                    self.update(job_id, state="failed", error=str(exc) or exc.__class__.__name__)

        threading.Thread(target=target, daemon=True).start()


def create_app(settings: Settings | None = None) -> Flask:
    settings = settings or Settings.from_env()
    settings.ensure_dirs()

    app = Flask(__name__, static_folder=str(STATIC_DIR), template_folder=str(TEMPLATE_DIR))
    # Local administrator authentication; account data stays outside source.
    import importlib.util as _local_auth_importlib
    from pathlib import Path as _LocalAuthPath
    _local_auth_spec = _local_auth_importlib.spec_from_file_location(
        "_local_admin_auth", _LocalAuthPath(__file__).with_name("local_admin_auth.py")
    )
    _local_auth_module = _local_auth_importlib.module_from_spec(_local_auth_spec)
    _local_auth_spec.loader.exec_module(_local_auth_module)
    _local_auth_module.protect_app(app, __file__)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024
    jobs = JobStore()
    # Built on first use so the page still loads when no model is configured.
    pipeline_holder: dict[str, Pipeline] = {}
    pipeline_lock = threading.Lock()

    def pipeline(language: Language | None = None) -> Pipeline:
        """The pipeline for one language, each with its own speech model."""
        language = language or get_language(DEFAULT_LANGUAGE)
        with pipeline_lock:
            if language.code not in pipeline_holder:
                pipeline_holder[language.code] = Pipeline(settings_for(language, settings))
            return pipeline_holder[language.code]

    def translate_upload(temporary: Path, language: Language) -> dict:
        try:
            if settings.api_url:
                payload = translate_audio_remote(settings.api_url, temporary)
                return {
                    "source_text": payload.get("kikuyu", ""),
                    "language": language.code,
                    "language_name": language.name,
                    "english": payload.get("english", ""),
                    "session_id": payload.get("session_id"),
                    "bible_match": payload.get("bible_match"),
                    "duration": payload.get("duration"),
                    "files": list((payload.get("files") or {})),
                }
            result = pipeline(language).run(temporary)
            return {
                "source_text": result.kikuyu,
                "language": language.code,
                "language_name": language.name,
                "english": result.english,
                "session_id": result.session_id,
                "bible_match": asdict(result.bible_match) if result.bible_match else None,
                "duration": round(result.duration, 2),
                "files": list(result.files),
            }
        finally:
            temporary.unlink(missing_ok=True)

    resources: dict[str, dict] = {}

    def language_resources(language: Language) -> dict:
        """The verse index and spelling table for one language, built once."""
        with pipeline_lock:
            if language.code not in resources:
                folder = language.bible_dir(settings.bible)
                resources[language.code] = {
                    "aligner": BibleAligner(folder),
                    "lexicon": load_lexicon(folder),
                }
            return resources[language.code]

    def translate_written_text(source: str, language: Language) -> dict:
        # Phone keyboards lack the accented letters these languages use, so the
        # spelling is repaired from that language's own corpus first. The model
        # treats "ruciu" and "rũciũ" as different words and tends to echo the
        # unaccented one back untranslated.
        bundle = language_resources(language)
        prepared, changes = prepare_for_translation(source, bundle["lexicon"])
        read_as = prepared if changes else None

        if settings.api_url:
            payload = translate_text_remote(settings.api_url, prepared)
            return {
                "source_text": source,
                "language": language.code,
                "language_name": language.name,
                "read_as": read_as,
                "english": payload.get("english", ""),
                "bible_match": payload.get("bible_match"),
                "files": [],
            }

        # A verse that is already published is returned as published, rather
        # than being machine translated a second time.
        match = bundle["aligner"].match(prepared)
        if match:
            english = match.english
        else:
            active = pipeline()
            # The model is multilingual; the source tag is what tells it which
            # language it is reading.
            active.translator.src_lang = language.nllb_code
            try:
                english, _confidence = active.translator.translate(prepared)
            finally:
                active.translator.src_lang = settings.translation_src_lang
            if not settings.keep_translation_loaded:
                active.translator.unload()
        return {
            "source_text": source,
            "language": language.code,
            "language_name": language.name,
            "read_as": read_as,
            "english": english,
            "bible_match": asdict(match) if match else None,
            "files": [],
        }

    def speech_error(language: Language) -> str:
        if language.speech_note:
            return language.speech_note
        return (
            f"Speech is not available for {language.name}: its speech model is not "
            f"installed. See the README for how to download it."
        )

    def page(template: str, language: Language, **context):
        return render_template(
            template,
            remote=settings.api_url,
            max_mb=settings.max_upload_mb,
            language=language,
            languages=ordered_languages(),
            speech_ok=speech_available(language, settings),
            speech_error=speech_error(language),
            **context,
        )

    @app.get("/")
    def index():
        return page("index.html", get_language(request.args.get("lang")))

    @app.post("/translate-text")
    def translate_text():
        language = get_language(request.form.get("lang") or request.args.get("lang"))
        source = (request.form.get("text") or "").strip()
        if not source:
            return page("index.html", language, error=f"Type some {language.name} text first.", text=source), 400
        if len(source) > MAX_TEXT_CHARS:
            return page(
                "index.html",
                language,
                error=f"That text is longer than {MAX_TEXT_CHARS} characters. Split it into smaller pieces.",
                text=source,
            ), 400

        job_id = jobs.create(source, kind="text")
        jobs.run(job_id, lambda: translate_written_text(source, language))
        return redirect(url_for("result", job_id=job_id))

    @app.post("/translate")
    def translate():
        language = get_language(request.form.get("lang") or request.args.get("lang"))
        if not speech_available(language, settings):
            return page("index.html", language, error=speech_error(language)), 400

        upload = request.files.get("audio")
        if not upload or not upload.filename:
            return page("index.html", language, error="Choose a recording first."), 400

        # The name is only a hint; the pipeline identifies the format from the bytes.
        suffix = safe_suffix(Path(upload.filename))
        handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temporary = Path(handle.name)
        handle.close()
        upload.save(temporary)

        if temporary.stat().st_size == 0:
            temporary.unlink(missing_ok=True)
            return page("index.html", language, error="That file is empty."), 400

        job_id = jobs.create(upload.filename, kind="audio")
        jobs.run(job_id, lambda: translate_upload(temporary, language))
        return redirect(url_for("result", job_id=job_id))

    @app.get("/result/<job_id>")
    def result(job_id: str):
        job = jobs.get(job_id)
        if not job:
            abort(404)
        active = get_language((job.get("result") or {}).get("language"))
        return page("result.html", active, job=job, job_id=job_id)

    @app.get("/status/<job_id>")
    def status(job_id: str):
        job = jobs.get(job_id)
        if not job:
            abort(404)
        return jsonify(job)

    @app.post("/corrections")
    def corrections():
        source = (request.form.get("source") or "").strip()
        corrected = (request.form.get("corrected") or "").strip()
        machine = (request.form.get("machine") or "").strip()
        job_id = request.form.get("job_id", "")
        if source and corrected:
            save_correction(settings.corrections_db, source, machine, corrected)
        return redirect(url_for("result", job_id=job_id, saved=1) if job_id else url_for("index"))

    @app.get("/sessions/<session_id>/<filename>")
    def session_file(session_id: str, filename: str):
        folder = (settings.sessions / session_id).resolve()
        if settings.sessions.resolve() not in folder.parents or not folder.is_dir():
            abort(404)
        # send_from_directory rejects any filename that escapes the folder.
        return send_from_directory(folder, filename, as_attachment=True)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return page(
            "index.html",
            get_language(request.args.get("lang")),
            error=f"That recording is larger than the {settings.max_upload_mb} MB limit.",
        ), 413

    return app


def run() -> None:
    parser = argparse.ArgumentParser(description="Flask upload page for the Kikuyu translator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5246)
    parser.add_argument("--lan", action="store_true", help="bind 0.0.0.0 so other devices can reach it")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    from .api import local_addresses

    host = "0.0.0.0" if args.lan else args.host
    banner = [f"\nOpen the upload page at http://localhost:{args.port}"]
    if host == "0.0.0.0":
        banner += [f"  on your phone: http://{address}:{args.port}" for address in local_addresses()]
    print("\n".join(banner) + "\n", flush=True)

    # Threaded so the status poll is answered while a translation is running.
    create_app().run(host=host, port=args.port, debug=args.debug, threaded=True)


app = create_app() if __name__ != "__main__" else None

if __name__ == "__main__":
    run()
