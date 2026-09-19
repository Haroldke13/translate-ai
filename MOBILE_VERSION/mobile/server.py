"""The web app, served by the phone to itself.

Uses http.server from the standard library rather than Flask or FastAPI,
because the first promise this build makes is that it starts after nothing more
than "pkg install python". A framework would be more comfortable to write and
would break that promise.

Translation runs in a background worker with a queue of one, so a second
request waits instead of loading a second copy of a model into a phone's
memory. The browser polls for the result, which survives the screen turning off
mid-translation — an Android browser tab that is backgrounded stops running
JavaScript, but the worker thread keeps going.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
import socket
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import languages, multipart
from .engine import Engine

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEB = ROOT / "web"
MAX_TEXT_CHARS = 5000
JOB_RETENTION = 60 * 60  # an hour is long enough to reread a result
SAFE_ASSET = re.compile(r"^[A-Za-z0-9._/-]+$")


@dataclass
class Job:
    id: str
    kind: str
    language: str
    label: str
    state: str = "queued"
    result: dict | None = None
    error: str | None = None
    progress: str = ""
    created: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "language": self.language,
            "label": self.label,
            "state": self.state,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
        }


class JobStore:
    """Background work with a queue depth of one.

    A phone has room for one model at a time. The semaphore is what stops two
    translations from being loaded at once and taking the whole app down with an
    out-of-memory kill.
    """

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._slot = threading.Semaphore(1)

    def create(self, kind: str, language: str, label: str) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind, language=language, label=label)
        with self._lock:
            self._prune()
            self._jobs[job.id] = job
        return job

    def _prune(self) -> None:
        cutoff = time.monotonic() - JOB_RETENTION
        for key in [key for key, job in self._jobs.items() if job.created < cutoff]:
            del self._jobs[key]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job: Job, **changes) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(job, key, value)

    def run(self, job: Job, work) -> None:
        def target() -> None:
            with self._slot:
                self.update(job, state="working")
                try:
                    self.update(job, state="done", result=work(job))
                except Exception as error:  # surfaced to the user, not swallowed
                    traceback.print_exc()
                    self.update(job, state="failed", error=str(error) or error.__class__.__name__)

        threading.Thread(target=target, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    server_version = "KikuyuMobile/1.0"
    engine: Engine
    jobs: JobStore
    uploads: Path
    corrections: Path
    max_upload: int

    # -- plumbing ---------------------------------------------------------

    def log_message(self, fmt, *args):  # quieter than the default one-line-per-asset
        if self.path.startswith("/api/"):
            print(f"  {self.command} {self.path}", flush=True)

    def _send(self, status: HTTPStatus, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The app is served from the phone itself and changes when the folder is
        # replaced, so caching an asset across versions only causes confusion.
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _fail(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 1_000_000:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    # -- routing ----------------------------------------------------------

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            return self._health()
        if path.startswith("/api/job/"):
            return self._job(path.rsplit("/", 1)[-1])
        if path.startswith("/api/"):
            return self._fail(HTTPStatus.NOT_FOUND, "no such endpoint")
        return self._asset(path)

    do_HEAD = do_GET

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/translate":
            return self._translate_text()
        if path == "/api/audio":
            return self._translate_audio()
        if path == "/api/correction":
            return self._correction()
        return self._fail(HTTPStatus.NOT_FOUND, "no such endpoint")

    # -- handlers ---------------------------------------------------------

    def _asset(self, path: str) -> None:
        name = "index.html" if path in ("/", "") else path.lstrip("/")
        if not SAFE_ASSET.match(name) or ".." in name:
            return self._fail(HTTPStatus.FORBIDDEN, "bad path")
        target = (WEB / name).resolve()
        try:
            target.relative_to(WEB.resolve())
        except ValueError:
            return self._fail(HTTPStatus.FORBIDDEN, "bad path")
        if not target.is_file():
            return self._fail(HTTPStatus.NOT_FOUND, "not found")
        kind, _ = mimetypes.guess_type(str(target))
        self._send(HTTPStatus.OK, target.read_bytes(), kind or "application/octet-stream")

    def _health(self) -> None:
        capabilities = self.engine.capabilities()
        self._json(
            {
                "ok": True,
                "languages": [
                    {
                        "code": language.code,
                        "name": language.name,
                        "native": language.native,
                        "placeholder": language.placeholder,
                        **capabilities["languages"][language.code],
                    }
                    for language in languages.ordered()
                ],
                "neural_translation": capabilities["neural_translation"],
                "neural_reason": capabilities["neural_reason"],
                "speech_installed": capabilities["speech_installed"],
                "max_upload_mb": self.max_upload // (1024 * 1024),
            }
        )

    def _job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return self._fail(HTTPStatus.NOT_FOUND, "that translation has expired")
        self._json(job.as_dict())

    def _translate_text(self) -> None:
        payload = self._read_json()
        text = str(payload.get("text") or "").strip()
        language = languages.get(payload.get("lang"))
        if not text:
            return self._fail(HTTPStatus.BAD_REQUEST, "Type something to translate.")
        if len(text) > MAX_TEXT_CHARS:
            return self._fail(
                HTTPStatus.BAD_REQUEST,
                f"That is longer than {MAX_TEXT_CHARS} characters. Translate it in pieces.",
            )

        job = self.jobs.create("text", language.code, "Written text")
        self.jobs.run(job, lambda _job: self.engine.translate_text(text, language).as_dict())
        self._json({"job": job.id}, HTTPStatus.ACCEPTED)

    def _translate_audio(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        content_type = self.headers.get("Content-Type") or ""
        if "multipart/form-data" not in content_type.casefold():
            return self._fail(HTTPStatus.BAD_REQUEST, "expected a file upload")
        try:
            fields, files = multipart.parse(
                self.rfile, content_type, length, self.uploads, max_bytes=self.max_upload
            )
        except multipart.UploadTooLarge:
            return self._fail(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"That recording is bigger than {self.max_upload // (1024 * 1024)} MB.",
            )
        except multipart.MalformedUpload as error:
            return self._fail(HTTPStatus.BAD_REQUEST, str(error))

        upload = files.get("audio")
        language = languages.get(fields.get("lang"))
        if not upload or not upload.size:
            if upload:
                upload.path.unlink(missing_ok=True)
            return self._fail(HTTPStatus.BAD_REQUEST, "Choose a recording first.")

        if not self.engine.speech.available(language.mms_code):
            upload.path.unlink(missing_ok=True)
            return self._fail(HTTPStatus.SERVICE_UNAVAILABLE, self.engine.speech_error(language))

        # Give the file its real extension so ffmpeg has the same hint the phone
        # gave us; it sniffs the contents too, but the hint costs nothing.
        stored = upload.path.with_suffix(mobile_suffix(upload.filename))
        upload.path.replace(stored)

        job = self.jobs.create("audio", language.code, upload.filename)

        def work(current: Job) -> dict:
            def progress(index: int, total: int) -> None:
                self.jobs.update(current, progress=f"Listening… part {index} of {total}")

            try:
                return self.engine.translate_audio(stored, language, progress=progress).as_dict()
            finally:
                stored.unlink(missing_ok=True)

        self.jobs.run(job, work)
        self._json({"job": job.id}, HTTPStatus.ACCEPTED)

    def _correction(self) -> None:
        payload = self._read_json()
        source = str(payload.get("source") or "").strip()
        corrected = str(payload.get("corrected") or "").strip()
        if not source or not corrected:
            return self._fail(HTTPStatus.BAD_REQUEST, "Nothing to save.")
        row = {
            "language": languages.get(payload.get("lang")).code,
            "source": source[:MAX_TEXT_CHARS],
            "machine": str(payload.get("machine") or "")[:MAX_TEXT_CHARS],
            "corrected": corrected[:MAX_TEXT_CHARS],
        }
        self.corrections.parent.mkdir(parents=True, exist_ok=True)
        with self.corrections.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._json({"saved": True})


def mobile_suffix(name: str) -> str:
    from .audio import safe_suffix

    return safe_suffix(name)


def local_addresses() -> list[str]:
    """Addresses this phone can be reached on, for sharing over a hotspot."""
    found = set()
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("10.255.255.255", 1))
        found.add(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    return sorted(address for address in found if not address.startswith("127."))


def build(root: Path, uploads: Path, max_upload_mb: int) -> type[Handler]:
    engine = Engine(root)
    handler = type(
        "BoundHandler",
        (Handler,),
        {
            "engine": engine,
            "jobs": JobStore(),
            "uploads": uploads,
            "corrections": root / "data" / "corrections.jsonl",
            "max_upload": max_upload_mb * 1024 * 1024,
        },
    )
    return handler


def warm(root: Path) -> None:
    """Build the verse index in the background while the user is still reading.

    Indexing 31,000 verses takes a few seconds on a phone. Doing it now means the
    first translation is instant instead of looking like the app has hung.
    """
    def target() -> None:
        engine = Engine(root)
        for language in languages.ordered():
            resource = engine.resources(language)
            if resource.verses.available:
                started = time.monotonic()
                resource.verses.load()
                print(
                    f"  indexed {language.name}: {len(resource.verses)} verses "
                    f"in {time.monotonic() - started:.1f}s",
                    flush=True,
                )

    threading.Thread(target=target, daemon=True).start()


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Translate Kikuyu, Kamba, Oromo and Somali on your phone.")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to let other devices connect")
    parser.add_argument("--lan", action="store_true", help="shorthand for --host 0.0.0.0")
    parser.add_argument("--max-upload-mb", type=int, default=512)
    parser.add_argument("--no-warm", action="store_true", help="skip building the verse index at startup")
    args = parser.parse_args(argv)

    host = "0.0.0.0" if args.lan else args.host
    uploads = Path(tempfile.mkdtemp(prefix="kikuyu-uploads-"))
    handler = build(ROOT, uploads, args.max_upload_mb)

    server = ThreadingHTTPServer((host, args.port), handler)
    server.daemon_threads = True

    capabilities = handler.engine.capabilities()
    ready = [name for name, info in capabilities["languages"].items() if info["verses"] or info["neural"]]
    print("", flush=True)
    print("  Kikuyu / Kamba / Oromo / Somali → English", flush=True)
    print(f"  Open this on the phone:  http://localhost:{args.port}", flush=True)
    if host == "0.0.0.0":
        for address in local_addresses():
            print(f"  Or from another device:  http://{address}:{args.port}", flush=True)
    print(
        f"  Translation: {'neural model installed' if capabilities['neural_translation'] else 'offline tier only (verses + word list)'}",
        flush=True,
    )
    print(f"  Speech: {'installed' if capabilities['speech_installed'] else 'not installed'}", flush=True)
    print(f"  Offline data ready for: {', '.join(ready) or 'nothing — run pack_mobile.py on the computer'}", flush=True)
    print("  Stop with Ctrl-C.", flush=True)
    print("", flush=True)

    if not args.no_warm:
        warm(ROOT)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped", flush=True)
    finally:
        server.server_close()
        shutil.rmtree(uploads, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
