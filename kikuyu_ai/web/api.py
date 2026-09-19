import re
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ..audio import safe_suffix
from ..config import Settings
from ..orthography import load_lexicon, prepare_for_translation
from ..models import TranslationResult
from ..pipeline import Pipeline
from ..translator import save_correction

settings = Settings.from_env()
pipeline = Pipeline(settings)
# Built once: phone keyboards have no tilde vowels, so typed Kikuyu needs
# its accents restored before the model sees it.
LEXICON = load_lexicon(settings.bible)
app = FastAPI(title="Kikuyu AI Translator", version="0.1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSION_FILE_NAMES = {
    "translation.mp3",
    "audio_clean.wav",
    "asr_segments.jsonl",
    "english_segments.jsonl",
    "kikuyu.txt",
    "english.txt",
    "subtitles.srt",
    "english_subtitles.srt",
    "metadata.json",
}
# The original upload keeps whatever format it arrived in, so its extension
# cannot be listed ahead of time; the name is matched by shape instead.
ORIGINAL_AUDIO_RE = re.compile(r"^audio_original(\.[A-Za-z0-9]{1,6})?$")


def is_session_file(filename: str) -> bool:
    if filename in SESSION_FILE_NAMES:
        return True
    return bool(ORIGINAL_AUDIO_RE.fullmatch(filename))

ROOT_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kikuyu AI Translator</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f4ee;
      color: #1f2933;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }
    main {
      width: min(680px, 100%);
      border: 1px solid #d8d0c3;
      border-radius: 8px;
      background: #fffdf8;
      padding: 24px;
      box-shadow: 0 12px 32px rgb(31 41 51 / 0.10);
    }
    h1 {
      margin: 0 0 8px;
      font-size: 1.75rem;
      font-weight: 700;
    }
    p {
      margin: 0 0 20px;
      line-height: 1.5;
      color: #52606d;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 20px;
      font-weight: 700;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: #2f9e44;
    }
    nav {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
    }
    a {
      min-height: 44px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid #c7d2fe;
      border-radius: 6px;
      color: #1d4ed8;
      background: #eff6ff;
      text-decoration: none;
      font-weight: 700;
    }
    a:hover {
      background: #dbeafe;
    }
    code {
      font-size: 0.95em;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        background: #111827;
        color: #f9fafb;
      }
      main {
        border-color: #374151;
        background: #1f2937;
      }
      p {
        color: #d1d5db;
      }
      a {
        border-color: #1d4ed8;
        color: #bfdbfe;
        background: #1e3a8a;
      }
      a:hover {
        background: #1d4ed8;
      }
    }
  </style>
</head>
<body>
  <main>
    <div class="status"><span class="dot" aria-hidden="true"></span>API running</div>
    <h1>Kikuyu AI Translator</h1>
    <p>This server accepts uploads at <code>POST /translate</code> and text at <code>POST /translate-text</code>.</p>
    <nav aria-label="API links">
      <a href="/health">Health</a>
      <a href="/docs">Docs</a>
      <a href="/redoc">ReDoc</a>
    </nav>
  </main>
</body>
</html>"""


def session_url(session_id: str, filename: str) -> str:
    return f"/sessions/{session_id}/{filename}"


def result_payload(result: TranslationResult) -> dict:
    payload = result.json_dict()
    payload["file_urls"] = {name: session_url(result.session_id, name) for name in result.files}
    payload["audio"] = payload["file_urls"].get("translation.mp3")
    payload["subtitles"] = payload["file_urls"].get("subtitles.srt")
    payload["english_subtitles"] = payload["file_urls"].get("english_subtitles.srt")
    return payload


@app.get("/", include_in_schema=False)
def root():
    """Serve the installable mobile app; fall back to the status page."""
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        return HTMLResponse(ROOT_HTML)
    # No-store keeps a phone from pinning an old shell; the service worker
    # handles genuine offline use.
    return FileResponse(index, headers={"Cache-Control": "no-store"})


@app.get("/status", response_class=HTMLResponse, include_in_schema=False)
def status_page() -> HTMLResponse:
    return HTMLResponse(ROOT_HTML)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "offline": True,
        "asr_backend": settings.asr_backend,
        "asr_model": bool(settings.asr_model),
        "translation_model": bool(settings.translation_model),
        "bible_verses": len(pipeline.aligner.rows),
        "tts": bool(settings.tts_command),
        "max_upload_mb": settings.max_upload_mb,
    }


@app.post("/translate")
async def translate(file: UploadFile = File(...), session_id: str | None = None) -> dict:
    # The upload keeps its extension only as a hint; the pipeline identifies the
    # format from the bytes, so an odd or missing extension is not a problem.
    suffix = safe_suffix(Path(file.filename or "recording"))
    with NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(await file.read())
    try:
        result = pipeline.run(temporary, session_id=session_id)
        return result_payload(result)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        temporary.unlink(missing_ok=True)


class TextTranslation(BaseModel):
    text: str


@app.post("/translate-text")
def translate_text(item: TextTranslation) -> dict:
    text = item.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty")
    prepared, changes = prepare_for_translation(text, LEXICON)
    try:
        match = pipeline.aligner.match(prepared)
        if match:
            english, confidence = match.english, match.score
        else:
            english, confidence = pipeline.translator.translate(prepared)
        if not settings.keep_translation_loaded:
            pipeline.translator.unload()
        return {
            "kikuyu": text,
            "read_as": prepared if changes else None,
            "english": english,
            "translation_confidence": confidence,
            "translation_model": settings.translation_model,
            "translation_src_lang": settings.translation_src_lang,
            "translation_tgt_lang": settings.translation_tgt_lang,
            "bible_match": asdict(match) if match else None,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class Correction(BaseModel):
    source: str
    machine: str
    corrected: str


@app.post("/corrections")
def correction(item: Correction) -> dict:
    return {"id": save_correction(settings.corrections_db, item.source, item.machine, item.corrected)}


@app.get("/sessions/{session_id}/{filename}")
def session_file(session_id: str, filename: str):
    if not is_session_file(filename):
        raise HTTPException(status_code=404, detail="File not found")
    target = (settings.sessions / session_id / filename).resolve()
    if settings.sessions.resolve() not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


# Mounted last so every route above still wins. This serves the mobile app's
# assets from the site root, which keeps the service worker's scope at "/".
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")


def local_addresses() -> list[str]:
    """Best-effort list of addresses a phone on the same network can reach."""
    import socket

    addresses: list[str] = []
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.settimeout(0.2)
        # No packet is sent for UDP connect; this just asks the routing table
        # which local address would be used to reach the network.
        probe.connect(("192.0.2.1", 9))
        addresses.append(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if not address.startswith("127.") and address not in addresses:
                addresses.append(address)
    except (OSError, socket.gaierror):
        pass
    return addresses


def ensure_dev_certificate(directory: Path) -> tuple[Path, Path]:
    """Create a self-signed certificate so phones can use the microphone.

    Browsers only expose getUserMedia on HTTPS or localhost, so a plain LAN
    address cannot record without one. This uses the local `openssl` binary and
    never contacts a certificate authority.
    """
    import shutil
    import subprocess

    directory.mkdir(parents=True, exist_ok=True)
    certificate, key = directory / "dev-cert.pem", directory / "dev-key.pem"
    if certificate.is_file() and key.is_file():
        return certificate, key
    if not shutil.which("openssl"):
        raise SystemExit(
            "--https needs the openssl command to create a local certificate.\n"
            "Install it (apt install openssl / pkg install openssl) or pass "
            "--certfile and --keyfile."
        )
    names = ["DNS:localhost", "IP:127.0.0.1"] + [f"IP:{address}" for address in local_addresses()]
    completed = subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(certificate),
            "-days", "825", "-subj", "/CN=kikuyu-ai-local",
            "-addext", "subjectAltName=" + ",".join(names),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise SystemExit(f"openssl failed: {completed.stderr.strip()[:500]}")
    print(f"created a self-signed certificate at {certificate}", flush=True)
    return certificate, key


def run() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Serve the Kikuyu AI translator and its mobile app")
    parser.add_argument("--host", default=settings.api_host, help="bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=settings.api_port, help="bind port (default: %(default)s)")
    parser.add_argument("--lan", action="store_true", help="bind 0.0.0.0 so phones on the same network can connect")
    parser.add_argument("--https", action="store_true", help="serve over HTTPS with a local self-signed certificate")
    parser.add_argument("--certfile", type=Path, help="use this certificate instead of generating one")
    parser.add_argument("--keyfile", type=Path, help="use this private key instead of generating one")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install the web extras first: python -m pip install -e '.[web]'") from exc

    host = "0.0.0.0" if args.lan else args.host
    ssl_kwargs: dict = {}
    if args.certfile and args.keyfile:
        ssl_kwargs = {"ssl_certfile": str(args.certfile), "ssl_keyfile": str(args.keyfile)}
    elif args.https:
        certificate, key = ensure_dev_certificate(settings.root / "data" / "certs")
        ssl_kwargs = {"ssl_certfile": str(certificate), "ssl_keyfile": str(key)}

    # Flushed explicitly: stdout is block-buffered when redirected to a file or a
    # pipe, which would otherwise hide the address the user needs to type.
    scheme = "https" if ssl_kwargs else "http"
    banner = [f"\nOpen the app at {scheme}://localhost:{args.port}"]
    if host == "0.0.0.0":
        banner += [f"  on your phone: {scheme}://{address}:{args.port}" for address in local_addresses()]
        banner.append(
            "  (accept the self-signed certificate warning once on the phone)"
            if scheme == "https"
            else "  (microphone recording needs --https; file upload works either way)"
        )
    print("\n".join(banner) + "\n", flush=True)

    uvicorn.run(app, host=host, port=args.port, **ssl_kwargs)


if __name__ == "__main__":
    run()
