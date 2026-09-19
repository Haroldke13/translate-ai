import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path


def _reexec_project_venv() -> None:
    root = Path(__file__).resolve().parents[1]
    venv = root / ".venv"
    venv_python = root / ".venv" / "bin" / "python"
    if not venv_python.exists() or Path(sys.prefix).resolve() == venv.resolve():
        return
    if os.environ.get("KIKUYU_NO_VENV_REEXEC"):
        return
    os.execv(str(venv_python), [str(venv_python), "-m", "kikuyu_ai.cli", *sys.argv[1:]])


_reexec_project_venv()

from .config import Settings
from .pipeline import Pipeline
from .remote import RemoteAPIError, translate_audio_remote


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate a Kikuyu recording into English")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--asr-only", action="store_true", help="write transcript/session files without loading the translation model")
    parser.add_argument("--api-url", help="Remote FastAPI base URL; overrides KIKUYU_API_URL")
    args = parser.parse_args()
    if args.api_url:
        os.environ["KIKUYU_API_URL"] = args.api_url
    settings = Settings.from_env()
    if settings.api_url:
        if args.asr_only:
            raise SystemExit("--asr-only is local-only; unset KIKUYU_API_URL or omit --api-url")
        try:
            payload = translate_audio_remote(settings.api_url, args.audio, session_id=args.session_id)
        except RemoteAPIError as exc:
            raise SystemExit(
                f"Remote API error: {exc}\n"
                "Start the API with ./translate.sh server on that machine, or choose the correct API URL."
            ) from None
        print(f"session: {payload.get('session_id')}")
        print(f"kikuyu: {payload.get('kikuyu', '')}")
        print(f"english: {payload.get('english', '')}")
        for name, path in (payload.get("files") or {}).items():
            print(f"{name}: {path}")
        for name in ("audio", "subtitles", "english_subtitles"):
            if payload.get(name):
                print(f"{name}: {payload[name]}")
        return
    if args.asr_only:
        settings = replace(settings, translation_model=None, tts_command=None)
    result = Pipeline(settings).run(args.audio, args.session_id)
    print(f"session: {result.session_id}")
    print(f"kikuyu: {result.kikuyu}")
    print(f"english: {result.english}")
    for name, path in result.files.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
