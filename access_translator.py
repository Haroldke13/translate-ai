import argparse
import json
from pathlib import Path

from kikuyu_ai.config import Settings
from kikuyu_ai.pipeline import Pipeline
from kikuyu_ai.remote import translate_audio_remote, translate_text_remote
from kikuyu_ai.translator import Translator


def translate_audio(audio_path: Path, session_id: str | None = None) -> dict:
    """Run the full audio-to-English translator pipeline."""
    settings = Settings.from_env()
    if settings.api_url:
        return translate_audio_remote(settings.api_url, audio_path, session_id=session_id)
    result = Pipeline(settings).run(audio_path, session_id=session_id)
    return result.json_dict()


def translate_text(kikuyu_text: str) -> dict:
    """Translate Kikuyu text directly with the configured translation model."""
    settings = Settings.from_env()
    if settings.api_url:
        return translate_text_remote(settings.api_url, kikuyu_text)
    translator = Translator(
        settings.translation_model,
        settings.translation_src_lang,
        settings.translation_tgt_lang,
    )
    english, confidence = translator.translate(kikuyu_text)
    return {
        "kikuyu": kikuyu_text,
        "english": english,
        "translation_confidence": confidence,
        "translation_model": settings.translation_model,
        "translation_src_lang": settings.translation_src_lang,
        "translation_tgt_lang": settings.translation_tgt_lang,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Access the Kikuyu AI translator")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path, help="Audio or video file to translate")
    source.add_argument("--text", help="Kikuyu text to translate directly")
    parser.add_argument("--session-id", help="Optional session id for audio translation")
    parser.add_argument("--api-url", help="Remote FastAPI base URL; overrides KIKUYU_API_URL")
    args = parser.parse_args()

    if args.api_url:
        import os

        os.environ["KIKUYU_API_URL"] = args.api_url

    if args.audio:
        payload = translate_audio(args.audio, session_id=args.session_id)
    else:
        payload = translate_text(args.text)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
