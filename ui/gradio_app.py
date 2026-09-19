from pathlib import Path
from kikuyu_ai.config import Settings
from kikuyu_ai.pipeline import Pipeline
from kikuyu_ai.remote import RemoteAPIError, translate_audio_remote


def _existing_file(value):
    if not value:
        return None
    path = Path(value)
    return str(path) if path.is_file() else None


def _payload_file(payload, filename, url_key=None):
    files = payload.get("files") if isinstance(payload.get("files"), dict) else {}
    local = _existing_file(files.get(filename))
    if local:
        return local
    if url_key:
        return payload.get(url_key)
    file_urls = payload.get("file_urls") if isinstance(payload.get("file_urls"), dict) else {}
    return file_urls.get(filename)


def build():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install gradio to run the local UI") from exc

    settings = Settings.from_env()
    if settings.api_url:
        def translate(audio):
            if not audio:
                return "", "", None, None
            try:
                payload = translate_audio_remote(settings.api_url, Path(audio))
            except RemoteAPIError as exc:
                return (
                    "Remote API error",
                    f"{exc}\n\nAPI URL: {settings.api_url}\nStart the API with ./translate.sh server on that machine, or choose the correct API URL.",
                    None,
                    None,
                )
            return (
                payload.get("kikuyu", ""),
                payload.get("english", ""),
                _payload_file(payload, "translation.mp3", "audio"),
                _payload_file(payload, "english_subtitles.srt", "english_subtitles")
                or _payload_file(payload, "subtitles.srt", "subtitles"),
            )
    else:
        pipeline = Pipeline(settings)

        def translate(audio):
            if not audio:
                return "", "", None, None
            result = pipeline.run(Path(audio))
            return result.kikuyu, result.english, result.files.get("translation.mp3"), result.files.get("subtitles.srt")

    return gr.Interface(
        fn=translate,
        inputs=gr.Audio(type="filepath", sources=["upload", "microphone"]),
        outputs=[
            gr.Textbox(label="Kikuyu"),
            gr.Textbox(label="English"),
            gr.Audio(label="English audio"),
            gr.File(label="Subtitles"),
        ],
        title="Kikuyu to English Translator",
    )


if __name__ == "__main__":
    build().launch()
