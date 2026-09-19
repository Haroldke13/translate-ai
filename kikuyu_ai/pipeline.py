from pathlib import Path
from uuid import uuid4
import json
import re
import shutil
import subprocess
import tempfile
from .asr import ASR
from .audio import FFMPEG_PROTOCOLS, ffmpeg_env, normalize_audio, safe_suffix
from .config import Settings
from .models import TranslationResult
from .outputs import write_metadata, write_srt
from .translator import BibleAligner, Translator
from .tts import synthesize


class Pipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.settings.ensure_dirs()
        self.asr = ASR(
            self.settings.asr_model,
            self.settings.asr_backend,
            self.settings.asr_language,
            self.settings.asr_chunk_seconds,
            self.settings.asr_retry_chunk_seconds,
            self.settings.asr_silence_rms,
            self.settings.asr_max_new_tokens,
            self.settings.asr_no_repeat_ngram_size,
            self.settings.asr_repetition_penalty,
        )
        self.translator = Translator(
            self.settings.translation_model,
            self.settings.translation_src_lang,
            self.settings.translation_tgt_lang,
            self.settings.translation_max_new_tokens,
        )
        self.aligner = BibleAligner(self.settings.bible)

    def run(self, source: Path, session_id: str | None = None) -> TranslationResult:
        source = source.resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        session_id = self._validate_session_id(session_id or uuid4().hex)
        folder = self.settings.sessions / session_id
        if folder.exists():
            raise FileExistsError(folder)

        work_folder = Path(tempfile.mkdtemp(prefix=f".{session_id}.", dir=self.settings.sessions))
        finalized = False
        try:
            original = work_folder / "audio_original.mp3"
            if source.suffix.casefold() == ".mp3":
                shutil.copyfile(source, original)
            else:
                # Keep a playable copy when ffmpeg can make one, otherwise keep
                # the upload byte for byte under a safe, recognisable name.
                converted = None
                if shutil.which("ffmpeg"):
                    converted = subprocess.run(
                        [
                            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                            "-protocol_whitelist", FFMPEG_PROTOCOLS,
                            "-y", "-i", str(source), "-vn", str(original),
                        ],
                        capture_output=True,
                        env=ffmpeg_env(),
                    )
                if not converted or converted.returncode or not original.is_file():
                    original.unlink(missing_ok=True)
                    original = work_folder / f"audio_original{safe_suffix(source)}"
                    shutil.copyfile(source, original)
            original_name = original.name

            clean = work_folder / "audio_clean.wav"
            duration = normalize_audio(source, clean)
            segments, asr_conf = self.asr.transcribe(clean)
            if not self.settings.keep_asr_loaded:
                self.asr.unload()

            kikuyu = " ".join(s.text for s in segments).strip()
            self._write_segments_jsonl(segments, work_folder / "asr_segments.jsonl")
            match = self.aligner.match(kikuyu)
            english_segments = []
            if match:
                english, tr_conf = match.english, match.score
            else:
                english_segments, english, tr_conf = self.translator.translate_segments(segments)
            if not self.settings.keep_translation_loaded:
                self.translator.unload()

            (work_folder / "kikuyu.txt").write_text(kikuyu, encoding="utf-8")
            (work_folder / "english.txt").write_text(english, encoding="utf-8")
            write_srt(segments or [], work_folder / "subtitles.srt")
            self._write_segments_jsonl(english_segments, work_folder / "english_segments.jsonl")
            if english_segments:
                write_srt(english_segments, work_folder / "english_subtitles.srt")
            tts_target = work_folder / "translation.mp3"
            synthesized = synthesize(english, tts_target, self.settings.tts_command)
            metadata = {
                "language": "kikuyu",
                "duration": round(duration, 3),
                "speaker_count": None,
                "translation_confidence": round(tr_conf, 3),
                "transcription_confidence": round(asr_conf, 3),
                "bible_match": bool(match),
                "tts_generated": synthesized,
            }
            write_metadata(work_folder / "metadata.json", metadata)
            if folder.exists():
                raise FileExistsError(folder)
            work_folder.rename(folder)
            finalized = True
        finally:
            if not finalized and work_folder.exists():
                shutil.rmtree(work_folder, ignore_errors=True)

        names = [
            original_name,
            "audio_clean.wav",
            "asr_segments.jsonl",
            "kikuyu.txt",
            "english.txt",
            "english_segments.jsonl",
            "english_subtitles.srt",
            "subtitles.srt",
            "metadata.json",
        ]
        files = {name: str(folder / name) for name in names if (folder / name).exists()}
        if synthesized:
            files["translation.mp3"] = str(folder / "translation.mp3")
        return TranslationResult(session_id, kikuyu, english, segments, match, asr_conf, tr_conf, duration, files)

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", session_id):
            raise ValueError("session_id must be 1-128 characters of letters, numbers, dot, underscore, or hyphen")
        if session_id in {".", ".."}:
            raise ValueError("session_id cannot be . or ..")
        return session_id

    @staticmethod
    def _write_segments_jsonl(segments, path: Path) -> None:
        rows = [
            {
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": segment.text,
                "confidence": segment.confidence,
            }
            for segment in segments
            if segment.text.strip()
        ]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
