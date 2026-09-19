from dataclasses import dataclass
from pathlib import Path
import os


DEFAULT_ASR_BACKEND = "transformers"
DEFAULT_ASR_MODEL = "Kiragu/whisper-small-kikuyu-v5"
DEFAULT_TRANSLATION_MODEL = "nickdee96/nllb-200-600m-kikuyu-english"
DEFAULT_TRANSLATION_SRC_LANG = "kik_Latn"
DEFAULT_TRANSLATION_TGT_LANG = "eng_Latn"
DEFAULT_ASR_CHUNK_SECONDS = 12.0
DEFAULT_ASR_RETRY_CHUNK_SECONDS = 6.0
DEFAULT_ASR_SILENCE_RMS = 0.001
DEFAULT_ASR_MAX_NEW_TOKENS = 96
DEFAULT_ASR_NO_REPEAT_NGRAM_SIZE = 3
DEFAULT_ASR_REPETITION_PENALTY = 1.15
DEFAULT_TRANSLATION_MAX_NEW_TOKENS = 128


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values


def _env_bool(env: dict[str, str], key: str, default: bool = False) -> bool:
    value = env.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(env: dict[str, str], key: str, default: float) -> float:
    value = env.get(key)
    if not value:
        return default
    return float(value)


def _env_int(env: dict[str, str], key: str, default: int) -> int:
    value = env.get(key)
    if not value:
        return default
    return int(value)


def _env_optional(env: dict[str, str], key: str, default: str | None = None) -> str | None:
    value = env.get(key, default)
    return value or None


def _env_model(env: dict[str, str], key: str, root: Path, default: str | None = None) -> str | None:
    value = _env_optional(env, key, default)
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    local_path = root / path
    if local_path.exists():
        return str(local_path)
    return value


@dataclass(frozen=True)
class Settings:
    root: Path
    sessions: Path
    bible: Path
    corrections_db: Path
    api_url: str | None
    api_host: str
    api_port: int
    asr_backend: str
    asr_model: str | None
    asr_language: str | None
    asr_chunk_seconds: float
    asr_retry_chunk_seconds: float
    asr_silence_rms: float
    asr_max_new_tokens: int
    asr_no_repeat_ngram_size: int
    asr_repetition_penalty: float
    translation_model: str | None
    translation_src_lang: str | None
    translation_tgt_lang: str | None
    translation_max_new_tokens: int
    keep_asr_loaded: bool
    keep_translation_loaded: bool
    tts_command: str | None
    max_upload_mb: int

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        base = (root or Path(__file__).resolve().parents[1]).resolve()
        file_env = _read_env_file(base / ".env")
        env = {**file_env, **os.environ}
        return cls(
            root=base,
            sessions=base / "data" / "sessions",
            bible=base / "data" / "bible",
            corrections_db=base / "data" / "corrections.db",
            api_url=_env_optional(env, "KIKUYU_API_URL"),
            api_host=env.get("KIKUYU_API_HOST", "127.0.0.1"),
            api_port=_env_int(env, "KIKUYU_API_PORT", 8000),
            asr_backend=env.get("KIKUYU_ASR_BACKEND", DEFAULT_ASR_BACKEND),
            asr_model=_env_model(env, "KIKUYU_ASR_MODEL", base, DEFAULT_ASR_MODEL),
            asr_language=_env_optional(env, "KIKUYU_ASR_LANGUAGE"),
            asr_chunk_seconds=_env_float(env, "KIKUYU_ASR_CHUNK_SECONDS", DEFAULT_ASR_CHUNK_SECONDS),
            asr_retry_chunk_seconds=_env_float(env, "KIKUYU_ASR_RETRY_CHUNK_SECONDS", DEFAULT_ASR_RETRY_CHUNK_SECONDS),
            asr_silence_rms=_env_float(env, "KIKUYU_ASR_SILENCE_RMS", DEFAULT_ASR_SILENCE_RMS),
            asr_max_new_tokens=_env_int(env, "KIKUYU_ASR_MAX_NEW_TOKENS", DEFAULT_ASR_MAX_NEW_TOKENS),
            asr_no_repeat_ngram_size=_env_int(env, "KIKUYU_ASR_NO_REPEAT_NGRAM_SIZE", DEFAULT_ASR_NO_REPEAT_NGRAM_SIZE),
            asr_repetition_penalty=_env_float(env, "KIKUYU_ASR_REPETITION_PENALTY", DEFAULT_ASR_REPETITION_PENALTY),
            translation_model=_env_model(env, "KIKUYU_TRANSLATION_MODEL", base, DEFAULT_TRANSLATION_MODEL),
            translation_src_lang=_env_optional(env, "KIKUYU_TRANSLATION_SRC_LANG", DEFAULT_TRANSLATION_SRC_LANG),
            translation_tgt_lang=_env_optional(env, "KIKUYU_TRANSLATION_TGT_LANG", DEFAULT_TRANSLATION_TGT_LANG),
            translation_max_new_tokens=_env_int(env, "KIKUYU_TRANSLATION_MAX_NEW_TOKENS", DEFAULT_TRANSLATION_MAX_NEW_TOKENS),
            keep_asr_loaded=_env_bool(env, "KIKUYU_KEEP_ASR_LOADED"),
            keep_translation_loaded=_env_bool(env, "KIKUYU_KEEP_TRANSLATION_LOADED"),
            tts_command=_env_optional(env, "KIKUYU_TTS_COMMAND"),
            max_upload_mb=int(env.get("KIKUYU_MAX_UPLOAD_MB", "512")),
        )

    def ensure_dirs(self) -> None:
        self.sessions.mkdir(parents=True, exist_ok=True)
        self.bible.mkdir(parents=True, exist_ok=True)
