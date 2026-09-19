"""The source languages the app can translate into English.

A language needs three things to work end to end:

* an **NLLB tag** so the translation model knows what it is reading,
* a **speech model** if you want to upload audio, and
* optionally a folder of **aligned Bible verses**, which powers exact verse
  lookup and the spelling repair that fills in missing accented letters.

Speech comes from Meta's MMS, which covers over a thousand languages with one
shared 3.9 GB encoder plus a ~9 MB adapter per language. Adding a language is
therefore an adapter download, not a training run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

DEFAULT_LANGUAGE = "kikuyu"

#: One shared encoder, swapped per-language adapters.
MMS_MODEL = "models/downloads/huggingface/facebook--mms-1b-all"

def kikuyu_engine() -> str:
    """Which speech model Kikuyu uses: "mms" (default) or "whisper".

    Read on each call rather than at import, so setting the variable after the
    module is loaded still takes effect. MMS transcribes more real Kikuyu words
    and restores the tilde vowels, but needs about three times the memory and
    runs about 2.5x slower, so both options stay open.
    """
    return os.environ.get("KIKUYU_ASR_ENGINE", "mms").strip().casefold()


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    nllb_code: str
    #: Folder under data/bible holding parallel.jsonl. Kikuyu keeps the top
    #: level so existing installs and paths keep working.
    bible_subdir: str = ""
    speech: bool = False
    speech_note: str = ""
    placeholder: str = ""
    #: Speech settings. None means "use whatever the environment configured".
    asr_model: str | None = None
    asr_backend: str | None = None
    asr_language: str | None = None

    def bible_dir(self, root_bible: Path) -> Path:
        return root_bible / self.bible_subdir if self.bible_subdir else root_bible


def _mms(code: str, name: str, nllb: str, adapter: str, subdir: str, placeholder: str = "") -> Language:
    return Language(
        code=code,
        name=name,
        nllb_code=nllb,
        bible_subdir=subdir,
        speech=True,
        placeholder=placeholder,
        asr_model=MMS_MODEL,
        asr_backend="mms",
        asr_language=adapter,
    )


KIKUYU_PLACEHOLDER = "Andĩka kana ũcookererie Gĩgĩkũyũ haha…"
#: Kikuyu keeps the top-level corpus folder that earlier versions wrote to.
KIKUYU_MMS = _mms("kikuyu", "Kikuyu", "kik_Latn", "kik", "", KIKUYU_PLACEHOLDER)
KIKUYU_WHISPER = Language(
    code="kikuyu",
    name="Kikuyu",
    nllb_code="kik_Latn",
    bible_subdir="",
    speech=True,
    placeholder=KIKUYU_PLACEHOLDER,
)

LANGUAGES: dict[str, Language] = {
    "kikuyu": KIKUYU_MMS,
    "luo": _mms("luo", "Luo", "luo_Latn", "luo", "luo", "Ndik kata mak weche mag Dholuo ka…"),
    "kamba": _mms("kamba", "Kamba", "kam_Latn", "kam", "kamba", "Andĩka Kĩkamba vaa…"),
    "swahili": _mms("swahili", "Swahili", "swh_Latn", "swh", "swahili", "Andika Kiswahili hapa…"),
    "somali": _mms("somali", "Somali", "som_Latn", "som", "somali", "Halkan ku qor Af-Soomaali…"),
    "oromo": _mms("oromo", "Oromo", "gaz_Latn", "orm", "oromo", "Afaan Oromoo asitti barreessi…"),
}


def _resolve(language: Language) -> Language:
    """Apply any choice that depends on the environment right now."""
    if language.code == "kikuyu":
        return KIKUYU_MMS if kikuyu_engine() == "mms" else KIKUYU_WHISPER
    return language


def get(code: str | None) -> Language:
    """Return the requested language, falling back to the default."""
    return _resolve(LANGUAGES.get((code or "").strip().casefold(), LANGUAGES[DEFAULT_LANGUAGE]))


def ordered() -> list[Language]:
    """Languages for display, with the default first."""
    default = get(DEFAULT_LANGUAGE)
    rest = [_resolve(language) for code, language in LANGUAGES.items() if code != DEFAULT_LANGUAGE]
    return [default, *sorted(rest, key=lambda language: language.name)]


def settings_for(language: Language, settings):
    """Settings adjusted for one language's models.

    The translation model is multilingual, so only its source tag changes. The
    speech model does not generalise, so a language that names its own ASR model
    gets that one instead of the configured default.
    """
    changes: dict = {"translation_src_lang": language.nllb_code}
    if language.asr_model:
        model = Path(language.asr_model)
        if not model.is_absolute():
            local = settings.root / model
            model = local if local.exists() else Path(language.asr_model)
        changes["asr_model"] = str(model)
    if language.asr_backend:
        changes["asr_backend"] = language.asr_backend
    if language.asr_language is not None:
        changes["asr_language"] = language.asr_language
    return replace(settings, **changes)


def adapter_path(language: Language, settings) -> Path | None:
    """Where this language's MMS adapter should live, if it uses MMS."""
    if language.asr_backend not in {"mms", "wav2vec2", "w2v"} or not language.asr_language:
        return None
    model = Path(settings_for(language, settings).asr_model or "")
    return model / f"adapter.{language.asr_language}.safetensors"


def speech_available(language: Language, settings) -> bool:
    """Whether this language's speech model is actually present on disk.

    A relative model path is resolved against the project root only. Resolving
    it against the current directory instead would make the answer depend on
    where the process happens to be started.
    """
    if not language.speech:
        return False
    if not language.asr_model:
        return bool(settings.asr_model)
    model = Path(language.asr_model)
    if not model.is_absolute():
        model = settings.root / model
    if not model.exists():
        return False
    # MMS needs the shared encoder *and* this language's adapter.
    adapter = adapter_path(language, settings)
    return adapter is None or adapter.exists()
