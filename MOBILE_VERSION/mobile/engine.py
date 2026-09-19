"""Decide which translator answers, and say honestly which one did.

Three tiers, best first:

1. **Verse** — the text is scripture, so the published English is returned
   verbatim. Perfect, and needs no model.
2. **Neural** — NLLB translates it properly. Needs the optional model files.
3. **Gloss** — word-by-word from the alignment table. Right words, wrong
   grammar, and labelled as such.

The point of the ordering is that the phone always answers *something*, and
never dresses up a weaker answer as a stronger one. A gloss is returned marked
as a gloss; if even that is not possible, the engine says so instead of
returning an empty string that looks like a failed translation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import languages, spelling
from .gloss import GlossTable
from .languages import Language
from .neural import NeuralSpeech, NeuralTranslator, Segment
from .verses import Verse, VerseIndex

#: A verse this close is worth offering as context even when it is not the
#: answer. Below it the overlap is common words rather than the same sentence.
NEAR_VERSE = 0.50
#: Below this share of recognisable words, a transcript is probably mis-heard
#: and the translation built on it should not be presented as trustworthy.
HEARD_WELL = 0.34


@dataclass
class Translation:
    language: str
    language_name: str
    source: str
    english: str
    engine: str
    engine_label: str
    confidence: float
    read_as: str | None = None
    verse: Verse | None = None
    near_verse: Verse | None = None
    note: str | None = None
    #: Set when a recording was transcribed into words the language does not
    #: have. Without it, a mis-heard recording still produces fluent English and
    #: nothing on screen suggests it is wrong.
    warning: str | None = None
    transcript_segments: list[Segment] = field(default_factory=list)
    duration: float = 0.0

    def as_dict(self) -> dict:
        def verse_dict(value: Verse | None) -> dict | None:
            if not value:
                return None
            return {
                "reference": value.reference,
                "book": value.book,
                "chapter": value.chapter,
                "verse": value.verse,
                "source": value.source,
                "english": value.english,
                "score": value.score,
            }

        return {
            "language": self.language,
            "language_name": self.language_name,
            "source": self.source,
            "read_as": self.read_as,
            "english": self.english,
            "engine": self.engine,
            "engine_label": self.engine_label,
            "confidence": self.confidence,
            "verse": verse_dict(self.verse),
            "near_verse": verse_dict(self.near_verse),
            "note": self.note,
            "warning": self.warning,
            "duration": round(self.duration, 2),
        }


class LanguageResources:
    """Everything one language needs, built the first time it is asked for."""

    def __init__(self, language: Language, root: Path):
        self.language = language
        self.data_dir = language.data_dir(root)
        self.verses = VerseIndex(self.data_dir)
        self.gloss = GlossTable(self.data_dir)
        self._spelling: dict[str, str] | None = None

    @property
    def spelling(self) -> dict[str, str]:
        if self._spelling is None:
            self._spelling = spelling.load_table(self.data_dir)
        return self._spelling


class Engine:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.translator = NeuralTranslator(self.models_dir)
        self.speech = NeuralSpeech(self.models_dir)
        self._resources: dict[str, LanguageResources] = {}

    def resources(self, language: Language) -> LanguageResources:
        if language.code not in self._resources:
            self._resources[language.code] = LanguageResources(language, self.root)
        return self._resources[language.code]

    # -- capability reporting -------------------------------------------

    def capabilities(self) -> dict:
        """What this particular phone can actually do, for the status line."""
        neural = self.translator.available
        per_language = {}
        for language in languages.ordered():
            resource = self.resources(language)
            per_language[language.code] = {
                "name": language.name,
                "native": language.native,
                "text": True,
                "neural": neural,
                "speech": self.speech.available(language.mms_code),
                "verses": resource.verses.available,
                "gloss": resource.gloss.available,
                "spelling": bool(resource.spelling),
            }
        return {
            "neural_translation": neural,
            "neural_reason": None if neural else self.translator.why_unavailable(),
            "speech_installed": self.speech.installed,
            "languages": per_language,
        }

    def speech_error(self, language: Language) -> str:
        return self.speech.why_unavailable(language.mms_code, language.name)

    # -- translation ------------------------------------------------------

    def translate_text(self, text: str, language: Language) -> Translation:
        resource = self.resources(language)
        prepared, changes = spelling.prepare(text, resource.spelling)
        read_as = prepared if changes else None

        verse = resource.verses.search(prepared) if resource.verses.available else None
        if verse:
            return Translation(
                language=language.code,
                language_name=language.name,
                source=text.strip(),
                read_as=read_as,
                english=verse.english,
                engine="verse",
                engine_label="Published Bible verse",
                confidence=verse.score,
                verse=verse,
                note="This is scripture, so the published English is shown rather than a machine translation.",
            )

        near = resource.verses.search(prepared, threshold=NEAR_VERSE) if resource.verses.available else None

        if self.translator.available:
            english = self.translator.translate(prepared, language.nllb_code)
            if english:
                return Translation(
                    language=language.code,
                    language_name=language.name,
                    source=text.strip(),
                    read_as=read_as,
                    english=english,
                    engine="neural",
                    engine_label="Neural translation",
                    confidence=0.70,
                    near_verse=near,
                )

        glossed = resource.gloss.gloss(prepared)
        if glossed:
            return Translation(
                language=language.code,
                language_name=language.name,
                source=text.strip(),
                read_as=read_as,
                english=glossed.text,
                engine="gloss",
                engine_label="Word-by-word only",
                confidence=round(glossed.coverage * 0.5, 2),
                near_verse=near,
                note=(
                    "Word-by-word from the Bible word list — the words are right but the "
                    "grammar is not. Install the translation model for a real translation."
                ),
            )

        return Translation(
            language=language.code,
            language_name=language.name,
            source=text.strip(),
            read_as=read_as,
            english="",
            engine="none",
            engine_label="Not translated",
            confidence=0.0,
            near_verse=near,
            note=self._nothing_available(language, resource),
        )

    def _nothing_available(self, language: Language, resource: LanguageResources) -> str:
        if not self.translator.installed:
            if not resource.gloss.available:
                return (
                    f"{language.name} has no offline word list on this phone and the "
                    "translation model is not installed, so there is nothing to translate "
                    "with. Copy MOBILE_VERSION/models/translation across."
                )
            return (
                "Too few of these words are in the offline word list to give an honest "
                "answer. Install the translation model to translate any sentence."
            )
        return self.translator.why_unavailable()

    # -- speech ------------------------------------------------------------

    def transcribe(self, audio_path: Path, language: Language, progress=None) -> tuple[str, list[Segment], float]:
        from . import audio as audio_module

        if not self.speech.available(language.mms_code):
            raise RuntimeError(self.speech_error(language))

        work = audio_path.with_suffix(".16k.wav")
        prepared = audio_module.to_wav(audio_path, work)
        samples, rate = audio_module.read_samples(prepared)
        duration = len(samples) / rate if rate else 0.0
        segments = self.speech.transcribe(samples, rate, language.mms_code, progress=progress)
        if prepared != audio_path:
            prepared.unlink(missing_ok=True)
        return " ".join(segment.text for segment in segments).strip(), segments, duration

    def heard_well(self, transcript: str, language: Language) -> float | None:
        """Share of transcribed words this language actually uses.

        Speech models invent plausible-looking letter sequences when they are
        unsure, and the translator downstream will happily turn those into a
        fluent English sentence. Checking the transcript against the language's
        own vocabulary is the cheapest way to notice. Returns None when there is
        no word list to check against, which is not the same as "fine".
        """
        resource = self.resources(language)
        vocabulary = resource.gloss.load().words
        spellings = resource.spelling
        if not vocabulary and not spellings:
            return None
        words = [word for word in re.findall(r"[^\W\d_]+", transcript, re.UNICODE)]
        if not words:
            return 0.0
        known = 0
        for word in words:
            folded = spelling.fold(word)
            if word.casefold() in vocabulary or folded in spellings or folded in vocabulary:
                known += 1
        return known / len(words)

    def translate_audio(self, audio_path: Path, language: Language, progress=None) -> Translation:
        transcript, segments, duration = self.transcribe(audio_path, language, progress=progress)
        if not transcript:
            return Translation(
                language=language.code,
                language_name=language.name,
                source="",
                english="",
                engine="none",
                engine_label="Nothing heard",
                confidence=0.0,
                duration=duration,
                note="No speech was recognised in that recording.",
            )
        result = self.translate_text(transcript, language)
        result.transcript_segments = segments
        result.duration = duration

        share = self.heard_well(transcript, language)
        if share is not None and share < HEARD_WELL:
            result.warning = (
                f"Only {share:.0%} of what was heard looks like real {language.name}. "
                "The English below is probably wrong — check the transcript above. "
                "Speech works better on the computer version."
            )
        return result
