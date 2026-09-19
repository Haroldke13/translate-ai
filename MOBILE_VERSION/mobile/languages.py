"""The four languages this phone build translates into English.

Deliberately a smaller list than the desktop app. Every language here has been
checked to have the two things a phone needs: an NLLB tag so written text can be
translated, and an MMS adapter so a recording can be transcribed. What differs
between them is the *third* thing — a parallel Bible corpus — which is what
powers exact verse lookup, spelling repair and the offline word gloss.

Kamba has no openly licensed Bible, so it ships without a corpus and is honest
about that rather than pretending to have one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_LANGUAGE = "kikuyu"


@dataclass(frozen=True)
class Language:
    code: str
    #: What to call it in English, for the interface.
    name: str
    #: What speakers call it, shown on the language button.
    native: str
    #: Source tag for NLLB. Target is always eng_Latn.
    nllb_code: str
    #: MMS adapter name, which is how MMS picks a language.
    mms_code: str
    #: False when no parallel Bible text exists, which disables verse lookup,
    #: spelling repair and the offline gloss for this language.
    corpus: bool
    placeholder: str

    def data_dir(self, root: Path) -> Path:
        return root / "data" / self.code


LANGUAGES: dict[str, Language] = {
    "kikuyu": Language(
        code="kikuyu",
        name="Kikuyu",
        native="Gĩgĩkũyũ",
        nllb_code="kik_Latn",
        mms_code="kik",
        corpus=True,
        placeholder="Andĩka kana ũcookererie Gĩgĩkũyũ haha…",
    ),
    "kamba": Language(
        code="kamba",
        name="Kamba",
        native="Kĩkamba",
        nllb_code="kam_Latn",
        mms_code="kam",
        corpus=False,
        placeholder="Andĩka Kĩkamba vaa…",
    ),
    "oromo": Language(
        code="oromo",
        name="Oromo",
        native="Afaan Oromoo",
        nllb_code="gaz_Latn",
        mms_code="orm",
        corpus=True,
        placeholder="Afaan Oromoo asitti barreessi…",
    ),
    "somali": Language(
        code="somali",
        name="Somali",
        native="Af-Soomaali",
        nllb_code="som_Latn",
        mms_code="som",
        corpus=True,
        placeholder="Halkan ku qor Af-Soomaali…",
    ),
}


def get(code: str | None) -> Language:
    """The requested language, falling back to the default rather than failing.

    A bad code arrives from a stale bookmark or a hand-typed URL far more often
    than from a real mistake, so it is not worth an error page.
    """
    return LANGUAGES.get((code or "").strip().casefold(), LANGUAGES[DEFAULT_LANGUAGE])


def ordered() -> list[Language]:
    """Languages for display, default first, the rest alphabetical."""
    default = LANGUAGES[DEFAULT_LANGUAGE]
    rest = sorted(
        (language for code, language in LANGUAGES.items() if code != DEFAULT_LANGUAGE),
        key=lambda language: language.name,
    )
    return [default, *rest]
