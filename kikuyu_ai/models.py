from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Segment:
    start: float
    end: float
    text: str
    confidence: float | None = None


@dataclass
class BibleMatch:
    book: str
    chapter: int
    verse: int
    kikuyu: str
    english: str
    score: float


@dataclass
class TranslationResult:
    session_id: str
    kikuyu: str
    english: str
    segments: list[Segment]
    bible_match: BibleMatch | None
    transcription_confidence: float
    translation_confidence: float
    duration: float
    files: dict[str, str]

    def json_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.bible_match:
            result["bible_match"] = asdict(self.bible_match)
        return result

