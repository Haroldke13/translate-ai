import sqlite3
from pathlib import Path
import os
from .ml_runtime import configure_torch, local_pretrained_kwargs, release_torch_memory
from .models import BibleMatch, Segment


class Translator:
    def __init__(
        self,
        model_name: str | None = None,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        max_new_tokens: int = 128,
    ):
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_new_tokens = max(16, int(max_new_tokens or 128))
        self.progress = os.environ.get("KIKUYU_TRANSLATION_PROGRESS", "").casefold() in {"1", "true", "yes", "on"}
        self._model = None

    def unload(self) -> None:
        self._model = None
        try:
            import torch
        except ImportError:
            torch = None
        release_torch_memory(torch)

    def translate(self, text: str) -> tuple[str, float]:
        if not text.strip():
            return "", 0.0
        if not self.model_name:
            return "[Translation model not configured]", 0.0
        try:
            import torch
            tokenizer, model = self._load_model(torch)
            output = self._translate_loaded(text, tokenizer, model, torch)
            return output, 0.70
        except ImportError as exc:
            raise RuntimeError("Install transformers to use the configured translation model") from exc

    def translate_segments(self, segments: list[Segment]) -> tuple[list[Segment], str, float]:
        usable = [segment for segment in segments if segment.text.strip()]
        if not usable:
            return [Segment(0.0, 0.0, "", None)], "", 0.0
        if not self.model_name:
            placeholder = [Segment(segment.start, segment.end, "[Translation model not configured]", 0.0) for segment in usable]
            return placeholder, "\n".join(segment.text for segment in placeholder), 0.0
        try:
            import torch
            tokenizer, model = self._load_model(torch)
            translated: list[Segment] = []
            total = len(usable)
            for index, segment in enumerate(usable, 1):
                english = self._translate_loaded(segment.text, tokenizer, model, torch)
                if english:
                    translated.append(Segment(segment.start, segment.end, english, 0.70))
                if self.progress:
                    print(f"translation segment {index}/{total}: {english[:80]!r}", flush=True)
            english_text = "\n".join(segment.text for segment in translated).strip()
            return translated or [Segment(0.0, 0.0, "", None)], english_text, 0.70 if english_text else 0.0
        except ImportError as exc:
            raise RuntimeError("Install transformers to use the configured translation model") from exc

    def _load_model(self, torch):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self._model is None:
            configure_torch(torch)
            pretrained_kwargs = local_pretrained_kwargs()
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **pretrained_kwargs)
            model.to("cpu")
            model.eval()
            self._model = (tokenizer, model)

        tokenizer, model = self._model
        if self.src_lang and hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = self.src_lang
        if self.tgt_lang and hasattr(tokenizer, "tgt_lang"):
            tokenizer.tgt_lang = self.tgt_lang
        return tokenizer, model

    def _translate_loaded(self, text: str, tokenizer, model, torch) -> str:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        generate_kwargs = {"max_length": self.max_new_tokens}
        forced_bos_token_id = self._lang_token_id(tokenizer, self.tgt_lang)
        if forced_bos_token_id is not None:
            generate_kwargs["forced_bos_token_id"] = forced_bos_token_id
        with torch.inference_mode():
            generated = model.generate(**inputs, **generate_kwargs)
        output = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        del inputs, generated
        release_torch_memory(torch)
        return output.strip()

    @staticmethod
    def _lang_token_id(tokenizer, lang: str | None) -> int | None:
        if not lang:
            return None
        lang_code_to_id = getattr(tokenizer, "lang_code_to_id", None)
        if isinstance(lang_code_to_id, dict) and lang in lang_code_to_id:
            return int(lang_code_to_id[lang])
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if not convert:
            return None
        token_id = convert(lang)
        if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
            return None
        return int(token_id)


class BibleAligner:
    def __init__(self, bible_dir: Path, threshold: float = 0.82):
        self.bible_dir, self.threshold = bible_dir, threshold
        self.rows: list[dict] = []
        self._load()

    def _load(self) -> None:
        import json
        path = self.bible_dir / "parallel.jsonl"
        if path.exists():
            self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {word.casefold().strip(".,!?;:()[]{}\"'") for word in value.split() if len(word) > 1}

    def match(self, text: str) -> BibleMatch | None:
        source = self._tokens(text)
        if not source:
            return None
        best = None
        for row in self.rows:
            target = self._tokens(row.get("kikuyu", ""))
            score = len(source & target) / max(1, len(source | target))
            if best is None or score > best[0]:
                best = (score, row)
        if not best or best[0] < self.threshold:
            return None
        row = best[1]
        return BibleMatch(str(row["book"]), int(row["chapter"]), int(row["verse"]), row["kikuyu"], row["english"], round(best[0], 4))


def save_correction(db_path: Path, source: str, machine: str, corrected: str) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS corrections (id INTEGER PRIMARY KEY, source TEXT NOT NULL, machine TEXT NOT NULL, corrected TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        cursor = db.execute("INSERT INTO corrections(source, machine, corrected) VALUES (?, ?, ?)", (source, machine, corrected))
        db.commit()
        return int(cursor.lastrowid)
