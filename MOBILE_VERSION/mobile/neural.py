"""The optional neural tier: real translation, and speech when it is installed.

This is what turns the phone build from a phrasebook into a translator, and it
is deliberately the only part that needs anything installed. Every entry point
answers "am I here?" before it answers anything else, so a phone without these
files degrades to the offline tier instead of failing.

**Why CTranslate2 and not PyTorch.** Termux's Python is 3.14 and there is no
prebuilt `safetensors` for Android anywhere, so `pip install transformers` has
to compile Rust — a 576 MB toolchain — before it will even import. CTranslate2
is packaged for Termux at 5.5 MB, runs the same NLLB weights quantised to int8,
and needs about a third of the memory. The tokenizer comes from the `tokenizers`
library reading tokenizer.json directly, which is prebuilt for Android and makes
sentencepiece unnecessary.

PyTorch is still supported as a second choice, because the same folder should
work on the computer that packed it.

Nothing here touches the network.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

TRANSLATION_DIR = "translation"
SPEECH_DIR = "speech"
RUNTIME_NAME = "runtime.json"
SOURCE_TAG = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")
TARGET_LANGUAGE = "eng_Latn"
MAX_INPUT_TOKENS = 512
#: Long audio is processed in windows so memory stays flat on a whole sermon.
CHUNK_SECONDS = 12.0
SILENCE_RMS = 0.001


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


def _has(module: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def read_runtime(models_dir: Path) -> dict:
    """Loading choices the packer recorded, rather than guessed at here."""
    path = Path(models_dir) / RUNTIME_NAME
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _threads() -> int:
    requested = os.environ.get("KIKUYU_THREADS")
    if requested and requested.isdigit():
        return max(1, int(requested))
    # Half the cores keeps the phone responsive and, more importantly, keeps CPU
    # use below the level that makes Android's phantom-process killer step in.
    return max(1, (os.cpu_count() or 4) // 2)


class NeuralTranslator:
    """NLLB-200, translating any of the four languages into English."""

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.path = self.models_dir / TRANSLATION_DIR
        self.runtime = read_runtime(self.models_dir)
        self.max_new_tokens = int(self.runtime.get("translation_max_new_tokens", 128))
        self.beam_size = int(self.runtime.get("translation_beam_size", 4))
        self._loaded = None

    # -- what is actually here -------------------------------------------

    @property
    def backend(self) -> str | None:
        """"ctranslate2", "transformers", or None if nothing usable is here."""
        if (self.path / "model.bin").is_file() and (self.path / "tokenizer.json").is_file():
            return "ctranslate2" if _has("ctranslate2") and _has("tokenizers") else None
        if (self.path / "config.json").is_file():
            return "transformers" if _has("torch") and _has("transformers") else None
        return None

    @property
    def installed(self) -> bool:
        return (self.path / "model.bin").is_file() or (self.path / "config.json").is_file()

    @property
    def available(self) -> bool:
        return self.backend is not None

    def why_unavailable(self) -> str:
        if not self.installed:
            return (
                "The translation model is not on this phone. Copy "
                "MOBILE_VERSION/models/translation across to translate any "
                "sentence; verses and the word list work without it."
            )
        if (self.path / "model.bin").is_file():
            return (
                "The translation model is here but its runtime is not. Run "
                "./install.sh, or: pkg install python-ctranslate2 tur-repo "
                "&& pkg install python-tokenizers"
            )
        return "This model needs PyTorch. Repack it with --translation for the lighter runtime."

    def unload(self) -> None:
        self._loaded = None

    # -- translation -------------------------------------------------------

    def translate(self, text: str, source_tag: str) -> str:
        if not text.strip():
            return ""
        backend = self.backend
        if backend == "ctranslate2":
            return self._translate_ct2(text, source_tag)
        if backend == "transformers":
            return self._translate_torch(text, source_tag)
        raise RuntimeError(self.why_unavailable())

    def _load_ct2(self):
        if self._loaded is None:
            import ctranslate2
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(self.path / "tokenizer.json"))
            tokenizer.enable_truncation(MAX_INPUT_TOKENS)
            translator = ctranslate2.Translator(
                str(self.path),
                device="cpu",
                compute_type=self.runtime.get("translation_compute_type", "int8"),
                inter_threads=1,
                intra_threads=_threads(),
            )
            self._loaded = (tokenizer, translator)
        return self._loaded

    def _encode(self, tokenizer, text: str, source_tag: str) -> list[str]:
        """Tokens with the source language forced to the one we asked for.

        This checkpoint was fine-tuned on Kikuyu, so its tokenizer has kik_Latn
        baked into the post-processor and stamps it on every input regardless of
        language. Left alone, Somali gets translated as though it were Kikuyu
        and comes back as confident nonsense, so the tag is replaced rather than
        trusted.
        """
        tokens = tokenizer.encode(text).tokens
        if tokens and SOURCE_TAG.match(tokens[0]):
            tokens[0] = source_tag
        else:
            tokens = [source_tag, *tokens]
        return tokens

    def _translate_ct2(self, text: str, source_tag: str) -> str:
        tokenizer, translator = self._load_ct2()
        results = translator.translate_batch(
            [self._encode(tokenizer, text, source_tag)],
            target_prefix=[[TARGET_LANGUAGE]],
            beam_size=self.beam_size,
            max_decoding_length=self.max_new_tokens,
        )
        tokens = list(results[0].hypotheses[0])
        if tokens and tokens[0] == TARGET_LANGUAGE:
            tokens = tokens[1:]
        ids = [tokenizer.token_to_id(token) for token in tokens]
        return tokenizer.decode([i for i in ids if i is not None]).strip()

    def _translate_torch(self, text: str, source_tag: str) -> str:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self._loaded is None:
            torch.set_grad_enabled(False)
            torch.set_num_threads(_threads())
            tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(str(self.path), local_files_only=True)
            model.to("cpu")
            model.eval()
            self._loaded = (tokenizer, model)
        tokenizer, model = self._loaded
        tokenizer.src_lang = source_tag
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS)
        forced = tokenizer.convert_tokens_to_ids(TARGET_LANGUAGE)
        kwargs = {"max_length": self.max_new_tokens}
        if forced is not None and forced != getattr(tokenizer, "unk_token_id", None):
            kwargs["forced_bos_token_id"] = int(forced)
        with torch.inference_mode():
            generated = model.generate(**inputs, **kwargs)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


class NeuralSpeech:
    """Speech recognition through ONNX Runtime.

    Uses a single multilingual CTC model rather than MMS. MMS bakes its
    per-language adapter into the weights, so covering four languages would mean
    four separate exports of about a gigabyte each; one shared model is a
    fraction of that and covers all four.
    """

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.path = self.models_dir / SPEECH_DIR
        self.runtime = read_runtime(self.models_dir)
        self.model_file = self.path / self.runtime.get("speech_file", "model.int8.onnx")
        self.tokens_file = self.path / "tokens.txt"
        self._session = None
        self._tokens: list[str] = []

    @property
    def installed(self) -> bool:
        return self.model_file.is_file() and self.tokens_file.is_file()

    def available(self, mms_code: str | None = None) -> bool:
        return self.installed and _has("onnxruntime") and _has("numpy")

    def why_unavailable(self, mms_code: str, name: str) -> str:
        if not self.installed:
            return (
                "Speech recognition is not installed on this phone. It adds "
                "about 370 MB. Typed text still translates."
            )
        return (
            "Speech recognition needs its runtime. Run ./install.sh, or: "
            "pkg install python-onnxruntime python-numpy ffmpeg"
        )

    def unload(self) -> None:
        self._session = None

    def _load(self):
        if self._session is None:
            import onnxruntime

            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = _threads()
            options.inter_op_num_threads = 1
            self._session = onnxruntime.InferenceSession(
                str(self.model_file), options, providers=["CPUExecutionProvider"]
            )
            self._tokens = [
                line.rsplit(" ", 1)[0]
                for line in self.tokens_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return self._session

    def _decode(self, ids) -> str:
        """Greedy CTC: drop blanks and collapse runs of the same symbol."""
        pieces: list[str] = []
        previous = -1
        for token_id in ids:
            token_id = int(token_id)
            if token_id != previous and token_id != 0 and token_id < len(self._tokens):
                pieces.append(self._tokens[token_id])
            previous = token_id
        return "".join(pieces).replace("▁", " ").strip()

    def transcribe(self, samples, rate: int, mms_code: str, progress=None) -> list[Segment]:
        import numpy

        from .audio import rms

        if not len(samples):
            return []
        session = self._load()
        window = max(1, int(rate * CHUNK_SECONDS))
        offsets = list(range(0, len(samples), window))
        inputs = {item.name: item for item in session.get_inputs()}
        segments: list[Segment] = []
        for index, offset in enumerate(offsets, 1):
            chunk = samples[offset:offset + window]
            if not chunk or rms(chunk) < SILENCE_RMS:
                continue
            audio = numpy.asarray(chunk, dtype=numpy.float32).reshape(1, -1)
            feed = {}
            for name in inputs:
                lowered = name.casefold()
                if "len" in lowered:
                    feed[name] = numpy.array([audio.shape[1]], dtype=numpy.int64)
                else:
                    feed[name] = audio
            logits = session.run(None, feed)[0]
            text = self._decode(numpy.argmax(logits[0], axis=-1))
            if text:
                segments.append(Segment(offset / rate, (offset + len(chunk)) / rate, text))
            if progress:
                progress(index, len(offsets))
        return segments
