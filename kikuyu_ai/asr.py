from pathlib import Path
from collections import Counter
import os
from .audio import read_wav_array, wav_duration
from .ml_runtime import configure_torch, local_pretrained_kwargs, release_torch_memory
from .models import Segment


class ASR:
    def __init__(
        self,
        model_name: str | None = None,
        backend: str = "transformers",
        language: str | None = None,
        chunk_seconds: float = 15.0,
        retry_chunk_seconds: float = 6.0,
        silence_rms: float = 0.001,
        max_new_tokens: int = 96,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.15,
    ):
        self.model_name = model_name
        self.backend = backend
        self.language = language
        self.chunk_seconds = max(1.0, float(chunk_seconds or 15.0))
        self.retry_chunk_seconds = max(1.0, float(retry_chunk_seconds or 6.0))
        self.silence_rms = max(0.0, float(silence_rms or 0.0))
        self.max_new_tokens = max(8, int(max_new_tokens or 96))
        self.no_repeat_ngram_size = max(0, int(no_repeat_ngram_size or 0))
        self.repetition_penalty = max(1.0, float(repetition_penalty or 1.0))
        self.progress = os.environ.get("KIKUYU_ASR_PROGRESS", "").casefold() in {"1", "true", "yes", "on"}
        self._model = None

    def unload(self) -> None:
        self._model = None
        try:
            import torch
        except ImportError:
            torch = None
        release_torch_memory(torch)

    def transcribe(self, audio: Path) -> tuple[list[Segment], float]:
        if not self.model_name:
            return [Segment(0.0, 0.0, "", None)], 0.0
        if self.backend in {"transformers", "hf", "huggingface"}:
            return self._transcribe_transformers(audio)
        if self.backend in {"faster-whisper", "faster_whisper"}:
            return self._transcribe_faster_whisper(audio)
        if self.backend in {"mms", "wav2vec2", "w2v"}:
            return self._transcribe_mms(audio)
        raise RuntimeError(f"Unsupported ASR backend: {self.backend}")

    def _transcribe_mms(self, audio: Path) -> tuple[list[Segment], float]:
        """Meta MMS (wav2vec2-CTC) with a per-language adapter.

        MMS covers over a thousand languages by keeping one shared encoder and
        swapping a small adapter per language, so the language code is required
        rather than optional. Decoding is a single forward pass with no
        autoregressive loop, which makes it markedly faster than Whisper on CPU
        and immune to the repetition loops Whisper falls into.
        """
        if not self.language:
            raise RuntimeError(
                "The mms backend needs a language code (for example KIKUYU_ASR_LANGUAGE=luo); "
                "MMS selects its adapter by language."
            )
        audio_input = read_wav_array(audio)
        samples = audio_input["array"]
        sampling_rate = int(audio_input["sampling_rate"])
        if len(samples) == 0:
            return [Segment(0.0, 0.0, "", None)], 0.0

        try:
            import torch
            from transformers import AutoProcessor, Wav2Vec2ForCTC
        except ImportError as exc:
            raise RuntimeError("Install the optional ML dependencies to use the MMS backend") from exc

        if self._model is None:
            configure_torch(torch)
            processor = AutoProcessor.from_pretrained(
                self.model_name, target_lang=self.language, local_files_only=True
            )
            model = Wav2Vec2ForCTC.from_pretrained(
                self.model_name,
                target_lang=self.language,
                ignore_mismatched_sizes=True,
                local_files_only=True,
            )
            # The adapter carries the language-specific output layer.
            model.load_adapter(self.language)
            model.to("cpu")
            model.eval()
            self._model = (processor, model)

        processor, model = self._model
        chunk_size = max(1, int(sampling_rate * self.chunk_seconds))
        segments: list[Segment] = []
        offsets = list(range(0, len(samples), chunk_size))
        for index, offset in enumerate(offsets, 1):
            chunk = samples[offset:offset + chunk_size]
            if len(chunk) == 0 or self._rms(chunk) < self.silence_rms:
                continue
            inputs = processor(chunk, sampling_rate=sampling_rate, return_tensors="pt")
            with torch.inference_mode():
                logits = model(**inputs).logits
            predicted = torch.argmax(logits, dim=-1)
            text = processor.batch_decode(predicted)[0].strip()
            del inputs, logits, predicted
            release_torch_memory(torch)
            if text:
                segments.append(
                    Segment(offset / sampling_rate, (offset + len(chunk)) / sampling_rate, text, None)
                )
            if self.progress:
                print(f"asr chunk {index}/{len(offsets)}: {text[:80]!r}", flush=True)

        combined = " ".join(segment.text for segment in segments).strip()
        return segments or [Segment(0.0, 0.0, "", None)], 0.70 if combined else 0.0

    def _transcribe_transformers(self, audio: Path) -> tuple[list[Segment], float]:
        audio_input = read_wav_array(audio)
        samples = audio_input["array"]
        sampling_rate = int(audio_input["sampling_rate"])
        if len(samples) == 0:
            return [Segment(0.0, 0.0, "", None)], 0.0

        chunk_size = max(1, int(sampling_rate * self.chunk_seconds))
        offsets = list(range(0, len(samples), chunk_size))
        active_offsets = [
            offset
            for offset in offsets
            if self.silence_rms <= 0.0 or self._rms(samples[offset:offset + chunk_size]) >= self.silence_rms
        ]
        if not active_offsets:
            return [Segment(0.0, 0.0, "", None)], 0.0

        try:
            import torch
            from transformers import AutoProcessor, WhisperForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Install the optional ML dependencies to use the configured Transformers ASR model") from exc

        if self._model is None:
            configure_torch(torch)
            pretrained_kwargs = local_pretrained_kwargs()
            processor = AutoProcessor.from_pretrained(self.model_name, local_files_only=True)
            model = WhisperForConditionalGeneration.from_pretrained(self.model_name, **pretrained_kwargs)
            model.to("cpu")
            model.eval()
            self._model = (processor, model)

        processor, model = self._model
        segments: list[Segment] = []
        forced_decoder_ids = None
        if self.language and hasattr(processor, "get_decoder_prompt_ids"):
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=self.language, task="transcribe")

        total_chunks = len(active_offsets)
        for chunk_index, offset in enumerate(active_offsets, 1):
            chunk = samples[offset:offset + chunk_size]
            chunk_segments = self._transcribe_chunk(
                processor,
                model,
                torch,
                chunk,
                sampling_rate,
                offset,
                forced_decoder_ids,
                depth=0,
            )
            segments.extend(chunk_segments)
            if self.progress:
                preview = " ".join(segment.text for segment in chunk_segments).strip()[:80]
                print(
                    f"asr chunk {chunk_index}/{total_chunks}: {len(chunk_segments)} segment(s) {preview!r}",
                    flush=True,
                )

        combined = " ".join(segment.text for segment in segments).strip()
        return segments or [Segment(0.0, 0.0, "", None)], 0.70 if combined else 0.0

    def _transcribe_chunk(
        self,
        processor,
        model,
        torch,
        chunk,
        sampling_rate: int,
        offset: int,
        forced_decoder_ids,
        depth: int,
    ) -> list[Segment]:
        if len(chunk) == 0 or self._rms(chunk) < self.silence_rms:
            return []

        text = self._generate_chunk_text(processor, model, torch, chunk, sampling_rate, forced_decoder_ids)
        chunk_seconds = len(chunk) / sampling_rate
        if (
            self._looks_repetitive(text)
            and depth < 2
            and chunk_seconds > self.retry_chunk_seconds
        ):
            midpoint = max(1, len(chunk) // 2)
            return [
                *self._transcribe_chunk(
                    processor,
                    model,
                    torch,
                    chunk[:midpoint],
                    sampling_rate,
                    offset,
                    forced_decoder_ids,
                    depth + 1,
                ),
                *self._transcribe_chunk(
                    processor,
                    model,
                    torch,
                    chunk[midpoint:],
                    sampling_rate,
                    offset + midpoint,
                    forced_decoder_ids,
                    depth + 1,
                ),
            ]

        text = self._trim_repetitive_tail(text)
        if not text:
            return []
        start = offset / sampling_rate
        end = (offset + len(chunk)) / sampling_rate
        return [Segment(start, end, text, None)]

    def _generate_chunk_text(self, processor, model, torch, chunk, sampling_rate: int, forced_decoder_ids) -> str:
        inputs = processor(chunk, sampling_rate=sampling_rate, return_tensors="pt")
        generate_kwargs = {
            "forced_decoder_ids": forced_decoder_ids,
            "max_new_tokens": self.max_new_tokens,
            "num_beams": 1,
            "do_sample": False,
        }
        if self.no_repeat_ngram_size:
            generate_kwargs["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        if self.repetition_penalty > 1.0:
            generate_kwargs["repetition_penalty"] = self.repetition_penalty
        with torch.inference_mode():
            generated = model.generate(inputs.input_features, **generate_kwargs)
        text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        del inputs, generated
        release_torch_memory(torch)
        return text

    @staticmethod
    def _rms(samples) -> float:
        if len(samples) == 0:
            return 0.0
        return float((samples * samples).mean() ** 0.5)

    @staticmethod
    def _looks_repetitive(text: str) -> bool:
        words = text.split()
        if len(words) < 12:
            return False
        if Counter(words).most_common(1)[0][1] / len(words) >= 0.28:
            return True
        for size in (3, 4):
            if len(words) < size * 3:
                continue
            grams = Counter(tuple(words[index:index + size]) for index in range(len(words) - size + 1))
            if grams and grams.most_common(1)[0][1] >= 3:
                return True
        return False

    @staticmethod
    def _trim_repetitive_tail(text: str) -> str:
        words = text.split()
        if len(words) < 18:
            return text
        for size in range(1, 7):
            if len(words) < size * 3:
                continue
            tail = words[-size:]
            repeats = 1
            position = len(words) - size * 2
            while position >= 0 and words[position:position + size] == tail:
                repeats += 1
                position -= size
            if repeats >= 3:
                return " ".join(words[: len(words) - repeats * size]).strip()
        return text

    @staticmethod
    def _segments_from_transformers(output: dict, text: str, duration: float) -> list[Segment]:
        segments: list[Segment] = []
        for chunk in output.get("chunks") or []:
            chunk_text = str(chunk.get("text", "")).strip()
            timestamp = chunk.get("timestamp") or (0.0, duration)
            start = float(timestamp[0] or 0.0)
            end = float(timestamp[1] or duration)
            if chunk_text:
                segments.append(Segment(start, end, chunk_text, None))
        if not segments and text:
            segments.append(Segment(0.0, duration, text, None))
        return segments or [Segment(0.0, 0.0, "", None)]

    def _transcribe_faster_whisper(self, audio: Path) -> tuple[list[Segment], float]:
        try:
            from faster_whisper import WhisperModel
            self._model = self._model or WhisperModel(self.model_name, device="auto", compute_type="int8")
            options = {"vad_filter": True, "beam_size": 5}
            if self.language:
                options["language"] = self.language
            chunks, info = self._model.transcribe(str(audio), **options)
            segments = [Segment(float(x.start), float(x.end), x.text.strip(), getattr(x, "avg_logprob", None)) for x in chunks if x.text.strip()]
            confidence = min(1.0, max(0.0, sum(1.0 + (s.confidence or -1.0) / 2 for s in segments) / len(segments))) if segments else 0.0
            return segments, confidence
        except ImportError as exc:
            raise RuntimeError("Install the optional ML dependencies or set KIKUYU_ASR_MODEL only after installing faster-whisper") from exc
