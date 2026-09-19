#!/usr/bin/env python3
"""Fine-tune a Whisper-compatible ASR model from a JSONL manifest.

Manifest rows must contain:
{"audio": "relative/or/absolute/path.wav", "text": "transcript"}

Rows may optionally contain {"split": "train"} or {"split": "eval"}.
Without explicit splits, pass --eval-manifest or use --validation-split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import random
import re
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kikuyu_ai.audio import normalize_audio
from kikuyu_ai.config import Settings


@dataclass(frozen=True)
class AsrExample:
    audio: Path
    text: str
    split: str | None = None


def read_manifest(path: Path) -> list[AsrExample]:
    base = path.resolve().parent
    examples: list[AsrExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc

        audio_value = str(row.get("audio", "")).strip()
        text = str(row.get("text", "")).strip()
        if not audio_value or not text:
            raise SystemExit(f"{path}:{line_number}: every row needs non-empty audio and text")

        audio = Path(audio_value).expanduser()
        if not audio.is_absolute():
            audio = base / audio
        if not audio.is_file():
            raise SystemExit(f"{path}:{line_number}: audio file does not exist: {audio}")

        split = row.get("split")
        examples.append(AsrExample(audio.resolve(), text, str(split).casefold() if split else None))
    if not examples:
        raise SystemExit(f"{path}: no training rows found")
    return examples


def split_examples(
    examples: list[AsrExample],
    eval_manifest: Path | None,
    validation_split: float,
    seed: int,
) -> tuple[list[AsrExample], list[AsrExample]]:
    if eval_manifest:
        return examples, read_manifest(eval_manifest)

    train = [x for x in examples if x.split in {None, "", "train", "training"}]
    eval_rows = [x for x in examples if x.split in {"eval", "dev", "valid", "validation", "test"}]
    if eval_rows:
        return train, eval_rows

    if validation_split <= 0:
        return examples, []
    if validation_split >= 1:
        raise SystemExit("--validation-split must be less than 1")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, round(len(shuffled) * validation_split))
    eval_count = min(eval_count, len(shuffled) - 1) if len(shuffled) > 1 else 0
    return shuffled[eval_count:], shuffled[:eval_count]


def write_validated_manifest(path: Path, examples: list[AsrExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"audio": str(item.audio), "text": item.text, "split": item.split} for item in examples]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def dependency_error(exc: Exception) -> SystemExit:
    return SystemExit(
        "ASR training needs the optional ML training stack. Install it with:\n"
        "  python -m pip install -e '.[training]'\n\n"
        f"Original import error: {type(exc).__name__}: {exc}"
    )


def require_training_dependencies() -> dict[str, Any]:
    try:
        import torch
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
    except ImportError as exc:
        raise dependency_error(exc) from exc

    if importlib.util.find_spec("accelerate") is None:
        raise dependency_error(ImportError("accelerate is not installed"))

    return {
        "torch": torch,
        "Seq2SeqTrainer": Seq2SeqTrainer,
        "Seq2SeqTrainingArguments": Seq2SeqTrainingArguments,
        "WhisperForConditionalGeneration": WhisperForConditionalGeneration,
        "WhisperProcessor": WhisperProcessor,
    }


def cached_wav(source: Path, cache_dir: Path) -> Path:
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
    target = cache_dir / f"{source.stem}-{digest}.wav"
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    normalize_audio(source, target)
    return target


def read_pcm16_wav(path: Path) -> list[float]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if channels != 1 or width != 2 or rate != 16000:
        raise RuntimeError(f"expected normalized mono PCM16 16 kHz WAV, got {path}")

    import struct

    samples = struct.unpack("<" + "h" * (len(frames) // 2), frames)
    return [sample / 32768.0 for sample in samples]


class WhisperManifestDataset:
    def __init__(
        self,
        examples: list[AsrExample],
        processor: Any,
        cache_dir: Path,
        max_input_seconds: float,
    ):
        self.examples = examples
        self.processor = processor
        self.cache_dir = cache_dir
        self.max_input_samples = int(max_input_seconds * 16000) if max_input_seconds > 0 else None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.examples[index]
        wav_path = cached_wav(item.audio, self.cache_dir)
        audio = read_pcm16_wav(wav_path)
        if self.max_input_samples:
            audio = audio[: self.max_input_samples]

        inputs = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        labels = self.processor.tokenizer(item.text).input_ids
        return {
            "input_features": inputs.input_features[0],
            "labels": labels,
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int | None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        input_features = [{"input_features": feature["input_features"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (
            self.decoder_start_token_id is not None
            and labels.shape[1] > 0
            and (labels[:, 0] == self.decoder_start_token_id).all().item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def training_args_kwargs(training_args_cls: Any, values: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(training_args_cls.__init__).parameters
    result = dict(values)
    if "eval_strategy" in parameters:
        result["eval_strategy"] = result.pop("evaluation_strategy")
    elif "evaluation_strategy" not in parameters:
        result.pop("evaluation_strategy")
    return {key: value for key, value in result.items() if key in parameters}


def configure_generation(processor: Any, model: Any, language: str | None, task: str) -> None:
    if not language or not hasattr(processor, "get_decoder_prompt_ids"):
        return
    try:
        forced_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    except Exception as exc:
        print(f"warning: could not set Whisper language prompt {language!r}: {exc}")
        return
    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = forced_ids
    if hasattr(model, "config"):
        model.config.forced_decoder_ids = forced_ids


def configure_trainable_parameters(model: Any, trainable_regex: str | None, freeze_encoder: bool) -> None:
    if trainable_regex:
        pattern = re.compile(trainable_regex)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = bool(pattern.search(name))
    elif freeze_encoder:
        if hasattr(model, "freeze_encoder"):
            model.freeze_encoder()
        else:
            encoder = getattr(getattr(model, "model", None), "encoder", None)
            if encoder is None:
                raise SystemExit("model does not expose a Whisper encoder; use --trainable-regex instead")
            for parameter in encoder.parameters():
                parameter.requires_grad = False

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"trainable parameters: {trainable:,}/{total:,}")


def build_parser() -> argparse.ArgumentParser:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Fine-tune a Whisper ASR model on Kikuyu audio")
    parser.add_argument("manifest", type=Path, help="JSONL rows with audio and text fields")
    parser.add_argument("--eval-manifest", type=Path, help="Optional JSONL eval manifest")
    parser.add_argument("--output", type=Path, default=Path("models/asr"))
    parser.add_argument("--model-name", default=settings.asr_model or "openai/whisper-small")
    parser.add_argument("--language", default=settings.asr_language, help="Optional Whisper language prompt")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-input-seconds", type=float, default=30.0)
    parser.add_argument("--optim", default="adafactor", help="Trainer optimizer; adafactor uses less memory than adamw_torch")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trainable-regex", help="Only parameters whose names match this regex remain trainable")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-from-checkpoint", type=str)
    parser.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hub-model-id")
    parser.add_argument("--online", action="store_true", help="Allow downloading model files that are not already cached")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifests and build one batch only")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    train_examples, eval_examples = split_examples(
        read_manifest(args.manifest),
        args.eval_manifest,
        args.validation_split,
        args.seed,
    )
    if not train_examples:
        raise SystemExit("no training rows after split")

    args.output.mkdir(parents=True, exist_ok=True)
    write_validated_manifest(args.output / "manifest.train.validated.json", train_examples)
    if eval_examples:
        write_validated_manifest(args.output / "manifest.eval.validated.json", eval_examples)

    deps = require_training_dependencies()
    torch = deps["torch"]
    processor_cls = deps["WhisperProcessor"]
    model_cls = deps["WhisperForConditionalGeneration"]
    training_args_cls = deps["Seq2SeqTrainingArguments"]
    trainer_cls = deps["Seq2SeqTrainer"]

    processor_kwargs = {}
    if args.language:
        processor_kwargs["language"] = args.language
        processor_kwargs["task"] = args.task
    pretrained_kwargs = {"local_files_only": not args.online}
    processor = processor_cls.from_pretrained(args.model_name, **processor_kwargs, **pretrained_kwargs)
    model = model_cls.from_pretrained(args.model_name, **pretrained_kwargs)
    configure_generation(processor, model, args.language, args.task)

    if args.gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False
    configure_trainable_parameters(model, args.trainable_regex, args.freeze_encoder)

    cache_dir = args.output / "cache" / "wav"
    train_dataset = WhisperManifestDataset(train_examples, processor, cache_dir, args.max_input_seconds)
    eval_dataset = WhisperManifestDataset(eval_examples, processor, cache_dir, args.max_input_seconds) if eval_examples else None

    decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor, decoder_start_token_id)

    if args.dry_run:
        batch = data_collator([train_dataset[0]])
        print(f"validated train rows: {len(train_examples)}")
        print(f"validated eval rows: {len(eval_examples)}")
        print(f"dry-run batch input_features: {tuple(batch['input_features'].shape)}")
        print(f"dry-run batch labels: {tuple(batch['labels'].shape)}")
        return

    training_values = {
        "output_dir": str(args.output),
        "do_train": True,
        "do_eval": bool(eval_dataset),
        "evaluation_strategy": "steps" if eval_dataset else "no",
        "eval_steps": args.eval_steps if eval_dataset else None,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "use_cpu": args.use_cpu or not torch.cuda.is_available(),
        "predict_with_generate": bool(eval_dataset),
        "generation_max_length": 225,
        "optim": args.optim,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": "none",
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "seed": args.seed,
    }

    trainer = trainer_cls(
        model=model,
        args=training_args_cls(**training_args_kwargs(training_args_cls, training_values)),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output))
    processor.save_pretrained(str(args.output))
    print(f"saved ASR model and processor to {args.output}")


if __name__ == "__main__":
    main()
