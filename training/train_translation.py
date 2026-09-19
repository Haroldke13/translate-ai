#!/usr/bin/env python3
"""Fine-tune a local Seq2Seq Kikuyu-English translation model from JSONL pairs.

Input rows must contain:
{"kikuyu": "source text", "english": "target text"}

Rows may optionally contain {"split": "train"} or {"split": "eval"}.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kikuyu_ai.config import Settings


@dataclass(frozen=True)
class TranslationExample:
    kikuyu: str
    english: str
    split: str | None = None


def read_parallel(path: Path) -> list[TranslationExample]:
    examples: list[TranslationExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        kikuyu = str(row.get("kikuyu", "")).strip()
        english = str(row.get("english", "")).strip()
        if not kikuyu or not english:
            raise SystemExit(f"{path}:{line_number}: every row needs non-empty kikuyu and english")
        split = row.get("split")
        examples.append(TranslationExample(kikuyu, english, str(split).casefold() if split else None))
    if not examples:
        raise SystemExit(f"{path}: no parallel rows found")
    return examples


def split_examples(
    examples: list[TranslationExample],
    eval_parallel: Path | None,
    validation_split: float,
    seed: int,
) -> tuple[list[TranslationExample], list[TranslationExample]]:
    if eval_parallel:
        return examples, read_parallel(eval_parallel)

    train = [item for item in examples if item.split in {None, "", "train", "training"}]
    eval_rows = [item for item in examples if item.split in {"eval", "dev", "valid", "validation", "test"}]
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


def write_validated_parallel(path: Path, examples: list[TranslationExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item.__dict__, ensure_ascii=False) + "\n" for item in examples),
        encoding="utf-8",
    )


def dependency_error(exc: Exception) -> SystemExit:
    return SystemExit(
        "Translation training needs the optional training stack. Install it with:\n"
        "  python -m pip install -e '.[training]'\n\n"
        f"Original import error: {type(exc).__name__}: {exc}"
    )


def require_training_dependencies() -> dict[str, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise dependency_error(exc) from exc
    if importlib.util.find_spec("accelerate") is None:
        raise dependency_error(ImportError("accelerate is not installed"))
    return {
        "torch": torch,
        "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
        "Seq2SeqTrainer": Seq2SeqTrainer,
        "Seq2SeqTrainingArguments": Seq2SeqTrainingArguments,
    }


class ParallelDataset:
    def __init__(
        self,
        examples: list[TranslationExample],
        tokenizer: Any,
        src_lang: str | None,
        tgt_lang: str | None,
        max_source_length: int,
        max_target_length: int,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.examples[index]
        if self.src_lang and hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = self.src_lang
        model_inputs = self.tokenizer(
            item.kikuyu,
            truncation=True,
            max_length=self.max_source_length,
        )
        if self.tgt_lang and hasattr(self.tokenizer, "tgt_lang"):
            self.tokenizer.tgt_lang = self.tgt_lang
        labels = self.tokenizer(
            text_target=item.english,
            truncation=True,
            max_length=self.max_target_length,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def training_args_kwargs(training_args_cls: Any, values: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(training_args_cls.__init__).parameters
    result = dict(values)
    if "eval_strategy" in parameters:
        result["eval_strategy"] = result.pop("evaluation_strategy")
    elif "evaluation_strategy" not in parameters:
        result.pop("evaluation_strategy")
    return {key: value for key, value in result.items() if key in parameters and value is not None}


def lang_token_id(tokenizer: Any, lang: str | None) -> int | None:
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


def configure_trainable_parameters(model: Any, trainable_regex: str | None, freeze_encoder: bool) -> None:
    if trainable_regex:
        pattern = re.compile(trainable_regex)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = bool(pattern.search(name))
    elif freeze_encoder:
        encoder = model.get_encoder() if hasattr(model, "get_encoder") else None
        if encoder is None:
            raise SystemExit("model does not expose get_encoder(); use --trainable-regex instead")
        for parameter in encoder.parameters():
            parameter.requires_grad = False

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"trainable parameters: {trainable:,}/{total:,}")


def build_parser() -> argparse.ArgumentParser:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Fine-tune a Seq2Seq Kikuyu-English translation model")
    parser.add_argument("parallel", type=Path, help="JSONL rows with kikuyu and english fields")
    parser.add_argument("--eval-parallel", type=Path, help="Optional eval JSONL file")
    parser.add_argument("--output", type=Path, default=Path("models/translation"))
    parser.add_argument("--model-name", default=settings.translation_model)
    parser.add_argument("--src-lang", default=settings.translation_src_lang)
    parser.add_argument("--tgt-lang", default=settings.translation_tgt_lang)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
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
    parser.add_argument("--dry-run", action="store_true", help="Validate data and build one batch only")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model_name:
        raise SystemExit("set --model-name or KIKUYU_TRANSLATION_MODEL")

    train_examples, eval_examples = split_examples(
        read_parallel(args.parallel),
        args.eval_parallel,
        args.validation_split,
        args.seed,
    )
    if not train_examples:
        raise SystemExit("no training rows after split")

    args.output.mkdir(parents=True, exist_ok=True)
    write_validated_parallel(args.output / "parallel.train.validated.jsonl", train_examples)
    if eval_examples:
        write_validated_parallel(args.output / "parallel.eval.validated.jsonl", eval_examples)

    deps = require_training_dependencies()
    torch = deps["torch"]
    tokenizer = deps["AutoTokenizer"].from_pretrained(args.model_name, local_files_only=not args.online)
    model = deps["AutoModelForSeq2SeqLM"].from_pretrained(args.model_name, local_files_only=not args.online)
    forced_bos_token_id = lang_token_id(tokenizer, args.tgt_lang)
    if forced_bos_token_id is not None and hasattr(model, "config"):
        model.config.forced_bos_token_id = forced_bos_token_id
    if args.gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False
    configure_trainable_parameters(model, args.trainable_regex, args.freeze_encoder)

    train_dataset = ParallelDataset(
        train_examples,
        tokenizer,
        args.src_lang,
        args.tgt_lang,
        args.max_source_length,
        args.max_target_length,
    )
    eval_dataset = ParallelDataset(
        eval_examples,
        tokenizer,
        args.src_lang,
        args.tgt_lang,
        args.max_source_length,
        args.max_target_length,
    ) if eval_examples else None
    collator = deps["DataCollatorForSeq2Seq"](tokenizer=tokenizer, model=model)

    if args.dry_run:
        batch = collator([train_dataset[0]])
        print(f"validated train rows: {len(train_examples)}")
        print(f"validated eval rows: {len(eval_examples)}")
        print(f"dry-run batch input_ids: {tuple(batch['input_ids'].shape)}")
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
        "generation_max_length": args.max_target_length,
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

    trainer = deps["Seq2SeqTrainer"](
        model=model,
        args=deps["Seq2SeqTrainingArguments"](
            **training_args_kwargs(deps["Seq2SeqTrainingArguments"], training_values)
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"saved translation model and tokenizer to {args.output}")


if __name__ == "__main__":
    main()
