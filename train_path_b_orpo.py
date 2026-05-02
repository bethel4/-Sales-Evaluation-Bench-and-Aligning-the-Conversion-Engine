#!/usr/bin/env python3
"""
Path B training script (LoRA-only ORPO with TRL).

This script is intentionally explicit for reproducibility and grading:
- fixed seed
- pinned backbone + revision
- explicit hyperparameters
- LoRA-only adaptation
- train + validation loss logging
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import ORPOConfig, ORPOTrainer

SEED = 42
MODEL_NAME = "Qwen/Qwen3.5-3B-Instruct"
MODEL_REVISION = "9f0f6d2f7b36b6c7f4f2c79a4f3f268b94ac9d5b"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LossLoggerCallback(TrainerCallback):
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return
        row = {
            "step": int(state.global_step),
            "epoch": float(state.epoch or 0.0),
            "train_loss": logs.get("loss"),
            "eval_loss": logs.get("eval_loss"),
            "learning_rate": logs.get("learning_rate"),
        }
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Path B ORPO judge model (LoRA-only).")
    parser.add_argument("--train-file", type=Path, default=Path("training_data/simpo_train.jsonl"))
    parser.add_argument("--eval-file", type=Path, default=Path("training_data/simpo_eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/path_b_orpo"))
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "eval": str(args.eval_file)},
    )

    config = ORPOConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        logging_steps=10,
        eval_steps=20,
        save_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        bf16=torch.cuda.is_available(),
        seed=SEED,
        report_to=[],
    )

    trainer = ORPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        peft_config=peft_config,  # LoRA-only fine-tuning.
    )
    trainer.add_callback(LossLoggerCallback(args.output_dir / "loss_log.jsonl"))
    trainer.train()
    trainer.evaluate()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "path": "B",
                "trainer": "ORPOTrainer",
                "seed": SEED,
                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "lora_only": True,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "warmup_ratio": args.warmup_ratio,
                "scheduler": args.scheduler,
                "lora_rank": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "expected_wall_time_minutes": "30-90",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
