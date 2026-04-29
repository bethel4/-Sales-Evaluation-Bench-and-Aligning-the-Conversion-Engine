#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

SEED = 42
random.seed(SEED)

INPUT_PATH = Path("training_data/preference_pairs.jsonl")
TRAIN_PATH = Path("training_data/simpo_train.jsonl")
EVAL_PATH = Path("training_data/simpo_eval.jsonl")


def read_pairs() -> list[dict[str, Any]]:
    if not INPUT_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in INPUT_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("prompt") and row.get("chosen") and row.get("rejected"):
            rows.append(
                {
                    "prompt": row["prompt"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                    "source": row.get("source"),
                    "task_id": row.get("task_id"),
                }
            )
    return rows


def main() -> int:
    pairs = read_pairs()
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    split_idx = max(1, int(len(pairs) * 0.8)) if pairs else 0
    train_rows = pairs[:split_idx]
    eval_rows = pairs[split_idx:]

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_PATH.open("w", encoding="utf-8") as train_handle:
        for row in train_rows:
            train_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with EVAL_PATH.open("w", encoding="utf-8") as eval_handle:
        for row in eval_rows:
            eval_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "seed": SEED,
                "n_pairs": len(pairs),
                "n_train": len(train_rows),
                "n_eval": len(eval_rows),
                "train_path": str(TRAIN_PATH),
                "eval_path": str(EVAL_PATH),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
