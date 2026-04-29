"""Convert preference_pairs.jsonl into TRL SimPOTrainer JSONL files."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

SEED = 42
random.seed(SEED)

INPUT_PATH = Path("training_data/preference_pairs.jsonl")
TRAIN_PATH = Path("training_data/simpo_train.jsonl")
EVAL_PATH = Path("training_data/simpo_eval.jsonl")


def main() -> None:
    pairs: List[Dict[str, Any]] = []
    with INPUT_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("prompt") and row.get("chosen") and row.get("rejected"):
                pairs.append(
                    {
                        "prompt": row["prompt"],
                        "chosen": row["chosen"],
                        "rejected": row["rejected"],
                    }
                )

    random.shuffle(pairs)
    split = int(len(pairs) * 0.8)
    train, eval_ = pairs[:split], pairs[split:]

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_PATH.open("w") as f:
        for row in train:
            f.write(json.dumps(row, default=str) + "\n")
    with EVAL_PATH.open("w") as f:
        for row in eval_:
            f.write(json.dumps(row, default=str) + "\n")

    print(f"SimPO train pairs: {len(train)} -> {TRAIN_PATH}")
    print(f"SimPO eval pairs: {len(eval_)} -> {EVAL_PATH}")


if __name__ == "__main__":
    main()
