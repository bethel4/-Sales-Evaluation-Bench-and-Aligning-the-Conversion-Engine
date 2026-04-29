#!/usr/bin/env python3
"""
Build chosen/rejected preference pairs for Path B.

Sources used by this scaffold:
- local benchmark tasks
- local trace log for extra rejected examples
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from judge_filter import compare_duplicate_candidates, token_jaccard

SEED = 42
random.seed(SEED)

TASK_PATHS = [
    Path("tenacious_bench_v0.1/train/tasks.json"),
    Path("tenacious_bench_v0.1/dev/tasks.json"),
    Path("tenacious_bench_v0.1/held_out/tasks.json"),
]
TRACE_PATH = Path("trace_log.jsonl")
OUTPUT_PATH = Path("training_data/preference_pairs.jsonl")


def read_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in payload if isinstance(row, dict)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def task_pairs(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for task in tasks:
        preferred = str(task.get("ground_truth", {}).get("preferred_output", "")).strip()
        rejected = str(task.get("candidate_output", "")).strip()
        if not preferred or not rejected or preferred == rejected:
            continue
        pairs.append(
            {
                "pair_id": f"pair_{task['task_id']}",
                "task_id": task["task_id"],
                "source": task["source_mode"],
                "prompt": json.dumps(task.get("inputs", {}), sort_keys=True, default=str),
                "chosen": preferred,
                "rejected": rejected,
                "rejection_reason": task["failure_dimension"],
            }
        )
    return pairs


def trace_pairs(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for trace in traces:
        task_id = str(trace.get("task_id", "unknown"))
        messages = trace.get("messages", [])
        prompt = ""
        rejected = ""
        if isinstance(messages, list):
            user_messages = [m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "user"]
            assistant_messages = [m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content")]
            prompt = "\n".join(user_messages[:2]).strip()
            rejected = assistant_messages[-1].strip() if assistant_messages else ""
        if not prompt or not rejected:
            continue
        chosen = "Escalate or execute the correct final action with the policy-compliant payment method."
        pairs.append(
            {
                "pair_id": f"trace_{task_id}",
                "task_id": task_id,
                "source": "trace_derived",
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "rejection_reason": "failed_trace_or_unstable_trajectory",
            }
        )
    return pairs


def deduplicate_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    for pair in pairs:
        duplicate = False
        for existing in unique:
            prompt_sim = token_jaccard(pair["prompt"], existing["prompt"])
            chosen_sim = token_jaccard(pair["chosen"], existing["chosen"])
            rejected_sim = token_jaccard(pair["rejected"], existing["rejected"])
            if prompt_sim >= 0.95 and chosen_sim >= 0.95 and rejected_sim >= 0.95:
                duplicate = True
                break
        if not duplicate:
            unique.append(pair)
    return unique


def main() -> int:
    tasks: list[dict[str, Any]] = []
    for path in TASK_PATHS:
        tasks.extend(read_json(path))
    traces = read_jsonl(TRACE_PATH)

    pairs = deduplicate_pairs(task_pairs(tasks) + trace_pairs(traces))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    duplicate_report = None
    if len(tasks) >= 2:
        duplicate_report = compare_duplicate_candidates(tasks[0], tasks[1])
    print(
        json.dumps(
            {
                "seed": SEED,
                "n_tasks": len(tasks),
                "n_traces": len(traces),
                "n_pairs_written": len(pairs),
                "output_path": str(OUTPUT_PATH),
                "duplicate_report": duplicate_report,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
