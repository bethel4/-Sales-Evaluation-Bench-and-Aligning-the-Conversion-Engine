#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(".")
SIMPO_TRAIN = ROOT / "training_data" / "simpo_train.jsonl"
SIMPO_EVAL = ROOT / "training_data" / "simpo_eval.jsonl"
TRAIN_TASKS = ROOT / "tenacious_bench_v0.1" / "train" / "tasks.json"
DEV_TASKS = ROOT / "tenacious_bench_v0.1" / "dev" / "tasks.json"
HELD_TASKS = ROOT / "tenacious_bench_v0.1" / "held_out" / "tasks.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _validate_task_list(path: Path) -> tuple[int, list[str], list[str]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        return 0, [], [f"{path}: expected JSON array"]
    ids: list[str] = []
    errors: list[str] = []
    for i, row in enumerate(payload):
        if not isinstance(row, dict):
            errors.append(f"{path}: row {i} is not an object")
            continue
        tid = row.get("task_id")
        if not isinstance(tid, str) or not tid.strip():
            errors.append(f"{path}: row {i} missing task_id")
        else:
            ids.append(tid)
        for key in ("chosen", "rejected", "scoring_rubric", "leakage_prevention"):
            if key not in row:
                errors.append(f"{path}: task {tid or i} missing '{key}'")
        lp = row.get("leakage_prevention")
        if isinstance(lp, dict) and lp.get("same_model") is True:
            errors.append(f"{path}: task {tid or i} has leakage_prevention.same_model=true")
    return len(payload), ids, errors


def _validate_simpo(path: Path) -> tuple[int, list[str]]:
    rows = _read_jsonl(path)
    errors: list[str] = []
    for i, row in enumerate(rows):
        for key in ("prompt", "chosen", "rejected"):
            val = row.get(key)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"{path}: row {i} invalid '{key}'")
    return len(rows), errors


def main() -> int:
    missing = [str(p) for p in [SIMPO_TRAIN, SIMPO_EVAL, TRAIN_TASKS, DEV_TASKS, HELD_TASKS] if not p.exists()]
    if missing:
        print(json.dumps({"status": "FAIL", "error": "missing required files", "missing": missing}, indent=2))
        return 1

    train_n, train_ids, train_err = _validate_task_list(TRAIN_TASKS)
    dev_n, dev_ids, dev_err = _validate_task_list(DEV_TASKS)
    held_n, held_ids, held_err = _validate_task_list(HELD_TASKS)
    simpo_train_n, simpo_train_err = _validate_simpo(SIMPO_TRAIN)
    simpo_eval_n, simpo_eval_err = _validate_simpo(SIMPO_EVAL)

    all_ids = train_ids + dev_ids + held_ids
    dup_ids = sorted({tid for tid in all_ids if all_ids.count(tid) > 1})
    errors = train_err + dev_err + held_err + simpo_train_err + simpo_eval_err
    if dup_ids:
        errors.append(f"duplicate task_id values across splits: {len(dup_ids)}")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "counts": {
            "train_tasks": train_n,
            "dev_tasks": dev_n,
            "held_out_tasks": held_n,
            "simpo_train_rows": simpo_train_n,
            "simpo_eval_rows": simpo_eval_n,
        },
        "task_id_uniqueness": {
            "n_total": len(all_ids),
            "n_unique": len(set(all_ids)),
            "n_duplicates": len(dup_ids),
        },
        "errors": errors[:200],
    }
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
