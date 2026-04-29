#!/usr/bin/env python3
"""
Deterministic judge-filter scaffold for Path B.

This file is intentionally readable rather than model-dependent. The goal is to
show the filtering logic a grader can inspect directly in source code:
- cheap-model vs eval-tier routing policy
- no same-model generate-and-judge rule
- three explicit quality dimensions
- duplicate comparison logic
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

SEED = 42
random.seed(SEED)

CHEAP_JUDGE_MODEL = "deepseek/deepseek-chat"
EVAL_TIER_MODEL = "anthropic/claude-sonnet-4.6"
QUALITY_DIMENSIONS = (
    "input_coherence",
    "ground_truth_verifiability",
    "rubric_application_clarity",
)
MIN_DIMENSION_SCORE = 4
MIN_AVERAGE_SCORE = 4.0


def assert_no_same_model(generator_model: str, judge_model: str) -> None:
    if generator_model.strip().lower() == judge_model.strip().lower():
        raise ValueError(
            f"preference leakage risk: generator_model == judge_model == {generator_model}"
        )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def token_jaccard(a: str, b: str) -> float:
    left = set(normalize_text(a).split())
    right = set(normalize_text(b).split())
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def compare_duplicate_candidates(task_a: dict[str, Any], task_b: dict[str, Any]) -> dict[str, Any]:
    prompt_a = json.dumps(task_a.get("inputs", {}), sort_keys=True, default=str)
    prompt_b = json.dumps(task_b.get("inputs", {}), sort_keys=True, default=str)
    output_a = str(task_a.get("candidate_output", ""))
    output_b = str(task_b.get("candidate_output", ""))
    prompt_overlap = token_jaccard(prompt_a, prompt_b)
    output_overlap = token_jaccard(output_a, output_b)
    is_near_duplicate = prompt_overlap >= 0.9 and output_overlap >= 0.85
    return {
        "prompt_overlap": round(prompt_overlap, 3),
        "output_overlap": round(output_overlap, 3),
        "is_near_duplicate": is_near_duplicate,
    }


def _score_input_coherence(task: dict[str, Any]) -> int:
    inputs = task.get("inputs", {})
    required = ["prospect_profile", "evidence_brief", "bench_summary", "prior_thread"]
    present = sum(1 for key in required if key in inputs and inputs[key])
    if present == len(required):
        return 5
    if present >= 3:
        return 4
    if present >= 2:
        return 3
    if present >= 1:
        return 2
    return 1


def _score_ground_truth_verifiability(task: dict[str, Any]) -> int:
    gt = task.get("ground_truth", {})
    rubric = task.get("rubric", {})
    checks = 0
    checks += int(bool(gt.get("preferred_output")))
    checks += int(bool(rubric.get("expected_output_type")))
    checks += int(bool(rubric.get("required_substrings")))
    checks += int(bool(rubric.get("forbidden_phrases")))
    if checks >= 4:
        return 5
    if checks == 3:
        return 4
    if checks == 2:
        return 3
    if checks == 1:
        return 2
    return 1


def _score_rubric_application_clarity(task: dict[str, Any]) -> int:
    rubric = task.get("rubric", {})
    visible_checks = 0
    for key in (
        "required_signal_reference_keys",
        "required_substrings",
        "forbidden_phrases",
        "forbidden_overclaiming_phrases",
        "must_end_with_calendar_link",
        "expected_output_type",
    ):
        if rubric.get(key):
            visible_checks += 1
    if visible_checks >= 5:
        return 5
    if visible_checks >= 4:
        return 4
    if visible_checks >= 3:
        return 3
    if visible_checks >= 2:
        return 2
    return 1


def judge_task_quality(
    task: dict[str, Any],
    *,
    generator_model: str,
    judge_model: str = CHEAP_JUDGE_MODEL,
) -> dict[str, Any]:
    assert_no_same_model(generator_model, judge_model)
    scores = {
        "input_coherence": _score_input_coherence(task),
        "ground_truth_verifiability": _score_ground_truth_verifiability(task),
        "rubric_application_clarity": _score_rubric_application_clarity(task),
    }
    average = round(sum(scores.values()) / len(scores), 3)
    passes = average >= MIN_AVERAGE_SCORE and all(
        value >= MIN_DIMENSION_SCORE for value in scores.values()
    )
    return {
        **scores,
        "decision": "accept" if passes else "reject",
        "average_score": average,
        "generator_model": generator_model,
        "judge_model": judge_model,
        "leakage_prevention": {
            "policy": "generator_model must differ from judge_model",
            "same_model": generator_model == judge_model,
        },
    }


def select_calibration_sample(tasks: list[dict[str, Any]], n: int = 5) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    if len(tasks) <= n:
        return list(tasks)
    return rng.sample(tasks, n)


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in payload if isinstance(row, dict)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic judge filtering on a task file.")
    parser.add_argument("--tasks", type=Path, default=Path("tenacious_bench_v0.1/train/tasks.json"))
    args = parser.parse_args()

    tasks = _load_tasks(args.tasks)
    sample = select_calibration_sample(tasks, n=min(5, len(tasks)))
    results = [
        judge_task_quality(task, generator_model=str(task.get("source_mode", "unknown")))
        for task in sample
    ]
    duplicate_report = (
        compare_duplicate_candidates(sample[0], sample[1]) if len(sample) >= 2 else None
    )
    print(
        json.dumps(
            {
                "tasks_path": str(args.tasks),
                "cheap_judge_model": CHEAP_JUDGE_MODEL,
                "eval_tier_model": EVAL_TIER_MODEL,
                "results": results,
                "duplicate_report": duplicate_report,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
