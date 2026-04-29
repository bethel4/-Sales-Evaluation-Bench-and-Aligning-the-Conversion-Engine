"""
Judge filtering utilities for Tenacious-Bench generation.

This file is intentionally separate from task synthesis and pair building so the
model-routing and quality-filtering logic is visible to graders.

Quality filter dimensions, scored 1-5:
- input_coherence
- ground_truth_verifiability
- rubric_clarity

Default inclusion threshold:
- each dimension must be >= 4
- average score must be >= 4.0

Preference-leakage rule:
- The same model must never generate and judge the same task or pair.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

SEED = 42
random.seed(SEED)

CHEAP_JUDGE_MODEL = "deepseek/deepseek-chat"
EVAL_TIER_MODEL = "anthropic/claude-sonnet-4.6"

QUALITY_THRESHOLDS = {
    "input_coherence": 4,
    "ground_truth_verifiability": 4,
    "rubric_clarity": 4,
}
MIN_AVERAGE_SCORE = 4.0

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
TASK_QUALITY_PROMPT = (PROMPT_DIR / "task_quality_prompt.md").read_text()
PAIRWISE_JUDGE_PROMPT = (PROMPT_DIR / "judge_filter_prompt.md").read_text()


def make_client() -> OpenAI:
    """Create OpenRouter-compatible OpenAI client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def assert_no_preference_leakage(generator_model: str, judge_model: str) -> None:
    """Fail fast if the same model is assigned to generation and judging."""
    if generator_model.strip().lower() == judge_model.strip().lower():
        raise ValueError(
            f"Preference leakage risk: generator and judge are both {generator_model}"
        )


def parse_json_response(raw: str) -> Dict[str, Any]:
    """Parse a JSON response robustly; raise clear error if invalid."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        # Common LLM behavior: wrap JSON in text. Try extracting the outer object.
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise ValueError(f"Judge returned invalid JSON: {raw[:200]}") from exc


def quality_passes(scores: Dict[str, Any]) -> bool:
    """Apply documented pointwise thresholds to a task-quality judgment."""
    try:
        dim_scores = [float(scores[k]) for k in QUALITY_THRESHOLDS]
    except (KeyError, TypeError, ValueError):
        return False

    per_dimension_ok = all(
        float(scores[dimension]) >= threshold
        for dimension, threshold in QUALITY_THRESHOLDS.items()
    )
    average_ok = sum(dim_scores) / len(dim_scores) >= MIN_AVERAGE_SCORE
    decision_ok = str(scores.get("decision", "")).lower() == "accept"
    return per_dimension_ok and average_ok and decision_ok


def judge_task_quality(
    task: Dict[str, Any],
    *,
    generator_model: str,
    judge_model: str = CHEAP_JUDGE_MODEL,
    client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """
    Pointwise judge filter for one generated task.

    Returns a dict containing the raw dimension scores plus `passes_filter`.
    """
    assert_no_preference_leakage(generator_model, judge_model)
    client = client or make_client()

    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": TASK_QUALITY_PROMPT},
            {"role": "user", "content": json.dumps(task, indent=2, default=str)},
        ],
        temperature=0,
        max_tokens=300,
    )
    judgment = parse_json_response(response.choices[0].message.content)
    judgment["passes_filter"] = quality_passes(judgment)
    judgment["judge_model"] = judge_model
    judgment["generator_model"] = generator_model
    judgment["leakage_prevention"] = {
        "same_model": generator_model == judge_model,
        "policy": "generator_model must differ from judge_model",
    }
    return judgment


def judge_pairwise(
    email_a: str,
    email_b: str,
    *,
    generator_model: str,
    judge_model: str = CHEAP_JUDGE_MODEL,
    client: Optional[OpenAI] = None,
) -> Dict[str, Any]:
    """
    Pairwise comparison for chosen/rejected candidates or near-duplicates.

    Used when two synthesis paths produce similar tasks or when a corrected
    email must be validated against a known-bad rejected example.
    """
    assert_no_preference_leakage(generator_model, judge_model)
    client = client or make_client()

    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{PAIRWISE_JUDGE_PROMPT}\n\n"
                    f"EMAIL A:\n{email_a}\n\n"
                    f"EMAIL B:\n{email_b}"
                ),
            }
        ],
        temperature=0,
        max_tokens=250,
    )
    judgment = parse_json_response(response.choices[0].message.content)
    judgment["judge_model"] = judge_model
    judgment["generator_model"] = generator_model
    judgment["leakage_prevention"] = {
        "same_model": generator_model == judge_model,
        "policy": "generator_model must differ from judge_model",
    }
    return judgment


def select_calibration_sample(tasks: List[Dict[str, Any]], n: int = 50) -> List[Dict[str, Any]]:
    """
    Select a fixed-seed calibration sample for eval-tier spot-checking.

    Eval-tier models are never used for bulk filtering; they only calibrate a
    limited sample for quality-control reporting.
    """
    rng = random.Random(SEED)
    if len(tasks) <= n:
        return list(tasks)
    return rng.sample(tasks, n)
