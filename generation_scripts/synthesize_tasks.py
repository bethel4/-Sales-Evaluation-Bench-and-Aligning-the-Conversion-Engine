#!/usr/bin/env python3
"""
Deterministic task synthesis scaffold.

This script creates placeholder-ready candidate tasks for two authoring modes:
- programmatic variants
- hand-authored adversarial / multi-LLM placeholder-ready variants

The committed repo keeps this deterministic so evaluators can run it locally.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from judge_filter import CHEAP_JUDGE_MODEL, EVAL_TIER_MODEL, judge_task_quality

SEED = 42
random.seed(SEED)

OUTPUT_DIR = Path("generation_scripts/generated")
OUTPUT_PATH = OUTPUT_DIR / "synthesized_candidates.jsonl"


def _candidate_task(task_id: str, source_mode: str, failure_dimension: str, difficulty: str, company: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "source_mode": source_mode,
        "failure_dimension": failure_dimension,
        "difficulty": difficulty,
        "inputs": {
            "prospect_profile": {"company": company, "role": "VP Engineering", "timezone": "Africa/Nairobi"},
            "evidence_brief": {
                "jobs": {"engineering_roles": "3 engineering roles open", "signal_strength": "medium"},
                "funding": {"summary": "no recent funding event"},
            },
            "bench_summary": {"python": 8, "data": 5, "ml": 3, "go": 0},
            "prior_thread": ["Prospect asked for direct evidence and no hype."],
            "scheduling_context": {"calendar_link": "https://cal.com/tenacious/discovery"},
            "action_context": {"expected_output_type": "text"},
        },
        "candidate_output": f"{company} is aggressively scaling engineering. https://cal.com/tenacious/discovery",
        "ground_truth": {
            "preferred_output": f"{company}: are the 3 open engineering roles tied to a specific delivery bottleneck? https://cal.com/tenacious/discovery",
            "expected_output_type": "text",
            "notes": "Placeholder-ready synthesis candidate.",
        },
        "rubric": {
            "required_signal_reference_keys": ["evidence_brief.jobs.engineering_roles"],
            "required_substrings": ["engineering", "https://cal.com/tenacious/discovery"],
            "forbidden_phrases": ["aggressively"],
            "forbidden_overclaiming_phrases": ["tripling", "clearly behind"],
            "must_end_with_calendar_link": True,
            "requires_cta": True,
            "expected_output_type": "text",
            "expected_action_name": "",
            "min_tone_score": 0.7,
            "weights": {
                "signal_reference": 0.2,
                "required_elements": 0.2,
                "forbidden_phrases": 0.15,
                "cta_calendar": 0.15,
                "output_type_or_action": 0.2,
                "no_overclaiming": 0.1,
                "tone": 0.0,
            },
        },
    }


def synthesize_candidates() -> list[dict[str, Any]]:
    seeds = [
        ("SYN-PG-001", "programmatic", "signal_grounding", "medium", "Consolety"),
        ("SYN-PG-002", "programmatic", "bench_truthfulness", "hard", "AcmeCloud"),
        ("SYN-ML-001", "multi_llm_synthesis", "tone_voice", "hard", "Northstar AI"),
        ("SYN-HA-001", "hand_authored", "thread_integrity", "hard", "Aster Data"),
    ]
    return [_candidate_task(*seed) for seed in seeds]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = synthesize_candidates()
    results = []
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for task in tasks:
            judgment = judge_task_quality(task, generator_model=str(task["source_mode"]), judge_model=CHEAP_JUDGE_MODEL)
            task["judge_filter_result"] = judgment
            task["eval_tier_placeholder"] = EVAL_TIER_MODEL
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")
            results.append({"task_id": task["task_id"], "decision": judgment["decision"]})

    print(json.dumps({"seed": SEED, "output_path": str(OUTPUT_PATH), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
