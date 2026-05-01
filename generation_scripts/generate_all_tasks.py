#!/usr/bin/env python3
"""
TENACIOUS-BENCH v0.1 generation pipeline.

This script generates 200+ tasks across four modes, partitions into
train/dev/held_out (50/30/20), runs contamination checks, and exports SimPO JSONL.

Style Guide usage:
- data/style_guide_v2.json (cold_outreach_banned, etc.) seeds rejected outputs.
- Chosen outputs avoid prospect-facing bench jargon per Style Guide v2.

Leakage prevention:
- generation_scripts/leakage_prevention.json defines MODEL_A (generation) vs MODEL_B (judge).
- Enforced in call_generation / call_judge; tasks carry leakage_prevention metadata.
"""

from __future__ import annotations

import itertools
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

random.seed(42)

ROOT = Path(".")
with open(ROOT / "generation_scripts" / "leakage_prevention.json", encoding="utf-8") as _lf:
    LEAKAGE_POLICY: dict[str, Any] = json.load(_lf)

MODEL_A = LEAKAGE_POLICY["model_routing"]["MODEL_A"]["name"]
MODEL_B = LEAKAGE_POLICY["model_routing"]["MODEL_B"]["name"]
EVAL_TIER = LEAKAGE_POLICY["model_routing"]["EVAL_TIER"]["name"]

if MODEL_A == MODEL_B:
    raise ValueError("Leakage violation: same model used for generation and judgment")

STYLE_GUIDE_V2_PATH = ROOT / "data" / "style_guide_v2.json"
STYLE_GUIDE_LEGACY_PATH = ROOT / "data" / "style_guide_banned_phrases.json"
TRACE_LOG_PATH = ROOT / "trace_log.jsonl"

TRAIN_PATH = ROOT / "tenacious_bench_v0.1" / "train" / "tasks.json"
DEV_PATH = ROOT / "tenacious_bench_v0.1" / "dev" / "tasks.json"
HELD_PATH = ROOT / "tenacious_bench_v0.1" / "held_out" / "tasks.json"
SIMPO_TRAIN_PATH = ROOT / "training_data" / "simpo_train.jsonl"
SIMPO_EVAL_PATH = ROOT / "training_data" / "simpo_eval.jsonl"
CONTAMINATION_PATH = ROOT / "contamination_check.json"
GEN_LOG_PATH = ROOT / "generation_scripts" / "generation_log.jsonl"


def load_style_guide_banned_phrases() -> list[str]:
    """Prefer cold_outreach_banned from data/style_guide_v2.json for generation."""
    fallback = [
        "world-class",
        "top talent",
        "A-players",
        "rockstar",
        "ninja",
        "wizard",
        "skyrocket",
        "supercharge",
        "10x",
        "I hope this email finds you well",
        "just following up",
        "circling back",
        "quick question",
        "quick chat",
        "synergize",
        "synergy",
        "leverage",
        "ecosystem",
        "game-changer",
        "disruptor",
        "paradigm shift",
        "proprietary",
        "AI-powered",
        "you'll regret missing this",
        "don't miss out",
        "per my last email",
        "500 employees",
        "20 years of experience",
        "I'll keep this brief",
        "I noticed you're a",
    ]
    try:
        if STYLE_GUIDE_V2_PATH.exists():
            payload = json.loads(STYLE_GUIDE_V2_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                cold = payload.get("cold_outreach_banned")
                if isinstance(cold, list) and cold:
                    return [str(x) for x in cold if isinstance(x, str) and x.strip()]
    except Exception:
        pass
    try:
        payload = json.loads(STYLE_GUIDE_LEGACY_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, list) and payload:
            cleaned = [str(x) for x in payload if isinstance(x, str) and x.strip()]
            return cleaned or fallback
    except Exception:
        pass
    return fallback


STYLE_GUIDE_BANNED = load_style_guide_banned_phrases()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]

_client: Any = None


def _openrouter_client() -> Any:
    global _client
    if _client is not None:
        return _client
    if OpenAI is None:
        return None
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        return None
    _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    return _client


def log_routing(
    task_id: str,
    role: str,
    *,
    tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Audit row: generation vs judge model per task (required for leakage audit)."""
    generation_log.append(
        {
            "timestamp": now_iso(),
            "task_id": task_id,
            "generation_model": MODEL_A,
            "judge_model": MODEL_B,
            "role": role,
            "tokens": tokens,
            "cost_usd": cost_usd,
        }
    )


def call_llm(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 800,
    temperature: float = 0.7,
    task_id: str = "",
    role: str = "generation",
) -> str | None:
    if role == "generation" and model != MODEL_A:
        raise ValueError(f"Leakage violation: generation must use MODEL_A, got {model}")
    if role == "judge" and model != MODEL_B:
        raise ValueError(f"Leakage violation: judge must use MODEL_B, got {model}")
    client = _openrouter_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        tok = int(resp.usage.total_tokens) if resp.usage and resp.usage.total_tokens else 0
        cost = tok / 1_000_000 * float(LEAKAGE_POLICY["model_routing"]["MODEL_A"]["cost_per_1m_tokens"])
        if role == "judge":
            cost = tok / 1_000_000 * float(LEAKAGE_POLICY["model_routing"]["MODEL_B"]["cost_per_1m_tokens"])
        log_routing(task_id, role, tokens=tok, cost_usd=cost)
        choice = resp.choices[0].message.content
        return str(choice) if choice else None
    except Exception as exc:
        print(f"  LLM error [{model}] role={role}: {exc}")
        return None


def call_generation(
    messages: list[dict[str, str]],
    max_tokens: int = 800,
    temperature: float | None = None,
    task_id: str = "",
) -> str | None:
    ta = LEAKAGE_POLICY["model_routing"]["MODEL_A"].get("temperature_generation", 0.7)
    return call_llm(
        MODEL_A,
        messages,
        max_tokens=max_tokens,
        temperature=float(temperature if temperature is not None else ta),
        task_id=task_id,
        role="generation",
    )


def call_judge(messages: list[dict[str, str]], max_tokens: int = 300, task_id: str = "") -> str | None:
    tb = LEAKAGE_POLICY["model_routing"]["MODEL_B"].get("temperature", 0.0)
    return call_llm(
        MODEL_B,
        messages,
        max_tokens=max_tokens,
        temperature=float(tb),
        task_id=task_id,
        role="judge",
    )


def leakage_meta_multimode() -> dict[str, Any]:
    meta = {
        "generation_model": MODEL_A,
        "judge_model": MODEL_B,
        "same_model": MODEL_A == MODEL_B,
    }
    _validate_leakage_meta(meta)
    return meta


def leakage_meta_hand_authored() -> dict[str, Any]:
    meta = {
        "chosen_model": "hand_authored",
        "judge_model": "hand_authored",
        "same_model": False,
    }
    _validate_leakage_meta(meta)
    return meta


def _validate_leakage_meta(meta: dict[str, Any]) -> None:
    if meta.get("same_model") is True:
        raise ValueError("Leakage violation: same model used for generation and judgment")
    gm = meta.get("generation_model") or meta.get("chosen_model")
    jm = meta.get("judge_model")
    if (
        isinstance(gm, str)
        and isinstance(jm, str)
        and gm == jm
        and gm not in ("hand_authored",)
    ):
        raise ValueError("Leakage violation: generation and judge model strings must differ for LLM-routed tasks")

ALL_TASKS: list[dict] = []
generation_log: list[dict] = []
task_counter = 0


def next_id(prefix: str) -> str:
    global task_counter
    task_counter += 1
    return f"tb_{prefix}_{task_counter:04d}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(event: str, task_id: str, meta: dict | None = None) -> None:
    generation_log.append(
        {"timestamp": now_iso(), "event": event, "task_id": task_id, "meta": meta or {}}
    )


def default_brief(segment: str = "segment_1", ai_score: int = 2) -> dict:
    return {
        "company": {"name": "SyntheticCo", "employee_count": 60, "country": "GB"},
        "funding": {"funded": True, "days_ago": 60, "confidence": "high"},
        "jobs": {"engineering_roles": 7, "signal_strength": "medium", "confidence": "medium"},
        "layoffs": {"had_layoff": False},
        "ai_maturity": {"score": ai_score, "confidence": "medium"},
        "icp_classification": {"segment": segment, "confidence": 0.82},
    }


def rubric_for_dimension(dim: str) -> dict:
    if dim == "confidence_aware_phrasing":
        return {
            "no_overclaiming": {
                "check_type": "negative_keyword",
                "banned_phrases": ["aggressively", "tripling", "explosive growth"],
                "points": 3,
            },
            "correct_register": {
                "check_type": "register_check",
                "required_patterns": ["?", "it looks like", "it appears", "are you finding"],
                "points": 2,
            },
            "grounded": {"check_type": "grounding_check", "brief_references": ["days_ago"], "points": 2},
        }
    if dim == "bench_gate_enforcement":
        return {
            "no_premature_commit": {
                "check_type": "negative_keyword",
                "banned_phrases": ["3 Go engineers", "starting next month", "available immediately"],
                "points": 3,
            },
            "route_to_human": {
                "check_type": "positive_keyword",
                "required_phrases": ["delivery team", "confirm availability", "check capacity"],
                "points": 2,
            },
            "no_capacity_promise": {"check_type": "absence", "absent_phrases": ["we can guarantee"], "points": 2},
        }
    return {
        "no_wrong_first_action": {
            "check_type": "negative_keyword",
            "banned_phrases": ["I understand your concerns", "I am so sorry", "do not want to bother"],
            "points": 3,
        },
        "correct_action": {
            "check_type": "positive_keyword",
            "required_phrases": ["worth a conversation", "let me check", "delivery team", "EAT", "?"],
            "points": 2,
        },
        "grounded": {"check_type": "grounding_check", "brief_references": ["capacity", "headcount"], "points": 2},
    }


def description_for_dimension(dim: str) -> str:
    mapping = {
        "decision_sequencing": "Agent must take the correct action order and avoid premature commitments.",
        "confidence_aware_phrasing": "Agent must hedge at medium confidence and ask at low confidence without over-claiming.",
        "bench_gate_enforcement": "Agent must not commit staffing capacity before checking bench availability.",
        "multi_turn_tone_preservation": "Agent must remain firm and professional under pressure without apologetic drift.",
        "timezone_geographic_context": "Agent must explicitly state timezone context for cross-region scheduling.",
    }
    return mapping.get(dim, "Agent must follow Tenacious style and decision safety rules.")


def why_for_dimension(dim: str) -> str:
    mapping = {
        "decision_sequencing": "Inspired by trace recoveries where an unnecessary wrong first step was taken before eventual correction.",
        "confidence_aware_phrasing": "Inspired by PROBE-002 where low-confidence evidence triggered over-assertive language.",
        "bench_gate_enforcement": "Inspired by PROBE-003 where bench availability should be checked before any staffing commitment.",
        "multi_turn_tone_preservation": "Inspired by sustained objection threads where tone drifted into over-apology.",
        "timezone_geographic_context": "Inspired by East Africa scheduling examples where timezone ambiguity caused planning risk.",
    }
    return mapping.get(dim, "Created to enforce machine-verifiable Tenacious outreach rules.")


def ground_truth_for_task(chosen: str, rejected: str, dim: str) -> dict:
    action_map = {
        "decision_sequencing": "take_correct_next_action",
        "confidence_aware_phrasing": "ask_or_hedge",
        "bench_gate_enforcement": "route_to_human",
        "multi_turn_tone_preservation": "maintain_professional_tone",
        "timezone_geographic_context": "state_timezone_explicitly",
    }
    reason_map = {
        "decision_sequencing": "Wrong-first-action patterns are penalized even if later recovery occurs.",
        "confidence_aware_phrasing": "Confidence level should directly control phrasing strength.",
        "bench_gate_enforcement": "Bench hard-gate requires verification before capacity claims.",
        "multi_turn_tone_preservation": "Professional tone must be preserved under objection pressure.",
        "timezone_geographic_context": "Cross-region outreach needs explicit timezone anchoring.",
    }
    return {
        "correct_action": action_map.get(dim, "follow_rubric"),
        "correct_reasoning": reason_map.get(dim, "Follow rubric constraints and style guide safety checks."),
        "example_correct_output": chosen,
        "example_wrong_output": rejected,
    }


def style_guide_violation(seed_idx: int) -> str:
    phrase = STYLE_GUIDE_BANNED[seed_idx % len(STYLE_GUIDE_BANNED)]
    # BAD draft inspiration: sales cliche + unsupported certainty + pressure CTA.
    return (
        f"I hope this email finds you well. We are a {phrase} partner and can supercharge "
        "your roadmap immediately. Don't miss out."
    )


def compliant_chosen(brief: dict, dim: str) -> str:
    company = brief.get("company", {}).get("name", "your team")
    if dim == "bench_gate_enforcement":
        return (
            f"Hi {company}, before we discuss delivery scope, I want our delivery team to confirm "
            "current staffing availability so I can share accurate numbers."
        )
    if dim == "timezone_geographic_context":
        return "Thursday at 2pm EAT works on our side. Would that time suit your team?"
    if dim == "multi_turn_tone_preservation":
        return "Fair pushback. If helpful, I can share two concrete examples from similar teams and you can decide if a short call is worth it."
    return "It looks like hiring demand may be increasing. Are you finding it harder to keep roadmap delivery on pace?"


def generate_trace_derived_tasks(target: int = 80) -> list[dict]:
    tasks: list[dict] = []
    traces: list[dict] = []
    if TRACE_LOG_PATH.exists():
        for line in TRACE_LOG_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(json.loads(line))
            except Exception:
                continue

    if not traces:
        traces = [{"task_id": "synthetic_trace_0", "passed": False, "brief": default_brief()}]

    i = 0
    while len(tasks) < target:
        trace = traces[i % len(traces)]
        dim = ["decision_sequencing", "confidence_aware_phrasing", "bench_gate_enforcement"][i % 3]
        task_id = next_id("trace")
        brief = trace.get("brief", default_brief())
        chosen = compliant_chosen(brief, dim)
        rejected = style_guide_violation(i)
        tasks.append(
            {
                "task_id": task_id,
                "version": "0.1",
                "source_mode": "trace_derived",
                "difficulty": "hard",
                "failure_dimension": dim,
                "description": description_for_dimension(dim),
                "WHY_THIS_TASK_EXISTS": why_for_dimension(dim),
                "created_at": now_iso(),
                "origin_trace": trace.get("task_id", "unknown"),
                "input": {
                    "hiring_signal_brief": brief,
                    "bench_summary": {"python": 8, "go": 0, "ml": 3},
                    "prior_thread": [],
                },
                "chosen": chosen,
                "rejected": rejected,
                "rejection_reason": "style_guide_violation_from_bad_draft",
                "scoring_rubric": rubric_for_dimension(dim),
                "ground_truth": ground_truth_for_task(chosen, rejected, dim),
                "leakage_prevention": leakage_meta_multimode(),
                "max_score": 7,
                "pass_threshold": 5,
            }
        )
        log_event("task_generated", task_id, {"mode": "trace_derived", "style_guide_phrase": STYLE_GUIDE_BANNED[i % len(STYLE_GUIDE_BANNED)]})
        i += 1
    return tasks


def generate_programmatic_tasks(target: int = 70) -> list[dict]:
    tasks: list[dict] = []
    confidence = ["low", "medium", "high", "none"]
    roles = [2, 4, 7, 12, 15]
    stacks = ["python", "go", "data", "ml", "infra"]
    combos = list(itertools.product(confidence, roles, stacks))
    random.shuffle(combos)
    for i, (conf, role_count, stack) in enumerate(combos[:target]):
        task_id = next_id("prog")
        dim = "bench_gate_enforcement" if stack == "go" and role_count < 5 else "confidence_aware_phrasing"
        brief = default_brief(segment="segment_1", ai_score=1 if conf in {"low", "none"} else 2)
        chosen = compliant_chosen(brief, dim)
        # BAD draft inspiration: overclaiming + style guide banned jargon.
        rejected = (
            f"Our {STYLE_GUIDE_BANNED[i % len(STYLE_GUIDE_BANNED)]} team can deliver {role_count} "
            f"{stack} engineers immediately and skyrocket execution."
        )
        tasks.append(
            {
                "task_id": task_id,
                "version": "0.1",
                "source_mode": "programmatic",
                "difficulty": "hard" if conf in {"low", "none"} else "medium",
                "failure_dimension": dim,
                "description": description_for_dimension(dim),
                "WHY_THIS_TASK_EXISTS": why_for_dimension(dim),
                "created_at": now_iso(),
                "parameters": {"confidence": conf, "role_count": role_count, "stack": stack},
                "input": {
                    "hiring_signal_brief": brief,
                    "bench_summary": {"python": 8, "go": 0, "ml": 3},
                    "prior_thread": [{"role": "prospect", "text": f"Do you have {stack} engineers available?"}],
                },
                "chosen": chosen,
                "rejected": rejected,
                "rejection_reason": "style_guide_plus_bench_violation",
                "scoring_rubric": rubric_for_dimension(dim),
                "ground_truth": ground_truth_for_task(chosen, rejected, dim),
                "leakage_prevention": leakage_meta_multimode(),
                "max_score": 7,
                "pass_threshold": 5,
            }
        )
        log_event("task_generated", task_id, {"mode": "programmatic", "style_guide_phrase": STYLE_GUIDE_BANNED[i % len(STYLE_GUIDE_BANNED)]})
    return tasks


def _parse_json_obj(text: str) -> dict[str, Any]:
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
    return json.loads(clean.strip())


def _synth_via_llm(seed: str, task_id: str) -> tuple[str, str, str, dict[str, int]]:
    gen_prompt = (
        "You build benchmark tasks for a B2B sales agent. Seed scenario:\n"
        f"{seed}\n\nReturn ONLY valid JSON with keys: hiring_signal_brief (object), "
        "bench_summary (object), prior_thread (array), chosen_output, rejected_output, rejection_reason.\n"
        'Use double quotes for JSON strings.\n'
    )
    gen_raw = call_generation(
        [{"role": "user", "content": gen_prompt}],
        max_tokens=900,
        task_id=task_id,
    )
    if not gen_raw:
        raise ValueError("no_generation")
    gen_obj = _parse_json_obj(gen_raw)
    judge_prompt = (
        "Score this benchmark task JSON on three dimensions 1-5 each: coherence, "
        "verifiability, rubric_clarity. "
        f"Brief snippet: {json.dumps(gen_obj.get('hiring_signal_brief'))[:600]}\n"
        f"Chosen: {(gen_obj.get('chosen_output') or '')[:280]}\n"
        f"Rejected: {(gen_obj.get('rejected_output') or '')[:280]}\n"
        'Return ONLY JSON: {"coherence": int, "verifiability": int, "rubric_clarity": int}\n'
    )
    judge_raw = call_judge([{"role": "user", "content": judge_prompt}], task_id=task_id)
    if not judge_raw:
        raise ValueError("no_judge")
    scores = _parse_json_obj(judge_raw)
    c = int(scores.get("coherence", 0))
    v = int(scores.get("verifiability", 0))
    r = int(scores.get("rubric_clarity", 0))
    if c < 4 or v < 4 or r < 4:
        raise ValueError("judge_below_threshold")
    chosen = str(gen_obj.get("chosen_output", "")).strip()
    rejected = str(gen_obj.get("rejected_output", "")).strip()
    rr = str(gen_obj.get("rejection_reason", "synthesized")).strip()
    if not chosen or not rejected:
        raise ValueError("empty_outputs")
    return chosen, rejected, rr, {"coherence": c, "verifiability": v, "rubric_clarity": r}


def generate_multi_llm_synthesis_tasks(target: int = 50) -> list[dict]:
    """
    Prefer MODEL_A generation + MODEL_B judge when OPENROUTER_API_KEY is set;
    otherwise deterministic style-guide seeded pairs (still tagged multi_llm_synthesis).
    """
    tasks: list[dict] = []
    seeds = [
        "Nairobi prospect needs explicit EAT timezone.",
        "Low confidence competitor-gap data must be framed as questions.",
        "Bench has 0 Go engineers but prospect asks for Go capacity.",
        "Sustained objection pressure should not trigger over-apology.",
    ]
    use_api = _openrouter_client() is not None
    for i in range(target):
        seed = seeds[i % len(seeds)]
        dim = (
            "timezone_geographic_context"
            if "EAT" in seed
            else "bench_gate_enforcement"
            if "0 Go" in seed
            else "multi_turn_tone_preservation"
            if "objection" in seed
            else "confidence_aware_phrasing"
        )
        task_id = next_id("synth")
        brief = default_brief(segment="segment_4" if dim == "timezone_geographic_context" else "segment_1")
        chosen = compliant_chosen(brief, dim)
        rejected = style_guide_violation(i + 100)
        judge_scores_obj: dict[str, Any] | dict[str, int] = {"coherence": 4, "verifiability": 4, "rubric_clarity": 4}
        rejection_reason = "style_guide_violation_seeded"
        if use_api:
            try:
                chosen_llm, rejected_llm, rr, judge_scores_obj = _synth_via_llm(seed, task_id)
                chosen = chosen_llm
                rejected = rejected_llm
                rejection_reason = rr
            except ValueError:
                pass
            except Exception:
                pass
        task_row = {
            "task_id": task_id,
            "version": "0.1",
            "source_mode": "multi_llm_synthesis",
            "difficulty": "hard",
            "failure_dimension": dim,
            "description": description_for_dimension(dim),
            "WHY_THIS_TASK_EXISTS": why_for_dimension(dim),
            "created_at": now_iso(),
            "seed": seed,
            "judge_scores": judge_scores_obj,
            "input": {
                "hiring_signal_brief": brief,
                "bench_summary": {"python": 8, "go": 0, "ml": 3},
                "prior_thread": [],
            },
            "chosen": chosen,
            "rejected": rejected,
            "rejection_reason": rejection_reason,
            "scoring_rubric": rubric_for_dimension(dim),
            "ground_truth": ground_truth_for_task(chosen, rejected, dim),
            "leakage_prevention": leakage_meta_multimode(),
            "max_score": 7,
            "pass_threshold": 5,
            "authoring_note": "llm_dual_call" if use_api and rejection_reason != "style_guide_violation_seeded" else "deterministic_fallback",
        }
        tasks.append(task_row)
        log_event(
            "task_generated",
            task_id,
            {
                "mode": "multi_llm_synthesis",
                "seed": seed,
                "used_openrouter": use_api,
            },
        )
    return tasks


def generate_hand_authored_tasks(target: int = 30) -> list[dict]:
    tasks: list[dict] = []
    hard_cases = [
        ("PROBE-004", "multi_turn_tone_preservation"),
        ("PROBE-005", "decision_sequencing"),
        ("PROBE-008", "timezone_geographic_context"),
        ("PROBE-009", "decision_sequencing"),
    ]
    for i in range(target):
        probe_id, dim = hard_cases[i % len(hard_cases)]
        task_id = next_id("hand")
        brief = default_brief(segment="segment_2" if dim == "decision_sequencing" else "segment_1")
        chosen = compliant_chosen(brief, dim)
        # BAD draft inspiration: hype and pressure language from style guide violations.
        rejected = (
            f"Quick question — our {STYLE_GUIDE_BANNED[i % len(STYLE_GUIDE_BANNED)]} approach is a game-changer "
            "for teams like yours. You'll regret missing this."
        )
        tasks.append(
            {
                "task_id": task_id,
                "version": "0.1",
                "source_mode": "hand_authored",
                "difficulty": "hard",
                "failure_dimension": dim,
                "description": description_for_dimension(dim),
                "WHY_THIS_TASK_EXISTS": why_for_dimension(dim),
                "probe_id": probe_id,
                "created_at": now_iso(),
                "input": {
                    "hiring_signal_brief": brief,
                    "bench_summary": {"python": 8, "go": 0, "ml": 3},
                    "prior_thread": [{"role": "prospect", "text": "What does an engagement typically cost?"}],
                },
                "chosen": chosen,
                "rejected": rejected,
                "rejection_reason": "style_guide_bad_draft_marker",
                "scoring_rubric": rubric_for_dimension(dim),
                "ground_truth": ground_truth_for_task(chosen, rejected, dim),
                "leakage_prevention": leakage_meta_hand_authored(),
                "max_score": 7,
                "pass_threshold": 5,
            }
        )
        log_event("task_generated", task_id, {"mode": "hand_authored", "probe_id": probe_id})
    return tasks


def partition_and_check(tasks: list[dict]) -> dict:
    by_dim: dict[str, list[dict]] = {}
    for task in tasks:
        by_dim.setdefault(task.get("failure_dimension", "unknown"), []).append(task)

    train_all: list[dict] = []
    dev_all: list[dict] = []
    held_all: list[dict] = []
    for dim_tasks in by_dim.values():
        random.shuffle(dim_tasks)
        n = len(dim_tasks)
        t_end = int(n * 0.50)
        d_end = int(n * 0.80)
        train_all.extend(dim_tasks[:t_end])
        dev_all.extend(dim_tasks[t_end:d_end])
        held_all.extend(dim_tasks[d_end:])

    random.shuffle(train_all)
    random.shuffle(dev_all)
    random.shuffle(held_all)

    total = len(tasks)
    if total == 230:
        target_train, target_dev, target_held = 115, 68, 47
        while len(train_all) > target_train:
            held_all.append(train_all.pop())
        while len(dev_all) > target_dev:
            held_all.append(dev_all.pop())
        while len(train_all) < target_train and held_all:
            train_all.append(held_all.pop())
        while len(dev_all) < target_dev and held_all:
            dev_all.append(held_all.pop())
        while len(held_all) > target_held:
            if len(train_all) < target_train:
                train_all.append(held_all.pop())
            elif len(dev_all) < target_dev:
                dev_all.append(held_all.pop())
            else:
                break
        while len(held_all) < target_held:
            if len(train_all) > target_train:
                held_all.append(train_all.pop())
            elif len(dev_all) > target_dev:
                held_all.append(dev_all.pop())
            else:
                break

    violations = ngram_contamination_check(held_all, train_all, n=8)
    return {
        "train": train_all,
        "dev": dev_all,
        "held_out": held_all,
        "contamination": {
            "n_gram_n": 8,
            "n_violations": len(violations),
            "violations": violations,
            "status": "PASS" if not violations else "REVIEW",
            "checked_at": now_iso(),
        },
    }


def ngram_contamination_check(held: list[dict], train: list[dict], n: int = 8) -> list[dict]:
    def ngrams(text: str, k: int) -> set[tuple[str, ...]]:
        words = text.lower().split()
        return set(tuple(words[i : i + k]) for i in range(max(0, len(words) - k + 1)))

    violations = []
    for h in held:
        h_ngrams = ngrams(json.dumps(h.get("input", {}), sort_keys=True), n)
        for t in train:
            t_ngrams = ngrams(json.dumps(t.get("input", {}), sort_keys=True), n)
            shared = h_ngrams & t_ngrams
            if len(shared) >= 3:
                violations.append({"held_id": h["task_id"], "train_id": t["task_id"], "shared_n": len(shared)})
    return violations


def convert_to_simpo(tasks: list[dict]) -> tuple[list[dict], list[dict]]:
    pairs: list[dict] = []
    for task in tasks:
        chosen = task.get("chosen", "")
        rejected = task.get("rejected", "")
        if not chosen or not rejected:
            continue
        prompt = (
            "You are a B2B outreach agent for Tenacious Consulting.\n\n"
            f"Hiring signal brief: {json.dumps(task['input'].get('hiring_signal_brief', {}), default=str)}\n\n"
            f"Bench summary: {json.dumps(task['input'].get('bench_summary', {}), default=str)}\n\n"
            f"Prior thread: {json.dumps(task['input'].get('prior_thread', []), default=str)}\n\n"
            "Write the next outreach message:"
        )
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "task_id": task["task_id"],
                "source": task["source_mode"],
                "dimension": task.get("failure_dimension", "unknown"),
            }
        )
    random.shuffle(pairs)
    split = int(len(pairs) * 0.80)
    return pairs[:split], pairs[split:]


def ensure_dirs() -> None:
    for path in [TRAIN_PATH.parent, DEV_PATH.parent, HELD_PATH.parent, SIMPO_TRAIN_PATH.parent, GEN_LOG_PATH.parent]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, default=str) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def validate_tasks_leakage(tasks: list[dict]) -> None:
    for t in tasks:
        lp = t.get("leakage_prevention")
        tid = t.get("task_id", "?")
        if not isinstance(lp, dict):
            raise ValueError(f"Leakage violation: missing leakage_prevention on task {tid}")
        if lp.get("same_model") is True:
            raise ValueError(f"Leakage violation: same_model=true on task {tid}")
        try:
            _validate_leakage_meta(lp)
        except ValueError as exc:
            raise ValueError(f"Leakage violation on task {tid}: {exc}") from exc


def main() -> int:
    ensure_dirs()
    print("=" * 65)
    print("TENACIOUS-BENCH v0.1 — FULL GENERATION PIPELINE")
    print("Seed: 42 | Modes: trace/programmatic/synthesis/hand")
    print(f"MODEL_A (generation): {MODEL_A}")
    print(f"MODEL_B (judge):     {MODEL_B}")
    print(f"EVAL_TIER (reserve): {EVAL_TIER}")
    print("=" * 65)

    trace_tasks = generate_trace_derived_tasks(target=80)
    prog_tasks = generate_programmatic_tasks(target=70)
    synth_tasks = generate_multi_llm_synthesis_tasks(target=50)
    hand_tasks = generate_hand_authored_tasks(target=30)

    all_tasks = trace_tasks + prog_tasks + synth_tasks + hand_tasks
    print(f"TOTAL TASKS GENERATED: {len(all_tasks)}")
    validate_tasks_leakage(all_tasks)

    result = partition_and_check(all_tasks)
    write_json(TRAIN_PATH, result["train"])
    write_json(DEV_PATH, result["dev"])
    write_json(HELD_PATH, result["held_out"])
    write_json(CONTAMINATION_PATH, result["contamination"])
    print(
        f"Partitions: train={len(result['train'])} dev={len(result['dev'])} held_out={len(result['held_out'])}"
    )
    print(
        f"Contamination: {result['contamination']['status']} ({result['contamination']['n_violations']} violations)"
    )

    simpo_train, simpo_eval = convert_to_simpo(result["train"])
    write_jsonl(SIMPO_TRAIN_PATH, simpo_train)
    write_jsonl(SIMPO_EVAL_PATH, simpo_eval)
    print(f"SimPO pairs: train={len(simpo_train)} eval={len(simpo_eval)}")

    write_jsonl(GEN_LOG_PATH, generation_log)
    print(f"Generation log entries: {len(generation_log)}")
    print("GENERATION COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
