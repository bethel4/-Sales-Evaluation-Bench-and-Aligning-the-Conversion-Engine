#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

# “Style Guide → Rubric → Code → Score”
# Tenacious Style Guide v2 is the source of truth for banned phrase checks,
# tone safety, confidence-aware phrasing, bench overcommitment prevention,
# and grounded outreach.
#
# Calibration convention used by this scaffold:
# - band 5 / score 1.0: dimension is cleanly satisfied
# - band 3 / score 0.5: partially satisfied or borderline
# - band 1 / score 0.0: clear failure
# The final numerical score is a weighted average over these normalized bands.

DEFAULT_TASKS = Path("tenacious_bench_v0.1/dev/tasks.json")
STYLE_GUIDE_V2_PATH = Path("data/style_guide_v2.json")
STYLE_GUIDE_BANNED_PHRASES_PATH = Path("data/style_guide_banned_phrases.json")
CALENDAR_RE = re.compile(r"https?://(www\.)?cal\.com/[^\s]+$")
CTA_PHRASES = [
    "book a discovery call",
    "schedule a discovery call",
    "review this together",
    "would you be open",
    "can we schedule",
]
GLOBAL_OVERCLAIMING_PHRASES = [
    "aggressively",
    "tripling",
    "guarantee",
    "definitely",
    "clearly behind",
    "explosive growth",
]
GENERIC_TONE_PHRASES = [
    "best-in-class",
    "synergy",
    "digital transformation",
    "world-class",
]
APOLOGETIC_PHRASES = ["sorry", "i apologize", "apologies"]
GLOBAL_BANNED_PHRASES = [
    "world-class",
    "top talent",
    "a-players",
    "rockstar",
    "ninja",
    "wizard",
    "skyrocket",
    "supercharge",
    "10x",
    "i hope this email finds you well",
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
    "ai-powered",
    "you'll regret missing this",
    "don't miss out",
    "per my last email",
    "500 employees",
    "20 years of experience",
    "i'll keep this brief",
    "i noticed you're a",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _contains_any(text: str, phrases: list[str]) -> bool:
    lowered = _normalize(text)
    return any(_normalize(phrase) in lowered for phrase in phrases)


def _flatten_style_guide_v2_lists(payload: dict[str, Any]) -> list[str]:
    keys = (
        "cold_outreach_banned",
        "subject_line_banned",
        "low_confidence_banned",
        "bench_external_banned",
        "over_apologetic_banned",
        "condescending_banned",
        "fabrication_risk",
        "pricing_banned",
    )
    out: list[str] = []
    for key in keys:
        blob = payload.get(key)
        if isinstance(blob, list):
            out.extend(str(x) for x in blob if isinstance(x, str) and x.strip())
    return out


def load_style_guide_banned_phrases() -> list[str]:
    """
    Load style-guide banned phrases from disk.

    Prefer data/style_guide_v2.json (structured Style Guide v2). Fall back to
    legacy flat list data/style_guide_banned_phrases.json, then GLOBAL_BANNED_PHRASES.
    Never raises.
    """
    try:
        if STYLE_GUIDE_V2_PATH.exists():
            payload_any: Any = json.loads(STYLE_GUIDE_V2_PATH.read_text(encoding="utf-8"))
            if isinstance(payload_any, dict):
                merged = _flatten_style_guide_v2_lists(payload_any)
                seen: set[str] = set()
                deduped: list[str] = []
                for p in merged:
                    n = _normalize(p)
                    if n and n not in seen:
                        seen.add(n)
                        deduped.append(str(p))
                return deduped or list(GLOBAL_BANNED_PHRASES)
    except Exception:
        pass

    try:
        payload = json.loads(STYLE_GUIDE_BANNED_PHRASES_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return list(GLOBAL_BANNED_PHRASES)
        parsed = [str(item) for item in payload if isinstance(item, str) and item.strip()]
        return parsed or list(GLOBAL_BANNED_PHRASES)
    except Exception:
        return list(GLOBAL_BANNED_PHRASES)


STYLE_GUIDE_BANNED_PHRASES = load_style_guide_banned_phrases()
DECODING_STRATEGIES = ("reranker", "best_of_n", "rejection_sampling")


def _check_negative_keyword(output: str, rubric_item: dict[str, Any]) -> list[str]:
    """
    Evaluate negative-keyword checks against both task-level and style-guide bans.

    This enforces the Tenacious Style Guide v2 as a global rubric source while
    preserving per-task banned phrases from each benchmark item.
    """
    task_banned = list(rubric_item.get("banned_phrases", []))
    combined = [*task_banned, *STYLE_GUIDE_BANNED_PHRASES]
    output_norm = _normalize(output)
    hits: list[str] = []
    seen: set[str] = set()
    for phrase in combined:
        norm_phrase = _normalize(str(phrase))
        if not norm_phrase or norm_phrase in seen:
            continue
        if norm_phrase in output_norm:
            hits.append(str(phrase))
            seen.add(norm_phrase)
    return hits


def _resolve_path(obj: dict[str, Any], dotted_path: str) -> Any:
    cur: Any = obj
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted_path)
        cur = cur[part]
    return cur


def _safe_signal_values(task: dict[str, Any], paths: list[str]) -> list[str]:
    values: list[str] = []
    for path in paths:
        try:
            resolved = _resolve_path(task["inputs"], path)
        except Exception:
            continue
        if isinstance(resolved, str):
            values.append(resolved)
        elif isinstance(resolved, (int, float)):
            values.append(str(resolved))
        elif isinstance(resolved, list):
            values.extend(str(item) for item in resolved)
    return values


def _signal_reference_score(task: dict[str, Any], output: str) -> tuple[float, str | None]:
    signal_paths = list(task.get("rubric", {}).get("required_signal_reference_keys", []))
    if not signal_paths:
        return 1.0, None
    output_norm = _normalize(output)
    values = _safe_signal_values(task, signal_paths)
    if any(_normalize(value) in output_norm for value in values if value.strip()):
        return 1.0, None
    if any(path.split(".")[-1].replace("_", " ") in output_norm for path in signal_paths):
        return 0.5, "referenced signal type but not the concrete signal value"
    return 0.0, "missing required signal reference"


def _required_substring_score(task: dict[str, Any], output: str) -> tuple[float, list[str]]:
    required = list(task.get("rubric", {}).get("required_substrings", []))
    if not required:
        return 1.0, []
    output_norm = _normalize(output)
    missing = [needle for needle in required if _normalize(needle) not in output_norm]
    if not missing:
        return 1.0, []
    if len(missing) < len(required):
        return 0.5, missing
    return 0.0, missing


def _forbidden_phrase_score(task: dict[str, Any], output: str) -> tuple[float, list[str]]:
    forbidden = list(task.get("rubric", {}).get("forbidden_phrases", []))
    hits = _check_negative_keyword(output, {"banned_phrases": forbidden})
    if not hits:
        return 1.0, []
    return 0.0, hits


def _calendar_cta_score(task: dict[str, Any], output: str) -> tuple[float, str | None]:
    rubric = task.get("rubric", {})
    requires_cta = bool(rubric.get("requires_cta"))
    must_end_with_calendar_link = bool(rubric.get("must_end_with_calendar_link"))
    cta_ok = True if not requires_cta else _contains_any(output, CTA_PHRASES)
    calendar_ok = True if not must_end_with_calendar_link else bool(CALENDAR_RE.search(output.strip()))
    if cta_ok and calendar_ok:
        return 1.0, None
    if cta_ok or calendar_ok:
        return 0.5, "partial CTA/calendar compliance"
    return 0.0, "missing CTA and/or calendar link"


def _extract_action_name(output: str) -> str | None:
    match = re.match(r"\s*ACTION:\s*([a-zA-Z0-9_]+)", output)
    return match.group(1) if match else None


def _output_type_or_action_score(task: dict[str, Any], output: str) -> tuple[float, str | None]:
    rubric = task.get("rubric", {})
    expected_type = str(rubric.get("expected_output_type", "text"))
    expected_action = rubric.get("expected_action_name")
    action_name = _extract_action_name(output)

    if expected_type == "action":
        if action_name is None:
            return 0.0, "expected action output but text was provided"
        if expected_action and action_name != expected_action:
            return 0.0, f"wrong action name: {action_name}"
        return 1.0, None

    if action_name is not None and expected_type == "text":
        return 0.0, "expected text output but action string was provided"
    return 1.0, None


def _overclaiming_score(task: dict[str, Any], output: str) -> tuple[float, list[str]]:
    extra = list(task.get("rubric", {}).get("forbidden_overclaiming_phrases", []))
    banned = [*GLOBAL_OVERCLAIMING_PHRASES, *extra]
    hits = [phrase for phrase in banned if _contains_any(output, [phrase])]
    if hits:
        return 0.0, hits

    signal_strength = ""
    try:
        signal_strength = str(_resolve_path(task["inputs"], "evidence_brief.jobs.signal_strength")).casefold()
    except Exception:
        signal_strength = ""

    if signal_strength in {"low", "medium"} and _contains_any(output, ["will deliver", "we can deliver", "is definitely"]):
        return 0.0, ["unsupported certainty for low/medium signal"]
    return 1.0, []


def _tone_score(output: str, min_tone_score: float) -> tuple[float, dict[str, Any]]:
    penalties = 0.0
    if _contains_any(output, APOLOGETIC_PHRASES):
        penalties += 0.25
    if _contains_any(output, GENERIC_TONE_PHRASES):
        penalties += 0.35
    if len(output.split()) < 8:
        penalties += 0.15
    raw = max(0.0, 1.0 - penalties)
    if raw >= min_tone_score:
        return 1.0, {"raw_tone_score": round(raw, 3), "band": 5}
    if raw >= max(0.5, min_tone_score - 0.15):
        return 0.5, {"raw_tone_score": round(raw, 3), "band": 3}
    return 0.0, {"raw_tone_score": round(raw, 3), "band": 1}


def _weights(rubric: dict[str, Any]) -> dict[str, float]:
    default = {
        "signal_reference": 0.2,
        "required_elements": 0.2,
        "forbidden_phrases": 0.15,
        "cta_calendar": 0.15,
        "output_type_or_action": 0.2,
        "no_overclaiming": 0.1,
        "tone": 0.0,
    }
    incoming = rubric.get("weights", {})
    return {key: float(incoming.get(key, value)) for key, value in default.items()}


def _resolve_agent_output(
    task: dict[str, Any], provided_output: Any, score_field: str = "rejected"
) -> tuple[str | None, str | None, str | None]:
    """
    Resolve output text with schema fallback:
    agent_output -> candidate_output.
    """
    if isinstance(provided_output, str) and provided_output.strip():
        return provided_output, "provided_output", None
    if not isinstance(task, dict):
        return None, None, "malformed_input: task must be a dict"

    preferred_simpo = "chosen" if score_field == "chosen" else "rejected"
    secondary_simpo = "rejected" if preferred_simpo == "chosen" else "chosen"
    fallback_sources = [
        ("agent_output", task.get("agent_output")),
        ("candidate_output", task.get("candidate_output")),
        (preferred_simpo, task.get(preferred_simpo)),
        (secondary_simpo, task.get(secondary_simpo)),
    ]
    field_name: str | None = None
    fallback: Any = None
    for source_name, value in fallback_sources:
        if isinstance(value, str) and value.strip():
            field_name = source_name
            fallback = value
            break

    if not isinstance(fallback, str):
        return None, None, "malformed_input: missing output fields (agent_output/candidate_output/chosen/rejected)"
    if not fallback.strip():
        return None, None, "malformed_input: resolved output field is empty"
    return fallback, field_name, None


def select_best_of_n(
    prompt: Any,
    generator: Callable[[Any], str],
    judge: Callable[[Any, str], float],
    *,
    n_candidates: int = 3,
) -> dict[str, Any]:
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")
    candidates = [generator(prompt) for _ in range(n_candidates)]
    judge_scores = [float(judge(prompt, c)) for c in candidates]
    selected_idx = max(range(len(candidates)), key=lambda idx: judge_scores[idx])
    return {
        "selected_output": candidates[selected_idx],
        "selected_index": selected_idx,
        "selected_score": judge_scores[selected_idx],
        "strategy": "best_of_n",
        "n_candidates": len(candidates),
        "scores": judge_scores,
        "threshold": None,
        "max_tries": None,
    }


def select_reranker(
    prompt: Any,
    candidate_fn: Callable[[Any, int], list[str]],
    judge: Callable[[Any, str], float],
    *,
    n_candidates: int = 3,
) -> dict[str, Any]:
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")
    candidates = candidate_fn(prompt, n_candidates)
    if not candidates:
        raise ValueError("candidate_fn returned zero candidates")
    judge_scores = [float(judge(prompt, c)) for c in candidates]
    selected_idx = max(range(len(candidates)), key=lambda idx: judge_scores[idx])
    return {
        "selected_output": candidates[selected_idx],
        "selected_index": selected_idx,
        "selected_score": judge_scores[selected_idx],
        "strategy": "reranker",
        "n_candidates": len(candidates),
        "scores": judge_scores,
        "threshold": None,
        "max_tries": None,
    }


def select_rejection_sampling(
    prompt: Any,
    generator: Callable[[Any], str],
    judge: Callable[[Any, str], float],
    *,
    threshold: float = 0.8,
    max_tries: int = 8,
) -> dict[str, Any]:
    if max_tries < 1:
        raise ValueError("max_tries must be >= 1")
    last_output = ""
    last_score = 0.0
    for attempt in range(1, max_tries + 1):
        candidate = generator(prompt)
        score = float(judge(prompt, candidate))
        last_output = candidate
        last_score = score
        if score >= threshold:
            return {
                "selected_output": candidate,
                "selected_index": attempt - 1,
                "selected_score": score,
                "strategy": "rejection_sampling",
                "n_candidates": attempt,
                "scores": [score],
                "threshold": threshold,
                "max_tries": max_tries,
                "tries": attempt,
                "fallback_used": False,
            }
    return {
        "selected_output": last_output,
        "selected_index": max_tries - 1,
        "selected_score": last_score,
        "strategy": "rejection_sampling",
        "n_candidates": max_tries,
        "scores": [last_score],
        "threshold": threshold,
        "max_tries": max_tries,
        "tries": max_tries,
        "fallback_used": True,
    }


def select_output(
    prompt: Any,
    generator: Callable[[Any], str],
    judge: Callable[[Any, str], float],
    strategy: str = "best_of_n",
    *,
    n_candidates: int = 3,
    threshold: float = 0.8,
    max_tries: int = 8,
    candidate_fn: Callable[[Any, int], list[str]] | None = None,
) -> dict[str, Any]:
    """
    Strategy wrapper that makes generator-judge coupling explicit and swappable.
    """
    if strategy == "best_of_n":
        return select_best_of_n(prompt, generator, judge, n_candidates=n_candidates)
    if strategy == "rejection_sampling":
        return select_rejection_sampling(prompt, generator, judge, threshold=threshold, max_tries=max_tries)
    if strategy == "reranker":
        if candidate_fn is None:
            candidate_fn = lambda p, n: [generator(p) for _ in range(n)]
        return select_reranker(prompt, candidate_fn, judge, n_candidates=n_candidates)
    raise ValueError(f"unsupported decoding strategy: {strategy}")


def select_output_from_scores(
    candidates: list[str],
    judge_scores: list[float],
    strategy: str = "best_of_n",
    *,
    threshold: float = 0.8,
    max_tries: int = 8,
) -> dict[str, Any]:
    """
    Couple generator candidates to a scalar judge at inference time.

    Supported strategies:
    - reranker: score all provided candidates and return top-1
    - best_of_n: same top-1 behavior with an explicit N-cost interpretation
    - rejection_sampling: return first candidate whose score >= threshold
    """
    if strategy not in DECODING_STRATEGIES:
        raise ValueError(f"unsupported decoding strategy: {strategy}")
    if len(candidates) != len(judge_scores):
        raise ValueError("candidates and judge_scores must have equal length")
    if not candidates:
        raise ValueError("at least one candidate is required")

    if strategy in ("reranker", "best_of_n"):
        selected_idx = max(range(len(candidates)), key=lambda idx: judge_scores[idx])
        return {
            "selected_output": candidates[selected_idx],
            "selected_index": selected_idx,
            "selected_score": judge_scores[selected_idx],
            "strategy": strategy,
            "n_candidates": len(candidates),
            "threshold": None,
                "max_tries": None,
        }

    for idx, score in enumerate(judge_scores):
        if score >= threshold:
            return {
                "selected_output": candidates[idx],
                "selected_index": idx,
                "selected_score": score,
                "strategy": strategy,
                "n_candidates": len(candidates),
                "threshold": threshold,
                "max_tries": max_tries,
            }
    fallback_idx = max(range(len(candidates)), key=lambda i: judge_scores[i])
    return {
        "selected_output": candidates[fallback_idx],
        "selected_index": fallback_idx,
        "selected_score": judge_scores[fallback_idx],
        "strategy": strategy,
        "n_candidates": len(candidates),
        "threshold": threshold,
        "max_tries": max_tries,
        "fallback_used": True,
    }


def audit_decoding_strategy(payload: dict[str, Any]) -> dict[str, Any]:
    """
    One-function audit: infer configured strategy from run payload/config.
    """
    strategy = str(payload.get("decoding_strategy", "reranker"))
    n_candidates = int(payload.get("n_candidates", 1))
    threshold = payload.get("threshold", payload.get("rejection_threshold"))
    max_tries = payload.get("max_tries")
    inferred = strategy
    if strategy == "reranker" and n_candidates > 1:
        inferred = "best_of_n"
    return {
        "configured_strategy": strategy,
        "inferred_runtime_behavior": inferred,
        "n_candidates": n_candidates,
        "threshold": threshold,
        "max_tries": max_tries,
    }


def score_task(
    task: dict[str, Any],
    agent_output: Any = None,
    score_field: str = "rejected",
    decoding_strategy: str = "reranker",
) -> dict[str, Any]:
    """
    Mechanically score one task/output pair.

    Returns:
    - numerical score in [0.0, 1.0]
    - pass/fail using a default threshold of 0.8
    - visible failure reasons and component scores for debugging
    """
    resolved_output, resolved_field, output_error = _resolve_agent_output(task, agent_output, score_field=score_field)
    if output_error:
        return {
            "task_id": task.get("task_id", "unknown") if isinstance(task, dict) else "unknown",
            "score": 0.0,
            "pass": False,
            "scored_field": resolved_field or score_field,
            "failure_reasons": [output_error],
            "components": {},
        }

    if "scoring_rubric" in task:
        scored = _score_task_schema_v01(task, resolved_output, score_field=score_field)
        scored["decoding_strategy"] = decoding_strategy
        return scored

    try:
        if not isinstance(task, dict):
            raise TypeError("task must be a dict")
        if not isinstance(resolved_output, str):
            raise TypeError("agent_output must be a string")
        rubric = task["rubric"]
        _ = task["inputs"]
    except Exception as exc:
        return {
            "task_id": task.get("task_id", "unknown") if isinstance(task, dict) else "unknown",
            "score": 0.0,
            "pass": False,
            "failure_reasons": [f"malformed_input: {type(exc).__name__}: {exc}"],
            "components": {},
        }

    failure_reasons: list[str] = []
    weights = _weights(rubric)

    signal_score, signal_reason = _signal_reference_score(task, resolved_output)
    if signal_reason:
        failure_reasons.append(signal_reason)

    required_score, missing_required = _required_substring_score(task, resolved_output)
    if missing_required:
        failure_reasons.append(f"missing required elements: {missing_required}")

    forbidden_score, forbidden_hits = _forbidden_phrase_score(task, resolved_output)
    if forbidden_hits:
        failure_reasons.append(f"forbidden phrases: {forbidden_hits}")

    cta_score, cta_reason = _calendar_cta_score(task, resolved_output)
    if cta_reason:
        failure_reasons.append(cta_reason)

    action_score, action_reason = _output_type_or_action_score(task, resolved_output)
    if action_reason:
        failure_reasons.append(action_reason)

    overclaim_score, overclaim_hits = _overclaiming_score(task, resolved_output)
    if overclaim_hits:
        failure_reasons.append(f"overclaiming: {overclaim_hits}")

    tone_score, tone_meta = _tone_score(resolved_output, float(rubric.get("min_tone_score", 0.7)))
    if tone_score < 1.0:
        failure_reasons.append(f"tone below target: {tone_meta['raw_tone_score']}")

    components = {
        "signal_reference": round(signal_score * weights["signal_reference"], 4),
        "required_elements": round(required_score * weights["required_elements"], 4),
        "forbidden_phrases": round(forbidden_score * weights["forbidden_phrases"], 4),
        "cta_calendar": round(cta_score * weights["cta_calendar"], 4),
        "output_type_or_action": round(action_score * weights["output_type_or_action"], 4),
        "no_overclaiming": round(overclaim_score * weights["no_overclaiming"], 4),
        "tone": round(tone_score * weights["tone"], 4),
    }
    total = round(sum(components.values()), 4)
    passed = total >= 0.8

    return {
        "task_id": task.get("task_id", "unknown"),
        "score": total,
        "pass": passed,
        "scored_field": resolved_field or score_field,
        "decoding_strategy": decoding_strategy,
        "failure_reasons": failure_reasons,
        "components": components,
        "checks": {
            "signal_band": 5 if signal_score == 1.0 else 3 if signal_score == 0.5 else 1,
            "required_band": 5 if required_score == 1.0 else 3 if required_score == 0.5 else 1,
            "forbidden_band": 5 if forbidden_score == 1.0 else 1,
            "cta_band": 5 if cta_score == 1.0 else 3 if cta_score == 0.5 else 1,
            "action_band": 5 if action_score == 1.0 else 1,
            "overclaim_band": 5 if overclaim_score == 1.0 else 1,
            "tone_meta": tone_meta,
        },
    }


def _score_task_schema_v01(task: dict[str, Any], agent_output: Any = None, score_field: str = "rejected") -> dict[str, Any]:
    """
    Scoring path for Tenacious task schema that uses:
    - input
    - scoring_rubric
    - max_score / pass_threshold
    """
    resolved_output, resolved_field, output_error = _resolve_agent_output(task, agent_output, score_field=score_field)
    if output_error:
        return _schema_failure_result(task, output_error, scored_field=resolved_field or score_field)

    rubric = task.get("scoring_rubric", {})
    if not isinstance(rubric, dict) or not rubric:
        return _schema_failure_result(task, "task has no scoring_rubric")

    detail: dict[str, Any] = {}
    total_earned = 0
    total_possible = 0
    output_lower = resolved_output.lower()

    for rubric_name, rubric_item in rubric.items():
        check_type = str(rubric_item.get("check_type", ""))
        points_possible = int(rubric_item.get("points", 1))
        total_possible += points_possible
        try:
            passed, evidence = _run_schema_check(check_type, rubric_item, resolved_output, output_lower, task)
        except Exception as exc:
            passed, evidence = False, f"check_error: {exc}"
        points_earned = points_possible if passed else 0
        total_earned += points_earned
        detail[rubric_name] = {
            "check_type": check_type,
            "passed": passed,
            "points_earned": points_earned,
            "points_possible": points_possible,
            "evidence": evidence,
        }

    max_score = int(task.get("max_score", total_possible))
    pass_threshold = int(task.get("pass_threshold", int(max_score * 0.7)))
    return {
        "task_id": task.get("task_id", "unknown"),
        "score": total_earned,
        "max_score": max_score,
        "pass": total_earned >= pass_threshold,
        "pass_threshold": pass_threshold,
        "pct": round(total_earned / max_score, 3) if max_score else 0.0,
        "scored_field": resolved_field or score_field,
        "detail": detail,
    }


def _run_schema_check(
    check_type: str,
    rubric_item: dict[str, Any],
    output: str,
    output_lower: str,
    task: dict[str, Any],
) -> tuple[bool, str]:
    if check_type == "negative_keyword":
        banned = list(rubric_item.get("banned_phrases", [])) + list(rubric_item.get("banned_phrases_at_zero_bench", []))
        hits = _check_negative_keyword(output, {"banned_phrases": banned})
        return (len(hits) == 0, f"banned_phrase_found: '{hits[0]}'" if hits else f"no_banned_phrases_found (checked {len(banned)} phrases)")
    if check_type == "positive_keyword":
        required = list(rubric_item.get("required_phrases", []))
        for phrase in required:
            if str(phrase).lower() in output_lower:
                return True, f"required_phrase_found: '{phrase}'"
        return False, f"no_required_phrase_found (needed one of: {required})"
    if check_type == "absence":
        absent = list(rubric_item.get("absent_phrases", []))
        for phrase in absent:
            if str(phrase).lower() in output_lower:
                return False, f"prohibited_phrase_found: '{phrase}'"
        return True, f"all_absent_phrases_absent (checked {len(absent)})"
    if check_type == "grounding_check":
        references = list(rubric_item.get("brief_references", []))
        brief = task.get("input", {}).get("hiring_signal_brief", {})
        company_name = brief.get("company", {}).get("name", "")
        if company_name and str(company_name).lower() in output_lower:
            return True, f"company_name_referenced: '{company_name}'"
        for ref in references:
            if str(ref).lower() in output_lower:
                return True, f"signal_reference_found: '{ref}'"
        return False, f"no_signal_references_found (checked: {references})"
    if check_type == "register_check":
        patterns = list(rubric_item.get("required_patterns", []))
        for pattern in patterns:
            if pattern in output:
                return True, f"register_pattern_found: '{pattern}'"
        return False, f"no_register_pattern_found (needed one of: {patterns})"
    return False, f"unknown check_type: {check_type}"


def _schema_failure_result(task: dict[str, Any], reason: str, scored_field: str = "rejected") -> dict[str, Any]:
    return {
        "task_id": task.get("task_id", "unknown"),
        "score": 0,
        "max_score": int(task.get("max_score", 0)),
        "pass": False,
        "pass_threshold": int(task.get("pass_threshold", 1)),
        "pct": 0.0,
        "scored_field": scored_field,
        "detail": {
            "error": {
                "check_type": "none",
                "passed": False,
                "points_earned": 0,
                "points_possible": 0,
                "evidence": reason,
            }
        },
    }


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Tasks file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Tasks file is not valid JSON: {path}") from exc
    if not isinstance(payload, list):
        raise SystemExit(f"Tasks file must contain a JSON list: {path}")
    return [task for task in payload if isinstance(task, dict)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Tenacious-Bench tasks mechanically.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS, help="Path to tasks JSON file.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit over the loaded tasks.")
    parser.add_argument(
        "--score-field",
        choices=["chosen", "rejected"],
        default="rejected",
        help="Which SimPO field to score when explicit output is missing.",
    )
    parser.add_argument(
        "--decoding-strategy",
        choices=list(DECODING_STRATEGIES),
        default="reranker",
        help="Inference-time coupling strategy between generator candidates and judge score.",
    )
    args = parser.parse_args()

    tasks = _load_tasks(args.tasks)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    results = [
        score_task(task, score_field=args.score_field, decoding_strategy=args.decoding_strategy)
        for task in tasks
    ]
    summary = {
        "tasks_path": str(args.tasks),
        "n_tasks": len(results),
        "n_pass": sum(1 for row in results if row["pass"]),
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
