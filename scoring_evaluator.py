#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# Calibration convention used by this scaffold:
# - band 5 / score 1.0: dimension is cleanly satisfied
# - band 3 / score 0.5: partially satisfied or borderline
# - band 1 / score 0.0: clear failure
# The final numerical score is a weighted average over these normalized bands.

DEFAULT_TASKS = Path("tenacious_bench_v0.1/dev/tasks.json")
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


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _contains_any(text: str, phrases: list[str]) -> bool:
    lowered = _normalize(text)
    return any(_normalize(phrase) in lowered for phrase in phrases)


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
    hits = [phrase for phrase in forbidden if _contains_any(output, [phrase])]
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


def score_task(task: dict[str, Any], agent_output: str) -> dict[str, Any]:
    """
    Mechanically score one task/output pair.

    Returns:
    - numerical score in [0.0, 1.0]
    - pass/fail using a default threshold of 0.8
    - visible failure reasons and component scores for debugging
    """
    try:
        if not isinstance(task, dict):
            raise TypeError("task must be a dict")
        if not isinstance(agent_output, str):
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

    signal_score, signal_reason = _signal_reference_score(task, agent_output)
    if signal_reason:
        failure_reasons.append(signal_reason)

    required_score, missing_required = _required_substring_score(task, agent_output)
    if missing_required:
        failure_reasons.append(f"missing required elements: {missing_required}")

    forbidden_score, forbidden_hits = _forbidden_phrase_score(task, agent_output)
    if forbidden_hits:
        failure_reasons.append(f"forbidden phrases: {forbidden_hits}")

    cta_score, cta_reason = _calendar_cta_score(task, agent_output)
    if cta_reason:
        failure_reasons.append(cta_reason)

    action_score, action_reason = _output_type_or_action_score(task, agent_output)
    if action_reason:
        failure_reasons.append(action_reason)

    overclaim_score, overclaim_hits = _overclaiming_score(task, agent_output)
    if overclaim_hits:
        failure_reasons.append(f"overclaiming: {overclaim_hits}")

    tone_score, tone_meta = _tone_score(agent_output, float(rubric.get("min_tone_score", 0.7)))
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
    args = parser.parse_args()

    tasks = _load_tasks(args.tasks)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    results = [score_task(task, str(task.get("candidate_output", ""))) for task in tasks]
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
