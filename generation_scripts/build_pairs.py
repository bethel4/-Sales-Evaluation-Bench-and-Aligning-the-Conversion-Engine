"""
Build Path B preference pairs for SimPO/ORPO/DPO training.

Inputs:
- eval/trace_log.jsonl: real Week 10 traces.
- generation_scripts/generated/synthetic_pairs.jsonl: synthesized/probe/programmatic pairs.

Output:
- training_data/preference_pairs.jsonl

The output format is:
{
  "prompt": "task input / prospect brief",
  "chosen": "good output",
  "rejected": "bad output",
  "source": "trace_derived | probe_derived | programmatic | multi_llm_synthesis"
}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SEED = 42
random.seed(SEED)

TRACE_PATH = Path("eval/trace_log.jsonl")
SYNTHETIC_PATH = Path("generation_scripts/generated/synthetic_pairs.jsonl")
OUTPUT_PATH = Path("training_data/preference_pairs.jsonl")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL robustly, skipping empty lines."""
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
    return rows


def trace_passed(trace: Dict[str, Any]) -> bool:
    """Normalize success flags across common trace shapes."""
    if "passed" in trace:
        return bool(trace["passed"])
    if "reward" in trace:
        return float(trace.get("reward", 0)) >= 1.0
    if "score" in trace:
        return float(trace.get("score", 0)) >= 1.0
    return False


def extract_prompt(trace: Dict[str, Any]) -> str:
    """Extract a readable prompt/input from a trace with fallbacks."""
    for key in ("input", "prompt", "user_request", "task", "task_input"):
        if trace.get(key):
            value = trace[key]
            return value if isinstance(value, str) else json.dumps(value, default=str)
    return json.dumps({"task_id": trace.get("task_id"), "trace_id": trace.get("trace_id")}, default=str)


def extract_output(trace: Dict[str, Any]) -> str:
    """Extract output/action trajectory from a trace with fallbacks."""
    for key in ("output", "final_output", "agent_output", "response"):
        if trace.get(key):
            value = trace[key]
            return value if isinstance(value, str) else json.dumps(value, default=str)

    # If the trace stores actions but no final text, serialize the action list.
    for key in ("actions", "steps", "trajectory"):
        if trace.get(key):
            return json.dumps(trace[key], default=str)

    return json.dumps(trace, default=str)


def build_trace_pairs(traces: List[Dict[str, Any]], limit_failed: int = 40) -> List[Dict[str, Any]]:
    """Pair failed traces with passed traces of the same task when available."""
    passed = [t for t in traces if trace_passed(t)]
    failed = [t for t in traces if not trace_passed(t)]

    pairs: List[Dict[str, Any]] = []
    if not passed or not failed:
        return pairs

    for failed_trace in failed[:limit_failed]:
        task_id = failed_trace.get("task_id")
        matched = next((p for p in passed if p.get("task_id") == task_id), None)
        if matched is None:
            matched = random.choice(passed)

        pairs.append(
            {
                "source": "trace_derived",
                "task_id": task_id,
                "trace_id_rejected": failed_trace.get("trace_id") or failed_trace.get("id"),
                "trace_id_chosen": matched.get("trace_id") or matched.get("id"),
                "prompt": extract_prompt(failed_trace),
                "chosen": extract_output(matched),
                "rejected": extract_output(failed_trace),
                "rejection_reason": "failed_week10_trace",
            }
        )
    return pairs


def validate_pair(pair: Dict[str, Any]) -> Optional[str]:
    """Return an error string if malformed, otherwise None."""
    for key in ("prompt", "chosen", "rejected"):
        if not pair.get(key) or not isinstance(pair.get(key), str):
            return f"missing_or_invalid_{key}"
    if pair["chosen"].strip() == pair["rejected"].strip():
        return "chosen_equals_rejected"
    return None


def deduplicate_pairs(pairs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simple exact dedupe over prompt/chosen/rejected."""
    seen = set()
    unique: List[Dict[str, Any]] = []
    for pair in pairs:
        key = (pair.get("prompt", ""), pair.get("chosen", ""), pair.get("rejected", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)
    return unique


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    traces = read_jsonl(TRACE_PATH)
    trace_pairs = build_trace_pairs(traces)
    synthetic_pairs = read_jsonl(SYNTHETIC_PATH)

    all_pairs = deduplicate_pairs([*trace_pairs, *synthetic_pairs])
    valid_pairs: List[Dict[str, Any]] = []
    rejected_count = 0

    for pair in all_pairs:
        error = validate_pair(pair)
        if error:
            rejected_count += 1
            continue
        pair.setdefault("seed", SEED)
        valid_pairs.append(pair)

    with OUTPUT_PATH.open("w") as f:
        for pair in valid_pairs:
            f.write(json.dumps(pair, default=str) + "\n")

    print(f"Trace-derived pairs: {len(trace_pairs)}")
    print(f"Synthetic/programmatic/probe pairs: {len(synthetic_pairs)}")
    print(f"Valid training pairs written: {len(valid_pairs)}")
    print(f"Malformed/duplicate-equivalent pairs rejected: {rejected_count}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
