#!/usr/bin/env python3
"""
Contamination checks for Tenacious-Bench splits.

Implements three gates:
1) n-gram overlap check with strict <8-gram requirement on input fields.
2) embedding-similarity check with cosine threshold < 0.85.
3) time-shift verification for public-data references with provenance windows.

Covers:
- held_out vs train
- held_out vs dev
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

NGRAM_N = 7
EMBEDDING_COSINE_THRESHOLD = 0.85
TIME_SHIFT_DAYS = 30


@dataclass
class CheckConfig:
    train_path: Path
    dev_path: Path
    held_out_path: Path
    report_path: Path


def _read_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in payload if isinstance(row, dict)]


def _flatten_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for value in obj.values():
            out.extend(_flatten_strings(value))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(_flatten_strings(value))
    elif isinstance(obj, (str, int, float, bool)):
        out.append(str(obj))
    return out


def task_input_text(task: dict[str, Any]) -> str:
    inputs = task.get("inputs") or task.get("input") or {}
    pieces = _flatten_strings(inputs)
    text = " ".join(pieces).lower()
    return re.sub(r"\s+", " ", text).strip()


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = re.findall(r"\b\w+\b", text)
    return set(tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1)))


def ngram_overlap_violations(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for a in left:
        a_text = task_input_text(a)
        a_ngrams = ngrams(a_text, NGRAM_N)
        for b in right:
            b_text = task_input_text(b)
            b_ngrams = ngrams(b_text, NGRAM_N)
            shared = a_ngrams & b_ngrams
            if shared:
                violations.append(
                    {
                        "left_task_id": a.get("task_id", "unknown"),
                        "right_task_id": b.get("task_id", "unknown"),
                        "shared_7gram_count": len(shared),
                        "sample_shared_7gram": " ".join(next(iter(shared))),
                    }
                )
    return violations


def cheap_embedding(text: str, dims: int = 64) -> list[float]:
    # Deterministic hash embedding for cheap similarity screening.
    vec = [0.0] * dims
    for tok in re.findall(r"\b\w+\b", text.lower()):
        digest = hashlib.sha256(tok.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % dims
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def embedding_violations(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    left_cache = {str(t.get("task_id", "")): cheap_embedding(task_input_text(t)) for t in left}
    right_cache = {str(t.get("task_id", "")): cheap_embedding(task_input_text(t)) for t in right}
    for a in left:
        aid = str(a.get("task_id", "unknown"))
        for b in right:
            bid = str(b.get("task_id", "unknown"))
            sim = cosine(left_cache[aid], right_cache[bid])
            if sim >= EMBEDDING_COSINE_THRESHOLD:
                violations.append(
                    {
                        "left_task_id": aid,
                        "right_task_id": bid,
                        "cosine_similarity": round(sim, 4),
                    }
                )
    return violations


def _extract_public_reference_date(task: dict[str, Any]) -> datetime | None:
    hay = json.dumps(task, default=str)
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", hay)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d")
    except ValueError:
        return None


def time_shift_violations(held: list[dict[str, Any]], other: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for h in held:
        h_date = _extract_public_reference_date(h)
        if h_date is None:
            continue
        for o in other:
            o_date = _extract_public_reference_date(o)
            if o_date is None:
                continue
            delta_days = abs((h_date - o_date).days)
            if delta_days < TIME_SHIFT_DAYS:
                violations.append(
                    {
                        "held_task_id": h.get("task_id", "unknown"),
                        "other_task_id": o.get("task_id", "unknown"),
                        "delta_days": delta_days,
                        "required_min_delta_days": TIME_SHIFT_DAYS,
                    }
                )
    return violations


def run_checks(held: list[dict[str, Any]], other: list[dict[str, Any]], pair_name: str) -> dict[str, Any]:
    ngram_hits = ngram_overlap_violations(held, other)
    embed_hits = embedding_violations(held, other)
    time_hits = time_shift_violations(held, other)
    return {
        "pair": pair_name,
        "thresholds": {
            "n_gram_overlap_rule": f"no shared {NGRAM_N}-grams on input fields",
            "embedding_cosine_lt": EMBEDDING_COSINE_THRESHOLD,
            "time_shift_min_days": TIME_SHIFT_DAYS,
        },
        "counts": {
            "left_n": len(held),
            "right_n": len(other),
            "n_gram_violations": len(ngram_hits),
            "embedding_violations": len(embed_hits),
            "time_shift_violations": len(time_hits),
        },
        "status": "PASS" if not (ngram_hits or embed_hits or time_hits) else "REVIEW",
        "violations": {
            "n_gram": ngram_hits[:100],
            "embedding": embed_hits[:100],
            "time_shift": time_hits[:100],
        },
    }


def parse_args() -> CheckConfig:
    parser = argparse.ArgumentParser(description="Run contamination checks across Tenacious-Bench splits.")
    parser.add_argument("--train", type=Path, default=Path("tenacious_bench_v0.1/train/tasks.json"))
    parser.add_argument("--dev", type=Path, default=Path("tenacious_bench_v0.1/dev/tasks.json"))
    parser.add_argument("--held-out", type=Path, default=Path("tenacious_bench_v0.1/held_out/held_out_tasks.json"))
    parser.add_argument("--report", type=Path, default=Path("contamination_check.json"))
    args = parser.parse_args()
    return CheckConfig(
        train_path=args.train,
        dev_path=args.dev,
        held_out_path=args.held_out,
        report_path=args.report,
    )


def main() -> int:
    cfg = parse_args()
    train = _read_json(cfg.train_path)
    dev = _read_json(cfg.dev_path)
    held = _read_json(cfg.held_out_path)

    report = {
        "script": "generation_scripts/contamination_check.py",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "coverage": ["held_out_vs_train", "held_out_vs_dev"],
        "checks": [
            run_checks(held, train, "held_out_vs_train"),
            run_checks(held, dev, "held_out_vs_dev"),
        ],
    }
    cfg.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
