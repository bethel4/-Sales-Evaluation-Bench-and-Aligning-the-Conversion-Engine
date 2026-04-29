"""
Synthesize Tenacious-Bench Path B candidate tasks and preference examples.

Authoring modes represented here:
1. Probe-derived synthesis: known Week 10 failure modes become rejected examples.
2. Programmatic parameter sweeps: templates vary confidence, segment, AI maturity,
   bench capacity, and layoff status.
3. Multi-LLM synthesis: Qwen generates corrected chosen rewrites for hard probes;
   DeepSeek judges them in judge_filter.py.

This script does not read or train the model. It creates candidate examples that
`build_pairs.py` can combine with trace-derived pairs.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from judge_filter import CHEAP_JUDGE_MODEL, judge_pairwise

SEED = 42
random.seed(SEED)

GENERATOR_MODEL = "qwen/qwen3-30b-a3b"
OUTPUT_DIR = Path("generation_scripts/generated")
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
CHOSEN_REWRITE_PROMPT = (PROMPT_DIR / "chosen_rewrite_prompt.md").read_text()


def make_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


PROBE_SEEDS: List[Dict[str, Any]] = [
    {
        "scenario": {
            "company": {"name": "SmallCo", "employee_count": 25},
            "jobs": {"engineering_roles": 3, "signal_strength": "weak", "confidence": "low"},
            "funding": {"funded": True, "days_ago": 45, "confidence": "high"},
            "ai_maturity": {"score": 1, "confidence": "low"},
            "icp_classification": {"segment": "segment_1", "confidence": 0.82},
        },
        "rejected": "Hi Jordan, I can see SmallCo is aggressively scaling its engineering team — tripling headcount this quarter. With your Series B momentum, now is the time to act.",
        "reason": "over_claiming_aggressive_hiring_at_low_signal",
    },
    {
        "scenario": {
            "company": {"name": "TechCo", "employee_count": 380},
            "bench_summary": {"go": 0, "python": 8, "ml": 3},
            "jobs": {"engineering_roles": 14, "confidence": "high"},
            "ai_maturity": {"score": 1},
            "icp_classification": {"segment": "segment_3", "confidence": 0.88},
        },
        "rejected": "We can provide 3 Go engineers starting next month to help with your infrastructure scaling.",
        "reason": "bench_over_commitment_go_unavailable",
    },
    {
        "scenario": {
            "company": {"name": "DataPulse", "employee_count": 62},
            "jobs": {"engineering_roles": 9, "ai_ml_roles": 5, "confidence": "high"},
            "ai_maturity": {"score": 0, "confidence": "low"},
            "icp_classification": {"segment": "segment_4", "confidence": 0.71},
        },
        "rejected": "DataPulse is clearly ready for ML platform migration. We specialize in exactly the inference infrastructure gap you're facing.",
        "reason": "segment_4_pitched_at_ai_maturity_0",
    },
    {
        "scenario": {
            "company": {"name": "ScaleBase", "employee_count": 920},
            "layoffs": {"had_layoff": True, "days_ago": 45, "percentage_cut": 18},
            "funding": {"funded": True, "days_ago": 60, "confidence": "high"},
            "icp_classification": {"segment": "segment_1", "confidence": 0.65},
        },
        "rejected": "Congratulations on the recent Series B! With fresh funding you're in a great position to scale your engineering team rapidly.",
        "reason": "segment_1_pitch_to_post_layoff_company",
    },
    {
        "scenario": {
            "company": {"name": "Consolety", "employee_count": 45},
            "jobs": {"engineering_roles": 7, "ai_ml_roles": 3, "confidence": "medium"},
            "funding": {"funded": True, "amount_usd": 14000000, "days_ago": 45, "confidence": "high"},
            "ai_maturity": {"score": 2, "confidence": "medium"},
            "icp_classification": {"segment": "segment_1", "confidence": 0.85},
        },
        "rejected": "I completely understand your concerns about offshore teams. I know you probably get a lot of emails like this and I'm sorry to bother you. I don't want to push you.",
        "reason": "tone_drift_overly_apologetic_defensive_reply",
    },
]


def generate_chosen_rewrite(seed: Dict[str, Any], client: OpenAI) -> str:
    """Use Qwen generator to correct a known-bad rejected draft."""
    scenario_json = json.dumps(seed["scenario"], default=str)
    response = client.chat.completions.create(
        model=GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": CHOSEN_REWRITE_PROMPT},
            {
                "role": "user",
                "content": (
                    "This outreach draft FAILED the rubric:\n\n"
                    f"BAD DRAFT: {seed['rejected']}\n"
                    f"REASON IT FAILED: {seed['reason']}\n\n"
                    f"Hiring signal brief: {scenario_json}\n\n"
                    "Write a CORRECTED version that passes all rubric checks."
                ),
            },
        ],
        temperature=0,
        max_tokens=400,
    )
    return response.choices[0].message.content


def synthesize_probe_pairs() -> List[Dict[str, Any]]:
    """Generate chosen rewrites and validate them with a different judge family."""
    client = make_client()
    pairs: List[Dict[str, Any]] = []

    for seed in PROBE_SEEDS:
        chosen = generate_chosen_rewrite(seed, client)
        judgment = judge_pairwise(
            chosen,
            seed["rejected"],
            generator_model=GENERATOR_MODEL,
            judge_model=CHEAP_JUDGE_MODEL,
            client=client,
        )

        if judgment.get("winner") == "A" and float(judgment.get("confidence", 0)) >= 0.70:
            pairs.append(
                {
                    "source": "probe_derived",
                    "probe_reason": seed["reason"],
                    "prompt": f"Write outreach email for: {json.dumps(seed['scenario'], default=str)}",
                    "chosen": chosen,
                    "rejected": seed["rejected"],
                    "judge_confidence": judgment.get("confidence"),
                    "b_violations": judgment.get("b_violations", []),
                    "generator_model": GENERATOR_MODEL,
                    "judge_model": CHEAP_JUDGE_MODEL,
                }
            )
    return pairs


def synthesize_programmatic_pairs() -> List[Dict[str, Any]]:
    """Create deterministic template pairs by sweeping one parameter at a time."""
    pairs: List[Dict[str, Any]] = []
    confidence_levels = ["low", "medium", "high"]

    chosen_map = {
        "low": "Are you finding it harder to hire engineering talent at the pace your roadmap needs?",
        "medium": "It looks like your engineering hiring may be accelerating — you have 7 roles open this week.",
        "high": "Your engineering roles have grown 2.4x since February — 7 positions posted now.",
    }
    rejected_map = {
        "low": "I can see SyntheticCo is aggressively scaling engineering — tripling headcount this quarter.",
        "medium": "SyntheticCo is clearly in a major hiring surge with explosive growth.",
        "high": "Are you perhaps considering hiring? I noticed you might have some open roles.",
    }

    for confidence in confidence_levels:
        brief = {
            "company": {"name": "SyntheticCo", "employee_count": 45},
            "jobs": {
                "engineering_roles": 3 if confidence == "low" else 7,
                "signal_strength": confidence,
                "confidence": confidence,
            },
            "funding": {"funded": True, "days_ago": 60, "confidence": "high"},
            "icp_classification": {"segment": "segment_1", "confidence": 0.82},
        }
        pairs.append(
            {
                "source": "programmatic",
                "scenario": f"confidence={confidence}",
                "prompt": f"Write outreach for: {json.dumps(brief, default=str)}",
                "chosen": f"Hi Jordan, {chosen_map[confidence]} [Tenacious CTA]",
                "rejected": f"Hi Jordan, {rejected_map[confidence]} [Tenacious CTA]",
                "rejection_reason": f"wrong_phrasing_at_{confidence}_confidence",
                "generator_model": "deterministic_template",
                "judge_model": "not_required_template_ground_truth",
            }
        )
    return pairs


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = synthesize_programmatic_pairs()

    # Probe synthesis requires OpenRouter. If no key is available, still emit
    # deterministic programmatic pairs so the pipeline remains runnable.
    if os.environ.get("OPENROUTER_API_KEY"):
        pairs.extend(synthesize_probe_pairs())
    else:
        print("OPENROUTER_API_KEY not set; skipping multi-LLM probe synthesis.")

    out_path = OUTPUT_DIR / "synthetic_pairs.jsonl"
    with out_path.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, default=str) + "\n")

    print(f"Saved {len(pairs)} synthesized pairs to {out_path}")


if __name__ == "__main__":
    main()
