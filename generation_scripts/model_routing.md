# Model Routing for Path B

This repo chooses **Path B: preference-tuned judge / critic using SimPO**. The generation pipeline is intentionally split into generator-facing and judge-facing roles so the same model is never assigned to both jobs for the same example.

## Routing Tiers

| Tier | Label in source | Intended use |
|---|---|---|
| Cheap judge tier | `deepseek/deepseek-chat` | bulk filtering and pairwise comparison |
| Eval-tier judge | `anthropic/claude-sonnet-4.6` | small calibration sample only |
| Generator family placeholder | `qwen/qwen3-30b-a3b` | future rewrite / synthesis generation |

## Leakage Rule

- generator model must differ from judge model
- the scripts fail fast if `generator_model == judge_model`
- trace-derived rejected examples are treated as coming from the Week 10 system, not the judge

## Why Two Judge Tiers

- cheap tier keeps bulk filtering affordable
- eval-tier is reserved for spot-checking a small calibration sample
- this separation makes the cost and trust tradeoff visible in source code

## Duplicate Policy

The judge layer also carries a pairwise duplicate comparison helper. It does not rely on embeddings in this interim repo; instead it uses normalized token overlap so the logic stays grader-readable.
