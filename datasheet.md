# Datasheet for Tenacious-Bench v0.1

## Telescopic Summary

Tenacious-Bench v0.1 is a small benchmark scaffold for evaluating a B2B sales conversion agent on failure modes that public retail/support benchmarks do not grade well. Current scaffold contains 24 validated tasks; target is 200–300 tasks.

## Periscopic Overview

The dataset is designed for Week 11 Path B work: a preference-tuned judge / critic using SimPO. The current split is intentionally partial:

- train: 12 tasks
- dev: 7 tasks
- held_out: 5 tasks

Source composition:

- 8 trace-derived
- 7 programmatic
- 5 multi-LLM synthesis placeholder-ready
- 4 hand-authored adversarial

Failure dimensions covered:

- signal grounding
- bench truthfulness
- thread integrity
- timezone scheduling
- claim safety
- final action selection
- escalation judgment
- output type control

## Microscopic Schema / Sample Detail

Every task contains:

- `task_id`
- `source_mode`
- `failure_dimension`
- `difficulty`
- `inputs`
- `candidate_output`
- `ground_truth`
- `rubric`

The schema is committed in [schema.json](schema.json). The benchmark split files are committed under [tenacious_bench_v0.1](tenacious_bench_v0.1/).

## Motivation

The benchmark exists because Week 10 evidence showed a gap between public benchmark success and Tenacious readiness. The agent often completed lookup and authentication steps correctly but remained inconsistent on final action choice, overclaiming control, tone under skepticism, and sales-specific scheduling or thread-memory constraints.

## Composition

The current scaffold contains 24 tasks. It is not yet a full benchmark release. The current scaffold is balanced enough to demonstrate evaluator logic and generation-pipeline structure, but not large enough to support strong statistical claims.

Composition by source mode:

- trace-derived: benchmark abstractions of real Week 10/11 traces, especially final-action misses after successful reads or auth
- programmatic: mechanically generated edge cases that vary signal strength, bench coverage, timezone constraints, and output-type expectations
- multi-LLM synthesis placeholder-ready: reserved slots for later model-assisted authoring, committed now as human-readable placeholders with the final schema
- hand-authored adversarial: manually written traps aimed at escalation judgment, unsupported certainty, and thread-discipline failures

Current composition table:

| Slice | Count |
|---|---:|
| total tasks | 24 |
| train | 12 |
| dev | 7 |
| held_out | 5 |
| trace_derived | 8 |
| programmatic | 7 |
| multi_llm_synthesis | 5 |
| hand_authored | 4 |

Current failure-dimension counts:

| Failure dimension | Count |
|---|---:|
| thread_integrity | 4 |
| claim_safety | 4 |
| final_action_selection | 3 |
| signal_grounding | 3 |
| bench_truthfulness | 3 |
| timezone_scheduling | 3 |
| escalation_judgment | 2 |
| output_type_control | 2 |

Target composition at full scale:

| Target scaffold goal | Approximate count at 200 | Approximate count at 300 |
|---|---:|---:|
| train (50%) | 100 | 150 |
| dev (30%) | 60 | 90 |
| held_out (20%) | 40 | 60 |
| trace-derived (about one-third) | 64-72 | 96-108 |
| programmatic (about one-third) | 56-64 | 84-96 |
| synthesis + adversarial remainder | 64-80 | 96-120 |

The exact 200–300-task mix is not fixed yet, but the intended scaling rule is to preserve all eight failure dimensions and avoid letting any one source mode dominate the corpus.

## Collection

Collection was done from four channels:

1. trace review over local and historical Week 10 / Week 11 logs
2. programmatic authoring from recurring rule failures
3. placeholder-ready synthesis prompts for later model-assisted expansion
4. manual adversarial writing for grader-visible edge cases

No claim is made that this dataset is complete. Current scaffold contains 24 validated tasks; target is 200–300 tasks.

## Preprocessing

Preprocessing steps for the current scaffold:

- normalized task structure to one schema
- added explicit rubrics with mechanical checks
- partitioned into train/dev/held_out
- ran simple contamination checks: id uniqueness, exact duplicate output detection, normalized prompt overlap, and split leakage review

The Tenacious Style Guide v2 is used as the rubric source of truth. Its banned phrases are loaded into scoring_evaluator.py, its good/bad drafts seed generation modes, and its tone markers define preference labels for SimPO.

Structured bans and tone-marker metadata live in [data/style_guide_v2.json](data/style_guide_v2.json); the scorer prefers this file automatically over the legacy flat list.

The generation pipeline enforces a strict model-rotation policy (Qwen for generation, DeepSeek for judgment) to prevent preference leakage (Li et al., 2025).

The current preprocessing is sufficient for an interim scaffold but not the final release standard.

## Uses

Intended uses:

- develop and inspect a mechanical evaluator
- generate chosen/rejected preference pairs
- convert pairs to SimPO train/eval format
- run small calibration reviews on sales-domain failure modes

Not intended uses:

- claiming final production readiness
- claiming broad coverage of all Tenacious sales interactions
- reporting final leaderboard-style metrics

## Distribution

The scaffold is distributed inside this repo as JSON and JSONL artifacts. No external package index or hosted benchmark release is claimed yet. The committed format is intended for direct review by program evaluators and local script execution.

Current license position:

- proposed license: `CC BY-NC 4.0` for benchmark tasks and documentation
- rationale: the dataset is meant to be inspectable, reusable for academic or portfolio evaluation, and attributable, while avoiding unreviewed commercial reuse before the larger 200–300-task version is sealed

This is an explicit interim license recommendation for `v0.1`; the repo should add a top-level `LICENSE` file before the final benchmark release.

## Maintenance

Maintenance plan:

- expand from 24 tasks toward 200–300 tasks
- add second-pass human review
- revise rubric weights after judge calibration
- freeze a cleaner held-out set before final evaluation

Version `v0.1` should be treated as an interim scaffold, not a final benchmark release.
