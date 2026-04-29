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

- trace-derived tasks anchor the benchmark in observed Week 10 failures
- programmatic tasks sweep known rule-based edge cases
- multi-LLM synthesis placeholder-ready tasks reserve space for later scaled authoring
- hand-authored adversarial tasks target brittle failure boundaries

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

## Maintenance

Maintenance plan:

- expand from 24 tasks toward 200–300 tasks
- add second-pass human review
- revise rubric weights after judge calibration
- freeze a cleaner held-out set before final evaluation

Version `v0.1` should be treated as an interim scaffold, not a final benchmark release.
