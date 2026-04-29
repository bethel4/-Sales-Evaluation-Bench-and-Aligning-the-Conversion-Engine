# Tenacious-Bench v0.1

Tenacious-Bench is a small sales-evaluation benchmark scaffold for auditing and improving the Conversion Engine on Tenacious-specific failure modes that public support benchmarks do not grade well. The benchmark focuses on signal-grounded outreach, bench-capacity truthfulness, thread integrity, timezone scheduling, claim safety, escalation judgment, output-type control, and final-action selection.

## Current Status

- Complete at interim:
  - 24-task benchmark scaffold with `train/dev/held_out` splits
  - root schema, scoring evaluator, contamination check, datasheet, methodology, report, and audit memo
  - Path B generation pipeline scaffold for preference pairs and SimPO-format export
- In progress:
  - scaling from 24 tasks to the target 200-300 tasks
  - second-pass human agreement review
  - full SimPO training run and calibration loop

Current scaffold contains 24 validated tasks; target is 200–300 tasks.

## Setup

Environment requirements:
- Python `3.11+`
- Standard library is sufficient for the current scaffold scripts
- Optional future live generation/judging can add API-backed model clients later, but the committed commands below do not require them

Install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## Commands

Run the scoring evaluator on the default sample split:

```bash
python scoring_evaluator.py
```

Run the scoring evaluator on a small explicit sample:

```bash
python scoring_evaluator.py --tasks tenacious_bench_v0.1/dev/tasks.json --limit 3
```

Generate deterministic candidate tasks:

```bash
python generation_scripts/synthesize_tasks.py
```

Build preference pairs:

```bash
python generation_scripts/build_pairs.py
```

Convert preference pairs into SimPO train/eval JSONL:

```bash
python generation_scripts/convert_to_simpo_format.py
```

## Major Artifacts

- [Audit Memo](audit_memo.md)
- [Methodology](methodology.md)
- [Datasheet](datasheet.md)
- [Inter-Rater Agreement](inter_rater_agreement.md)
- [Contamination Check](contamination_check.json)
- [Report](report.md)
- [SimPO Memo](synthesis_memos/simpo_memo.md)
- [Preference Leakage Memo](synthesis_memos/preference_leakage_memo.md)

## Directory Structure

- `tenacious_bench_v0.1/`: benchmark task splits for `train`, `dev`, and `held_out`
- `generation_scripts/`: deterministic generation, filtering, pairing, and SimPO-format export scripts
- `training_data/`: generated preference-pair outputs and SimPO train/eval files
- `synthesis_memos/`: short path-selection and preference-leakage rationale memos
- `trace_log.jsonl`: local Week 10/11 trace evidence used for trace-derived tasks and pair building
- `probe_library.md`: Week 10 probe inventory and intended failure-mode coverage
- `failure_taxonomy.md`: short failure-mode reference used in report framing

## Forward Plan

- Act IV: expand from the 24-task scaffold to a broader 200-300 task pool with the same schema
- Act V: train the Path B judge / critic with SimPO on preference pairs and monitor early loss behavior
- Act VI: recalibrate rubric weights, run human agreement review, and seal the held-out split for final evaluation
