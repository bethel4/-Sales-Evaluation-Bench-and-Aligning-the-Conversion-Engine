# Tenacious-Bench v0.1

Tenacious-Bench evaluates sales-agent behavior that public support benchmarks under-measure: signal grounding, bench truthfulness, thread integrity, timezone-safe scheduling, and trajectory consistency at decisive actions.

## Status

- Declared path: **Path B** (preference-tuned judge/critic).
- Current dataset scaffold: `24` tasks across `train/dev/held_out`.
- Four authoring modes are implemented in generation code: trace-derived, programmatic sweeps, multi-LLM synthesis, hand-authored adversarial.
- In progress: scaling to 200-300 tasks and full training/evaluation loop.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies are pinned in `requirements.txt`.

## Quickstart (Reproduce Headline Number)

Headline number for this scaffold: **24 validated tasks**.

```bash
python generation_scripts/generate_all_tasks.py
python verify_training_data.py
```

The quickstart reproduces split/task artifacts and validates counts/integrity.

## Reproduction Commands

```bash
python generation_scripts/judge_filter.py --tasks tenacious_bench_v0.1/train/tasks.json
python generation_scripts/contamination_check.py
python train_path_b_orpo.py --train-file training_data/simpo_train.jsonl --eval-file training_data/simpo_eval.jsonl
python ablation_harness.py --run all
```

## Public Artifacts

- HuggingFace dataset URL: [https://huggingface.co/datasets/tenacious-labs/tenacious-bench-v0.1](https://huggingface.co/datasets/tenacious-labs/tenacious-bench-v0.1)
- Blog post URL: [https://tenacious-labs.github.io/blog/tenacious-bench-week11](https://tenacious-labs.github.io/blog/tenacious-bench-week11)
- Community engagement URL: [https://github.com/tenacious-labs/tenacious-bench/discussions/1](https://github.com/tenacious-labs/tenacious-bench/discussions/1)

Path B does not require publishing a trained model artifact in this checkpoint.

## Core Documents

- [Audit Memo](audit_memo.md)
- [Methodology](methodology.md)
- [Methodology Rationale](methodology_rationale.md)
- [Datasheet](datasheet.md)
- [Inter-Rater Agreement](inter_rater_agreement.md)
- [Report](report.md)

## License and Attribution

- License: `CC BY-NC 4.0` (see `LICENSE`).
- Attribution: benchmark design and scaffolding for Tenacious Conversion Engine evaluation.
- Credits: Week 10 probe/trace analysis, generation pipeline implementation, and judge-filter methodology documented in this repo.
