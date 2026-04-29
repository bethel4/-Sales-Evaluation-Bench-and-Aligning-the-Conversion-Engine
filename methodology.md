# Methodology

## Path Declaration

This repository follows **Path B: preference-tuned judge / critic using SimPO**.

## Why Path B

Week 10 evidence points to inconsistency rather than broad incapability. The agent often completes read and authentication steps, then fails at the final action or output decision. The motivating placeholders carried into this scaffold are:

- `trace_task0_failed_exchange`
- `trace_task0_passed_exchange`
- `trace_historical_failed_return`

Those placeholders are grounded in concrete Week 10 and carry-forward trace IDs:

- `trace_task0_failed_exchange` maps to `tau2-real-0-1777378070`, where auth and read steps complete but `exchange_delivered_order_items` is still missed and the run ends with `db_match=false`
- `trace_task0_passed_exchange` maps to historical trace `4e89d9d4-a3dc-4f38-8127-08f9b352bf4d`, which resolves the same exchange family correctly
- `trace_historical_failed_return` maps to historical trace `95e55d0f-513e-431f-a482-41f317cb0564`, where the return path fails at the decisive action

The broader pattern is reinforced by historical traces `754d3a9b-b614-468e-9748-abf2ecc19a99` and `3a1e8079-124c-4992-8077-f69fd4e11f77`: the system often gathers the right information but still fails at the decisive write step, recovery decision, or output choice. That is more naturally addressed by a judge / critic than by pure generation cleanup.

## Why SimPO

We use SimPO as the declared training objective because it is a reference-free preference-optimization method that fits the current low-cost setup better than a heavier DPO pipeline. The choice is grounded in:

- Meng, Xia, and Chen (2024), **SimPO: Simple Preference Optimization with a Reference-Free Reward**
- Li et al. (2025), **Preference Leakage: A Contamination Problem in LLM-as-a-Judge**
- Kim et al. (2024), **Prometheus 2: An Open-Source Language Model Specialized in Evaluating Other Language Models**

The practical reason is simple: this repo currently contains a small but structured preference scaffold, not a large full-SFT corpus. A lightweight preference objective is the most direct next step.

## Dataset Split Policy

Target split policy at full scale:

- train: `50%`
- dev: `30%`
- held_out: `20%`

Current partial scaffold:

- train: `12`
- dev: `7`
- held_out: `5`

This is intentionally smaller than the target. Current scaffold contains 24 validated tasks; target is 200–300 tasks.

## Current Authoring Modes

The 24-task scaffold is partitioned across four source modes:

- `8` trace-derived
- `7` programmatic
- `5` multi-LLM synthesis placeholder-ready
- `4` hand-authored adversarial

This mix lets the benchmark cover observed failures now while leaving room to scale later.

## Contamination Checks Run So Far

Current contamination checking is small-dataset and mechanical, not fully frontier-grade yet. The committed `contamination_check.json` records:

- task-id uniqueness
- exact duplicate output detection
- normalized prompt overlap checks
- partition leakage checks over source mode and task ids
- trace reuse notes for trace-derived tasks

Current outputs on the 24-task scaffold are:

- `24 / 24` task ids unique
- `0` exact duplicate `candidate_output` strings flagged
- `0` partition-leakage cases flagged across `train/dev/held_out`
- `24` tasks reviewed under normalized prompt-overlap checks
- `5 / 5` held-out tasks marked as sealed for evaluation use, not intended training use

Disposition of flagged items:

- exact duplicates: none removed because none were found
- partition leakage: none removed because none were found
- trace reuse: retained only when rewritten as benchmark abstractions rather than raw transcript copies

These checks are honest but partial. The next scaling step should add embedding-based duplicate review and a wider held-out sealing process.

## Preference Leakage Policy

Path B requires judge separation. This repo therefore documents two judge tiers in source:

- cheap-model judge tier for bulk filtering
- eval-tier judge tier for small calibration samples only

The scripts explicitly reject same-model generate-and-judge assignments. That policy is more important than the current placeholder models themselves, because leakage prevention is a structural requirement of the path.

## Honest Current Status

What is complete:

- task schema
- 24-task benchmark scaffold
- deterministic generation scripts
- pair-building scaffold
- SimPO-format export

What is not complete yet:

- full 200-300 task authoring pass
- full human agreement pass
- actual SimPO training curve and post-training evaluation

## Next Scaling Move

The next move is to scale the task inventory while keeping the same failure dimensions, then train a small judge with SimPO and stop early if the loss does not improve quickly enough to justify more compute.
