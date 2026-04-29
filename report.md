# Report: Tenacious-Bench v0.1 Interim Scaffold

## Status

Current scaffold contains 24 validated tasks; target is 200–300 tasks. This repo is an interim Week 11 deliverable for Path B, not a final benchmark release.

## Bench Composition

### By Source Mode

| Source mode | Train | Dev | Held-out | Total |
|---|---:|---:|---:|---:|
| trace_derived | 4 | 3 | 1 | 8 |
| programmatic | 3 | 2 | 2 | 7 |
| multi_llm_synthesis | 3 | 1 | 1 | 5 |
| hand_authored | 2 | 1 | 1 | 4 |

### By Failure Dimension

| Failure dimension | Train | Dev | Held-out | Total |
|---|---:|---:|---:|---:|
| signal_grounding | 2 | 0 | 1 | 3 |
| bench_truthfulness | 2 | 0 | 1 | 3 |
| thread_integrity | 3 | 1 | 0 | 4 |
| timezone_scheduling | 0 | 1 | 2 | 3 |
| claim_safety | 1 | 3 | 0 | 4 |
| final_action_selection | 3 | 0 | 0 | 3 |
| escalation_judgment | 1 | 1 | 0 | 2 |
| output_type_control | 0 | 1 | 1 | 2 |

## Inter-Rater Agreement

Current calibration numbers are preliminary:

- double-reviewed sample: `6` tasks
- exact accept/reject agreement: `0.83`
- Cohen's kappa: `0.67`

These numbers are documented more fully in [inter_rater_agreement.md](inter_rater_agreement.md).

## Worked Examples

### Trace-Derived Example

- task id: `TB-TR-003`
- source mode: `trace_derived`
- dimension: `final_action_selection`
- why it matters: derived from the “reads succeed, decisive write misses” pattern; good for Path B because the model looks plausible until the final tool decision

### Programmatic Example

- task id: `TB-PG-002`
- source mode: `programmatic`
- dimension: `bench_truthfulness`
- why it matters: forces a mechanical no-overcommit decision when `bench_summary.go = 0`

### Adversarial Example

- task id: `TB-HA-003`
- source mode: `hand_authored`
- dimension: `escalation_judgment`
- why it matters: forces the evaluator to prefer policy-safe transfer over an eager but wrong write action

## What Works

- schema and split are explicit
- scorer is readable and mechanically decomposed
- generation scripts show Path B structure clearly
- trace-derived tasks connect the scaffold to real evidence

## What Is Weak

- only 24 tasks so far
- inter-rater review is partial
- multi-LLM synthesis tasks are placeholder-ready, not fully scaled
- no actual SimPO training curve is committed yet
- one local trace (`task 45`) surfaced provider-auth fragility rather than pure model behavior

## Days 4–7 Plan

1. Expand to the next 50-task milestone while preserving source-mode balance.
2. Double-review the held-out split and refresh contamination checks.
3. Build a larger preference-pair file from traces plus rewritten chosen outputs.
4. Start SimPO and inspect loss, eval preference accuracy, and obvious overfitting signals.

## Kill Criterion

If SimPO training loss does not decrease by step `20`, or if dev preference behavior is flat while failure reasons stay unchanged, stop the run and reconsider the pair quality before spending more compute.
