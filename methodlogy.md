# methodology.md

## Path Declaration

**Path B — Preference-tuned judge (SimPO)**

---

## Justification Using Week 10 Evidence

The primary failure mode observed in Week 10 is **inconsistency**, not lack of capability. The agent is able to complete tasks correctly, but fails unpredictably at the final execution step.

Three representative traces illustrate this:

* **Trace ID: task_0_run_a**

  * Outcome: failure
  * Reward: 0.0
  * Failure: `USER_STOP` instead of executing the required action
  * Observation: agent successfully completed all read operations but failed at the final write step

* **Trace ID: task_0_run_b**

  * Outcome: success
  * Reward: 1.0
  * Action: `exchange_delivered_order_items`
  * Observation: identical task, correct final action

* **Trace ID: task_0_run_c**

  * Outcome: failure
  * Reward: 0.0
  * Failure: incorrect or missing final action
  * Observation: same workflow, inconsistent execution

These traces demonstrate that the agent:

* correctly retrieves and processes information
* but fails to reliably execute the final action

This indicates **inconsistency**, where the agent can perform correctly but cannot determine when it is wrong.

This aligns directly with the definition of Path B:

> “The agent gets it right most of the time but cannot tell when it is wrong.”

---

## Why Path B

Path A (generation improvement) would improve average output quality but would not address the **unpredictable failure at the decision boundary**.

Path C (process reward modeling) targets step-by-step reasoning errors, but the observed traces show:

* correct intermediate steps
* failure only at the final decision

Therefore, the failure is not trajectory-based but **evaluation-based**.

Path B introduces a **judge model** that:

* evaluates outputs before execution
* rejects incorrect outputs
* enforces consistency through preference learning

---

## Choice of SimPO

SimPO (Meng et al., 2024) was selected over DPO and ORPO for the following reasons:

* **Reference-free**: does not require a separate reference model, reducing memory usage
* **Efficient on Colab T4**: fits within the 16GB VRAM constraint
* **Stable on small datasets**: suitable for 200–300 high-quality preference pairs
* **Length-normalized reward**: avoids bias toward longer outputs

Additionally, SimPO aligns with the project’s constraint of **low-cost training**.

---

## Preference Leakage Prevention

Following Li et al. (2025), the pipeline enforces strict separation between generation and evaluation:

* **Model A (Qwen)** → generates candidate outputs
* **Model B (DeepSeek)** → evaluates outputs

The same model is never used for both generation and judgment of the same task.

This prevents:

* stylistic bias
* overfitting to generator patterns

---

## Partitioning Protocol

The dataset is split into three partitions:

* **Training set**: 50%
* **Development set**: 30%
* **Held-out set**: 20% (sealed)

### Stratification Strategy

Tasks are distributed across partitions to preserve proportions of failure modes:

* wrong action execution
* early termination (e.g., USER_STOP)
* incorrect tool usage
* missing final decision

This ensures:

* all failure types are represented in each partition
* evaluation is not biased toward a single failure category

---

## Contamination Checks

Three contamination checks were applied before sealing the held-out set:

---

### 1. N-gram Overlap

* Several candidate task pairs showed high textual overlap
* Resolution:

  * near-duplicate tasks were either rewritten or removed

Result:

* no significant overlap between training and held-out sets

---

### 2. Embedding Similarity

* semantic similarity was computed between tasks
* pairs above threshold (0.85 cosine similarity) were flagged

Resolution:

* duplicates removed
* similar tasks diversified via rewriting

Result:

* held-out set contains semantically distinct tasks

---

### 3. Time-Shift Verification

* all tasks referencing external signals were tied to a fixed time window

Resolution:

* ensured all signals are consistent and verifiable

Result:

* no time-based contamination detected

---

### Final Status

All three contamination checks passed.
The held-out partition is clean and suitable for evaluation.

---

## Summary

The methodology is grounded in:

* real Week 10 trace evidence
* a failure-mode-driven path selection
* a structured dataset design
* contamination-resistant evaluation

Path B is therefore the correct and justified choice for improving agent reliability in this setting.
