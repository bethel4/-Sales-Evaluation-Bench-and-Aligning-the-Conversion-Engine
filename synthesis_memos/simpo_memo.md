# synthesis_memos/simpo_memo.md

## Paper

SimPO: Simple Preference Optimization with a Reference-Free Reward
(Meng, Xia, and Chen, NeurIPS 2024)

---

## Design Choice

The paper proposes that **reference-free preference optimization (SimPO)** can effectively replace DPO by computing rewards directly from sequence log-probabilities, reducing complexity and computational cost.

---

## Disagreement

I partially disagree with the implicit assumption that **simplifying the training objective alone is sufficient to improve model behavior**.

---

## Evidence from My Project

In my Week 10 traces, the primary failure mode was not poor generation quality but **inconsistency in final decision execution**. For example:

* The agent successfully retrieved all required information (user ID, order details, product details)
* But failed at the final step by producing `USER_STOP` instead of executing `exchange_delivered_order_items`
* On identical or near-identical inputs, the same agent sometimes succeeded and sometimes failed

This demonstrates that the issue is not:

* lack of data
* or poor optimization

but rather:

> the agent cannot reliably distinguish correct vs incorrect outputs

---

## Why This Matters

SimPO optimizes preference learning, but **it assumes that preference data already captures the correct signal**.

In my case:

* If preference pairs are generated purely synthetically
* Or without grounding in real trace failures

then the model may learn:

* stylistic improvements
* but not decision reliability

---

## My Adjustment

Instead of relying purely on synthetic preference data, I:

* derived pairs from **real trace failures (chosen vs rejected)**
* used LLM synthesis only to expand edge cases
* applied a judge filter to enforce quality

This aligns SimPO training with **actual failure modes**, not just general preferences.

---

## Conclusion

While SimPO is an efficient and appropriate training method, I disagree that **algorithmic simplification alone drives improvement**.

In practice, the dominant factor is:

> the quality and grounding of the preference data

SimPO works well in my pipeline **only because it is paired with trace-derived, failure-specific training data**, not because of the optimization method alone.
