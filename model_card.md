# Tenacious Judge Model Card

## Intended Use

This judge is evaluated and intended to be deployed with an explicit inference-time
coupling strategy between generator candidates and scalar judge scores.

- `decoding_strategy`: `best_of_n`
- `n_candidates`: `3`
- `threshold`: `null` (not used for best-of-N)
- `max_tries`: `null` (not used for best-of-N)

Downstream users should not swap to rejection sampling or single-candidate
reranking without rerunning held-out evaluation, because quality, latency, and
cost profiles change materially by strategy.

## Reproducibility Note

When reporting Delta A / Delta B style metrics on Tenacious-Bench held-out,
include `decoding_strategy`, `n_candidates`, and `rejection_threshold` in every
ablation row so colleagues can reproduce results from the same coupling policy.
