# synthesis_memos/generation_design_memo.md

## Paper / Design Claim

A common benchmark-construction claim is that strict stratification by failure dimension is always preferable because it preserves balanced evaluation slices.

## Disagreement

I disagree with applying strict per-dimension stratification as an absolute rule for every split.

## Why

- Strict stratification can lock in synthetic artifacts if one mode overproduces similar templates.
- A small amount of controlled global reshuffling can improve distribution realism while still preserving overall dimension coverage.
- In practice, model behavior is stressed by mixed-context batches, not perfectly balanced minibatches.

## Evidence from this repo

- The generator combines trace-derived, programmatic, synthesis, and hand-authored tasks.
- If split assignment is fully deterministic per dimension, lexical overlap may remain high in held-out despite no exact duplicates.
- Contamination checks improve when we allow bounded rebalancing toward target split counts after dimension-wise initialization.

## Position

Dimension-aware splitting is necessary, but rigid stratification is not sufficient for robust evaluation. The better design is:

1. initialize with dimension-aware buckets
2. rebalance to exact split targets
3. run contamination checks and adjust if needed

This keeps benchmark coverage while reducing structural artifacts.
