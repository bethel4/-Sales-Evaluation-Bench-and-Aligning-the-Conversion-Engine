# Methodology Rationale (Path B)

## Declared Path and Failure Mode Match

Chosen path: **Path B** (preference-tuned judge / critic).

Primary Week 10 failure mode: **trajectory inconsistency at decisive steps**, not broad inability to do retrieval/auth. This aligns with Path B, which directly optimizes comparative preferences on action quality and reasoning quality.

## Week 10 Trace Evidence

Key traces grounding the decision:

- `tau2-real-0-1777378070`: reaches auth/read steps, then misses decisive exchange write.
- `95e55d0f-513e-431f-a482-41f317cb0564`: return trajectory fails at completion step.
- `3a1e8079-124c-4992-8077-f69fd4e11f77`: fails exchange family where related traces can pass.
- `4e89d9d4-a3dc-4f38-8127-08f9b352bf4d`: matched-pass comparator for same family.
- `754d3a9b-b614-468e-9748-abf2ecc19a99`: unstable write-step sequence after plausible dialogue.

This pattern argues for preference-based calibration of decision quality, abstention quality, and final-action correctness.

## Literature Grounding

1. **Li et al., 2025, "Preference Leakage: A Contamination Problem in LLM-as-a-Judge"**
   - Cited sections: Sec. 3 (leakage mechanism) and Sec. 5 (mitigation via model-family separation).
   - Applied here as strict rotation policy: same model family cannot both generate and judge the same task.

2. **Meng, Xia, Chen, 2024, "SimPO: Simple Preference Optimization with a Reference-Free Reward"**
   - Cited sections: Sec. 2 (objective intuition) and Sec. 4 (empirical behavior under limited preference data).
   - Applied here because the repo has preference-pair scaffolding and a judge-centric objective, not a large SFT corpus.

3. **Kim et al., 2024, "Prometheus 2"**
   - Cited section: evaluation specialization discussion for judge behavior calibration.
   - Applied here as rationale for explicit judge prompts, dimension scoring, and spot-check calibration tiering.

## Why Not Path A or Path C

- **Path A (generator quality first) dismissed:** Week 10 traces show many failures after coherent intermediate steps; this is less a drafting-quality deficit and more a final-decision consistency deficit.
- **Path C (trajectory/reward modeling first) deferred:** potentially strong long-term fit, but higher implementation overhead for current timeline; Path B yields faster diagnostic signal with lower compute and clearer leakage controls.

## Operational Consequence

Path B is selected because it best targets the observed inconsistency failure mode while preserving auditable anti-leakage constraints and practical training cost for Week 11 timelines.
