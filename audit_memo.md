# Audit Memo: Week 10 Gap Analysis

Public benchmark contrast: `tau2-bench retail` mainly measures transactional support reliability (auth flow correctness, tool API usage, DB state transitions). Tenacious-specific readiness requires additional dimensions that retail does not directly score: signal-grounded outreach claims, bench-capacity truthfulness, thread-memory discipline, timezone-safe scheduling, and tone stability under objection pressure.

Week 10 probes show why this is a structural measurement gap, not a generic “benchmark is weak” complaint:

- **Signal-grounding and segmentation fidelity:** `PROBE-001`, `PROBE-002`, `PROBE-003`
- **Abstention under uncertainty:** `PROBE-035`
- **Claim safety vs noisy enrichment:** `PROBE-005`, `PROBE-006`, `PROBE-008`
- **Bench-capacity gatekeeping:** `PROBE-009`, `PROBE-010`, `PROBE-011`
- **Tone/voice integrity under skepticism:** `PROBE-012`, `PROBE-013`, `PROBE-014`
- **Thread isolation:** `PROBE-015`, `PROBE-017`
- **Timezone scheduling correctness:** `PROBE-018`, `PROBE-019`, `PROBE-020`, `PROBE-021`

These probe IDs are dimension-specific and map to business failure modes (trust loss, false staffing commitments, thread leakage, missed meetings), not just style preferences.

Minimum Week 10 probe set explicitly covered: `PROBE-001`, `PROBE-002`, `PROBE-003`, `PROBE-005`, `PROBE-006`, `PROBE-008`, `PROBE-009`, `PROBE-010`, `PROBE-011`.

Week 10 traces reinforce the same gap with trajectory evidence. Distinct trace IDs:

- `754d3a9b-b614-468e-9748-abf2ecc19a99`
- `95e55d0f-513e-431f-a482-41f317cb0564`
- `3a1e8079-124c-4992-8077-f69fd4e11f77`
- `4e89d9d4-a3dc-4f38-8127-08f9b352bf4d`
- `tau2-real-0-1777378070`
- `9f1bceea-557f-4086-b5f0-ddebed571544`
- `3bb05cae-be14-405a-866c-7355eccde196`
- `85051d0d-3245-4ddb-b366-2ecb00df4ece`

Observed pattern: trajectories can complete early read/auth steps but still fail at decisive end-of-trajectory actions (`exchange_delivered_order_items`, return completion, or final write consistency). A retail-support pass can coexist with wrong intermediate action ordering, while some user-visible completions still receive fail outcomes. That means benchmark success/failure is not a sufficient proxy for Tenacious conversion-engine reliability.

Dimension-level gap statement:

1. **Signal-grounded factuality gap:** public retail does not penalize unsupported hiring/growth inference from weak evidence.
2. **Bench-truthfulness gap:** public retail does not enforce “no staffing commitment without bench verification.”
3. **Tone governance gap:** public retail underweights non-defensive, specific tone under multi-turn skepticism.
4. **Thread-integrity gap:** public retail does not explicitly test contact-to-thread memory isolation in outbound sales.
5. **Timezone execution gap:** public retail does not robustly grade cross-region scheduling correctness under ambiguity.
6. **Trajectory consistency gap:** public pass/fail can miss wrong-step recovery dynamics that matter for trust and conversion.

Conclusion: Tenacious-Bench must be scored on these sales-specific dimensions in addition to generic tool-call reliability. Re-running public retail alone cannot close the Week 10 failure modes.
