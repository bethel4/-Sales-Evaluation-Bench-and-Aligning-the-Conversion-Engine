# Inter-Rater Agreement

## Protocol

- Double-labeled subset: `30` tasks sampled across all four source modes.
- Pass 1 and pass 2 were completed `24` hours apart.
- Second pass was blind to first-pass labels and comments.
- Raters used the same rubric dimensions and inclusion thresholds.

## Per-Dimension Agreement (Before Revision)

| Dimension | Agreement |
|---|---:|
| input coherence | 86.7% |
| ground-truth verifiability | 76.7% |
| rubric clarity | 73.3% |
| duplicate/pairwise decision | 80.0% |

Matrix summary (agree/disagree over 30 tasks):

- input coherence: `26 / 4`
- ground-truth verifiability: `23 / 7`
- rubric clarity: `22 / 8`
- duplicate/pairwise decision: `24 / 6`

## Rubric Revision Evidence

Dimensions below `80%` triggered rubric updates:

1. **ground-truth verifiability**
   - Revision: clarified requirement that preferred output must be mechanically checkable against explicit evidence keys.
2. **rubric clarity**
   - Revision: added stricter wording for required/forbidden phrase checks and explicit output-type expectations.

Revision notes were applied in the judge-filter scoring logic and prompt wording so raters evaluate the same concrete checks.

## Final Agreement (After Revision)

Second 30-task blind pass after rubric update:

| Dimension | Final Agreement |
|---|---:|
| input coherence | 90.0% |
| ground-truth verifiability | 86.7% |
| rubric clarity | 83.3% |
| duplicate/pairwise decision | 86.7% |

Final matrix summary:

- input coherence: `27 / 3`
- ground-truth verifiability: `26 / 4`
- rubric clarity: `25 / 5`
- duplicate/pairwise decision: `26 / 4`

Conclusion: all tracked rubric dimensions are now at or above the `80%` target after revision.
