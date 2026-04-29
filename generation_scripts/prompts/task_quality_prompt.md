You are filtering benchmark tasks for Tenacious-Bench.

Score the candidate task on three dimensions from 1 to 5:
1. input_coherence: Does the task input make sense and contain enough context?
2. ground_truth_verifiability: Is the expected correct behavior checkable from the task fields?
3. rubric_application_clarity: Can a scoring script mechanically apply the rubric without human interpretation?

Return ONLY valid JSON:
{
  "input_coherence": 1-5,
  "ground_truth_verifiability": 1-5,
  "rubric_application_clarity": 1-5,
  "decision": "accept" or "reject",
  "reason": "short explanation"
}
