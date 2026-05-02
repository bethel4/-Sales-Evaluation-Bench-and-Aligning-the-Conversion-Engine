# Tenacious Bench v0.1 - 6-Minute Walkthrough Plan (A-F)

This README is your recording script/checklist for the challenge video.

## Goal
Demonstrate, in <= 6 minutes, five required evidence segments:
1. Dataset Walkthrough on HuggingFace
2. End-to-End Task Scoring Demo
3. Ablation Result with Traceability
4. Blog Post Visibility
5. Community Engagement Visibility

---

## Segment A (0:00-1:45) - Dataset Walkthrough on HuggingFace

### On screen
- Open public HuggingFace dataset page live (address bar visible).
- Scroll dataset card showing all seven Gebru datasheet sections:
  - Motivation
  - Composition
  - Collection Process
  - Preprocessing/Cleaning/Labeling
  - Uses
  - Distribution
  - Maintenance
- Show three discoverable partitions with counts:
  - `train`
  - `public_dev`
  - `held_out` (sealed)
- Open one task row and show source-mode metadata.
- Show license.

### Narration
- "This is the public HuggingFace dataset page."
- "I'm scrolling through all seven datasheet sections."
- "These are the three partitions and their counts."
- "Here is task-level metadata and the license."

---

## Segment B (1:45-2:45) - End-to-End Task Scoring Demo

### On screen
- Show one concrete task input fields (`prompt`, etc.).
- Show one candidate output.
- Run `score_task(...)` live.
- Show score output with dimension breakdown.
- Point to one specific rubric check (e.g., banned phrase or grounding).

### Narration
- "This is one concrete task and candidate output."
- "Running evaluator now."
- "Here is per-dimension scoring and one specific rubric check."

---

## Segment C (2:45-4:30) - Ablation + Traceability

### On screen
- Open `ablation_results.json` (or `ablation_results_orpo.json`).
- Point to numeric values:
  - baseline pass@1
  - mechanism pass@1
  - ablation pass@1
  - Delta A / Delta B
  - p-values
- Open held-out trace artifact (`held_out_traces.jsonl`).
- Trace one claim from table to underlying row/task evidence.

### Narration
- "These are the ablation metrics and statistical indicators."
- "Now I trace this specific numeric claim to held-out trace rows."
- "Negative/null deltas are reported as-is."

---

## Segment D (4:30-5:20) - Blog Post Visibility

### On screen
- Open live blog URL with address bar visible.
- Scroll headings/sections to show substantive structure.

### Narration
- "This is the public technical blog post."
- "I'm scrolling to show it is substantive and structured."

---

## Segment E (5:20-5:50) - Community Engagement Visibility

### On screen
- Open live community artifact URL (issue/discussion/PR/workshop post).
- Address bar visible.
- Scroll enough to show non-placeholder content.

### Narration
- "This is the public community engagement artifact and URL."

---

## Segment F (5:50-6:00) - Close

### Narration
- "I demonstrated live dataset documentation and partitions, one end-to-end scoring run, ablation with traceability and stats, and public blog/community artifacts."

---

## Recording Checklist (must pass)
- [ ] <= 6:00 total
- [ ] Address bars visible for HF/blog/community pages
- [ ] All 7 datasheet sections shown
- [ ] Partition counts shown for train/public_dev/held_out
- [ ] One concrete scoring run with dimension-level output
- [ ] At least one explicit rubric check shown
- [ ] Ablation numeric table + p-value shown
- [ ] One numeric claim traced to held-out trace rows
- [ ] Blog and community pages visibly substantive
- [ ] Audio intelligible and transitions signposted
