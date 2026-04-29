# Model routing and preference-leakage prevention

This pipeline builds Path B preference data for Tenacious-Bench. It separates generation and judging across model families so no task is generated and judged by the same model.

## Roles

| Role | Model | Family | Usage |
|---|---|---|---|
| Chosen rewrite generator | `qwen/qwen3-30b-a3b` | Qwen | Generates corrected/chosen outreach drafts from failed probes. |
| Cheap judge filter | `deepseek/deepseek-chat` | DeepSeek | Scores task quality and compares chosen/rejected pairs. |
| Eval-tier calibration judge | `anthropic/claude-sonnet-4.6` or GPT-5-class model | Frontier eval | Used only on a small calibration sample, never for bulk generation. |

## Rotation policy

- A model that generates a task or chosen rewrite must not judge that same task.
- Qwen-generated chosen rewrites are judged by DeepSeek.
- Trace-derived rejected examples come from the Week 10 agent, not the rewrite model.
- If a model family is reused in the future, the script raises an error when `generator_model == judge_model`.

## Cost policy

- Cheap dev-tier models are used for bulk synthesis and judge filtering.
- Eval-tier models are reserved for a small calibration sample only.
- All calls should be logged in `cost_log.csv` with timestamp, bucket, model, and purpose.
