# Metric Definition Audit

## Decision

Official HotpotQA supporting-fact and joint metrics cannot be reconstructed from the completed reader run. The source dataset supplies gold sentence-level supporting facts, and v2 exports them, but the system did not produce predicted support sentence IDs. Reinterpreting retrieved document titles as sentence predictions would be invalid. The v2 paper therefore reports only explicitly named custom title-level metrics.

| Paper name | Legacy artifact field | Formula | Official HotpotQA? | Gold use | Used at inference? |
|---|---|---|---|---|---|
| Answer access@5 | `answer_access_at_k` | Indicator that the selected context satisfies the project answer-access check | No | Evaluation only | No |
| Title-level support recall@5 | `support_recall_at_k` | Number of gold support titles in selected top-5 titles divided by number of gold support titles | No | Evaluation and train-query labels only | No target-query gold use |
| Title-level support F1 | `sp_f1` | Title-set precision/recall F1, with the implementation's full-recall adjustment | No | Evaluation and train-query labels only | No target-query gold use |
| Answer EM | `answer_em` | Normalized short-answer exact match | Matches the answer-side normalization family, but not an official full Hotpot result by itself | Evaluation and train-query labels only | No |
| Answer F1 | `answer_f1` | Token overlap F1 between generated and gold answer | Answer-side diagnostic | Evaluation and train-query labels only | No |
| Answer-title-support product | `joint_f1` | Per-query answer F1 multiplied by title-level support F1, then averaged | **No** | Evaluation and train-query labels only | No |

## Full-recall adjustment

The stored title-level support F1 implementation applies an adjustment when all gold support titles are retrieved. Because this differs from official sentence-level scoring, the paper reports the metric as a project-defined title-set measure and points readers to the evaluation artifact. It must never be abbreviated to bare “support F1” in a table header without the title-level qualifier.

## Why official support/joint cannot be reported

Official HotpotQA scoring requires a predicted answer plus predicted `(title, sentence_id)` supporting facts for every example. The archived reader outputs contain predicted answers and selected document contexts, but no support-sentence predictor or predicted sentence IDs. Gold sentence labels can be recovered, but gold labels cannot substitute for predictions. Therefore:

- `official_hotpotqa_metrics.json`: not created;
- `official_hotpotqa_significance.json`: not created;
- `official_metric_table.md`: not created.

This absence is intentional and documented rather than silently filled with title proxies.

## Permitted claim

“Under fully nested query-level cross-fitting on a HotpotQA-derived 1,000-query sample, the selector improves title-level support recall and title-level support F1; answer F1 and the answer-title-support product change positively but not significantly.”

## Prohibited claim

“The method improves official HotpotQA supporting-fact F1 or joint F1.”
