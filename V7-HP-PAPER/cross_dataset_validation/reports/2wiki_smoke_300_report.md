# 2WikiMultiHopQA Dev-300 Smoke Validation Report

## Status

Completed on the prepared dev split. This is a retrieval/access smoke test, not a reader QA evaluation.

## Configuration

- Input: `outputs/2wiki_adapter/2wiki_converted.json`
- Sample: stratified dev `300` examples, seed `20260622`
- Top-K: `5`
- Ranking methods: `context_order_top5`, `lexical_bm25_top5`
- No answer/support labels are used for ranking; labels are used only for metric computation.

## Field Gate

- answer present: `True`
- context docs present: `True`
- support titles present: `True`

## Metrics

| method | answer_access@5 | support_recall@5 | all_support_access@5 | joint_access@5 |
|---|---:|---:|---:|---:|
| context_order_top5 | 0.5433 | 0.4775 | 0.1667 | 0.1533 |
| lexical_bm25_top5 | 0.7133 | 0.7967 | 0.5467 | 0.5067 |

## Delta vs Context Order

| metric | mean_delta | wins | losses | ties |
|---|---:|---:|---:|---:|
| answer_access@5 | 0.1700 | 87 | 36 | 177 |
| support_recall@5 | 0.3192 | 185 | 19 | 96 |
| all_support_access@5 | 0.3800 | 129 | 15 | 156 |
| joint_access@5 | 0.3533 | 121 | 15 | 164 |

## Interpretation

The 2Wiki dev adapter is usable for cross-dataset validation: every sampled example has answer, context documents, and support titles. The lexical smoke metrics provide a non-leaky retrieval/access baseline before running any reader or learned selector.

Next step: connect the frozen selector/reader pipeline to this dev adapter and run a reader-backed smoke only after confirming the retrieval index and prompt schema consume `context` and `supporting_titles` correctly.
