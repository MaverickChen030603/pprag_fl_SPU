# 2WikiMultiHopQA Dev-300 Reader-Backed Smoke Report

## Status

Completed. This is a reader-backed cross-dataset smoke check using the prepared 2Wiki dev adapter.

## Configuration

- Input: `outputs/2wiki_adapter/2wiki_converted.json`
- Sample: stratified dev `300` examples, seed `20260622`
- Reader: `google/flan-t5-large`
- Top-K: `5`
- Methods: `context_order_top5`, `frozen_selector_bm25_top5`
- Selector ranking uses only query/context text. Labels are used only for metric computation.

## Metrics

| method | n | access@k | support_recall@k | sp_f1 | answer_em | answer_f1 | joint_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| context_order_top5 | 300 | 0.5867 | 0.4775 | 0.3761 | 0.3133 | 0.3714 | 0.1669 |
| frozen_selector_bm25_top5 | 300 | 0.7633 | 0.7967 | 0.7290 | 0.4300 | 0.4977 | 0.4176 |

## Delta vs Context Order

| metric | mean_delta | wins | losses | ties |
|---|---:|---:|---:|---:|
| answer_access_at_k | 0.1767 | 87 | 34 | 179 |
| support_recall_at_k | 0.3192 | 185 | 19 | 96 |
| sp_f1 | 0.3529 | 185 | 19 | 96 |
| answer_em | 0.1167 | 54 | 19 | 227 |
| answer_f1 | 0.1263 | 70 | 30 | 200 |
| joint_f1 | 0.2507 | 133 | 30 | 137 |

## Claim Boundary

Reader-backed smoke only. It validates 2Wiki adapter compatibility and reader-facing behavior; it is not a formal 1000-sample cross-dataset claim.
