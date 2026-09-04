# V7-HP-PAPER 2WikiMultiHopQA Cross-Dataset Current Report

## 1. Current Status

2WikiMultiHopQA cross-dataset preparation and dev-300 smoke validation are complete. No 2Wiki/V7-HP-PAPER process is currently running on the server.

Completed artifacts:

- data preparation with answer, context docs, and dev support/evidence labels
- retrieval/access dev-300 smoke
- reader-backed dev-300 smoke with `google/flan-t5-large`

Not yet completed:

- formal 1000-sample 2Wiki reader validation
- direct migration of Hotpot frozen selector v2.3 action-table model to 2Wiki action features
- statistical significance report for 2Wiki final validation

## 2. Data Readiness

- Adapter status: `ready`
- Preferred eval split: `dev`
- Dev examples: `12576`
- Dev answer/context/support/evidence: `12576` / `12576` / `12576` / `12576`
- Dev avg context docs: `10.0`
- Dev avg support docs: `2.4375`
- Test examples: `12576`
- Test support/evidence sentence labels: `0` / `0`

Conclusion: use dev for support/evidence/joint metrics. Test is prepared but should not be used for sentence-level support F1 unless an ID-level metric is added.

## 3. Retrieval/Access Smoke

Configuration: stratified dev 300, Top-5, no reader. Labels are used only for metric computation.

| method | n | answer_access@5 | support_recall@5 | all_support_access@5 | joint_access@5 |
|---|---:|---:|---:|---:|---:|
| context_order_top5 | 300 | 0.5433 | 0.4775 | 0.1667 | 0.1533 |
| lexical_bm25_top5 | 300 | 0.7133 | 0.7967 | 0.5467 | 0.5067 |

Delta vs context order:

| metric | mean_delta | wins | losses | ties |
|---|---:|---:|---:|---:|
| answer_access@5 | 0.1700 | 87 | 36 | 177 |
| support_recall@5 | 0.3192 | 185 | 19 | 96 |
| all_support_access@5 | 0.3800 | 129 | 15 | 156 |
| joint_access@5 | 0.3533 | 121 | 15 | 164 |

Interpretation: the 2Wiki adapter is not only field-complete; lexical/BM25 ranking substantially improves support and joint access, so the dataset is suitable for reader-backed cross-dataset validation.

## 4. Reader-Backed Smoke

Configuration: stratified dev `300`, `600` reader prompts, Top-5, reader `google/flan-t5-large`, batch size `4`.

| method | n | access@k | support_recall@k | sp_f1 | answer_em | answer_f1 | joint_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| context_order_top5 | 300 | 0.5867 | 0.4775 | 0.3761 | 0.3133 | 0.3714 | 0.1669 |
| frozen_selector_bm25_top5 | 300 | 0.7633 | 0.7967 | 0.7290 | 0.4300 | 0.4977 | 0.4176 |

Delta vs context order:

| metric | mean_delta | wins | losses | ties |
|---|---:|---:|---:|---:|
| answer_access_at_k | 0.1767 | 87 | 34 | 179 |
| support_recall_at_k | 0.3192 | 185 | 19 | 96 |
| sp_f1 | 0.3529 | 185 | 19 | 96 |
| answer_em | 0.1167 | 54 | 19 | 227 |
| answer_f1 | 0.1263 | 70 | 30 | 200 |
| joint_f1 | 0.2507 | 133 | 30 | 137 |

Interpretation: the reader-backed smoke shows a clear positive transfer signal for the fixed lexical/BM25 routing baseline over raw context order. The strongest gains appear in support routing (`sp_f1 +0.3529`) and propagate into answer and joint scores (`answer_f1 +0.1263`, `joint_f1 +0.2507`).

## 5. Claim Boundary

This is a cross-dataset smoke result, not a formal external generalization endpoint. The current 2Wiki method is a frozen, no-training lexical/BM25 selector connected to the reader pipeline. It is not yet the Hotpot frozen selector v2.3 action-table model directly transferred to 2Wiki.

Paper-safe wording:

> We prepared 2WikiMultiHopQA and verified that the V7-HP-PAPER reader/evaluation path can consume a new multi-hop QA dataset. In a dev-300 smoke check, a non-leaky lexical routing baseline substantially improves reader-backed support and joint metrics over raw context order. Formal cross-dataset claims require a larger validation run and selector-feature alignment.

## 6. Artifacts

```text
V7-HP-PAPER/cross_dataset_validation/outputs/2wiki_adapter/adapter_summary.json
V7-HP-PAPER/cross_dataset_validation/outputs/2wiki_smoke_300/summary.json
V7-HP-PAPER/cross_dataset_validation/outputs/2wiki_reader_smoke_300/reader_summary.json
V7-HP-PAPER/cross_dataset_validation/outputs/2wiki_reader_smoke_300/per_example_reader.jsonl
V7-HP-PAPER/cross_dataset_validation/reports/2wiki_data_preparation_report.md
V7-HP-PAPER/cross_dataset_validation/reports/2wiki_smoke_300_report.md
V7-HP-PAPER/cross_dataset_validation/reports/2wiki_reader_smoke_300_report.md
```

## 7. Recommended Next Step

Run a bounded 1000-sample 2Wiki reader validation only after deciding whether the paper needs external validation as a main claim or an appendix robustness check. If the goal is method transfer rather than dataset plumbing, first build a 2Wiki action-feature table compatible with selector v2.3; otherwise the current result should be described as a reader-backed lexical routing smoke.
