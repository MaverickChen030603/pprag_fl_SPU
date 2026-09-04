# Paper Experiment Outline

## 1. Main Experiment: HotpotQA

Describe the strict no-leak query-level cross-fitting setup, the reader-backed evaluation protocol, and the baseline.

## 2. Main Result: v2.3 Answer-Neutral Positive Selector

Report the formal final-1000 result. Emphasize significant gains in joint_f1, support_recall@5, and sp_f1, with answer_f1 preserved but not significantly improved.

## 3. Ablations

Compare two-stage selection, paper-positive classification, safety removal, support-feature removal, and earlier support-first variants. Use the ablations to motivate answer-neutral action selection.

## 4. Oracle Diagnostic

Report oracle rows only as upper-bound analyses and explicitly separate them from inference-time methods.

## 5. External Diagnostic: 2WikiMultiHopQA

Present 2Wiki as an external sanity check. BM25 lexical routing validates the adapter and reader path, while selector alignment and detectability audits expose cross-dataset limitations.

## 6. Limitation and Future Work

State that cross-dataset selector generalization is limited by candidate exposure, feature detectability, and safety calibration. Future work should improve candidate generation beyond BM25 and dataset-robust answer-neutral calibration.
