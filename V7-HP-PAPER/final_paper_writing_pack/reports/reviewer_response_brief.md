# Reviewer Response Brief

## Q1: Why is HotpotQA the main dataset?

Because it provides answer, support, and joint metrics required for our no-leak action-selection evaluation.

## Q2: Why not claim strong cross-dataset transfer?

2Wiki diagnostics show that selector-level transfer beyond a strong BM25 baseline is limited by candidate exposure, feature detectability, and safety calibration.

## Q3: Why is answer_f1 gain small?

The method is designed to preserve answer quality while converting routing-side support signals into joint/support gains.

## Q4: Is oracle used in inference?

No. Oracle is diagnostic only. Formal results use strict no-leak query-level cross-fitting.

## Q5: Why no additional large-scale validation?

Reviewer proof analysis indicates current experiments are sufficient for a HotpotQA-centered paper with 2Wiki diagnostic limitation; additional large-scale validation would not change the central claim.
