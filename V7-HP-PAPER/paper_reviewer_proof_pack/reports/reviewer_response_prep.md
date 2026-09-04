# Reviewer Response Prep

## Q1: Why only HotpotQA as the main result?

HotpotQA provides supporting facts and joint metrics required for no-leak action selection evaluation. We include 2WikiMultiHopQA as an external diagnostic, but selector-level generalization beyond a strong BM25 baseline remains limited, so we do not present it as a main success claim.

## Q2: Why is answer_f1 gain small?

The method is designed to preserve answer quality while improving joint/support evidence utility, not to directly optimize answer generation. Accordingly, we describe answer_f1 as preserved with a small non-significant positive delta and focus the main claim on significant joint/support improvements.

## Q3: Why not claim 2Wiki success?

Against a strong BM25 baseline, selector-level improvements on 2Wiki are not reliable. BM25-anchor repair nearly matches BM25, but the gain is too small for a formal generalization claim. Reporting it as a limitation is more scientifically accurate.

## Q4: Is oracle used in inference?

No. Oracle results are diagnostic upper bounds only. The formal selector uses strict no-leak query-level cross-fitting, and oracle rows are separated from inference-time method claims.

## Q5: Why no 2Wiki 1000?

Smoke and detectability diagnostics show bottlenecks in candidate exposure and feature separability. Expanding sample size would likely confirm the limitation rather than strengthen the main claim, so it is deferred as future work.
