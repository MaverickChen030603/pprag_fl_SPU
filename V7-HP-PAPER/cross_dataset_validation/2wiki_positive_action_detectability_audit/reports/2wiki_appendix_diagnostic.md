# Appendix: Cross-Dataset Diagnostic on 2WikiMultiHopQA

## BM25 lexical smoke result

The 2Wiki adapter and reader-backed evaluation pipeline were validated on dev-300. BM25 / lexical routing is a strong baseline and substantially improves over raw context order.

## Selector alignment failure

Direct selector transfer and the original 2Wiki crossfit selector underperform the strong BM25 baseline. This should not be written as selector-level generalization success.

## BM25-anchor repair result

The BM25-anchor repair preserves BM25 top-1/top-2/top-3 anchors and reduces negative transfer. The best no-leak repair (`bm25_anchor_answer_neutral_selector`) obtains joint delta 0.0002 versus BM25, which is too small to justify 1000-sample expansion.

## Oracle gap

Oracle positive actions exist for 73 / 300 queries, with oracle joint delta 0.1533. This is diagnostic only.

## Failure analysis

The dominant failure mode is candidate-pool limitation: 227 / 300 queries have no oracle positive action beyond BM25, and only 33 / 300 expose strict positive actions in the BM25-anchor table. Selector recall over strict available positives is 0.3939.

## Feature detectability

positive actions are weakly distinguishable with current features. Safety calibration is weak, with answer-safe AUC 0.5567 and paper-positive AUC 0.5451.

## Claim boundary

2Wiki is reported as pipeline validation and limitation analysis. It is not used as a main method success claim, and oracle rows are not inference-time evidence.
