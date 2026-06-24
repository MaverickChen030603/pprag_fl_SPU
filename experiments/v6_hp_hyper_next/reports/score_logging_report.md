# Score Logging Report

## Scope

- Score records: 150
- Summary groups: 3

## Summary Table

| method | subset | records | jaccard vs anchor | JS divergence | avg margin | pooler ratio | layer8 ratio | entropy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hypernet_v4 | hotpot_hard_1000 | 50 | 1.0 | 0.06238648654591139 | -1.0179465077437653 | 0.5000 | 0.4600 | 0.6931471805579453 |
| hypernet_v5 | hotpot_hard_1000 | 50 | 1.0 | 0.06238648654591139 | -1.017946490202394 | 0.5000 | 0.4600 | 0.6931471805579453 |
| hypernet_v6 | hotpot_hard_1000 | 50 | 1.0 | 0.0 | -1.592493506663876 | 0.5000 | 0.4600 | 0.6931471805579453 |

## Diagnostic Answers

1. Candidate score distribution difference: see JS divergence against the anchor after score-log reruns.
2. Identical top blocks despite different scores would indicate a large selected-vs-next score margin or topk bottleneck.
3. Large positive margins indicate selector collapse caused by dominant high-score blocks.
4. Pooler and encoder.layer.8 dominance is measured by selected ratios.
5. Layerwise-budget effect is measured by entropy and block-set diversity across ablations.
