# V20 R4 Statistics Reproduction

## Result

Independent aggregation and 5,000-round deterministic paired bootstrap were run from the two immutable prediction JSONLs only. No reader, retrieval, evaluator, or final-test code was invoked.

- Source rows: 7200
- Main-result rows: 24; numeric match at 1e-12: **True**
- Paired-bootstrap rows: 54; numeric match at 1e-12: **True**
- Support transition table match: **True**
- Positive Logistic-ProbeRoute-vs-federated-baseline Joint-F1 cells: **6/6**
- Macro Answer-F1 delta: **+0.037664735591**
- Macro SP-F1 delta: **+0.075071428571**
- Macro Joint-F1 delta: **+0.038666748670**

## Logistic ProbeRoute vs Federated Baseline: Primary Joint F1

| Dataset | Reader | Delta | 95% paired-bootstrap CI | p | W/T/L |
|---|---|---:|---|---:|---|
| 2wikimultihopqa | flan | +0.063194551695 | [+0.040392975118, +0.086210699986] | 0.0003999200159968 | 47/247/6 |
| 2wikimultihopqa | unifiedqa | +0.054742063492 | [+0.033094708995, +0.077260681217] | 0.0003999200159968 | 47/245/8 |
| hotpotqa | flan | +0.037402745403 | [+0.020496708847, +0.055080591631] | 0.0003999200159968 | 33/258/9 |
| hotpotqa | unifiedqa | +0.032859788360 | [+0.016846726190, +0.049441699735] | 0.0003999200159968 | 32/254/14 |
| musique | flan | +0.026056032827 | [+0.013182353202, +0.040961883369] | 0.0003999200159968 | 21/273/6 |
| musique | unifiedqa | +0.017745310245 | [+0.005575000000, +0.030870147908] | 0.0011997600479904 | 15/280/5 |

The minimum attainable observed plus-one two-sided p-value is `2/5001 = 0.0003999200159968`. BH-FDR is applied to the 36 secondary Answer-F1/SP-F1 comparisons only; Joint F1 remains uncorrected as primary. Macro averages are equal-weight means across the six dataset-reader cells.

The published CSV byte-identity checks are recorded in `audit_recompute/reproduction_summary.json`. Numeric equality is authoritative because independent CSV formatting may differ without changing values.

## Post-outcome Analysis Changes

The prediction files predate commits `4fb90fa` and `13091c6`. Both commits are analyzer-only. The final change corrected the plus-one two-sided bootstrap p-value formula; predictions and aggregate outcome values were not regenerated.
