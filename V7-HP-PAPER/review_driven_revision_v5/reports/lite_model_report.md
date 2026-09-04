# Lite Model Report

## Design

The Lite generator removes the missing-hop estimator, learned document-opportunity model, and cross-encoder. `Lite-Lexical-Pair` uses lexical/entity features plus a learned pair-complementarity model; `Lite-Semantic-Pair` adds one cached query-document cosine; `PairChain-Ablation` retains pair scoring and bounded two-document chains only. All learned models and selector thresholds use fully nested outer/inner folds.

- Pre-registered Joint-F1 non-inferiority margin: `0.002`
- Pending contexts requiring new reader outcomes: `1850`
- Revision holdout opened during architecture selection: `false`

## Development Results

| Method | Answer F1 | SP F1 | Joint F1 | Joint vs Full | Point non-inferior | CI non-inferior |
|---|---:|---:|---:|---:|---:|---:|
| frozen_top5_baseline | 0.6114 | 0.4920 | 0.3241 | reference | reference | reference |
| full_v4 | 0.6247 | 0.4973 | 0.3305 | reference | reference | reference |
| lite_lexical_pair | 0.6183 | 0.4922 | 0.3290 | -0.0015 | true | false |
| lite_semantic_pair | 0.6068 | 0.4960 | 0.3251 | -0.0053 | false | false |
| pairchain_ablation | 0.6167 | 0.4937 | 0.3293 | -0.0012 | true | false |

## Architecture Freeze Decision

A Lite architecture is not promoted from development metrics alone. Eligibility requires the pre-registered non-inferiority test and at least 30% lower measured latency or cross-encoder calls. The final variant is then frozen before evaluating the untouched 3,405-query revision holdout.

## Untouched Revision Holdout (3,405)

| Method | Answer F1 | SP F1 | Joint F1 |
|---|---:|---:|---:|
| frozen_top5_baseline | 0.6129 | 0.4862 | 0.3201 |
| full_v4 | 0.6244 | 0.4923 | 0.3280 |
| lite_lexical_pair | 0.6149 | 0.4860 | 0.3217 |

Lite vs Full Joint F1: `-0.0063`; point-estimate non-inferior: `false`; CI non-inferior: `false`.
