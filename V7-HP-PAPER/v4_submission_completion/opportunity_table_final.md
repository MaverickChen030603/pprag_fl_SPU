# Table 1. Action opportunity under the frozen development protocol

| Method | Effective actions | Positive-action density | Overall positive-query coverage | Non-ceiling coverage | Newly covered vs predecessor | New-query efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 fixed actions | 4,000 | 9.48% | 20.3% | 32.90% | n/a | n/a |
| V3 heuristic expansion | 7,882 | 9.43% | 23.4% | 38.30% | 81 V2-uncovered queries | 0.0209 |
| V4 semantic generation | 7,934 | 14.71% | 29.2% | 47.63% | 81 V3-uncovered queries | 0.0143 |

V3 adds 3,882 actions relative to V2. V4 exposes 5,655 contexts absent from the V3 table. "Newly covered" is a set difference, not the net coverage change: V3 newly covers 81 V2-negative queries but fails to recover 50 V2-positive queries. V4 passes three of five pre-specified opportunity criteria. Overall coverage (29.2% versus a 30% target) and new-query efficiency do not pass.
