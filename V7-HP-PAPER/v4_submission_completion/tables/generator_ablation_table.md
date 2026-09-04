# Generator Ablation Table

| Variant | Effective actions | New actions vs V3 | Positive density | Overall coverage | Non-ceiling coverage | New queries vs V3 | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 7934 | 5655 | 0.147 | 0.292 | 0.476 | 81 | 0.0143 |
| without_missing_hop_estimator | 7952 | 5619 | 0.145 | 0.290 | 0.473 | 81 | 0.0144 |
| without_mpnet_features | 7948 | 5622 | 0.144 | 0.295 | 0.481 | 83 | 0.0148 |
| without_cross_encoder_features | 7940 | 5691 | 0.147 | 0.306 | 0.499 | 91 | 0.0160 |
| without_semantic_document_model | 7934 | 6484 | 0.149 | 0.326 | 0.532 | 110 | 0.0170 |
| without_pair_complementarity | 7934 | 5461 | 0.103 | 0.277 | 0.452 | 71 | 0.0130 |
| without_two_document_actions | 5547 | 3563 | 0.104 | 0.251 | 0.409 | 54 | 0.0152 |
| without_anchor_preservation | 5909 | 4088 | 0.166 | 0.274 | 0.447 | 73 | 0.0179 |
| without_redundancy_actions | 7397 | 5298 | 0.148 | 0.292 | 0.476 | 81 | 0.0153 |
| lexical_only_generator | 7952 | 5652 | 0.139 | 0.307 | 0.502 | 89 | 0.0157 |
| semantic_only_generator | 7952 | 5929 | 0.147 | 0.306 | 0.499 | 97 | 0.0164 |

All learned ablations are retrained on each outer-training split and applied to the disjoint outer test fold. Structural action-family removals use the same frozen fold model. No 3,000-query holdout outcome is used.
