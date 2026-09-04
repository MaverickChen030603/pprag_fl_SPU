# Table 3. Frozen same-source confirmatory holdout (3,000 queries)

| Reader | N | Intervention coverage | Answer F1 delta | SP F1 delta | Joint F1 delta | Joint F1 95% CI | Joint F1 p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FLAN-T5-Large | 3,000 | 0.258 | +0.0088 | +0.0056 | +0.0064 | [+0.0027, +0.0104] | 0.0004 |
| UnifiedQA-T5-Large* | 3,000 | 0.258 | +0.0110 | +0.0056 | +0.0085 | [+0.0049, +0.0122] | <0.0002 |

The 3,000 queries are disjoint from the 1,000-query development sample but come from the same HotpotQA distractor validation source. Generator models, selector thresholds and coverage, reader prompts and decoding, and the support threshold were frozen. FLAN answer F1 p=0.0096; FLAN supporting-fact F1 p=0.0004. *UnifiedQA reuses the frozen selected contexts and the same sentence-support predictor; it is an answer-reader replication, not an independent support-pipeline replication.
