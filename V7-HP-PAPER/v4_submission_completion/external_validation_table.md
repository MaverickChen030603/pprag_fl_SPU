# Table 6. Frozen external validation on 2WikiMultiHopQA development (1,000 queries)

| System | Answer EM | Answer F1 | Supporting-fact EM | Supporting-fact F1 | Joint EM | Joint F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | 0.402 | 0.4709 | 0.080 | 0.4545 | 0.049 | 0.2463 |
| Frozen V4 transfer | 0.407 | 0.4794 | 0.078 | 0.4539 | 0.047 | 0.2496 |
| Delta | +0.0050 | +0.0086 | -0.0020 | -0.0006 | -0.0020 | +0.0033 |

Answer F1: [-0.0021, +0.0191], p=0.1116. Supporting-fact F1: [-0.0036, +0.0025], p=0.6928. Joint F1: [-0.0031, +0.0098], p=0.3296. The HotpotQA generator, selector, thresholds, coverage, reader, and support predictor are frozen; only the data adapter changes. The result is directionally positive for answer and joint F1, statistically flat for support F1, and not significant. It is external validation evidence, not proof of broad cross-dataset generalization. Opportunity density is 14.29%; positive-query coverage is 31.7%; selection coverage is 26.0%; selected-action answer-drop rate is 6.92%.
