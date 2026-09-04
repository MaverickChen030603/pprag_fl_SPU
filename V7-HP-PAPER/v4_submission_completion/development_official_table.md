# Table 2. Official HotpotQA development evaluation (1,000 queries)

| System | Answer EM | Answer F1 | Supporting-fact EM | Supporting-fact F1 | Joint EM | Joint F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline | 0.483 | 0.6114 | 0.073 | 0.4920 | 0.052 | 0.3241 |
| V4 semantic generation + reader-safe selection | 0.493 | 0.6247 | 0.074 | 0.4973 | 0.051 | 0.3305 |
| Delta | +0.0100 | +0.0133 | +0.0010 | +0.0053 | -0.0010 | +0.0064 |

Paired bootstrap, 5,000 resamples: answer F1 [+0.0024, +0.0249], p=0.0176; supporting-fact F1 [+0.0003, +0.0106], p=0.0372; joint F1 [-0.0005, +0.0132], p=0.0752. Answer F1 and supporting-fact F1 improve significantly at the unadjusted 0.05 level. Joint F1 is positive but not significant. The selector intervenes on 260/1,000 queries and its selected-action answer-drop rate is 5.0%.
