# Related Work Comparison Matrix

This matrix is conceptual unless a frozen experimental comparison is explicitly named. It does not rank unimplemented methods.

| Method/direction | Retrieval stage | Changes pool | Reader/generator feedback | Online selection calls | Action structure | Risk mechanism | Closest difference from Full |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| GenDec | Pre-retrieval decomposition | Yes | LLM decomposition | At least decomposition plus QA | Subquestions | None central to method | Full keeps query and upstream pool fixed |
| MDR | Multi-step retrieval | Yes | Retrieval supervision | Iterative retriever | Hop-wise retrieval | None central to method | Full reorganizes a bounded post-retrieval pool |
| HGN | Reader/reasoner | No fixed comparison | Joint QA supervision | One graph reader | Paragraph-sentence-entity graph | None central to method | Full does not change reader architecture |
| RECOMP | Post-retrieval compression | No | Compressor training | Compressor plus reader | Sentence compression | No answer-preservation gate | Full preserves full passages and structured actions |
| RankRAG | Ranking plus generation | No/depends on upstream | Joint ranking-generation instruction tuning | Ranking/generation model | Ranked contexts | No explicit fallback risk gate | Full separates frozen reader from selector |
| SetR | Passage-set selection | No/within retrieved pool | LLM reasoning over set needs | Selection LLM plus reader | Variable set | No empirical preservation head | Full uses bounded learned actions and exact fallback |
| R-CPS | Reader-centered selection | No | Reader distributions | Reader-dependent reranking/clustering | Ranked clusters | Reader consistency heuristic | Full uses offline outcomes, no online candidate-reader loop |
| BAR-RAG | Boundary-aware selection | Noisy retrieved pool | Generator reward/RL | Selector and generator pipeline | Evidence subset | Reward-defined boundary | Recent preprint; Full uses supervised nested fallback |
| Conformal RAG filtering | Retrieval/answer set calibration | Often variable set | Calibration scores | Method-dependent | Prediction/retrieval sets | Marginal finite-sample guarantee under assumptions | Full reports empirical intervention risk only |

Sources are official proceedings pages where available: MDR and HGN via ACL Anthology, RankRAG via NeurIPS, SetR and R-CPS via ACL Anthology, RECOMP via ACL Anthology, TRAQ via ACL Anthology, and C-RAG via PMLR. GenDec and BAR-RAG are treated as conceptual preprint comparisons only.
