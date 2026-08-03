# Federated Search Contract Matrix

| Work | Main decision object | Supervision / signal | Budget focus | Direct use in V20 | Non-equivalence / constraint |
|---|---|---|---|---|---|
| FeB4RAG | source selection and returned-result merging | relevance assessments | resources and returned results | benchmark contract; source/merge metrics | BEIR subcollections, not multi-hop QA silos |
| RAGRoute | lightweight source classifier | source relevance labels | contacts, communication, latency | budget-matched student-router baseline | not a multi-hop completeness objective |
| ReSLLM / SLAT | LLM source relevance | zero-shot or synthetic labels | selected resources | teacher / train-only soft-label source | teacher is not the low-latency deployed router |
| ReDDE | resource utility estimate | collection statistics and sampled results | source selection | size-aware resource prior | classical text retrieval assumptions |
| Classification-based selection | learned resource ranking | multiple resource features | selected resources | multi-feature router baseline | needs explicit label contract |
| Language-model result merging | joint resource/document probability | score distributions | returned result merge | source-aware calibrated merge | must not centralize client corpus |
| MKP-QA | joint domain and passage relevance | QA-domain data | domain search | source-passage joint scorer | enterprise product QA, not arbitrary silos |
| FedMosaic | parameter adapter aggregation | local training objectives | model-update payload | collision audit only | not query-time source selection |
| V17/V19 | context action / adapter perturbation | reader or contrastive signals | K=5 / adapter payload | negative diagnostic asset | no stable reader-visible context gain |
