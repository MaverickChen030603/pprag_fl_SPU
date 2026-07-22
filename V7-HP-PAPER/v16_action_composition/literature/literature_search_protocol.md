# V16 Literature Search Protocol

## Scope and date

Search frozen on 2026-07-22 for methods at the intersection of RAG context selection, multi-hop retrieval, reader-aware utility, set/subset prediction, structured passage sequences, dynamic context size, and offline reinforcement learning.

## Allowed evidence

Only original papers and supplements from ACL Anthology, OpenReview, proceedings sites, arXiv, and author-linked official GitHub repositories are used. Blogs, vendor summaries, survey-only descriptions, and search snippets are not evidence for novelty claims.

## Search questions

1. Does the method learn a trajectory of edits to an already retrieved, ordered, fixed-cardinality context?
2. Is each decision conditioned on the context produced by earlier edits?
3. Is document cardinality fixed or variable?
4. Is output a passage, pair, set, sequence, or edit trajectory?
5. Is supervision relevance-, evidence-, answer-, reader-outcome-, influence-, or reward-based?
6. Does it compare the best composed trajectory with **all legal single edits** under one frozen contract?
7. Does it define interaction or synergy beyond leave-one-out marginal utility?
8. Are reader/LLM calls required during test-time search?

## Queries and seed works

Seed works: SetR; Modeling Contextual Passage Utility for Multihop QA; Generative Context Pair Selection; Influence Guided Context Selection; R-CPS; Context-Picker; RankRAG; RECOMP; MDR; Beam Retrieval; GenDec; adaptive context size. Citation chasing then covers Deep Sets, Set Transformer, Conservative Q-Learning, and Advantage-Weighted Regression.

Representative search strings include `multi-hop QA context set selection`, `reader-aware passage utility`, `RAG leave-one-out context influence`, `sequential passage subset reinforcement learning`, `fixed-k context edit trajectory`, `synergy context selection RAG`, and `action composition document reranking`.

## Collision rule

A full collision requires all of: (a) a frozen retrieved pool, (b) fixed context cardinality, (c) state-dependent multi-step atomic edits, (d) direct reader-outcome training or evaluation, and (e) explicit best-single-versus-composed synergy analysis. A partial match narrows claims but does not establish novelty.

No exact collision was found in this search. This is a bounded literature audit, not proof of global novelty. `CompoRepair` remains an internal codename until title, method-name, Semantic Scholar, DBLP, arXiv, and GitHub collision checks are repeated immediately before submission.

## Primary sources

- [SetR, ACL 2025](https://aclanthology.org/2025.acl-long.861/)
- [Contextual Passage Utility, IJCNLP-AACL 2025](https://aclanthology.org/2025.ijcnlp-short.37/)
- [Generative Context Pair Selection, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.561/)
- [Influence Guided Context Selection, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/2a07497c94cd24b20faa3fd14d847037-Abstract-Conference.html)
- [R-CPS, COLING 2025](https://aclanthology.org/2025.coling-main.67/)
- [Context-Picker, arXiv 2025](https://arxiv.org/abs/2512.14465)
- [RankRAG, arXiv 2024](https://arxiv.org/abs/2407.02485)
- [RECOMP, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/bda88ed2892f5e61c9a9bf215c566913-Paper-Conference.pdf)
- [MDR, ICLR 2021](https://openreview.net/forum?id=EMHoBG0avc1)
- [Beam Retrieval, NAACL 2024](https://aclanthology.org/2024.naacl-long.96/)
- [GenDec, arXiv 2024](https://arxiv.org/abs/2402.11166)
- [Adaptive-k Retrieval, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1017/)
- [Deep Sets, NeurIPS 2017](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)
- [Set Transformer, ICML 2019](https://proceedings.mlr.press/v97/lee19d)
- [Conservative Q-Learning, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)
- [Advantage-Weighted Regression, ICLR 2021](https://openreview.net/forum?id=ToWi1RjuEr8)
