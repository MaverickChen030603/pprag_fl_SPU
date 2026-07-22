# V17 Federated RAG Literature Search Protocol

**Search date:** 2026-07-22  
**Project:** V7-HP-PAPER-v17-federated-action-rag  
**Status:** primary-source search completed for Phase-A framing; living audit for later 2026 additions

## Research Question

Does prior work already federate the training of a reader-aware scorer that selects complete, fixed-budget, cross-client contexts for multi-hop QA, while jointly controlling client access and context budgets?

## Source Policy

Included sources are original papers or official artifacts hosted by ACL Anthology, OpenReview, arXiv, formal conference/proceedings sites, author institution pages, or paper-linked GitHub repositories. Blogs, vendor summaries, Reddit posts, and secondary explainers are excluded from method claims.

The search covers four intersecting literatures:

1. federated/decentralized/distributed RAG and federated search;
2. personalized federated retrieval and client routing;
3. reader-aware passage, pair, set, and context selection;
4. multi-hop retrieval, action selection, structured prediction, and offline RL.

## Search Strings

- `federated retrieval augmented generation`, `federated RAG`, `federated retrieval`
- `distributed RAG`, `decentralized RAG`, `federated search RAG`
- `personalized federated retrieval`, `federated reranking`, `cross-silo RAG`
- `client routing RAG`, `source routing RAG`, `peer discovery RAG`
- `multi-hop distributed knowledge retrieval`, `cross-client evidence`
- `reader-aware context selection`, `set-wise passage selection`, `context utility`
- `fixed budget context action`, `cross-client context synergy`

## Verified Primary Sources

The Phase-A boundary relies most directly on:

- [Federated RAG systematic mapping study, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.388/)
- [pFedRAG, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.769/)
- [MKP-QA, COLING Industry 2025](https://aclanthology.org/2025.coling-industry.33/)
- [FedE4RAG, arXiv 2025](https://arxiv.org/abs/2504.19101)
- [FedRAG framework, OpenReview/arXiv 2025](https://openreview.net/pdf?id=I2is1QsDuD)
- [Distributed RAG (DRAG), arXiv 2025](https://arxiv.org/abs/2505.00443)
- [FeB4RAG, arXiv 2024](https://arxiv.org/abs/2402.11891)
- [FD-RAG, arXiv 2026](https://arxiv.org/abs/2605.27432)
- [FedMosaic, arXiv 2026](https://arxiv.org/abs/2602.05235)
- [Raffle, OpenReview 2024](https://openreview.net/forum?id=MKd1SkDbbz)
- [SetR, ACL 2025](https://aclanthology.org/2025.acl-long.861/)
- [Contextual Passage Utility, IJCNLP-AACL 2025](https://aclanthology.org/2025.ijcnlp-short.37/)
- [Influence Guided Context Selection, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/2a07497c94cd24b20faa3fd14d847037-Abstract-Conference.html)
- [R-CPS, COLING 2025](https://aclanthology.org/2025.coling-main.67/)
- [Context-Picker, arXiv 2025](https://arxiv.org/abs/2512.14465)
- [RankRAG, arXiv 2024](https://arxiv.org/abs/2407.02485)
- [RECOMP, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/bda88ed2892f5e61c9a9bf215c566913-Paper-Conference.pdf)
- [MDR, ICLR 2021](https://openreview.net/forum?id=EMHoBG0avc1)
- [Beam Retrieval, NAACL 2024](https://aclanthology.org/2024.naacl-long.96/)
- [GenDec, arXiv 2024](https://arxiv.org/abs/2402.11166/)

## Search Stopping Rule

The novelty boundary is considered sufficiently audited for Phase A when searches stop revealing work that simultaneously satisfies all of the following: heterogeneous knowledge clients, raw-query/data-local federated model training, fixed-K complete-context action output, reader-outcome supervision, explicit cross-client complementarity, and joint client/context budgets.

No verified work in the current set satisfies the complete conjunction. This is a provisional absence claim, not proof that no such work exists. The collision audit must be rerun before a paper submission because the 2026 Federated RAG literature is changing quickly.

## Integrity Notes

- `federated` is used differently across papers. Federated search over sources is not automatically federated model training.
- Raw-data locality is not a formal privacy guarantee. V17 must not claim privacy preservation without an implemented privacy mechanism.
- A source-routing method is not a complete-context selector unless it chooses the final bounded context after cross-source retrieval.
- A centralized reader-aware selector is a technical neighbor, not a federated-training baseline unless its training contract is matched.
