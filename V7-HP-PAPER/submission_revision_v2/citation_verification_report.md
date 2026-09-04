# Citation Verification Report

Verification date: 2026-07-13. Publication metadata was checked against ACL Anthology, PMLR, NeurIPS Proceedings, JMLR, OpenReview, or arXiv primary records. Recent Federated RAG works are labeled as preprints unless a publication record was directly confirmed.

| Citation key | Title | Authors | Venue | Year | DOI / arXiv | Verified | Used claim |
|---|---|---|---|---:|---|---|---|
| `yang-etal-2018-hotpotqa` | HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering | Zhilin Yang et al. | EMNLP | 2018 | 10.18653/v1/D18-1259 | Yes | HotpotQA supplies multi-document questions and sentence-level supporting facts. |
| `raffel-etal-2020-t5` | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer | Colin Raffel et al. | JMLR 21(140) | 2020 | JMLR 20-074 | Yes | T5 is the base text-to-text model family. |
| `liu-etal-2024-lost` | Lost in the Middle: How Language Models Use Long Contexts | Nelson F. Liu et al. | TACL 12 | 2024 | 10.1162/tacl_a_00638 | Yes | Reader performance depends on where relevant information appears. |
| `shi-etal-2023-distracted` | Large Language Models Can Be Easily Distracted by Irrelevant Context | Freda Shi et al. | ICML / PMLR 202 | 2023 | PMLR v202/shi23a | Yes | Irrelevant context can reduce reasoning accuracy. |
| `geifman-elyaniv-2019-selectivenet` | SelectiveNet: A Deep Neural Network with an Integrated Reject Option | Yonatan Geifman; Ran El-Yaniv | ICML / PMLR 97 | 2019 | PMLR v97/geifman19a | Yes | Reject-option methods motivate intervention coverage and fallback. |
| `jiang-etal-2023-llmlingua` | LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models | Huiqiang Jiang et al. | EMNLP | 2023 | 10.18653/v1/2023.emnlp-main.825 | Yes | Context compression is related but more expressive than bounded extractive actions. |
| `jiang-etal-2024-longllmlingua` | LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression | Huiqiang Jiang et al. | ACL | 2024 | 10.18653/v1/2024.acl-long.91 | Yes | Compression and reordering address density and position effects. |
| `xu-etal-2024-recomp` | RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation | Fangyuan Xu; Weijia Shi; Eunsol Choi | ICLR | 2024 | OpenReview `mlJLVigNHp` | Yes | Reader/task-trained compression and selective augmentation are adjacent approaches. |
| `xin-etal-2025-aligning` | Aligning Retrieval with Reader Needs: Reader-Centered Passage Selection for Open-Domain Question Answering | Chunlei Xin et al. | COLING | 2025 | ACL Anthology 2025.coling-main.67 | Yes | Reader-centered passage utility motivates downstream context selection. |
| `lee-etal-2025-shifting` | Shifting from Ranking to Set Selection for Retrieval Augmented Generation | Dahyun Lee et al. | ACL | 2025 | 10.18653/v1/2025.acl-long.861 | Yes | Set-level selection addresses collective multi-hop evidence requirements. |
| `yu-etal-2024-rankrag` | RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs | Yue Yu et al. | NeurIPS 37 | 2024 | 10.52202/079017-3850 | Yes | Ranking can be aligned with answer generation. Not reproduced here. |
| `mao-etal-2025-fede4rag` | Privacy-Preserving Federated Embedding Learning for Localized Retrieval-Augmented Generation | Qianren Mao et al. | arXiv preprint | 2025 | arXiv:2504.19101 | Yes | Example of federated retriever training; not evidence that the submitted organizer is private. |
| `dhasade-etal-2026-ragroute` | Efficient Federated Search for Retrieval-Augmented Generation using Lightweight Routing | Akash Dhasade et al. | arXiv v2; DAIS 2026 forthcoming per record | 2026 | arXiv:2502.19280 | Yes | Federated search can route queries among distributed sources. Not reproduced here. |
| `chakraborty-etal-2025-fedrag-survey` | Federated Retrieval-Augmented Generation: A Systematic Mapping Study | Abhijit Chakraborty; Chahana Dahal; Vivek Gupta | arXiv preprint | 2025 | arXiv:2505.18906 | Yes | Maps emerging Federated RAG architectures and evaluation gaps. |

## Corrections from v1

- Replaced `[CITATION NEEDED]` around harmful context, position effects, compression, fallback, reader-aware selection, and set selection with verified entries.
- Corrected RAGRoute to the current arXiv-v2 title and current author list. Its DAIS status is reported only as “forthcoming per arXiv,” not as an already published 2025 venue paper.
- Kept FedE4RAG and the Federated RAG survey as arXiv preprints.
- Moved local “SetR-style” and “RankRAG-style” rows to appendix diagnostics. They are not exact method reproductions and support no external superiority claim.

## Submission citation rule

Every method comparison in the v2 manuscript is either a frozen in-project baseline or explicitly labeled a conceptual proxy. Citation establishes relatedness, not empirical equivalence.
