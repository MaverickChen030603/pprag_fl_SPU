# R2 Baseline Contract Matrix

| ID | Contract in R2 | Status / boundary |
|---|---|---|
| B0 | inherited origin-plus-topic-centroid Bc=3 | Exact inherited baseline |
| B1 | seeded random Bc=3 | Internal control |
| B2 | single-centroid independent top-3 | Reimplemented under frozen profile contract |
| B3 | multi-prototype independent top-3 | Reimplemented under frozen profile contract |
| B4 | lexical resource-sketch top-3 | Reimplemented under frozen profile contract |
| B5 | ReDDE-style collection-resource score | Contract-matched approximation; not official reproduction |
| B6 | multi-label classifier | To be trained on Router-Train only |
| B7 | RAGRoute-style lightweight router | Contract-matched approximation; not official reproduction |
| B8 | ReSLLM-style zero-shot teacher | Optional resource-card-only baseline |
| B9 | SLAT-style distilled student | Enabled only if B8 improves Router-Dev coverage |

No simplified implementation will be described as an official result. Every
comparison keeps Bc=3, partition, local dense ranking, A0 and 15 documents.
