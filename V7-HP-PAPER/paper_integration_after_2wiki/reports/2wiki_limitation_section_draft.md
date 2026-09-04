# 2Wiki Limitation Section Draft

We further tested the pipeline on 2WikiMultiHopQA as an external sanity check. A strong lexical/BM25 baseline substantially improved reader-backed evidence and joint metrics over the raw context order, indicating that the dataset adapter and reader evaluation pipeline transfer correctly. However, when evaluated against this strong BM25 baseline, the HotpotQA-trained selector and the 2Wiki cross-fitted selector did not establish reliable selector-level generalization.

A BM25-anchor repair reduced negative transfer and nearly matched BM25, with answer-F1 delta +0.0017, evidence-F1 delta +0.0073, and joint-F1 delta +0.0002 against BM25. This gain is too small to justify a full 1000-sample validation.

Oracle diagnostics show that positive actions beyond BM25 exist (73 / 300 queries), but only 33 / 300 queries expose strict positive actions in the current BM25-anchor action table. The current no-leak features and safety predictor do not identify these actions reliably: the answer-safe AUC is 0.5567 and the paper-positive AUC is 0.5451.

We therefore report 2Wiki as a diagnostic limitation rather than as a main generalization claim.
