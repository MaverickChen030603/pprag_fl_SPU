# Final SIGIR-AP Review Response

## 1. End-to-end ablations

**Concern.** The previous draft did not establish end-to-end effects of pair, two-document-chain, and CrossEncoder features.

**Agreement level.** We agree that frozen end-to-end removals would strengthen component attribution.

**Existing/new evidence.** The artifact inventory found development opportunity ablations but no corresponding removal checkpoint/action set frozen before holdout inspection.

**Paper revision.** We retain the development diagnostics, label them as such, add a protocol audit, and do not train post-inspection variants. The manuscript now states: a clean frozen end-to-end ablation is unavailable because the corresponding model was not frozen before holdout inspection.

**Remaining limitation.** End-to-end component necessity remains unresolved.

## 2. Reader and support robustness

**Concern.** SP and Joint may depend on the 0.7 support threshold, and reader evidence is narrow.

**Agreement level.** Agree.

**Existing/new evidence.** We add a fixed 0.5/0.6/0.7/0.8 support-threshold grid over frozen probabilities, contexts, and answer outputs. Full-baseline and CrossEncoder-baseline SP/Joint directions remain positive on both holdouts. FLAN Answer changes from .6183 to .6271 and UnifiedQA Answer from .5662 to .5772.

**Paper revision.** The primary threshold remains 0.7; the grid is labeled post-hoc sensitivity. UnifiedQA is described as Answer-only directional evidence.

**Remaining limitation.** The support predictor is shared, so SP and Joint are not independent reader replications.

## 3. CrossEncoder dual role

**Concern.** Full and CrossEncoder-Top5 share a relevance model.

**Agreement level.** Agree; the previous "independent" shorthand was insufficient.

**Existing/new evidence.** The same frozen checkpoint is used under two contracts: one feature among many inside Full versus the sole ranking criterion in CrossEncoder-Top5.

**Paper revision.** We use "protocol-matched shared-checkpoint CrossEncoder baseline" and add a role-isolation audit. The comparison tests the complete pair/action/selection pipeline beyond relevance-only ranking, not representation-level independence.

**Remaining limitation.** A frozen Full-without-CrossEncoder holdout variant is unavailable.

## 4. Latency variance

**Concern.** Mean latency alone hides variance and module cost.

**Agreement level.** Agree; the necessary artifacts already existed.

**Existing/new evidence.** Frozen Top-5 is 140.88/252.10 mean/P95, CrossEncoder 149.90/262.59, and Full 213.48/330.56 ms/query. Full components are 70.05 generator, 0.61 selector, and 142.59 reader/serialization ms.

**Paper revision.** We restore mean, P95, call counts, memory, and component breakdown. Semantic feature computation is identified as the main added generator cost.

**Remaining limitation.** Measurements cover one GPU, batch-one post-retrieval setup.

## 5. Conformal selection and risk wording

**Concern.** "Risk-controlled" could imply finite-sample safety.

**Agreement level.** Agree.

**Existing/new evidence.** Selective prediction, risk-controlling prediction sets, TRAQ, and C-RAG are now positioned from verified formal sources.

**Paper revision.** We state that the gate is outcome-supervised and empirically development-calibrated, with no finite-sample, per-query, or group-conditional guarantee. Conformal risk control is future work, not a last-minute method addition.

**Remaining limitation.** The current operating point reports observed average intervention risk only.

## 6. Larger candidate pools

**Concern.** A 20/50-document stress test would clarify scaling.

**Agreement level.** We agree with the motivation but not with manufacturing unrelated distractors as a substitute.

**Existing/new evidence.** The frozen pool audit finds 2,973/3,000 queries with at least ten candidates, one with at least twenty, and none with fifty.

**Paper revision.** We bound the claim to approximately ten-document post-retrieval pools and list adaptive Top-L, subquadratic pair proposal, ANN pairing, and changing-index calibration as future work.

**Remaining limitation.** No natural common large-pool evaluation is available in this benchmark artifact.
