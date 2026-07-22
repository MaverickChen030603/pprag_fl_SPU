# Method Novelty Hypotheses, Not Claims

## Provisional method boundary

The internal `CompoRepair` idea is a structured prediction problem over an ordered, fixed-cardinality context. It starts from a frozen Top-5 baseline inside a frozen Top-10/20 pool and selects at most three legal state-dependent edits before `STOP` or exact fallback. The scientific object is not “better reranking” in general; it is whether a trajectory reaches a reader-effective context unavailable to every single legal edit.

## Hypotheses that remain testable

**NH1: Fixed-budget trajectory gap.** Existing audited methods do not jointly instantiate a frozen pool, exact five-document budget, state-dependent atomic edit trajectory, and one-call reader contract.

**NH2: Strict interaction evidence.** Existing audited context-selection work does not make best-composed-versus-all-single-edit `StrictSyn` a primary query-level quantity across readers and datasets.

**NH3: Action-conditioned opportunity decomposition.** Pool absence, single-action absence, composition-search miss, scoring miss, gating miss, and harmful realization form an auditable diagnostic not supplied by one-shot ranking or variable-k selection.

**NH4: Reader-effective composition can be learned offline.** If the oracle gap exists, imitation and trajectory-level synergy supervision may realize it without reader calls during search.

These are hypotheses until the Oracle and Learnability checkpoints pass. They must not be written as established novelty or effectiveness claims.

## Direct answers to required collision questions

1. **Complete-context multi-step edit trajectory:** Multi-step retrieval and RL subset construction exist. The current audit did not find an exact fixed-K post-retrieval atomic-edit contract, but Context-Picker and Beam Retrieval are close enough that claims must be contract-specific.
2. **Best single edit versus composed edits:** Pair/set methods compare systems or selected sets, but no audited paper was found with V16's all-legal-single counterfactual denominator. This absence is bounded by the search protocol.
3. **Context-action synergy:** Passage complementarity, conditional utility, and influence are established concepts. `StrictSyn` is only a proposed operational test for higher-order edit gain; it is not a claim that interaction itself is new.
4. **Context-Picker distinction:** Context-Picker targets a minimal sufficient, variable-sized subset through RL. V16 holds document count at five, edits an ordered baseline, limits depth to three, includes exact fallback, and measures excess over every single edit. If its code already implements equivalent ordered fixed-K edit trajectories, V16 must be repositioned.
5. **Influence distinction:** Leave-one-out influence estimates a marginal deletion effect around a context. V16 synergy compares the outcome of a composed transition with the best attainable single edit; the latter can capture non-additive gains but requires much more complete offline action evaluation.
6. **Contextual Passage Utility distinction:** Conditional passage utility asks whether one passage is useful given previous passages or traces. V16 predicts actions over the whole current ordered context, unused/removed documents, and history, and assigns trajectory-level Answer/SP/Joint and harm values.

## Claims explicitly unavailable

- “First reader-aware context selector.”
- “First set-wise or pair-complementary RAG method.”
- “First RL method for context selection.”
- “First multi-step retrieval method.”
- “Formal risk control” without conformal/PAC assumptions and proof.
- “General multi-hop improvement” before three-dataset, two-reader final confirmation.

## Naming decision

`CompoRepair` remains internal. Exact-title and repository searches found no obvious direct collision in the first pass, but a publication name will be frozen only after Oracle Checkpoint 1 and a final collision audit. A neutral fallback is **Fixed-Budget Sequential Context Repair**.
