# V17 Closest Collision Audit

## Name Search

Searches for `FedAction-RAG`, `Federated Context Action Selection`, `cross-client context synergy`, and `cross-client context selection RAG` found no verified paper using the exact `FedAction-RAG` name as of 2026-07-22. The name remains an internal engineering label because absence from search results is not trademark or exhaustive bibliographic clearance.

## Closest Technical Collisions

### 1. Context-Picker

Context-Picker already makes RL-based context subset selection and sufficiency rewards non-novel. V17 cannot claim novelty from action selection, offline RL, or context utility alone. Its only defensible additional object is client-budgeted, fixed-K complete-context action utility trained with raw-query/data locality.

### 2. SetR

SetR already models context as a set and targets multi-hop adequacy. V17 must demonstrate that client fragmentation and communication constraints change the problem, not merely reproduce centralized set selection with client IDs attached.

### 3. Contextual Passage Utility and Influence-Guided Selection

Both model context-dependent or outcome-dependent utility. V17 cannot describe reader-aware utility as a new concept. The proposed distinction is complete cross-client action value plus explicit separation of routing, retrieval, action, scoring, personalization, and gating errors.

### 4. MKP-QA and DRAG

MKP-QA routes across product domains and DRAG discovers peers in decentralized RAG. They are strong routing baselines. V17 must not call source routing a novel layer. The proposed method begins after budgeted client retrieval and selects the final K-document context.

### 5. pFedRAG and FedE4RAG

pFedRAG already combines shared and personalized embedding layers; FedE4RAG federates retriever learning. V17 cannot claim the first personalized or federated RAG model. Its hypothesis concerns shared plus local adapters for action utility, while readers, evaluators, encoders, and local retrievers remain frozen.

### 6. FD-RAG and FedMosaic

FD-RAG aggregates compact QA memories from edge clients. FedMosaic selectively aggregates document adapters and explicitly considers relevance alignment and conflict. These works narrow the remaining novelty claim: V17 must be framed in evidence-context space, with supporting-fact/reader outcomes and fixed context budget, not as generic selective aggregation across clients.

## Answers to Required Collision Questions

1. **Complete fixed-budget context actions after cross-client retrieval?** No verified reviewed work in the matrix jointly satisfies this contract. Centralized set/action selectors and federated routers exist separately.
2. **Cross-client context synergy definition?** No verified source defines the exact `BestCrossComposition - max(BestSingleClient, BestSingleCrossAction)` quantity. Related work studies source coverage, passage complementarity, memory conflict, or adapter compatibility.
3. **Nine-stage opportunity gap?** No verified source uses the complete routing/retrieval/pool/action/search/scoring/personalization/gating/harm decomposition.
4. **Federated training of complete-context action scorer?** Not found in the reviewed sources. This remains a provisional absence claim.
5. **Shared global plus client adapter for cross-client action utility?** pFedRAG has global plus personalized embedding layers, but not for complete-context action utility.
6. **Joint client budget and context budget for multi-hop QA?** Federated search methods control source/message cost, and context selectors control passage budgets; their joint optimization with reader outcomes was not found in the reviewed set.

## Claim Guardrail

The Phase-A paper language may say “we study” or “we formulate.” It may not say “first” until a pre-submission rerun of the collision audit and method-level comparison against FD-RAG, FedMosaic, MKP-QA, DRAG, SetR, and Context-Picker.
