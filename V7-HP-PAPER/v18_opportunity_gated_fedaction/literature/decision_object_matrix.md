# Decision-Object Matrix for FedAction-RAG v2

This matrix prevents false numerical comparisons across incompatible contracts.
FedAction-RAG v2's prospective decision object is a **complete reader-context
action after budgeted cross-client retrieval**: a full K=5 document context,
including an exact fallback action. It is neither a generic model-update
selector nor a source router alone.

| Family / work | Decision object | Time | Locality and shared artifact | Retrieval/context unit | Reader supervision; personalization | Communication / latency | Answer/evidence metric | Official code; fair numeric comparison | Relationship to FedAction-RAG v2 |
|---|---|---|---|---|---|---|---|---|---|
| **A. FedAvg** | model parameters | training | local data; averaged model | task-specific; no source budget | task loss; no | update bytes / rounds | task-dependent | official; only Phase C under matched model | optimizer baseline, not context composition |
| Federated Dropout / Adaptive FD | parameter blocks/subnetworks | training | local data; partial updates | no document action | task loss; optional | payload / rounds | task-dependent | variants; Phase D only | update selection, unlike query-time action |
| FedPAQ | quantized model updates | training | local data; quantized updates | no document action | task loss; no | bits / rounds | task-dependent | official variants; Phase D reference | communication compression |
| V7-HP selective upload | retriever/scorer parameter blocks | training | client-local data; selected blocks | historical utility selection | proxy or reader-aligned; limited | payload ratio | QA proxy / reader QA | internal; Phase D only | predecessor update-selection line |
| **B. pFedRAG** | retriever embedding/personal layers | train + inference | local corpora; shared plus personalized embedding | local vector retrieval | downstream response; yes | updates/retrieval cost | response quality | paper artifact; reference not direct table | V18 personalizes action utility, not embeddings |
| FedRAG framework | RAG component parameters/config | training | configurable data; model updates | framework-level | task-dependent; supported | Flower-style updates | task-dependent | framework; infrastructure comparison | host framework, not action method |
| **C. RAGRoute / Efficient Federated Search** | source/peer set | inference | distributed sources; routing scores | source then documents | relevance/answer indirect; no required adapter | sources contacted / latency | source or QA | official only if contract-compatible | V18 router component and baseline |
| FeB4RAG | information resource list | inference benchmark | heterogeneous resources; ranked results | source selection/result merge | relevance judgments; no | requests/source cost | relevance, not full QA labels | dataset; routing/merge audit | source benchmark, not fixed-K actions |
| **D. FedMosaic** | selected parametric adapters | training/aggregation | silo documents via adapters | adapter aggregation | relevance utility; silo-aware | storage/update traffic | task quality | conceptual/ablation inspiration only | motivates context-level conflict head |
| Conventional result merging | document/result list | inference | local results; score list | flat or normalized union | typically none; no | docs/bytes/merge time | retrieval or QA | direct Fast Path baseline | non-compositional alternative |
| **E. FD-RAG** | memory/hypergraph route, slow trigger | inference + learning | edge memory; aggregated representation | memory matching then LLM | QA outcome; no fixed local adapter | memory/latency | QA | conceptual comparison | fast/slow inspiration; not fixed-K action |
| HyFedRAG | multimodal/anonymous representation | train/inference | heterogeneous sources; representations | representation fusion | task-dependent | representation traffic | task-specific | systems discussion only | incompatible modality/context contract |
| FICAL / Federated In-Context Agent Learning | knowledge compendium/agent state | training | distributed agent information | agent collaboration | agent/task reward | messages/updates | task reward | systems discussion only | different shared artifact |
| **F. RAGAS** | evaluation signal | evaluation | corpus agnostic | supplied context | automatic proxy; no | not primary | relevance/faithfulness | official; auxiliary only | cannot replace native metrics |
| MSRS | multi-source retrieval/synthesis benchmark | evaluation | multi-source documents | long-text synthesis | benchmark-dependent | source/latency | summary/faithfulness | external validation after core QA | post-core source/synthesis validation |

## Claim boundary

Do **not** claim first selective federated RAG, first federated aggregation,
first source routing, first personalized FedRAG, or first distributed evidence
composition. The only prospective contribution claim is conditional: if
Checkpoint-A establishes Bc-realizable opportunity and later checkpoints learn
it, FedAction-RAG v2 studies reader-aware complete-context actions under
budgeted cross-client retrieval and federated personalization.
