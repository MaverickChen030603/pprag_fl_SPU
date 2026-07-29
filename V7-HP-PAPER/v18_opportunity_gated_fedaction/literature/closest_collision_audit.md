# Closest-Collision Audit

The V18 design combines federated retrieval/personalization, distributed source
routing, and reader-aware context selection. None alone licenses a broad
novelty claim; every proposed distinction is conditional on evidence.

| Collision risk | What is already established | Required V18 evidence |
|---|---|---|
| pFedRAG / FedRAG | FL can personalize retrievers/RAG components under non-IID data. | The learned object is complete-context action utility and shared-plus-local action models improve matched client-level outcomes. |
| RAGRoute / FeB4RAG | Sources can be routed and results merged. | Bc<=3 source recall and reader-context actions improve over routing plus flat merge, not merely over weaker source access. |
| FedMosaic | Selective aggregation/conflict control stabilizes parametric silo knowledge. | Context-level conflict/complementarity protects answer anchors and improves reader outcomes; FedMosaic is not portrayed as a document baseline. |
| FD-RAG | Fast memory matching can be separated from slower reasoning. | A small opportunity-gated slow path beats random/uncertainty gates and lowers always-compose cost. |
| SetR, Context-Picker, utility selectors | Centralized reader-aware set selection is known. | Client locality, Bc-realizable opportunity, and federated action learning each contribute measured value. |
| V7-HP / V14 Full | Risk-gated post-retrieval context repair is known. | V18 improves over a budget-matched Fast Path/single-edit comparator and reports remaining harm. |

If those conditions fail, the correct output is a federated opportunity analysis
or targeted high-dispersion slow path, not a general FedAction-RAG method paper.
