# Cache and Index Consistency Audit

The Stage 0 runner performs fresh BGE encodes of every query and candidate document for each checkpoint and writes no embedding, FAISS, or retrieval-result cache. The only inherited candidate pool is a frozen sparse/routing candidate set; document embeddings are recomputed in the same adapter space as queries. Therefore no stale vector-index or cache namespace participates in this smoke.
