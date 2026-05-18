from llama_index.core.postprocessor import LongContextReorder


def get_postprocessor(cfg):
    # Keep optional rerankers lazily imported so the retrieval-only path
    # does not require every external postprocessor package to be installed.
    if cfg.postprocess_rerank == "long_context_reorder":
        return LongContextReorder()
    if cfg.postprocess_rerank == "colbertv2_rerank":
        from llama_index.postprocessor.colbert_rerank import ColbertRerank

        return ColbertRerank()
    if cfg.postprocess_rerank == "cohere_rerank":
        from llama_index.postprocessor.cohere_rerank import CohereRerank

        return CohereRerank()
    if cfg.postprocess_rerank == "bge-reranker-base":
        from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

        return FlagEmbeddingReranker(model="BAAI/bge-reranker-base")
    raise Exception("postprocess_rerank not supported: %s" % cfg.postprocess_rerank)
