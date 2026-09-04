# CrossEncoder Role and Fairness Audit

## Shared representation, different contracts

Both Full and CrossEncoder-Top5 use the frozen `cross-encoder/ms-marco-MiniLM-L-6-v2` checkpoint. This deliberately controls the document-level relevance representation, but it means the comparison is not representation-level independent.

### CrossEncoder inside Full

- Produces one query-document relevance feature.
- Is combined with BM25, lexical and title overlap, entity and bridge overlap, MPNet similarity, novelty, redundancy, source rank, missing-hop, opportunity, pair, preservation, and action-family signals.
- Does not directly choose or order the final five-document context.
- Is available at inference without answers, support labels, or reader outcomes.

### CrossEncoder-Top5 baseline

- Scores every candidate document in the same frozen approximately ten-document pool.
- Selects and orders five documents using only the CrossEncoder relevance score.
- Excludes pair complementarity, missing-hop, document opportunity, preservation, utility, and action-family logic.
- Uses the same document budget, context cap, answer reader, prompt, support model, and official metrics.
- The score-order variant was selected on development and frozen before the two reported holdout evaluations.

## Fair interpretation

The comparison isolates the value of the complete context-construction and selective-intervention pipeline beyond using the same relevance checkpoint as an independent document ranker. It does not isolate the CrossEncoder representation itself and does not show that Full universally outperforms relevance reranking. CrossEncoder achieves higher SP and Joint at lower latency; Full achieves higher Answer and positive Answer/Joint changes relative to Frozen Top-5.

## Required naming

Use **protocol-matched shared-checkpoint CrossEncoder baseline**. Do not use **fully independent model baseline**. A clean Full-without-CrossEncoder holdout comparison remains future work because no compatible variant was frozen before holdout inspection.
