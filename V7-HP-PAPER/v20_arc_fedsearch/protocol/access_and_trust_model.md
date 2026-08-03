# Access and Trust Model

## Setting

V20 uses cooperative cross-silo federated search.  A client holds raw document
text, its full local index, local dense embeddings, local BM25 statistics, and
its own query logs.  The coordinator may receive the client ID, a resource
profile derived from the train corpus, client size, non-sensitive term sketches,
score-distribution summaries, and query-time returned document IDs, scores, and
permitted passages.

## Explicitly Out of Scope

The coordinator does not centrally retain a full client corpus, client index,
or all document embeddings.  V20 does not implement differential privacy,
secure aggregation, a TEE, homomorphic encryption, or PIR.  We therefore claim
only **data-local federated search with communication-constrained retrieval**,
not formal privacy preservation.

## Query-Time Contract

1. The coordinator routes a query to a budgeted subset of clients.
2. Each contacted client performs local hybrid retrieval and returns an allowed
   number of document IDs, scores, and passages.
3. The coordinator calibrates and merges only returned results, then selects a
   global pool and a reader context.
4. A conditional second round may contact unvisited clients only under the
   frozen mean-contact and document budget.

No gold support, answer string, answer-presence indicator, reader target, or
final-test label is available in any query-time feature.
