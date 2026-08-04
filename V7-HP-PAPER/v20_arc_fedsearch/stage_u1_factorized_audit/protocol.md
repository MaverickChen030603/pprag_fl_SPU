# V20 U1 Frozen Audit Contract

U1 evaluates only the already frozen 2WikiMultiHopQA and MuSiQue N=300 development slices. Partition, query IDs, inherited Bc=3 route, local depth=10, communication budget=15, and global pool=10 remain fixed. No reader, learned router, learned retriever, learned calibrator, final-test data, or gold-derived inference feature is allowed.

The local ranker pool is built from the same per-client BM25 top-100 candidates. It stores top-10 lists for BGE dense, BM25, fixed `0.55 dense + 0.45 sparse`, and RRF. Only the audit script reads support labels, exclusively to construct a maximum-three-client Oracle route and to score offline metrics.
