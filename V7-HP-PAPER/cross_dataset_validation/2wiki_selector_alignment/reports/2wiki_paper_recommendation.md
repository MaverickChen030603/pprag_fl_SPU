# 2Wiki Paper Recommendation

Recommendation: `pipeline_validation_only`

2Wiki results validate dataset transfer and lexical routing effectiveness, but do not yet establish selector-level generalization beyond a strong BM25 baseline.

Current status:

- 2Wiki adapter: ready
- dev-300 lexical/BM25 reader smoke: positive
- 2Wiki action-feature alignment: complete
- selector smoke 300 gate: `stop_at_smoke_300`
- formal 1000: `skipped_gate_not_passed`

Paper-safe sentence:

> 2WikiMultiHopQA validates the cross-dataset data and reader pipeline and confirms that lexical routing is a strong external baseline; however, selector-level generalization beyond BM25 requires additional action-feature adaptation.
