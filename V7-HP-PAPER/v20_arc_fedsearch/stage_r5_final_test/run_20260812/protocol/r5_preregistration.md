# V20 Stage R5 Preregistration

- Nature: one-shot confirmatory evaluation; method development is closed.
- Data: V17 untouched train-derived final split, deterministic hash sample N=300/dataset.
- Methods: federated baseline, label-free ProbeRoute, logistic ProbeRoute, centralized retrieval reference.
- Primary: Logistic ProbeRoute minus federated baseline Joint F1.
- Readers: frozen FLAN-T5-Large and UnifiedQA-T5-Large.
- Statistics: 5,000 paired bootstrap resamples; primary Joint uncorrected; Answer/SP secondary BH-FDR.
- SP is context-level and reader-independent because both readers share the frozen V16 support predictor.
- Labels may be opened only after all 7,200 unscored predictions pass checksum validation.
- No observed result may trigger tuning, replacement, or rerun.
