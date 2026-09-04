# V20 Stage R4 Reader Preregistration

- Data: the three frozen R3 N=300 holdouts, unchanged query IDs.
- Conditions: federated baseline, label-free ProbeRoute, logistic ProbeRoute, and centralized retrieval reference.
- Primary outcome: query-level Joint F1, Logistic ProbeRoute versus federated baseline.
- Readers: frozen FLAN-T5-Large and UnifiedQA-T5-Large under the audit contract.
- Statistics: 5,000 paired bootstrap resamples; Joint primary uncorrected, remaining paired tests BH-FDR reported.
- Reader evaluation never changes packets, clients, raw merge, context K, prompt, decoder, or retrieval configuration.
