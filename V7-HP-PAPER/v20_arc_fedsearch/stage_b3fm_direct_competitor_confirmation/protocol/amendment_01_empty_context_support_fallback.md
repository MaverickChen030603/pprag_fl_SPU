# Amendment 01: Empty-Context Support Fallback

Date: 2026-09-30

The original B3F-M contract reused the C1/C2 support predictor but omitted its
behavior when `RAGRoute-Threshold` legally selects zero clients. The corrected,
deterministic rule is:

```text
if reader_context_docs == []: predicted_support = []
```

This does not alter M2, RAGRoute, static routing, candidate IDs, local depth,
document count, merge, Reader prompt, generation, or the zero-client/no-fallback
Threshold rule. It only defines the previously missing support-output schema for
an empty Reader context. All pre-amendment contexts and partial Reader outputs
are quarantined; blind routing, context materialization, both Reader runs, and
support prediction restart from scratch before any label unseal.
