# R3 Compact Probe Wire-Payload Audit

This audit does not rerun retrieval, modify a query, change a feature, or
change any P0--P5 client selection. It reads the two-run verified Probe-Dev
transcript and encodes each candidate client's existing scalar feature vector
in a fixed little-endian float32 schema.

## Formal returned payload

- One schema header per query: 16 bytes.
- One scalar vector per probed client: 18 float32 values, or 72 bytes.
- No title string, document text, passage ID, full local embedding, answer,
  support label, or reader information is sent in the formal probe payload.
- The server already knows the P0 candidate client IDs and static profile score;
  these are not returned by the client.

The verbose JSON transcript is retained only as a debugging artifact. It is not
the claimed communication wire format. The audit verifies that float32
round-trip reconstruction produces exactly the original P0--P5 selections for
both `L=5` and `L=8`, and it reuses the frozen retrieval outcome columns rather
than rerunning any local or global retrieval.

`ranker_training/` and reader evaluation remain prohibited in this audit.
