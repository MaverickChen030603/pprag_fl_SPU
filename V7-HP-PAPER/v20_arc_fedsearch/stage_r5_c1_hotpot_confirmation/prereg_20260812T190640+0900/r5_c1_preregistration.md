# V20 R5-C1 HotpotQA Sealed Confirmation Preregistration

## Status and boundary

`R5_C1_PREREGISTERED_WAITING_HUMAN_APPROVAL` at `2026-08-12T19:33:55+09:00`. This is a new sealed holdout sampled from unused HotpotQA official training IDs. It is not the official final test. R4 remains valid exploratory evidence only and is outside this primary family.

No formal retrieval, Reader generation, evaluation, bootstrap, R5-C1 label access, or official-test access occurred in Stage A.

## Frozen split

- Candidate source SHA-256: `205173be35443520aa89478ad7b1084a9be4f8ae4f542672c18cc25b5add1cdc`
- Eligible IDs: `69147`
- Selected N: `4200`
- Salt: `v20-r5-c1-hotpot-confirmation-20260812`
- Query-ID order SHA-256: `6e7e39e64b0b70ad7c76fe5e15ee982a95115c446e5b51e0dfe70f8034faf912`
- Question-only manifest SHA-256: `3c10940a73576b994c7cfb69c883a5cf60d0929207a00050e278a8afeabac092`
- Historical / old-R5-300 / Logistic-train-5000 overlap: `0 / 0 / 0`

## Primary confirmation

Frozen Logistic ProbeRoute is compared with the inherited federated baseline/H0 using `google/flan-t5-large@0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a`. The statistical unit is query and the primary outcome is Logistic-minus-baseline Joint F1. The single primary test uses 5,000 query-level paired bootstrap resamples, seed `20260812`, a two-sided plus-one-corrected p-value, and a 95% percentile CI. Confirmation requires both positive mean delta and CI lower bound above zero. No FDR is used because there is one primary comparison.

Secondary outputs are Answer EM/F1, SP EM/F1, Joint EM, complete-support rate, complete-support rescue/harm/unchanged transitions, and win/tie/loss. A negative Answer F1 delta must be reported as a trade-off even if the primary confirms.

## Frozen retrieval and Reader

The R3 wire accounting is `592 B = 16 + 8 x 72`: each of eight candidates returns 18 float32 probe values. `static_score` is already coordinator-resident and joins those 18 values as the frozen Logistic model's 19th input; it is not retransmitted. Logistic uses C=1, balanced classes, liblinear, seed 20260807, frozen checkpoint SHA-256 `0d07b5cd47c3b6c666c6a1fd70c1ca9e4185307bece207d43b8b6b21ecdcc166`. Three clients, local depth 10, five documents/client, 15 transmitted, stable raw-dense global Top-10, Reader first five, no fill, no percentile primary, and no Reader-aware rerank are frozen.

The Reader inherits the R4 prompt SHA-256 `069d4e6ed8c75e99d43dcd46e2a34f8a964e3ffeae4f2839ce78e6c85d856b40`, 4,000 context characters, 1,024 input tokens with right truncation, greedy decoding, one beam, no sampling, 32 new tokens, CUDA float16, batch size 4, and seed 20260812.

## Power and firewall

The P0 value N=4,110 is reproduced using query-level paired deltas, two-sided alpha 0.05, 80% power, the maximum observed query-delta SD, and a 50%-shrunken smallest positive R4 cell effect. Analytic N is 4,110; fixed-seed residual-bootstrap planning power is `0.8174` at N=4,200. No implementation error was found, so the mentor-fixed N=4,200 is unchanged.

Stage B cannot import the evaluator or open gold. Stage C refuses gold until exactly 8,400 unique predictions and matching prediction/context/split checksums are present. Synthetic-only validation passed. Formal paths remain empty.

## Frozen hashes

- Execution contract: `3c91002ba32d65e2b0672208de5ba0a73dab51560c9dfece330171a9e3edebdd`
- Expected artifacts: `1e0b305d0337abcfb371c3d089b7d6e2f9047626641285b45167c3bc277e905e`
- Artifact freeze audit: `7b2fab0be5472237c54ea9451086974425fb165fea2e0fcc44a57e8a5458e1a8`
- Label firewall audit: `0410124f5a0e1d3c3ce0444045a28e0df4d913c8ace642549c810abe193aed14`
- Sealed evaluator code: `a32733137122c8ebe9a2f7e47376b625fc211a25ef58ce0151abf6e731d0f256`
- Stage-B runner: `c48628a902ee0091268fe72e0e712842df529ca888714c57a34280f68872fadf`
- Stage-C runner: `28c30d1afeefc41f77dc33b5c47c6e73ea1a928ef4634851b365474025e8cf2b`

Formal execution remains blocked pending a separate human approval file that binds the execution-contract SHA-256.
