# V20 R4 Scope and Preregistration Audit

## Verdict

The earliest on-disk root protocol fixed a three-dataset, two-reader, four-method design before reader outcomes, but the available files do not prove that this root launcher received formal user authorization. The first explicit operational decision preserved in an audit report narrowed confirmatory R4 to HotpotQA, treated 2Wiki/MuSiQue as transfer evidence, and stopped the old root jobs. The old jobs were subsequently resumed by an unresolved actor and produced the 7,200-row result. Therefore the historical three-dataset plan existed before outcomes, but the actual execution did not honor the latest recoverable pre-outcome authorization.

## Ordered Scope Artifacts

| Mtime (JST) | Path | Role | Declared scope | SHA-256 |
|---|---|---|---|---|
| 2026-08-12T15:48:16+09:00 | `V7-HP-PAPER/v20_arc_fedsearch/stage_r4_frozen_reader/run_r4_full.sh` | root launcher | three datasets / 7200 | `b3f27734028cc61dce7d69513e53cc563bf1a5020a321b3e638bb7207786999b` |
| 2026-08-12T15:48:17+09:00 | `V7-HP-PAPER/v20_arc_fedsearch/stage_r4_frozen_reader/protocol/reader_preregistration.md` | root preregistration | three datasets / six primary dataset-reader cells | `e718af50ccdce6d90c555dbcc156083fe717f62b1f296d4ab2f641dc2e1cce83` |
| 2026-08-12T16:24:44+09:00 | `V7-HP-PAPER/v20_arc_fedsearch/stage_r4_frozen_reader/reports/v20_r4_frozen_dual_reader_progress_20260812T162143+0900.md` | preflight BLOCKED report | Hotpot-only / 2400; old root jobs stopped | `d97d3e692e3b82e626d96def3fbe1c524d93667f2ffd9e37f13816be7c5d261b` |
| 2026-08-12T17:26:18+09:00 | `V7-HP-PAPER/v20_arc_fedsearch/stage_r4_frozen_reader/reports/r4_reader_go_no_go.md` | root go/no-go | three datasets / six reported primary cells | `46e8b1c770dcfd5f76534e63adc740139524d2faec6850d984795303d195d318` |
| 2026-08-12T17:26:18+09:00 | `V7-HP-PAPER/v20_arc_fedsearch/stage_r4_frozen_reader/reports/r5_final_test_recommendation.md` | R5 recommendation | inherits root confirmed status; no split/N/gate | `ced092db0782b9b0d883f98f7a3cc0c8b2b8efaaae69ff1a5574a0d1ee2d8f3b` |
| 2026-08-12T17:29:43+09:00 | `V7-HP-PAPER/v20_arc_fedsearch/stage_r4_frozen_reader/provenance_seal_20260812/sealed_source_reports/v20_r4_frozen_dual_reader_complete_report_20260812.md` | sealed local complete report | three datasets / 7200 / confirmed | `17234984c21f0d074fa8d56e23bfbd0e34a20df1031d72f4aa0d278a89eb0b43` |
| 2026-08-12T17:35:14+09:00 | `V7-HP-PAPER/v20_arc_fedsearch/stage_r4_frozen_reader/u1_20260812/protocol/reader_protocol_frozen.json` | U1 frozen protocol | Hotpot-only / 2400 | `44cb088cd705503269503f9336453bc3cc65ac7cef471686ede83bc6ae7c6ce9` |
| 2026-08-12T17:38:37+09:00 | `V7-HP-PAPER/v20_arc_fedsearch/stage_r4_frozen_reader/u1_20260812/v20_r4_unblock_progress_20260812T173837.md` | U1 progress | Hotpot-only / 0 of 2400 | `b6eef4f18c2c2d7c6629d7697ac755a7693ff7fbafcc20c272099f2512ff8194` |

## Timing

- Root runner mtime: `2026-08-12T15:47:17+09:00`; origin commit `7cb8a783c35ac475f5c436817aaec282bd8d94cb`.
- Root preregistration mtime: `2026-08-12T15:48:17+09:00`.
- Hotpot-only BLOCKED report mtime: `2026-08-12T16:24:44+09:00`; its body records a 16:21 JST stop directive.
- Formal predictions completed: FLAN `2026-08-12T17:05:25+09:00`, UnifiedQA `2026-08-12T17:05:15+09:00`.
- Published statistics mtime: `2026-08-12T17:26:18+09:00`.

## Scope Boundary

- Historical root preregistration: all three datasets are confirmatory cells.
- Later operative contract before outcomes: HotpotQA-only confirmatory; 2Wiki/MuSiQue transfer-only.
- Actual root output: all three datasets, all 24 cells, 300 rows per cell.
- U1: a later, separate Hotpot-only unblock branch; it produced no formal prediction rows and cannot be the source of the 17:05 root outputs.

## Selection Audit

No post-outcome dataset, reader, or method cell was dropped: all 24 expected cells are complete. Logistic ProbeRoute versus federated baseline and Joint F1 were named before outcomes in the root protocol. The final report's six-primary-cell basis is that root protocol and `analyze_r4.py`; U1's noncompliance finding is based on the later Hotpot-only contract, missing question-only view, mixed generation/evaluation runner, and absent cell seals. The confirmatory problem is authorization/scope supersession and the generation-label firewall, not missing-cell cherry-picking.

2Wiki and MuSiQue cannot be promoted to jointly confirmatory primary evidence from the recovered record. They remain valid exploratory/cross-dataset transfer observations. The exact actor and authority that resumed the stopped root jobs is unresolved.
