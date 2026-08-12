# V20 R5 Server-Maintenance Handoff

## Status

- Interruption time: 2026-08-12 10:44:58 UTC.
- Cause: infrastructure maintenance delivered `SIGSTOP`/`SIGTERM` to the complete R5 process tree.
- Scientific base: `13091c6deb6b3868705a49041e89f578f14b4e0e`.
- Original R5 execution code: `5520039b448bc1798bdbb5a6c79202b7edcd3785`.
- Maintenance-resume code: `c20ba1c0b3bf0e609f2bce4caec6c64977969987`.
- Labels opened: **false**.
- Readers started: **false**.
- Evaluator started: **false**.
- Old stopped process tree: fully terminated after checkpointing.

## Preserved Immutable Outputs

| Artifact | Rows | SHA-256 |
|---|---:|---|
| 2Wiki probe packets | 300 | `f8ce9f32e3d151785cf593cc75a6d5a6c0442e9e6b9006671587724177bc497e` |
| HotpotQA probe packets | 300 | `6a7073175f4b911ffcb310fc58edded406cd7dc093aff66ea06cc17d8b93e256` |
| MuSiQue probe packets | 300 | `3c02533103e08bbba0303dbc0c5252ca7abec0ab9bbdf24157aaeac368dde1fd` |
| HotpotQA inherited routes | 300 | `8bc7eb6b99cbc5613382784bb5413b573521eeea3cdb67fcc17de2315db8c7aa` |

The interrupted centralized outputs were archived for audit only: 2Wiki 98 rows, HotpotQA 88 rows, and MuSiQue 92 rows. They will not be mixed with the resumed run because the centralized generator is non-resumable.

## Local Backup

- File: `stage_r5_final_test/checkpoints/v20_r5_premaintenance_checkpoint_20260812.tar.gz`
- SHA-256: `a14426ce0309610476e1599cffc05a24b1dad03437218be7579a65aaf3e72b46`
- Contents: no-leak protocol, frozen sample inputs, frozen artifact/cost contracts, completed probe packets, Hotpot route, partial centralized outputs, and checkpoint metadata.
- Sealed label files are not included.

## Recovery Contract

Run only:

```bash
nohup bash V7-HP-PAPER/v20_arc_fedsearch/stage_r5_final_test/resume_r5_after_maintenance.sh \
  > V7-HP-PAPER/v20_arc_fedsearch/stage_r5_final_test/run_20260812/logs/r5_resume_after_maintenance.nohup.log 2>&1 < /dev/null &
```

The recovery script validates that labels remain sealed and that all immutable Phase-1a files have exactly 300 rows. It archives and rebuilds incomplete centralized/context files, resumes reader predictions safely, and permits the evaluator to open labels only after both 3,600-row reader outputs and completion markers exist.

Do not rerun `run_r5_full.sh`; doing so would revisit preflight/freeze phases and violate the intended maintenance continuation path.
