# R5-C1 Maintenance Recovery

Status at `2026-08-12T19:44:45+09:00`: Stage B was intentionally interrupted for planned server maintenance.

## Preserved state

- Frozen execution contract SHA-256: `3c91002ba32d65e2b0672208de5ba0a73dab51560c9dfece330171a9e3edebdd`
- Inherited routes: 4,200 complete rows
- Probe packets: 54 valid unique rows
- Probe packet partial SHA-256: `c7925df66a333e89800ea55cb3cba81aee6d3b0d03fff4d43ed58b4afe6018c1`
- Contexts: not started
- Reader predictions: not started
- Sealed evaluation: not started
- Official test: not accessed
- Interrupted PGID `354262`: terminated after stop-and-snapshot

The existing partial packet file must not be deleted, truncated, or overwritten.

## Recovery command

After the server returns, first verify CUDA with the frozen environment. Then launch exactly once:

```bash
cd /home/iiserver31/projects/FedE4RAG-main
setsid nohup bash \
  V7-HP-PAPER/v20_arc_fedsearch/stage_r5_c1_hotpot_confirmation/prereg_20260812T190640+0900/code/resume_r5_c1_stage_b_after_maintenance.sh \
  > V7-HP-PAPER/v20_arc_fedsearch/stage_r5_c1_hotpot_confirmation/prereg_20260812T190640+0900/r5_c1_maintenance_resume_launcher.log \
  2>&1 < /dev/null &
```

The recovery runner validates the approval, contract, frozen split, all 4,200 inherited routes, and the uniqueness/subset integrity of the partial packets before using the existing `--resume` path. It will not overwrite contexts or predictions. Stage C remains separate and may run only after the 8,400-row atomic Stage B completion marker validates.
