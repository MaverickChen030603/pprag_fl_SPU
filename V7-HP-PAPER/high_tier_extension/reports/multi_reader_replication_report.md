# Multi-Reader Replication Report

Status: **partial_existing_reader_only**.

The frozen HotpotQA v2.3 outputs provide one completed reader result, `google/flan-t5-large`, but do not expose full baseline and selected context text needed to re-run `google/flan-t5-base` or `google/flan-t5-xl` safely. Therefore no new reader was launched in this extension pass.

Existing reader result:

- answer_f1_delta: +0.0023
- joint_f1_delta: +0.0150
- support_recall_delta: +0.0190
- sp_f1_delta: +0.0254

Paper implication: this extension **does not justify a multi-reader robustness claim**. The current paper may keep the existing main-reader HotpotQA claim and list multi-reader replication as a future/revision-ready experiment requiring frozen context snapshots.
