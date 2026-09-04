# V20 R4-U1 Frozen Dual-Reader Preflight Unblock

生成时间：2026-08-12T17:38:37.784929+09:00

## 结论

当前状态：**R4_U1_BLOCKED_QUERY_VIEW_AND_GPU**。不得启动正式 Reader cell。

- protocol_ready：true
- contexts_ready：true
- runner_ready：true
- query_view_ready：false
- gpu_blocked：true

## 已新增的 R4-U1 文件

所有新增文件位于 `/home/iiserver31/projects/FedE4RAG-main/V7-HP-PAPER/v20_arc_fedsearch/stage_r4_frozen_reader/u1_20260812`，包括：

- `code/run_r4_frozen_generation.py`
- `code/run_r4_sealed_evaluation.py`
- `code/materialize_r4_u1_contexts.py`
- `code/finalize_r4_u1_centralized.py`
- `protocol/reader_prompt_template.txt`
- `protocol/reader_protocol_frozen.json`
- 四组 `contexts/*.jsonl` 与 `input_manifests/*_manifest.json`
- process、label firewall、context、runner separation、resume、CUDA 与 preflight 审计文件。

未修改任何 R3 retrieval 文件；未执行 reset、checkout、clean、批量 stage 或无关格式化。

## Query View 与 Label Firewall

sealed R3 packet 为 300 条、`gold_or_answer_used=false`，但 packet **不含 question 字段**。R3-H 目录内唯一带 question 的 JSONL 是含 `answer` 和 `supporting_facts` 的 manifest。U1 遵守禁令，没有用它导出问题文本，因此 `r4_label_free_query_view.jsonl` 尚未创建。

generation runner 不 import evaluator，不接受 gold manifest 参数，并强制 query view schema 恰为 `query_id + question`。本轮 labels read=false，evaluation run=false，Reader metrics computed=false。

## Frozen Contexts

| Method | N | 状态 | SHA-256 |
|---|---:|---|---|
| inherited | 300 | frozen | `5abe8644d64bfcdc49b4360abb6780cb9580b32294cdb1fe161bf4f2448eb3fe` |
| label_free | 300 | frozen | `aae08adafc06d8494d563516c2d71e5e564e4dacbe4d1cf273fa8831ad7c3813` |
| logistic | 300 | frozen | `12b23ff6c1e28552cae1f9153929339c686cf3a2615da94b9044b643794db02f` |
| centralized | 300 | frozen | `0db49a5d140f6113a817df8941c36c6f5a22e950bdb8410898f6daaa4656af13` |

四组 query ID 集合与顺序完全一致；每条 global Top-10、Reader Top-5；逐条 context hash 和整体文件 hash 均已记录。前三组 selected clients 与 R3 sealed published results 逐 query 一致。

centralized pool 在 U1 初始快照时已是 300 条且 checksum 至今未变。旧 PID 当时已不存在，U1 没有 SIGCONT、没有补跑 retrieval。complete manifest、行完整性、唯一 ID、canonical 顺序、无 gold 字段和 centralized contract 均通过后，完整池被复制为 U1 只读 cache 并安全复用。

## Frozen Protocol 与 Runner

旧 `run_r4_reader.py` 中两个 prompt f-string literal 已逐字提取，prompt SHA-256 为 `069d4e6ed8c75e99d43dcd46e2a34f8a964e3ffeae4f2839ce78e6c85d856b40`。frozen protocol SHA-256 为 `44cb088cd705503269503f9336453bc3cc65ac7cef471686ede83bc6ae7c6ce9`，文件均为只读。

FLAN 与 UnifiedQA 的 model/tokenizer revision 已显式 pin；长度、greedy decoding、float16 CUDA、batch size 4、seed 20260812 已冻结。generation/evaluation 物理分离测试 PASS；Hotpot-only 八 cell、checksum resume、duplicate/missing key、atomic marker 和只读 seal 合成测试 PASS。

正式完成度仍为 cells 0/8、predictions 0/2400、cache hits 0。

## CUDA

`nvidia-smi` 仍失败：NVML userspace 575.57，而已加载 kernel module 为 535.54.03。PyTorch 2.3.1+cu121 能枚举 4 张 A100 且 `cuda_available=true`，但按验收合同 NVML 不健康仍是基础设施 blocker。U1 未使用 sudo、未卸载模块、未 reboot、未 kill 其他进程、未改系统 CUDA。

## 剩余解锁条件

1. 提供或明确批准一次性生成与 packet canonical order 对齐的 sealed question-only view；当前 packet 本身无法提供 question。
2. 由服务器管理员处理 575.57 userspace / 535.54.03 kernel driver mismatch，通常需要维护完成后的主机重启或一致化驱动。

由于验收项 1 和 16 未通过，当前不满足 `R4_PROTOCOL_READY_GPU_BLOCKED`，也不满足 `R4_PREFLIGHT_PASS_READY_FOR_FIRST_CELL`。Reader、evaluation、labels 与 final test 均保持未触碰。
