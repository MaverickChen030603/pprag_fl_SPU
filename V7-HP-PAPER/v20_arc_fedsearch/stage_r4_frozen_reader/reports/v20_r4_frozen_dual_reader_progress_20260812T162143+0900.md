# V20 R4 Frozen Dual-Reader Validation 最新进展

生成时间：2026-08-12 16:21:43 JST  
结论：**BLOCKED，禁止启动第一个 reader-method cell。**

## 1. 权威状态

当前权威起点是 `ready_for_frozen_reader`，并已确认 `probe_routing_method_confirmed`。HotpotQA fresh holdout N=300 的 retrieval-only 结果为：inherited 0.4700、label-free 0.5900、logistic 0.6067；logistic 相对 inherited 为 +13.67pp，95% CI 为 [+9.67pp, +18.00pp]。

旧状态 `hotpot_ranker_training_complete_holdout_not_started` 已过时。R3-H fresh holdout 已经完成并成为 R4 的冻结来源；从旧 checkpoint 重跑会破坏 holdout 不变性，也违反本轮“不得重跑或修改 R3 Hotpot routing/retrieval”的约束。

## 2. 仓库与查询集

- 远端仓库：`/home/iiserver31/projects/FedE4RAG-main`
- 分支：`main`
- HEAD：`7cb8a783c35ac475f5c436817aaec282bd8d94cb`
- worktree：dirty，共 448 个状态条目；完整状态摘要 SHA-256 为 `066c116181f26c9b04fbe0d1e2ea6d1874533a0900a64034cbfc67af20fbbd27`
- 未执行 `reset`、`checkout`、`clean`，也没有覆盖用户修改。
- Hotpot immutable query manifest：300 条，SHA-256 `dae40960c20e31c4286421968842f872102ead64a989c5871687aa9c864053a8`
- 冻结 R3 packet：300 条，SHA-256 `1b1393e49db0e8735983d9a549353bda577cfa09c7c347018c87fef6859bab3f`
- 两者 query ID 集合与顺序哈希一致。

query manifest 本身含 `answer` 和 `supporting_facts`，因此不能直接作为 Reader 输入。此次 preflight 仅检查字段名与哈希，没有输出或评分任何 label 值。

## 3. 四组 Frozen Contexts

四组上下文目前**不齐全**：

| Method | 当前状态 | N | 可用 SHA-256 |
|---|---|---:|---|
| inherited | R3 packet 可重建，但尚未物化并冻结为 R4 Reader context | 0/300 | 无 |
| label-free | R3 packet 可重建，但尚未物化并冻结为 R4 Reader context | 0/300 | 无 |
| logistic | R3 packet 可重建，但尚未物化并冻结为 R4 Reader context | 0/300 | 无 |
| centralized | 仅有中断的 raw pool，不是 frozen Reader context | 144/300 | 暂停点哈希 `9cb1002b...d3c0c31` |

`input_manifests/` 尚不存在，因此无法证明四组均为 300 条、query 顺序完全一致、每条文档数符合合同，也无法冻结文档内容及顺序哈希。

## 4. Reader Protocol

两套 checkpoint 在本地缓存中可定位：

- FLAN-T5-Large：`google/flan-t5-large`，revision `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a`
- UnifiedQA-T5-Large：`allenai/unifiedqa-v2-t5-large-1363200`，revision `1d3b8e13b29dbd161494b0b15428378f4713c418`

现有代码显示的配置为：context 4000 chars、source 1024 tokens、target 32 new tokens、greedy、`num_beams=1`、`do_sample=false`、CUDA float16、batch size 4。但协议仍未完全冻结：

- prompt 只嵌在 `run_r4_reader.py` 中，没有独立 immutable template 和 prompt hash；
- runner 未显式 pin model/tokenizer revision；
- seed 缺失；
- evaluator 虽可定位并哈希，但 runner 在生成过程中逐批读取 gold、评分并写入 gold 字段，违反“先完成并冻结 2400 predictions，再读取 labels”的顺序。

因此，两套 Reader protocol 当前只能视为“部分可定位”，不能视为 preflight pass。

## 5. 推理量、GPU 与断点恢复

- 正确矩阵：2 readers × 4 methods × 300 = **2400 predictions**，共 8 cells。
- 已完成 cells：0；剩余 cells：8。
- 已生成 predictions：0；cache 命中：0。
- 最后有效研究 checkpoint：R3-H retrieval holdout 完成，状态 `ready_for_frozen_reader`。
- Reader labels 已读取：否；Reader 评分已运行：否；final test 已触碰：否。
- GPU：当前 `nvidia-smi` 返回 `Failed to initialize NVML: Driver/library version mismatch`，NVML library 575.57。
- GPU 需求：通过 preflight 后，1 张 CUDA GPU 可顺序运行；2 张可各运行一个冻结 Reader。若 OOM，只能在固定小样本等价性验证后调整 batch size。
- 合规恢复方案：每个 `(reader, method)` 独立 JSONL，以 `(reader, method, query_id)` 为主键；只复用 checksum 匹配的记录；每个 cell 完成后原子写 marker 与 checksum。

现有 runner 不满足该恢复合同：它把多方法合并到单文件，未校验预测 checksum，没有 per-cell 原子完成标记；现有 launcher 还覆盖三个数据集，若继续会产生 7200 predictions，而非 Hotpot-only 2400。

## 6. 当前进程与保护动作

发现此前已自动启动的 R4 三数据集流水线。为阻止它在 preflight 前继续生成 retrieval/context/Reader 输出，已对 PID `303527, 303530, 303533, 303534, 303535, 303536, 303537, 303538` 执行可逆 `SIGSTOP`。

所有进程目前为 `T/Tl`；没有 kill，没有删除或覆盖任何产物。Hotpot centralized raw pool 停在 144/300；Reader 尚未启动，prediction 为 0。旧 R3、M0、CTD、Reader、final-test 产物均未修改。

## 7. Blockers 与最小解决方案

1. 缺少四组完整、同序、文档内容与顺序已哈希的 Hotpot Reader contexts。
2. 缺少完整冻结协议：prompt hash、显式 model/tokenizer revision、seed、严格的 prediction/evaluation 边界。
3. 当前 Reader runner 会在生成期间读取并写入 gold/metrics，label firewall 不通过。
4. 当前 launcher 的数据集范围、2400 预测矩阵、checksum resume 和原子 cell marker 均不符合本轮合同。
5. CUDA/NVML 在维护后仍不健康。

最小处理路径：不重跑 R3-H；仅从批准的冻结来源物化并哈希四组 Hotpot contexts；把**既有批准 prompt**冻结成不可变协议文件，不设计新 prompt；显式 pin 两套 revision 与 seed；将生成和评分拆开；实现 Hotpot-only 八 cell 的 checksum resume；修复 CUDA/NVML 后重新执行 preflight。

在以上项目全部通过前，不恢复当前流水线，不启动首个 Reader cell，也不读取 labels 或 final test。
