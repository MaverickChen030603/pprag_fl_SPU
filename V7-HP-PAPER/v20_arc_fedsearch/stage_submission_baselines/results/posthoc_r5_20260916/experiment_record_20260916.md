# ProbeRoute R5 事后基线实验记录

## 1. 实验定位

本批实验补充固定预算下的客户端选择基线，用于判断既有 M2
`logistic_proberoute` 的收益是否仅来自静态候选排序或随机波动。R5
标签在执行前已经揭示，因此证据等级固定为
`retrospective_post_hoc_r5_revealed`，仅作诊断，不构成新的确认实验。

## 2. 冻结协议

- 数据集：HotpotQA、2WikiMultiHopQA、MuSiQue，每个数据集 300 个 R5 查询。
- 候选与路由：静态候选 Top-8，固定选择 3 个客户端。
- 本地检索与通信：本地深度 10，每个客户端传 5 篇文档，共传 15 篇。
- 合并与 Reader：原始分数合并 Top-10，取 Top-5，输入限制 4,000 字符和 1,024 token。
- Reader：`google/flan-t5-large` 与
  `allenai/unifiedqa-v2-t5-large-1363200`，greedy 解码，最多 32 个新 token。
- B0：只按协调端静态分数选 Top-3，探针通信为 0 字节。
- B1：在同一静态 Top-8 内随机选 3 个客户端，使用 20 个确定性哈希种子，探针通信为 0 字节。
- B2：仅用 1 维静态分数训练标准化、类别平衡的逻辑回归。
- M2：既有方法，使用 18 维客户端探针与 1 维协调端静态分数独立打分并选 Top-3。

## 3. 环境与版本

- 代码提交：`78e031deb9e4b0bd9040f6d30d33f4f122d3fe82`。
- Python 3.10.21，PyTorch 2.3.1+cu121，Transformers 4.41.2，scikit-learn 1.7.2。
- FLAN-T5-Large revision：`0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a`。
- UnifiedQA revision：`1d3b8e13b29dbd161494b0b15428378f4713c418`。
- 上下文总数 18,900；按数据集、查询和上下文哈希去重后实际推理 10,303 个，节省 45.5% Reader 计算。
- 两套 Reader 各生成 18,900 条预测，最终评分 37,800 行。

## 4. 完整性与审计

- 标签重建后重新评分既有 7,200 条预测，指标不一致数为 0。
- 固化上下文 SHA-256：`af227a242d08fb8d9c7d3c7535620aca9705d6936ad9d4d4b780926e4ff26f58`。
- FLAN 预测 SHA-256：`1c180edcd01750f58adda95006f6e333aad5c6f62d6e1d4bba266ba230905b84`。
- UnifiedQA 预测 SHA-256：`9a39da0b820e521d896cfb2562793c81beddd1ab27f19b1653ef85ded976eda6`。
- B2 在三个数据集上的 900 个查询均与 B0 路由和上下文完全一致，无需重复运行 Reader。
- HotpotQA 的 B0 与历史 M0 有 76/300 个上下文不同；2WikiMultiHopQA 与 MuSiQue 则 300/300 完全一致。

## 5. 归档位置

- 服务器完整运行：`/srv/lab/projects/jiankangchen/pprag_fl_SPU/experiments/proberoute_submission_baselines_20260915/runs/posthoc_r5_20260915`。
- 本地完整备份：`/Users/iilab/ForAgent/paper_drafts/three_tasks_20260914/experiment_results/proberoute_submission_baselines_20260916`。
- Git 只保留本目录中的小型汇总表、manifest 与审计文件，不提交逐查询预测、上下文和模型权重。
