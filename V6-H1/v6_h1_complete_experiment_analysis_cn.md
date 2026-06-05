# V6-H1 完整实验分析

- 生成时间：2026-06-05T14:57:33

## 1. 实验目的

V6-H1 的核心目的不是继续扩大通信预算，而是在 V6 的同预算比较框架上加入 stable hard-query 子集，验证原先在全量下游集上不明显的策略差异，是否能在更难的检索样本上被放大。

## 2. 主要结论

- V6-H1 全流程已完成，包含全量 downstream 与 stable hard-query downstream 两类评估。
- 全量 downstream 上，各选择性上传策略的 `cos_3/recall_3/mrr/NDCG` 仍高度接近，说明原始下游集依然偏容易，难以区分 V3/V6/H1 这类细粒度选择策略。
- stable hard-query 子集确实更难，但在 `v6h1_budget_aligned_stable_hardquery` 与 `v6h1_hardquery_stable_hardquery` 中，多数方法指标同时为 0，说明该子集在主任务设定下过于苛刻，更多反映“共同失败样本”，没有充分形成可比较的梯度。
- 在 `v6h1_heterogeneity_stable_hardquery` 中，强异构任务下出现了更有信息量的差异：`delta_norm/hypernet_v3/hypernet_v6` 在部分 alpha 场景达到 `cos_3=0.3333, recall_3=0.3333, mrr=0.1111, NDCG=0.1667`，而 random 为 0。这说明 hard-query 方向是有效的，但 V6-H1 仍未让 `hypernet_v6` 明显超过 `hypernet_v3`。
- 通信效率方面，`hypernet_v3/hypernet_v6` 通常保持约 0.22-0.23 payload，明显低于 random 的约 0.2575，也低于部分 delta_norm 强异构场景的 0.30 左右；因此 H1 继续支持“选择性上传能显著压缩通信且保持下游效果”的结论。

## 3. 同预算主对照分析

### v6h1_budget_aligned 全量评估

| suite | task | strategy | payload | reduction | cos_3 | recall_3 | mrr | NDCG |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v6h1_budget_aligned | num5_dir_a03_imb00_ts0_v6h1 | delta_norm | 0.2431±0.0032 | 0.7569±0.0032 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_budget_aligned | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v3 | 0.2268±0.0019 | 0.7732±0.0019 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_budget_aligned | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v6 | 0.2268±0.0019 | 0.7732±0.0019 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_budget_aligned | num5_dir_a03_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |


在同预算主对照中，`hypernet_v3` 与 `hypernet_v6` 的 payload 基本一致，且下游指标完全相同；random 的 payload 更高但没有带来更高下游收益，delta_norm payload 也更高。这说明 H1 保持了 V6 的预算公平性，但没有证明 H1 的选择信号本身优于 V3。

### v6h1_budget_aligned stable hard-query 评估

| suite | task | strategy | payload | reduction | cos_3 | recall_3 | mrr | NDCG |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v6h1_budget_aligned_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | delta_norm | 0.2431±0.0032 | 0.7569±0.0032 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_budget_aligned_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v3 | 0.2268±0.0019 | 0.7732±0.0019 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_budget_aligned_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v6 | 0.2268±0.0019 | 0.7732±0.0019 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_budget_aligned_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |


该表显示 stable hard-query 子集在 budget_aligned 设置下对所有方法都过难，所有方法核心指标同时为 0。因此它不能作为证明 `hypernet_v6` 优于 `hypernet_v3` 的主要证据，只能说明 hard-query 构造捕捉到了共同失败样本。

## 4. 异构场景分析

### v6h1_heterogeneity 全量评估

| suite | task | strategy | payload | reduction | cos_3 | recall_3 | mrr | NDCG |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v6h1_heterogeneity | num5_dir_a005_imb00_ts0_v6h1 | delta_norm | 0.3071±0.0040 | 0.6929±0.0040 | 0.9600 | 0.9333 | 0.9067 | 0.8960 |
| v6h1_heterogeneity | num5_dir_a005_imb00_ts0_v6h1 | hypernet_v3 | 0.2240±0.0023 | 0.7760±0.0023 | 0.9600 | 0.9366 | 0.9067 | 0.8975 |
| v6h1_heterogeneity | num5_dir_a005_imb00_ts0_v6h1 | hypernet_v6 | 0.2240±0.0023 | 0.7760±0.0023 | 0.9600 | 0.9366 | 0.9067 | 0.8975 |
| v6h1_heterogeneity | num5_dir_a005_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a01_imb00_ts0_v6h1 | delta_norm | 0.3042±0.0060 | 0.6958±0.0060 | 0.9600 | 0.9433 | 0.9067 | 0.9016 |
| v6h1_heterogeneity | num5_dir_a01_imb00_ts0_v6h1 | hypernet_v3 | 0.2221±0.0019 | 0.7779±0.0019 | 0.9600 | 0.9333 | 0.9067 | 0.8960 |
| v6h1_heterogeneity | num5_dir_a01_imb00_ts0_v6h1 | hypernet_v6 | 0.2221±0.0019 | 0.7779±0.0019 | 0.9600 | 0.9333 | 0.9067 | 0.8960 |
| v6h1_heterogeneity | num5_dir_a01_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a03_imb00_ts0_v6h1 | delta_norm | 0.2431±0.0032 | 0.7569±0.0032 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v3 | 0.2268±0.0019 | 0.7732±0.0019 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v6 | 0.2268±0.0019 | 0.7732±0.0019 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a03_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a05_imb00_ts0_v6h1 | delta_norm | 0.2418±0.0015 | 0.7582±0.0015 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a05_imb00_ts0_v6h1 | hypernet_v3 | 0.2237±0.0009 | 0.7763±0.0009 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a05_imb00_ts0_v6h1 | hypernet_v6 | 0.2237±0.0009 | 0.7763±0.0009 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |
| v6h1_heterogeneity | num5_dir_a05_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.9400 | 0.9133 | 0.9000 | 0.8855 |


全量异构评估显示，强异构 alpha 较小时，选择性策略可以在更低 payload 下保持与全量/随机相近甚至略好的检索指标；但 `hypernet_v6` 与 `hypernet_v3` 仍基本重合。

### v6h1_heterogeneity stable hard-query 评估

| suite | task | strategy | payload | reduction | cos_3 | recall_3 | mrr | NDCG |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v6h1_heterogeneity_stable_hardquery | num5_dir_a005_imb00_ts0_v6h1 | delta_norm | 0.3071±0.0040 | 0.6929±0.0040 | 0.3333 | 0.3333 | 0.1111 | 0.1667 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a005_imb00_ts0_v6h1 | hypernet_v3 | 0.2240±0.0023 | 0.7760±0.0023 | 0.3333 | 0.3333 | 0.1111 | 0.1667 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a005_imb00_ts0_v6h1 | hypernet_v6 | 0.2240±0.0023 | 0.7760±0.0023 | 0.3333 | 0.3333 | 0.1111 | 0.1667 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a005_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a01_imb00_ts0_v6h1 | delta_norm | 0.3042±0.0060 | 0.6958±0.0060 | 0.3333 | 0.3333 | 0.1111 | 0.1667 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a01_imb00_ts0_v6h1 | hypernet_v3 | 0.2221±0.0019 | 0.7779±0.0019 | 0.3333 | 0.3333 | 0.1111 | 0.1667 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a01_imb00_ts0_v6h1 | hypernet_v6 | 0.2221±0.0019 | 0.7779±0.0019 | 0.3333 | 0.3333 | 0.1111 | 0.1667 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a01_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | delta_norm | 0.2431±0.0032 | 0.7569±0.0032 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v3 | 0.2268±0.0019 | 0.7732±0.0019 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | hypernet_v6 | 0.2268±0.0019 | 0.7732±0.0019 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a03_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a05_imb00_ts0_v6h1 | delta_norm | 0.2418±0.0015 | 0.7582±0.0015 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a05_imb00_ts0_v6h1 | hypernet_v3 | 0.2237±0.0009 | 0.7763±0.0009 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a05_imb00_ts0_v6h1 | hypernet_v6 | 0.2237±0.0009 | 0.7763±0.0009 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| v6h1_heterogeneity_stable_hardquery | num5_dir_a05_imb00_ts0_v6h1 | random | 0.2575±0.0015 | 0.7425±0.0015 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |


hard-query 异构评估是 V6-H1 中相对最有价值的部分。它显示 random 在 hard 子集上更容易完全失败，而 `delta_norm/hypernet_v3/hypernet_v6` 在 alpha=0.005/0.01 场景可以命中部分 hard query。不过 `hypernet_v6` 没有超过 `hypernet_v3`，说明当前 H1 的 hard-query 信息还没有有效转化为更优的参数块选择。

## 5. Hard Query 构建质量判断

- stable hard-query 数量：3
- 从结果看，hard-query 子集的区分力仍不足：一部分设置下所有方法均为 0，另一部分设置下非随机方法共同提升但 V6-H1 与 V3 重合。
- 这说明 hard-query 方向正确，但构建规则需要从“共同失败”转向“可被更好策略挽救的困难样本”。

## 6. 对论文价值的判断

- V6-H1 可以作为论文中的重要负结果/诊断实验：它证明仅仅把评估限制在 hard-query 子集上，不一定自然放大方法差距。
- V6-H1 支持的正向结论是：选择性上传在 0.22-0.23 payload 附近可以维持全量下游效果，并在强异构 hard-query 场景下优于 random。
- V6-H1 尚不能支持的强结论是：`hypernet_v6` 或 H1 方法显著优于 `hypernet_v3`。

## 7. 下一步建议

- 重新定义 hard-query：不要只取 baseline 共同失败样本，而应构建“random/delta 失败但 V3/V6 有机会命中”的 recoverable-hard 子集。
- 增强下游差异：扩大 HotpotQA fullwiki/多跳证据检索、增加 answer-level EM/F1 或 support fact F1，而不是只依赖当前 RAGTest 的 top-k 检索指标。
- 引入 query-level utility feedback：让 hard-query 信息参与训练或参数块选择，而不仅仅参与评估。
- 报告写法上，建议把 V6-H1 定位为“hard-query 诊断实验”，核心贡献是发现当前下游评估瓶颈，而不是宣称 H1 已经显著优于 V6/V3。
