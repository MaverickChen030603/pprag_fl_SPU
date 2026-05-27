# V7 起跑检查记录

检查日期：2026-05-27  
本地工作区：`/Users/chenpi/ForAgent`  
目标服务器：`iia100.slis.tsukuba.ac.jp`  
目标服务器项目根目录：`/home/iiserver31/projects/FedE4RAG-main`

## 1. 已完成

已在本地创建：

```text
/Users/chenpi/ForAgent/V7
```

已写入 V7 完整实验方案：

```text
V7/V7_experiment_set_cn.md
```

已写入 V7 实验矩阵：

```text
V7/configs/v7_experiment_matrix.yaml
```

已写入服务器自动化脚本：

```text
V7/scripts/bootstrap_v7_from_v6.sh
V7/scripts/deploy_v7_to_server.sh
V7/scripts/collect_v7_results.py
V7/scripts/write_v7_analysis.py
V7/scripts/sync_github_v7.sh
```

## 2. 本地语法检查

已通过：

```bash
bash -n V7/scripts/bootstrap_v7_from_v6.sh
bash -n V7/scripts/deploy_v7_to_server.sh
bash -n V7/scripts/sync_github_v7.sh
python3 -m py_compile V7/scripts/collect_v7_results.py V7/scripts/write_v7_analysis.py
```

## 3. 服务器连通性检查

已执行：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 iiserver31@iia100.slis.tsukuba.ac.jp
```

结果：

```text
ssh: connect to host iia100.slis.tsukuba.ac.jp port 22: Operation timed out
```

收到服务器账号密码后，已再次使用交互式密码自动化尝试部署：

```bash
V7_SSH_PASSWORD='********' bash V7/scripts/deploy_v7_to_server.sh
```

结果仍为：

```text
ssh: connect to host iia100.slis.tsukuba.ac.jp port 22: Operation timed out
```

说明当前阻塞发生在 TCP 连接阶段，尚未进入 password authentication。

DNS 解析正常：

```text
iia100.slis.tsukuba.ac.jp has address 133.51.231.67
```

判断：

当前本地环境无法连通服务器 22 端口，可能需要校园网、VPN、跳板机或服务器侧网络放行。

## 4. 网络恢复后的部署命令

在本地执行：

```bash
cd /Users/chenpi/ForAgent/V7
bash scripts/deploy_v7_to_server.sh
```

然后在服务器起跑 first-pass：

```bash
cd /home/iiserver31/projects/FedE4RAG-main
nohup ./run_v7_all.sh first_pass > v7_nohup.log 2>&1 &
```

检查状态：

```bash
./check_v7_status.sh
```

采集与分析：

```bash
/home/iiserver31/anaconda3/envs/supv2/bin/python V7/collect_v7_results.py
/home/iiserver31/anaconda3/envs/supv2/bin/python V7/write_v7_analysis.py
```

同步 GitHub：

```bash
./sync_github_v7.sh
```

## 5. 当前状态

V7 实验设计与自动化文件已准备完成，但尚未能在服务器上实际创建 V7 平行路径、起跑或 push GitHub。阻塞原因是当前环境到服务器 SSH 端口超时。
