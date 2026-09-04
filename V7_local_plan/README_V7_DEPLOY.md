# V7 Deployment and Automation Notes

本目录包含 V7 Agentic Federated RAG 的实验方案与服务器自动化脚本。

## 1. 当前目标

在服务器项目根目录：

```bash
/home/iiserver31/projects/FedE4RAG-main
```

建立与 `V6` 平行的：

```bash
V7/
run_v7_all.sh
check_v7_status.sh
sync_github_v7.sh
```

并先起跑 `first_pass`：

```text
54 upstream runs + 54 downstream RAG runs
```

## 2. 一键部署

在本地网络可访问服务器时运行：

```bash
cd /Users/chenpi/ForAgent/V7
bash scripts/deploy_v7_to_server.sh
```

如果需要指定参数：

```bash
HOST=iia100.slis.tsukuba.ac.jp \
REMOTE_USER=iiserver31 \
ROOT=/home/iiserver31/projects/FedE4RAG-main \
bash scripts/deploy_v7_to_server.sh
```

如果需要自动输入密码，可临时使用环境变量，不要写入脚本文件：

```bash
V7_SSH_PASSWORD='********' bash scripts/deploy_v7_to_server.sh
```

## 3. 服务器起跑

```bash
cd /home/iiserver31/projects/FedE4RAG-main
nohup ./run_v7_all.sh first_pass > v7_nohup.log 2>&1 &
```

完整实验：

```bash
nohup ./run_v7_all.sh full_pass > v7_nohup.log 2>&1 &
```

## 4. 状态检查

```bash
cd /home/iiserver31/projects/FedE4RAG-main
./check_v7_status.sh
```

重点看：

```text
run_metadata.json
rag_eval_stdout.log
v7_all.log
实验分析报告/V7
```

## 5. 结果采集与分析

```bash
cd /home/iiserver31/projects/FedE4RAG-main
/home/iiserver31/anaconda3/envs/supv2/bin/python V7/collect_v7_results.py
/home/iiserver31/anaconda3/envs/supv2/bin/python V7/write_v7_analysis.py
```

## 6. GitHub 同步

```bash
cd /home/iiserver31/projects/FedE4RAG-main
./sync_github_v7.sh
```

默认 commit message：

```text
Add V7 agentic federated RAG experiment automation
```

可覆盖：

```bash
MESSAGE="Run V7 first-pass agentic experiments" ./sync_github_v7.sh
```

## 7. 当前本地限制

当前本地环境尝试连接服务器：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 iiserver31@iia100.slis.tsukuba.ac.jp
```

结果为 22 端口超时。因此服务器端部署、起跑、实时记录和 GitHub push 需要在可访问服务器网络/VPN 后执行。

本地已经准备好所有部署材料，不需要重新设计 V7。
