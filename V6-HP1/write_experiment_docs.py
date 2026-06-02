from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write V6-HP1 experiment record and analysis documents.")
    parser.add_argument("--full-root", default="实验分析报告/V6-HP1")
    parser.add_argument("--output-record", default="V6-HP1/v6_hp1_complete_experiment_record_cn.md")
    parser.add_argument("--output-analysis", default="V6-HP1/v6_hp1_complete_experiment_analysis_cn.md")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_reports(root: Path) -> list[dict[str, Any]]:
    reports = []
    for report_path in sorted(root.rglob("report.json")):
        payload = load_json(report_path)
        if payload:
            payload["_report_dir"] = str(report_path.parent)
            reports.append(payload)
    return reports


def metric_text(group: dict[str, Any]) -> str:
    parts = []
    for key in ("cos_3", "recall_3", "mrr", "NDCG"):
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        if mean_key in group:
            parts.append(f"{key}={group[mean_key]:.4f}±{group.get(std_key, 0.0):.4f}")
    return ", ".join(parts) if parts else "下游指标未解析"


def build_record(reports: list[dict[str, Any]]) -> str:
    now = datetime.now().isoformat(timespec="seconds")
    full_reports = [item for item in reports if item.get("report_type") == "full_pipeline"]
    suite_reports = [item for item in reports if item.get("report_type") == "suite"]
    lines = [
        "# V6-HP1 完整实验结果记录",
        "",
        f"- 生成时间：{now}",
        "- 实验版本：V6-HP1",
        "- 数据集：HotpotQA (fullwiki)",
        "- 上游训练：Hotpot train split 生成的 question-supporting-facts 配对语料",
        "- 下游评测：Hotpot validation split，same-budget / heterogeneity / hardquery / ablation 全流程",
        f"- 全流程报告数量：{len(full_reports)}",
        f"- 上游 suite 报告数量：{len(suite_reports)}",
        "",
        "## 1. 报告目录",
        "",
    ]
    for item in full_reports:
        lines.append(
            f"- `{item.get('suite_name', '')}`: upstream={item.get('completed_upstream', 0)}, "
            f"downstream={item.get('completed_downstream', 0)}, dir=`{item.get('_report_dir', '')}`"
        )
    lines.extend(["", "## 2. Seed 聚合结果", ""])
    for item in full_reports:
        lines.append(f"### {item.get('suite_name', '')}")
        grouped = item.get("grouped_runs", [])
        if not grouped:
            lines.append("- 暂无可解析聚合结果。")
            continue
        for group in grouped:
            lines.append(
                f"- `{group.get('strategy', '')}` task=`{group.get('task_name', '')}`: "
                f"payload={group.get('overall_payload_ratio_mean', 0.0):.4f}±{group.get('overall_payload_ratio_std', 0.0):.4f}, "
                f"reduction={group.get('communication_reduction_mean', 0.0):.4f}±{group.get('communication_reduction_std', 0.0):.4f}, "
                f"{metric_text(group)}"
            )
    return "\n".join(lines) + "\n"


def build_analysis(reports: list[dict[str, Any]]) -> str:
    now = datetime.now().isoformat(timespec="seconds")
    full_reports = [item for item in reports if item.get("report_type") == "full_pipeline"]
    lines = [
        "# V6-HP1 完整实验分析",
        "",
        f"- 生成时间：{now}",
        "",
        "## 1. 实验问题",
        "",
        "V6-HP1 直接放弃了原本区分度较低的数据集，改用 HotpotQA fullwiki。目标是在更难、更多跳推理、更容易拉开排序差异的下游任务上，重新检验 same-budget 选择性上传是否真的比 V3/V5/V6 更值。",
        "",
        "## 2. 主要观察",
        "",
    ]
    if not full_reports:
        lines.append("- 当前尚未发现 full-pipeline 报告，说明自动化流程可能仍在运行或尚未进入报告阶段。")
    else:
        lines.append(f"- 已收集 {len(full_reports)} 个 full-pipeline 报告。")
        lines.append("- 重点先看 `v6hp1_budget_aligned`：如果 Hotpot 上同预算差异被放大，它会是最关键的证据。")
        lines.append("- 其次看 `v6hp1_heterogeneity`：如果在更强 non-IID 下 V6-HP1 比 `random/delta_norm` 更省且下游更稳，则能补足主结果差异不明显的问题。")
        lines.append("- `v6hp1_hardquery` 会进一步验证 harder query 上同预算策略是否优于启发式基线。")
    lines.extend(
        [
            "",
            "## 3. 判断标准",
            "",
            "- 优先比较 `v6hp1_budget_aligned` 中 `hypernet_v6` 与 `hypernet_v3/random/delta_norm` 的 `recall_3/mrr/NDCG`。",
            "- 若 payload 接近而下游指标更高，则说明 Hotpot 已经成功放大方法差异。",
            "- 若 Hotpot 上仍然几乎没有差距，则主要瓶颈更可能来自上游训练信号本身，而不只是旧数据集太简单。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    reports = collect_reports(Path(args.full_root))
    record_path = Path(args.output_record)
    analysis_path = Path(args.output_analysis)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(build_record(reports), encoding="utf-8")
    analysis_path.write_text(build_analysis(reports), encoding="utf-8")
    print(f"Wrote {record_path} and {analysis_path}")


if __name__ == "__main__":
    main()
