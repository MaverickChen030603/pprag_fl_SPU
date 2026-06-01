from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write V6-H1 experiment record and analysis documents.")
    parser.add_argument("--full-root", default="实验分析报告/V6-H1")
    parser.add_argument("--output-record", default="V6-H1/v6_h1_complete_experiment_record_cn.md")
    parser.add_argument("--output-analysis", default="V6-H1/v6_h1_complete_experiment_analysis_cn.md")
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
        "# V6-H1 完整实验结果记录",
        "",
        f"- 生成时间：{now}",
        "- 实验版本：V6-H1",
        "- 核心目的：在 V6 同预算框架上加入稳定 hard-query 子集构建，验证方法优势是否能在更困难的下游检索问题上被放大。",
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
    hard_reports = [item for item in full_reports if "hardquery" in item.get("suite_name", "")]
    lines = [
        "# V6-H1 完整实验分析",
        "",
        f"- 生成时间：{now}",
        "",
        "## 1. 实验问题",
        "",
        "V6-H1 针对 V3-V6 下游差距不明显的问题，专门构建稳定 hard-query 子集。其目标不是继续扩大通信预算，而是在更有判别力的查询集合上检验选择性上传策略是否真正改善检索排序质量。",
        "",
        "## 2. 主要观察",
        "",
    ]
    if not full_reports:
        lines.append("- 当前尚未发现 full-pipeline 报告，说明自动化流程可能仍在运行或尚未进入报告阶段。")
    else:
        lines.append(f"- 已收集 {len(full_reports)} 个 full-pipeline 报告，其中 hard-query 相关报告 {len(hard_reports)} 个。")
        lines.append("- 全集评估用于保持与 V6 的公平对照，hard-query 评估用于观察困难样本上的差异是否被放大。")
        lines.append("- 如果 V6-H1 在 hard-query 子集上提升 `recall_3/mrr/NDCG`，同时 payload 与 V6_budget_aligned 接近，则可作为更强论文证据。")
        lines.append("- 如果全集与 hard-query 子集仍都没有明显差距，主要瓶颈更可能来自数据集难度、检索候选构造或训练信号与下游指标耦合不足。")
    lines.extend(
        [
            "",
            "## 3. 后续判断标准",
            "",
            "- 优先比较 `v6h1_budget_aligned_stable_hardquery` 中 `hypernet_v6` 与 `hypernet_v3/random/delta_norm` 的同预算下游指标。",
            "- 其次查看 `v6h1_heterogeneity_stable_hardquery`，判断强异构场景下 hard-query 子集是否放大方法差异。",
            "- 最后结合 `hard_queries/stable_hard_queries.json` 的数量与原因分布，确认 hard-query 子集是否足够难且不是偶然噪声。",
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
