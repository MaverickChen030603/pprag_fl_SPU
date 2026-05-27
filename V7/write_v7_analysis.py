#!/usr/bin/env python3
"""Write a first-pass Chinese V7 analysis report from collected summaries."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/iiserver31/projects/FedE4RAG-main")
REPORT_ROOT = ROOT / "实验分析报告" / "V7"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latest_collection() -> Path | None:
    dirs = sorted(REPORT_ROOT.glob("v7_collected_*"))
    return dirs[-1] if dirs else None


def numeric(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def method_payload_table(rows: list[dict]) -> list[tuple[str, int, float | None]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    counts = Counter()
    for row in rows:
        method = row.get("method") or "unknown"
        counts[method] += 1
        value = numeric(row.get("avg_payload", ""))
        if value is not None:
            grouped[method].append(value)
    table = []
    for method, count in sorted(counts.items()):
        values = grouped.get(method, [])
        avg = sum(values) / len(values) if values else None
        table.append((method, count, avg))
    return table


def write_report(collection_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    upstream = read_csv(collection_dir / "v7_upstream_summary.csv")
    downstream = read_csv(collection_dir / "v7_downstream_summary.csv")
    upstream_by_suite = Counter(row.get("suite") or "unknown" for row in upstream)
    downstream_by_suite = Counter(row.get("suite") or "unknown" for row in downstream)
    payload_table = method_payload_table(upstream)

    lines = [
        "# V7 自动实验阶段分析报告",
        "",
        f"生成时间：{stamp}",
        f"结果采集目录：`{collection_dir}`",
        "",
        "## 1. 当前完成度",
        "",
        f"- upstream run_metadata 数量：`{len(upstream)}`",
        f"- downstream rag_eval_stdout 数量：`{len(downstream)}`",
        "",
        "### 1.1 Upstream suite 分布",
        "",
    ]
    for suite, count in sorted(upstream_by_suite.items()):
        lines.append(f"- `{suite}`: {count}")
    lines.extend(["", "### 1.2 Downstream suite 分布", ""])
    for suite, count in sorted(downstream_by_suite.items()):
        lines.append(f"- `{suite}`: {count}")

    lines.extend(["", "## 2. 通信预算初步检查", ""])
    lines.append("| method | runs | mean avg_payload |")
    lines.append("|---|---:|---:|")
    for method, count, avg in payload_table:
        avg_text = f"{avg:.4f}" if avg is not None else "N/A"
        lines.append(f"| `{method}` | {count} | {avg_text} |")

    lines.extend(
        [
            "",
            "## 3. 初步科研判断模板",
            "",
            "当前报告是自动生成的阶段性分析。人工写正式分析时，请围绕以下问题补全：",
            "",
            "1. `agent_rule_v7` / `agent_bandit_v7` / `agent_policy_v7` 是否在 `v7_budget_aligned` 中保持严格 same-budget。",
            "2. 在相同 payload 下，agentic 方法是否超过 `hypernet_v6` 和 `adaptive_v6`。",
            "3. 在 `v7_hardquery` 中，hard-query Recall/MRR/F1 是否有稳定提升。",
            "4. 在 `v7_heterogeneity` 中，rare-domain 或 hard-client 是否获得更多有效通信预算。",
            "5. 若主指标未提升，检查是否是 reward 延迟、proxy 噪声、memory 更新过慢或 agent action space 过窄导致。",
            "",
            "## 4. 下一步",
            "",
            "- 若 first-pass 已完成 54 upstream + 54 downstream，优先比较 `v7_budget_aligned`。",
            "- 若 agent 方法有正信号，再扩展到 full-pass 148 upstream + 148 downstream。",
            "- 若 agent 方法无正信号，先做 `v7_ablation_signal`，不要急于引入 LLM planner。",
        ]
    )

    report_path = REPORT_ROOT / f"v7_auto_analysis_{stamp}.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    collection = latest_collection()
    if collection is None:
        raise SystemExit("No v7_collected_* directory found. Run collect_v7_results.py first.")
    report = write_report(collection)
    print(json.dumps({"analysis_report": str(report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
