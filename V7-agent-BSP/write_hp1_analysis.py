#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = REPO_ROOT / "V7-agent-2" / "outputs" / "pprag_fl_v7agent2"
DEFAULT_STRICT = REPO_ROOT / "V7-agent-2" / "outputs" / "hp1_strict_eval"
DEFAULT_REPORT = REPO_ROOT / "实验分析报告" / "V7-agent-2"
METHOD_ORDER = ["hypernet_v6", "adaptive_v6", "agent_rule_v7", "agent_bandit_v7", "agent_tail_v7hp1", "agent_memory_v7hp1", "agent_oracle_v7hp1"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} ± {pstdev(values):.4f}"


def infer_suite(path: Path, strict_root: Path) -> str:
    try:
        return path.relative_to(strict_root).parts[0]
    except Exception:
        return path.parent.name


def load_strict(strict_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(strict_root.glob("*/hp1_strict_summary.csv")):
        suite = infer_suite(path, strict_root)
        if suite in {"all_hp1", "smoke"}:
            continue
        for row in read_csv(path):
            row = dict(row)
            row["suite"] = suite
            rows.append(row)
    return rows


def load_upstream(upstream_root: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for meta in upstream_root.glob("*/*/*/run_metadata.json"):
        counts[meta.parts[-4]] += 1
    return dict(counts)


def group(rows: list[dict[str, Any]], suite: str | None = None, key: str = "method") -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if suite and row.get("suite") != suite:
            continue
        label = row.get(key) or row.get("method") or "unknown"
        grouped[label].append(row)
    return grouped


def metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        val = as_float(row.get(key))
        if val is not None:
            out.append(val)
    return out


def best_baseline(grouped: dict[str, list[dict[str, Any]]], metric: str) -> float | None:
    vals = []
    for method in ("hypernet_v6", "adaptive_v6"):
        values = metric_values(grouped.get(method, []), metric)
        if values:
            vals.append(mean(values))
    return max(vals) if vals else None


def render_table(grouped: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines = [
        "| method/profile | n | avg_budget_topk_hp1 | bridge_block_recall_hp1 | early_evidence_recall_hp1 | selection_diversity_hp1 | hp1_multihop_score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    seen = set(grouped)
    ordered = [m for m in METHOD_ORDER if m in seen] + sorted(seen - set(METHOD_ORDER))
    for method in ordered:
        rows = grouped[method]
        lines.append(
            "| {method} | {n} | {budget} | {bridge} | {early} | {div} | {score} |".format(
                method=method,
                n=len(rows),
                budget=stats(metric_values(rows, "avg_budget_topk_hp1")),
                bridge=stats(metric_values(rows, "bridge_block_recall_hp1")),
                early=stats(metric_values(rows, "early_evidence_recall_hp1")),
                div=stats(metric_values(rows, "selection_diversity_hp1")),
                score=stats(metric_values(rows, "hp1_multihop_score")),
            )
        )
    return lines


def render_gaps(grouped: dict[str, list[dict[str, Any]]]) -> list[str]:
    metric = "hp1_multihop_score"
    base = best_baseline(grouped, metric)
    if base is None:
        return ["- baseline 数据不足，暂不能计算 agent gap。"]
    lines = []
    for method in ("agent_rule_v7", "agent_bandit_v7", "agent_tail_v7hp1", "agent_memory_v7hp1", "agent_oracle_v7hp1"):
        values = metric_values(grouped.get(method, []), metric)
        if not values:
            continue
        gap = mean(values) - base
        sign = "正" if gap > 1e-6 else "负/无"
        lines.append(f"- `{method}` 相对最佳 baseline 的 `{metric}` gap = {gap:+.4f}，方向：{sign}信号。")
    return lines or ["- agent 数据不足，暂不能判断 gap。"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--strict-root", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    rows = load_strict(args.strict_root)
    upstream_counts = load_upstream(args.upstream_root)
    suites = sorted({r.get("suite", "unknown") for r in rows})
    out = args.report_dir / f"v7_agent2_auto_analysis_{args.stamp}.md"
    lines: list[str] = []
    lines.append("# V7-agent-2 HotpotQA Agent-Rule 自动实验分析报告")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## 数据集与目的")
    lines.append("")
    lines.append("- 数据集切换为 HotpotQA fullwiki 派生的 `FedE/select_data_hotpot_train_5000.json`，保留 question、answer、supporting_titles 与 supporting context。")
    lines.append("- 目的：放弃当前对方法不敏感的旧数据设置，用多跳证据链、rare bridge client 与 hard-query 场景放大 agent memory / hard-query / rarity signal 的作用。")
    lines.append("- 当前 V7-agent-2/HP1 strict 指标是选择行为诊断；它不是 Hotpot 官方 answer F1/EM 或 supporting fact F1 的最终替代。")
    lines.append("")
    lines.append("## 结论摘要")
    lines.append("")
    if not rows:
        lines.append("当前尚未发现 V7-agent-2 strict eval 结果。实验可能仍在运行，或 strict evaluator 尚未执行。")
    else:
        all_grouped = group(rows)
        lines.extend(render_gaps(all_grouped))
        lines.append("")
        lines.append(f"- V7-agent-2 已收集 strict-eval 记录 {len(rows)} 条，平均预算 top-k：{stats(metric_values(rows, 'avg_budget_topk_hp1'))}。")
        lines.append("- 判断正信号时必须同步看预算：若 agent 预算显著高于 baseline，需要以 `hp1_budget_aligned` 为主结论。")
    lines.append("")
    lines.append("## Upstream 完成数")
    lines.append("")
    if upstream_counts:
        for suite, count in sorted(upstream_counts.items()):
            lines.append(f"- `{suite}`: {count} run")
    else:
        lines.append("- 尚未发现 upstream run_metadata。")
    lines.append("")
    if rows:
        lines.append("## 全局 strict 指标")
        lines.append("")
        lines.extend(render_table(group(rows)))
        lines.append("")
        for suite in suites:
            lines.append(f"## Suite: {suite}")
            lines.append("")
            suite_grouped = group(rows, suite=suite)
            lines.extend(render_table(suite_grouped))
            lines.append("")
            lines.extend(render_gaps(suite_grouped))
            lines.append("")
        ablation = [r for r in rows if r.get("suite") == "hp1_ablation_signal"]
        lines.append("## Ablation: profile 拆分")
        lines.append("")
        if ablation:
            lines.extend(render_table(group(ablation, key="agent_profile")))
            lines.append("")
            lines.append("- `full` 高于 `no_memory` 说明 memory/utility memory 对选择行为有实质贡献。")
            lines.append("- `full` 高于 `no_hard_query` 说明 hard-query weighting 改变了多跳压力下的 block 选择。")
            lines.append("- `full` 高于 `no_rarity` 说明 client rarity/embedding 对 rare bridge 场景有贡献。")
        else:
            lines.append("- ablation 尚无结果。")
        lines.append("")
        lines.append("## 下一步建议")
        lines.append("")
        lines.append("1. 先看 `hp1_budget_aligned`，确认 agent gap 是否在相同 top-k 预算下仍存在。")
        lines.append("2. 若 `hp1_rare_bridge_tail` gap 最大，把 rarity signal 固化进 agent policy，而不是只作为诊断标签。")
        lines.append("3. 若 strict 指标有正信号，再接 Hotpot 官方 answer/supporting-fact 评估，避免只证明选择器会变而没有证明问答收益。")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    latest = args.report_dir / "v7_agent2_auto_analysis_latest.md"
    latest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
