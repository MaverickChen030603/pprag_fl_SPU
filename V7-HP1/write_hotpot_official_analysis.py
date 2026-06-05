#!/usr/bin/env python3
"""Write an analysis report for V7-HP1 Hotpot QA/supporting-fact evaluation."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List

KEY_METRICS = [
    "answer_access_at_k",
    "support_title_recall_at_k",
    "all_gold_sp_retrieved_at_k",
    "sp_f1",
    "sp_em",
    "joint_f1",
    "joint_em",
    "answer_f1",
    "answer_em",
]
CLASSIC_BASELINES = {"hypernet_v6", "adaptive_v6"}
AGENT_METHOD_HINTS = ("agent", "memory", "rare", "hard")


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def metric_value(row: Dict[str, Any], metric: str) -> float:
    value = row.get(metric)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: float) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.4f}"


def collect_rows(output_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summary = output_root / "official_eval_all_summary.json"
    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict) and str(r.get("status", "")).startswith("completed")]
    for path in sorted(output_root.glob("**/official_metrics.json")):
        data = load_json(path)
        metrics = data.get("metrics", {})
        row = {k: data.get(k) for k in ["suite_tag", "task_name", "method", "profile", "seed", "n", "dataset_mode", "run_dir", "output_dir"]}
        row.update(metrics)
        rows.append(row)
    return rows


def aggregate(rows: List[Dict[str, Any]], group_keys: Iterable[str]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    keys = list(group_keys)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    out: List[Dict[str, Any]] = []
    for group, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rec = {k: v for k, v in zip(keys, group)}
        rec["runs"] = len(members)
        rec["n_mean"] = mean([float(m.get("n") or 0) for m in members]) if members else 0
        for metric in KEY_METRICS:
            vals = [metric_value(m, metric) for m in members]
            vals = [v for v in vals if not math.isnan(v)]
            rec[f"{metric}_mean"] = mean(vals) if vals else float("nan")
            rec[f"{metric}_std"] = pstdev(vals) if len(vals) > 1 else 0.0 if vals else float("nan")
        out.append(rec)
    return out


def best(rows: List[Dict[str, Any]], metric: str, predicate=lambda r: True) -> Dict[str, Any] | None:
    candidates = [r for r in rows if predicate(r) and not math.isnan(metric_value(r, f"{metric}_mean"))]
    if not candidates:
        return None
    return max(candidates, key=lambda r: metric_value(r, f"{metric}_mean"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("V7-HP1/outputs/hotpot_official_eval"))
    parser.add_argument("--report-dir", type=Path, default=Path("实验分析报告/V7-HP1"))
    args = parser.parse_args()

    rows = collect_rows(args.output_root)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    by_method = aggregate(rows, ["suite_tag", "method", "profile"])
    by_suite_method = aggregate(rows, ["suite_tag", "method"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.report_dir / f"v7_hp1_hotpot_official_eval_{ts}.md"
    latest_path = args.report_dir / "v7_hp1_hotpot_official_eval_latest.md"

    lines: List[str] = []
    lines.append("# V7-HP1 Hotpot 官方式 QA 与 Supporting-Fact 评估报告")
    lines.append("")
    lines.append(f"生成时间: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"评估输出目录: `{args.output_root}`")
    lines.append(f"完成 run 数: {len(rows)}")
    modes = sorted({str(r.get("dataset_mode")) for r in rows})
    lines.append(f"数据模式: {', '.join(modes) if modes else 'NA'}")
    lines.append("")
    lines.append("## 读数口径")
    lines.append("")
    lines.append("本评估接入 Hotpot 风格的 answer EM/F1、supporting-fact EM/F1、joint EM/F1，并额外记录 `answer_access_at_k`、`support_title_recall_at_k` 与 `all_gold_sp_retrieved_at_k`。当前默认使用 V7-HP1 compact Hotpot 派生数据；answer 采用检索可达性启发式，因此判断 agent 是否转化为真实收益时，优先看 answer access、support recall 与 joint 指标，而不是把 answer EM 视作完整 reader/generator 成绩。")
    lines.append("")
    if not rows:
        lines.append("## 当前结论")
        lines.append("")
        lines.append("尚未发现完成的 official_metrics.json。")
    else:
        lines.append("## 方法均值")
        lines.append("")
        header = ["suite", "method", "profile", "runs", "answer_access", "support_title_recall", "all_sp", "sp_f1", "joint_f1"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for rec in by_method:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(rec.get("suite_tag")),
                        str(rec.get("method")),
                        str(rec.get("profile")),
                        str(rec.get("runs")),
                        fmt(rec.get("answer_access_at_k_mean")),
                        fmt(rec.get("support_title_recall_at_k_mean")),
                        fmt(rec.get("all_gold_sp_retrieved_at_k_mean")),
                        fmt(rec.get("sp_f1_mean")),
                        fmt(rec.get("joint_f1_mean")),
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append("## 正信号判断")
        lines.append("")
        for suite in sorted({str(r.get("suite_tag")) for r in by_suite_method}):
            suite_rows = [r for r in by_suite_method if str(r.get("suite_tag")) == suite]
            baseline = best(suite_rows, "joint_f1", lambda r: str(r.get("method")) in CLASSIC_BASELINES)
            agent = best(suite_rows, "joint_f1", lambda r: any(h in str(r.get("method", "")) for h in AGENT_METHOD_HINTS))
            lines.append(f"### {suite}")
            if baseline and agent:
                gap_joint = metric_value(agent, "joint_f1_mean") - metric_value(baseline, "joint_f1_mean")
                gap_sp = metric_value(agent, "sp_f1_mean") - metric_value(baseline, "sp_f1_mean")
                gap_access = metric_value(agent, "answer_access_at_k_mean") - metric_value(baseline, "answer_access_at_k_mean")
                lines.append(f"最佳 classic baseline: `{baseline.get('method')}` joint_f1={fmt(metric_value(baseline, 'joint_f1_mean'))}, sp_f1={fmt(metric_value(baseline, 'sp_f1_mean'))}, answer_access={fmt(metric_value(baseline, 'answer_access_at_k_mean'))}。")
                lines.append(f"最佳 agent: `{agent.get('method')}` joint_f1={fmt(metric_value(agent, 'joint_f1_mean'))}, sp_f1={fmt(metric_value(agent, 'sp_f1_mean'))}, answer_access={fmt(metric_value(agent, 'answer_access_at_k_mean'))}。")
                lines.append(f"差值(agent - baseline): joint_f1={gap_joint:+.4f}, sp_f1={gap_sp:+.4f}, answer_access={gap_access:+.4f}。")
                if gap_joint > 0 and (gap_sp > 0 or gap_access > 0):
                    lines.append("判断: 存在正信号，agent 的选择优势至少部分转化到了 Hotpot supporting/QA 可达性指标。")
                elif abs(gap_joint) < 1e-9 and abs(gap_sp) < 1e-9 and abs(gap_access) < 1e-9:
                    lines.append("判断: 暂无可分辨正信号；当前指标与 baseline 持平。")
                else:
                    lines.append("判断: 正信号不足或不稳定，需要看更强 hard/rare 子集或接入 reader 后再确认。")
            else:
                lines.append("classic baseline 或 agent 组不完整，暂不能比较。")
            lines.append("")
        lines.append("## 后续建议")
        lines.append("")
        lines.append("1. 若 compact 指标仍饱和，应开启 `--prefer-official` 对 official fullwiki context 做子集评估，提升 supporting-fact 判别难度。")
        lines.append("2. 下一步应接 reader/generator，用同一 retrieved context 生成答案，再用官方 answer EM/F1 评估，避免 answer 启发式带来的上界口径。")
        lines.append("3. 优先比较 rare-bridge/tail 与 hard-query 子集，因为这些场景更可能放大 memory、rarity signal 与 client selection 的差异。")

    text = "\n".join(lines) + "\n"
    report_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    print(json.dumps({"report_path": str(report_path), "latest_path": str(latest_path), "runs": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
