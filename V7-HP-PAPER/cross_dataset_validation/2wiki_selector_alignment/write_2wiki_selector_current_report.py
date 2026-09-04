#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/iiserver31/projects/FedE4RAG-main")
ALIGN = ROOT / "V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment"
MIRROR = ROOT / "实验分析报告/V7-HP-PAPER"


def load(rel: str):
    return json.loads((ALIGN / rel).read_text(encoding="utf-8"))


def f(x) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def main() -> None:
    action = load("outputs/action_table_300/action_table_summary.json")
    feature = load("outputs/action_table_300/feature_summary.json")
    smoke = load("outputs/selector_smoke_300/summary.json")
    failure = load("outputs/selector_smoke_300/failure_summary.json")
    audit = load("outputs/audit/2wiki_no_leak_audit.json")
    crossfit = load("outputs/selector_crossfit_1000/final_1000_summary.json")
    methods = smoke["methods"]
    gate = smoke["gate"]
    order = [
        "context_order",
        "bm25_or_lexical_routing",
        "support_first_selector",
        "hotpot_v23_frozen_transfer",
        "2wiki_v23_crossfit_selector",
        "no_safety_predictor",
        "no_support_features",
        "oracle_diagnostic_only",
    ]
    lines = [
        "# V7-HP-PAPER 2Wiki Selector Alignment Current Report",
        "",
        "## 1. 当前状态",
        "",
        "服务器当前没有 `2wiki_selector_alignment` 相关进程在运行。本轮 selector alignment smoke 已完成，1000 formal validation 因 gate 未通过而按规则跳过。",
        "",
        "已完成：",
        "",
        "- 2Wiki action table 300",
        "- v2.3-compatible feature table 300",
        "- dev-300 selector smoke with reader outcomes",
        "- ablation summary",
        "- failure diagnostics",
        "- no-leak audit",
        "- gated 1000 skip decision",
        "- selector alignment report and paper recommendation",
        "",
        "## 2. Action Table 与 Feature Alignment",
        "",
        f"- num_queries: `{action['num_queries']}`",
        f"- num_actions: `{action['num_actions']}`",
        f"- actions_per_query: `{action['actions_per_query']}`",
        f"- effective_action_rate: `{f(action['effective_action_rate'])}`",
        f"- avg_added_docs: `{f(action['avg_added_docs'])}`",
        f"- avg_removed_docs: `{f(action['avg_removed_docs'])}`",
        f"- prefix2_preserve_rate: `{f(action['prefix2_preserve_rate'])}`",
        f"- prefix3_preserve_rate: `{f(action['prefix3_preserve_rate'])}`",
        f"- dense_feature_available: `{action['dense_feature_available']}`",
        f"- safe_answer_prob_mode: `{feature['safe_answer_prob_mode']}`",
        "",
        "说明：dense feature 当前不可用；`safe_answer_prob` 在 smoke 阶段是 heuristic safety proxy，不应被写成正式训练出的 safety predictor。",
        "",
        "## 3. Selector Smoke 300 结果",
        "",
        "主基准是 `bm25_or_lexical_routing`，不是 raw context order。",
        "",
        "| method | answer_f1 | evidence_f1 | joint_f1 | answer_delta_vs_bm25 | evidence_delta_vs_bm25 | joint_delta_vs_bm25 | effective_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        m = methods[name]
        lines.append(
            f"| {name} | {f(m.get('answer_f1', 0))} | {f(m.get('evidence_f1', 0))} | {f(m.get('joint_f1', 0))} | "
            f"{f(m.get('answer_f1_delta_vs_bm25', 0))} | {f(m.get('evidence_f1_delta_vs_bm25', 0))} | "
            f"{f(m.get('joint_f1_delta_vs_bm25', 0))} | {f(m.get('selected_effective_action_rate', 0))} |"
        )
    lines.extend([
        "",
        "## 4. Gate 结论",
        "",
        "```json",
        json.dumps(gate, ensure_ascii=False, indent=2),
        "```",
        "",
        "Gate 未通过。关键原因是 `2wiki_v23_crossfit_selector` 相对 BM25 strong baseline 在 answer、evidence、joint 三类指标上全部为负，且 `selected_effective_action_rate` 只有 `0.0233`。",
        "",
        "因此不运行 1000 reader validation；这符合实验说明中的 stop rule。",
        "",
        "## 5. 失败诊断",
        "",
        "```json",
        json.dumps(failure.get("failure_distribution", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "主要失败不是数据管线，而是 selector 在 2Wiki 上偏向 fallback / 非有效 action，无法稳定选择 BM25 已经找到的高质量上下文。",
        "",
        "## 6. No-Leak Audit",
        "",
        f"- audit status: `{audit['status']}`",
        f"- query_fold_disjoint: `{audit['query_fold_disjoint']}`",
        f"- held_out_outcome_not_used_for_inference: `{audit['held_out_outcome_not_used_for_inference']}`",
        f"- gold_answer_support_not_used_as_inference_feature: `{audit['gold_answer_support_not_used_as_inference_feature']}`",
        f"- oracle_separated_from_formal_method: `{audit['oracle_separated_from_formal_method']}`",
        "",
        "## 7. 1000 状态",
        "",
        "```json",
        json.dumps(crossfit, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 8. 当前论文口径",
        "",
        "可以写：",
        "",
        "> 2Wiki results validate dataset transfer and lexical routing effectiveness, but do not yet establish selector-level generalization beyond a strong BM25 baseline.",
        "",
        "不能写：",
        "",
        "> HotpotQA v2.3 selector generalizes to 2WikiMultiHopQA.",
        "",
        "## 9. 下一步建议",
        "",
        "不要直接启动 MuSiQue。当前瓶颈是 2Wiki action-feature alignment / selector decision rule 未超过 BM25，而不是缺另一个数据集。若继续推进，应优先修复 2Wiki action candidate generation 与 safety calibration，再考虑 1000 或 MuSiQue。",
        "",
        "## 10. 关键文件",
        "",
        "```text",
        "V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment/outputs/action_table_300/action_table_summary.json",
        "V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment/outputs/action_table_300/feature_summary.json",
        "V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment/outputs/selector_smoke_300/summary.json",
        "V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment/outputs/selector_smoke_300/significance_report.json",
        "V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment/outputs/selector_smoke_300/failure_summary.json",
        "V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment/outputs/audit/2wiki_no_leak_audit.md",
        "V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment/reports/2wiki_selector_alignment_report.md",
        "V7-HP-PAPER/cross_dataset_validation/2wiki_selector_alignment/reports/2wiki_paper_recommendation.md",
        "```",
        "",
    ])
    text = "\n".join(lines)
    for p in [
        ALIGN / "reports/2wiki_selector_current_report_20260623.md",
        ALIGN / "reports/2wiki_selector_current_report_latest.md",
        MIRROR / "2wiki_selector_current_report_20260623.md",
        MIRROR / "2wiki_selector_current_report_latest.md",
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    print(json.dumps({
        "status": "written",
        "gate_passed": gate.get("passed"),
        "decision": gate.get("decision"),
        "report": str(MIRROR / "2wiki_selector_current_report_latest.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
