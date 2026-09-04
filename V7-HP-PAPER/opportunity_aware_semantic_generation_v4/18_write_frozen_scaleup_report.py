#!/usr/bin/env python3
"""Assemble the same-source 3,000-query frozen scale-up report."""

from __future__ import annotations

import json

from v4_common import OUTPUTS, REPORTS, ensure_layout, read_json, write_json


def format_p(value: float) -> str:
    return "<0.0002" if float(value) == 0.0 else f"{float(value):.4f}"


def metric_table(reader: str, payload: dict) -> str:
    rows = payload["readers"][reader]
    lines = [
        "| Metric | Baseline | V4 selected | Delta | 95% CI | p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in ("answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1"):
        baseline = rows["metrics"]["baseline"][metric]
        selected = rows["metrics"]["v4_selected"][metric]
        sig = rows["significance"][metric]
        lines.append(
            f"| {metric} | {baseline:.4f} | {selected:.4f} | {selected - baseline:+.4f} | "
            f"[{sig['ci95_low']:+.4f}, {sig['ci95_high']:+.4f}] | {format_p(sig['p_value'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    ensure_layout()
    scale_dir = OUTPUTS / "scaleup"
    context = read_json(scale_dir / "same_source_context_audit.json")
    generator = read_json(scale_dir / "frozen_generator_audit.json")
    selector = read_json(scale_dir / "frozen_selector_manifest.json")
    flan = read_json(scale_dir / "readers/flan/summary.json")
    unified = read_json(scale_dir / "readers/unifiedqa/summary.json")
    official = read_json(scale_dir / "official_metrics/scaleup_official_summary.json")
    complete = all([
        context.get("status") == "pass",
        generator.get("status") == "pass",
        selector.get("status") == "pass",
        flan.get("status") == "complete",
        unified.get("status") == "complete",
        official.get("status") == "complete",
    ])
    payload = {
        "status": "complete" if complete else "incomplete",
        "n_queries": int(official.get("n_queries", 0)),
        "same_source": context.get("source"),
        "source_seed": context.get("source_seed"),
        "disjoint_from_development_1000": context.get("development_scaleup_overlap") == 0,
        "baseline_1000_reproduction_rate": context.get("baseline_1000_reproduction_rate"),
        "baseline_retriever": context.get("baseline_retriever"),
        "bm25_only_substitution_used": False,
        "thresholds_retuned": False,
        "generator_effective_actions": generator.get("n_effective_actions"),
        "selector_selected_count": selector.get("selected_count"),
        "selector_coverage": selector.get("coverage"),
        "official_dual_reader": official,
    }
    write_json(scale_dir / "scaleup_summary.json", payload)

    report = f"""# V7-HP V4 同源 3,000-query Frozen Scale-Up 报告

## 1. 目的与约束

本阶段检验 1,000-query 开发结果能否在更大、未参与调参的同源 HotpotQA 样本上复现。3,000 条 query 来自与原开发集完全相同的 `hotpot_qa/distractor/validation`，沿用 seed 44 的固定打乱顺序，并取开发 1,000 之后的互斥切片。

- 开发/scale-up query overlap: **{context['development_scaleup_overlap']}**。
- 原 1,000 source reconstruction: **{str(context['source_1000_exact_reconstruction']).lower()}**。
- 原 1,000 baseline title-order reproduction: **{context['baseline_1000_exact_title_order_matches']}/1000**。
- Baseline: `{context['baseline_retriever']}`。
- BM25-only top-5 substitution: **未使用**。
- Generator、selector threshold、reader prompt/decoding、support threshold: **均未在 3,000 条上调参**。

## 2. 冻结执行

- Frozen context queries: **{context['scaleup_size']:,}**。
- Generator effective actions: **{generator['n_effective_actions']:,}**。
- Selector interventions: **{selector['selected_count']:,}/{selector['n_queries']:,} ({selector['coverage']:.1%})**。
- Official sentence-support threshold: **{official['support_threshold']:.1f}**，来自原 1,000 五折一致阈值。

## 3. FLAN-T5-Large Official Metrics

{metric_table('flan', official)}

## 4. UnifiedQA-T5-Large Official Metrics

{metric_table('unifiedqa', official)}

## 5. 稳健性判断

- Dual-reader answer direction consistent: **{str(official['dual_reader_direction_consistent']).lower()}**。
- Systematic answer degradation: **{str(official['systematic_answer_degradation']).lower()}**。
- FLAN answer-drop rate: **{official['readers']['flan']['answer_drop_rate']:.2%}**。
- UnifiedQA answer-drop rate: **{official['readers']['unifiedqa']['answer_drop_rate']:.2%}**。

结果应按效应量、置信区间和双 reader 一致性解释。该 3,000-query 结果是同源规模化验证；它不等价于跨数据集 external validation。

## 6. 产物

- Context provenance audit: `{scale_dir / 'same_source_context_audit.json'}`
- Frozen generator audit: `{scale_dir / 'frozen_generator_audit.json'}`
- Frozen selector manifest: `{scale_dir / 'frozen_selector_manifest.json'}`
- Official dual-reader summary: `{scale_dir / 'official_metrics/scaleup_official_summary.json'}`
"""
    (REPORTS / "scaleup_report.md").write_text(report, encoding="utf-8")
    (REPORTS / "same_source_3000_frozen_scaleup_report_cn.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(REPORTS / "same_source_3000_frozen_scaleup_report_cn.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
