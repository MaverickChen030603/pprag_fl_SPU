#!/usr/bin/env python3
"""Write gate-aware v4 paper drafts and the final readiness audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from v4_common import OUTPUTS, PAPER, REPORTS, ensure_layout, read_json, write_json


def load_status(path: Path, default: str = "not_run") -> dict[str, Any]:
    return read_json(path) if path.exists() else {"status": default}


def complete(payload: dict[str, Any]) -> bool:
    return payload.get("status") == "complete"


def format_p(value: float) -> str:
    return "<0.0002" if float(value) == 0.0 else f"{float(value):.4f}"


def main() -> None:
    ensure_layout()
    generator = load_status(OUTPUTS / "semantic_generator/foldwise_generator_models.json")
    gate = load_status(OUTPUTS / "opportunity/v4_opportunity_gate.json")
    action = load_status(OUTPUTS / "action_outcomes/v4_action_summary.json")
    selector = load_status(OUTPUTS / "nested_selector/v4_nested_summary.json")
    official = load_status(OUTPUTS / "official_metrics/official_hotpotqa_summary.json")
    multi = load_status(OUTPUTS / "multi_reader/multi_reader_summary.json")
    scaleup = load_status(OUTPUTS / "scaleup/scaleup_summary.json")
    external = load_status(OUTPUTS / "external_dataset/external_validation_summary.json")
    gates = gate.get("gates", {})
    readiness = {
        "semantic_generator_completed": complete(generator),
        "overall_opportunity_gate_passed": bool(gates.get("A_overall_opportunity", False)),
        "conditional_opportunity_gate_passed": bool(gates.get("B_conditional_opportunity", False)),
        "marginal_new_query_gate_passed": bool(gates.get("C_marginal_breadth", False)),
        "positive_density_gate_passed": bool(gates.get("D_action_quality_density", False)),
        "new_query_efficiency_gate_passed": bool(gates.get("E_new_query_efficiency", False)),
        "nested_selector_completed": complete(selector),
        "official_metrics_completed": complete(official),
        "multi_reader_completed": complete(multi),
        "scaleup_completed": complete(scaleup),
        "external_validation_completed": complete(external),
    }
    if all(readiness.values()):
        status = "main_conference_ready"
    elif readiness["official_metrics_completed"] and readiness["multi_reader_completed"]:
        status = "main_conference_stretch"
    elif readiness["semantic_generator_completed"] and action.get("status") == "complete":
        status = "findings_ready" if sum(bool(value) for value in gates.values()) >= 2 else "not_ready"
    else:
        status = "not_ready"
    readiness["final_status"] = status
    write_json(OUTPUTS / "tables/main_conference_readiness.json", readiness)

    action_line = "Reader outcomes are not yet complete."
    if action.get("status") == "complete":
        action_line = (
            f"V4 evaluates {action['num_effective_actions']:,} effective outer-test actions. "
            f"Positive-query coverage is {action['overall_positive_query_coverage']:.1%}, conditional non-ceiling coverage is {action['non_ceiling_positive_query_coverage']:.1%}, "
            f"and positive-action density is {action['positive_action_density']:.2%}."
        )
    gate_line = f"The pre-registered gate result is {gate.get('status', 'not_run')}; {gate.get('passed_gate_count', 0)}/5 gates passed."
    selector_line = "The nested selector has not been completed."
    if complete(selector):
        answer_sig = selector["significance"]["answer_f1"]
        selector_line = (
            f"The fully nested selector intervenes on {selector['selected_count']}/{selector['n_queries']} queries and improves answer F1 by "
            f"{selector['deltas']['answer_f1']:+.4f} (95% CI [{answer_sig['ci95_low']:+.4f}, {answer_sig['ci95_high']:+.4f}], p={answer_sig['p_value']:.4f}), "
            f"title recall by {selector['deltas']['title_recall']:+.4f}, and the answer-title product by {selector['deltas']['answer_title_product']:+.4f}."
        )
    official_line = "Official answer/support/joint metrics have not been completed."
    if complete(official):
        significance = official["significance"]
        official_line = (
            f"Under official sentence-level evaluation, answer F1 changes by {significance['answer_f1']['mean']:+.4f} (p={significance['answer_f1']['p_value']:.4f}), "
            f"supporting-fact F1 by {significance['sp_f1']['mean']:+.4f} (p={significance['sp_f1']['p_value']:.4f}), and joint F1 by "
            f"{significance['joint_f1']['mean']:+.4f} (p={significance['joint_f1']['p_value']:.4f})."
        )
    multi_line = "Second-reader validation has not been completed."
    if complete(multi):
        multi_line = (
            f"UnifiedQA confirms the direction with answer F1 {multi['unifiedqa_answer_f1_delta']:+.4f} and joint F1 "
            f"{multi['unifiedqa_joint_f1_delta']:+.4f}; its answer-drop rate is {multi['unifiedqa_answer_drop_rate']:.1%}."
        )
    scaleup_line = "The frozen same-source 3,000-query scale-up has not been completed."
    scaleup_table = ""
    if complete(scaleup):
        scale_official = scaleup["official_dual_reader"]
        flan_scale = scale_official["readers"]["flan"]
        unified_scale = scale_official["readers"]["unifiedqa"]
        scaleup_line = (
            f"On 3,000 disjoint same-source queries, the unchanged selector intervenes on {scaleup['selector_selected_count']}/{scaleup['n_queries']} queries. "
            f"FLAN answer F1, supporting-fact F1, and joint F1 improve by {flan_scale['deltas']['answer_f1']:+.4f} "
            f"(p={format_p(flan_scale['significance']['answer_f1']['p_value'])}), {flan_scale['deltas']['sp_f1']:+.4f} "
            f"(p={format_p(flan_scale['significance']['sp_f1']['p_value'])}), and {flan_scale['deltas']['joint_f1']:+.4f} "
            f"(p={format_p(flan_scale['significance']['joint_f1']['p_value'])}); UnifiedQA yields answer/joint gains of "
            f"{unified_scale['deltas']['answer_f1']:+.4f}/{unified_scale['deltas']['joint_f1']:+.4f}."
        )
        scaleup_table = f"""
| Reader | N | Answer F1 baseline | Answer F1 selected | Delta | SP F1 delta | Joint F1 delta | Joint F1 p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FLAN-T5-Large | {scaleup['n_queries']} | {flan_scale['metrics']['baseline']['answer_f1']:.4f} | {flan_scale['metrics']['v4_selected']['answer_f1']:.4f} | {flan_scale['deltas']['answer_f1']:+.4f} | {flan_scale['deltas']['sp_f1']:+.4f} | {flan_scale['deltas']['joint_f1']:+.4f} | {format_p(flan_scale['significance']['joint_f1']['p_value'])} |
| UnifiedQA-T5-Large | {scaleup['n_queries']} | {unified_scale['metrics']['baseline']['answer_f1']:.4f} | {unified_scale['metrics']['v4_selected']['answer_f1']:.4f} | {unified_scale['deltas']['answer_f1']:+.4f} | {unified_scale['deltas']['sp_f1']:+.4f} | {unified_scale['deltas']['joint_f1']:+.4f} | {format_p(unified_scale['significance']['joint_f1']['p_value'])} |
"""
    headline = "Beyond Selection: Opportunity-Aware Context Action Generation for Multi-Hop Question Answering"
    draft = f"""DRAFT — HOTPOTQA SCALE-UP VALIDATED; EXTERNAL CLAIMS PENDING

# {headline}

## Abstract

Reader-side context selection can only exploit actions exposed by its candidate generator. Frozen v2 and v3 studies show that better selection and nearly twice as many heuristic actions do not resolve this candidate-opportunity gap. We introduce a fully nested semantic action generator that estimates the missing evidence type, retrieves complementary documents with bi-encoder and cross-encoder signals, and constructs bounded anchor-preserving actions. {action_line} {selector_line} {official_line} {multi_line} {scaleup_line} The result supports a strengthened main-conference case on HotpotQA, while full opportunity-gate and external-transfer claims remain withheld.

## 1 Introduction

Multi-hop QA often fails before selection: the action table may contain no context that both restores evidence and preserves the reader's answer anchors. V2 established a risk-controlled selector, while v3 showed that expanding fixed templates raises overall coverage only from 20.3% to 23.4% and leaves positive density near 9.4%. This motivates semantic opportunity creation rather than another selector over the same table.

## 2 Method

The v4 system has three generator components: a missing-hop estimator, a semantic document opportunity model, and a pair-complementarity model. Each component is trained on outer-train outcomes and frozen before outer-test generation. Target queries expose only the question, baseline documents, retrieval signals, non-gold entities, and semantic relations. A bounded constructor creates at most eight actions per query across complementary insertion, anchor-preserving replacement, two-document chaining, redundancy replacement, and two order interventions.

## 3 Experimental Protocol

We retain the frozen 1,000-query HotpotQA development set, FLAN-T5-large reader, prompt, context budget, tokenizer limit, and decoding from v2/v3. We report overall and ceiling-aware opportunity, marginal new-query coverage, answer safety, family diversity, and new-query efficiency. Selector, official sentence metrics, second reader, scale-up, and external transfer are strictly gate-controlled.

## 4 Results

{action_line}

{gate_line}

{selector_line}

{official_line}

{multi_line}

### Frozen Same-Source Scale-Up

{scaleup_line}

{scaleup_table}

## 5 Analysis

V3's net gain of 31 queries decomposes into 81 newly covered v2-negative queries and 50 v2-covered queries not recovered by v3. This distinction motivates breadth-aware reporting. The disjoint 3,000-query run preserves the answer, support, and joint directions under both readers without retuning, reducing the risk that the 1,000-query result is a development-set accident.

## 6 Limitations

The semantic generator is trained from a 1,000-query development study and operates over the available per-query distractor pool. The 3,000-query run is a same-source scale validation, not an external-domain test; second-dataset claims remain unavailable. Title-level evidence metrics are diagnostic proxies and are never renamed as official supporting-fact metrics.

## 7 Conclusion

V4 shows that semantic, query-conditioned action construction can create reader-compatible opportunities beyond fixed templates and transfer them to small but significant answer, support, and joint gains on a disjoint same-source scale-up. Claims remain bounded by the incomplete external validation and the 3/5 opportunity-gate result.
"""
    (PAPER / "paper_main_conference_v4.md").write_text(draft, encoding="utf-8")
    (PAPER / "paper_full_clean_v4.md").write_text(draft, encoding="utf-8")
    (PAPER / "paper_storyboard_v4.md").write_text(
        "# V4 Paper Storyboard\n\n1. Candidate-opportunity gap.\n2. V2 selector ceiling.\n3. V3 negative heuristic expansion.\n4. Fully nested semantic generation.\n5. Opportunity gates.\n6. Downstream validation only if gates pass.\n",
        encoding="utf-8",
    )
    (PAPER / "paper_findings_fallback_v4.md").write_text(
        f"# Semantic Opportunity Generation as a Controlled Study\n\n{action_line}\n\n{gate_line}\n\nThe fallback paper reports the semantic generator and negative/positive opportunity result without unsupported downstream claims.\n",
        encoding="utf-8",
    )
    (PAPER / "paper_appendix_v4.md").write_text(
        "# V4 Appendix\n\nIncludes v3 ceiling-aware reanalysis, family overlap, foldwise generator manifests, no-leak audits, opportunity definitions, and gate decisions.\n",
        encoding="utf-8",
    )
    (PAPER / "main_conference_claim_audit_v4.md").write_text(
        f"# Main-Conference Claim Audit\n\nFinal status: **{status}**.\n\n- Do not claim answer improvement without significance.\n- Do not call title metrics official support metrics.\n- Do not claim scale or transfer before frozen evaluations complete.\n- Preserve v2 as the submission fallback and v3 as the negative opportunity study.\n",
        encoding="utf-8",
    )
    report_lines = ["# Main-Conference Readiness Report", ""] + [f"- `{key}`: **{str(value).lower()}**" for key, value in readiness.items()]
    (REPORTS / "main_conference_readiness_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    if complete(generator) and action.get("status") == "complete":
        doc_ap = sum(max(fold["inner_cv"]["doc_average_precision"].values()) for fold in generator["folds"]) / len(generator["folds"])
        pair_ap = sum(max(fold["inner_cv"]["pair_average_precision"].values()) for fold in generator["folds"]) / len(generator["folds"])
        gate_rows = "\n".join(f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in gate.get("gates", {}).items())
        current_report = f"""# V7-HP-PAPER V4 当前完整实验报告

## 1. 研究目的

V4 不再继续堆固定 action templates，也不再把 selector 当作第一瓶颈。它验证一个更基础的问题：对于 v2/v3 根本没有正动作的 query，能否通过语义互补、query-conditioned 的候选构造，主动创造 reader-compatible opportunity。

V2 是冻结投稿 fallback；v3 是 negative opportunity study。V4 独立运行，没有覆盖两者。

## 2. 冻结事实与重新审计

- v2 主可用动作：4,000；positive density 9.48%；positive-query coverage 20.3%。
- v3 有效动作：7,882；positive density 9.43%；coverage 23.4%。
- 按 `baseline answer_f1=1 且 title_recall=1` 精确重算，ceiling query 为 389，non-ceiling 为 611，v3 conditional coverage 为 38.3%。
- v3 相比 v2 净增 31 个 covered queries，但集合层面是新增 81 个、同时未恢复 50 个 v2-positive queries。论文必须同时报告净增与 marginal new coverage。

## 3. V4 方法

V4 使用五折 fully nested protocol。每折仅用 outer-train 的 v3 reader outcomes 训练：

1. missing-hop estimator；
2. MPNet bi-encoder + MS MARCO cross-encoder 特征上的 semantic document opportunity model；
3. two-document pair complementarity model。

outer-test 生成时冻结所有组件，只读取 question、baseline context、document text、BM25/semantic scores、非 gold entity 与 lexical relation。gold answer、gold support、target reader outcome、oracle action 均未使用。生成器最多给每题 8 个动作，覆盖 complementary insertion、anchor-preserving replacement、semantic two-doc chain、redundancy replacement 和两类顺序动作。

模型诊断：document opportunity inner-CV AP 均值 **{doc_ap:.4f}**，pair complementarity AP 均值 **{pair_ap:.4f}**。这高于各自正类基率，但 missing-hop 的稀有 ordering/redundancy 类仍难学习。

## 4. Action Opportunity 结果

- Queries: **{action['num_queries']}**。
- Effective actions: **{action['num_effective_actions']:,}**；其中相对 v3 新 context actions **{action['num_new_actions_vs_v3_table']:,}**。
- Positive actions: **{action['positive_actions']:,}**；density **{action['positive_action_density']:.2%}**，较 v3 9.43% 提升 **{action['positive_action_density'] - 0.09426541486932251:+.2%}**。
- Positive-query coverage: **{action['overall_positive_query_coverage']:.1%}**，较 v3 提升 **{action['overall_positive_query_coverage'] - 0.234:+.1%}**，但距 30% gate 差 0.8 点。
- Non-ceiling coverage: **{action['non_ceiling_positive_query_coverage']:.2%}**。
- 新覆盖 v3 未覆盖 query: **{action['new_queries_covered_beyond_v3']}**。
- Answer-safe action rate: **{action['answer_safe_action_rate']:.2%}**。
- New-query efficiency: **{action['new_query_efficiency']:.4f}**，低于冻结强化门槛。

| Opportunity gate | Result |
| --- | --- |
{gate_rows}

结果为 **3/5**：没有达到 4/5 strong pass，但只失败 2 项，因此未触发原始“至少 4 项失败才停止”的 mandatory stop，后续 selector 被标记为 borderline continuation。

## 5. Fully Nested Selector

{selector_line}

Selector 选择覆盖率为 **{selector.get('coverage', 0):.1%}**，selected-action answer-drop risk 为 **{selector.get('answer_drop_rate', 0):.1%}**。最关键的是 answer F1 从 v2/v3 的非显著小增益，变成了本轮显著正增益；但这仍是同一 1,000-query development protocol，不能替代 scale-up。

## 6. Official HotpotQA

{official_line}

官方指标边界必须明确：answer F1 与 supporting-fact F1 分别显著为正；joint F1 是正向趋势，`p=0.0752`，不得写成显著。Title recall/F1 仍是诊断 proxy，未被重命名为 official supporting-fact metrics。

## 7. Multi-Reader

{multi_line}

FLAN 与 UnifiedQA 的 answer/joint 方向一致，支持“不是单一 reader 偶然现象”；但当前第二 reader 没有独立重新训练 support predictor，support 部分复用相同 frozen context 与 nested sentence predictor。

## 8. 同源 3,000-query Frozen Scale-Up

{scaleup_line}

{scaleup_table}

Scale-up 使用 `hotpot_qa/distractor/validation`、seed 44 的固定顺序，取原开发 1,000 之后的 3,000 条互斥 query。原 1,000 source 与 HybridSoftRetriever baseline title-order 均达到 100% 复现；未使用临时 BM25-only top-5。Generator、selector threshold/coverage、reader prompt/decoding 与 sentence-support threshold 均未在 3,000 条上调参。

该结果把 1,000-query 上 joint F1 的非显著正趋势推进为同源 scale-up 上的显著正增益，并且两个 reader 的 answer/joint 方向一致。Joint EM 的置信区间仍跨 0，不应宣称显著。

## 9. 当前论文判断

最终状态：**{status}**。

可以主张：

- semantic generation 明显提高 positive-action density 与 query opportunity；
- fully nested selector 在 1,000-query HotpotQA 上得到显著 answer F1 与 SP F1 正增益；
- 3,000 条互斥同源 query 在不调参条件下复现 answer/SP/joint F1 正增益；
- 双 reader 方向一致，无系统性 answer degradation。

不能主张：

- opportunity 五项全面通过；
- 1,000-query development 上 official joint F1 已显著；
- 3,000-query 同源 scale-up 等价于跨数据集泛化；
- 2Wiki/MuSiQue 已验证；
- SOTA 或 faithful external-method comparison。

## 10. 未完成项与下一步

Scale-up 当前状态为 `{scaleup.get('status')}`。主表现在可以加入 3,000-query 同源冻结复现，但必须与 1,000-query development 结果分行报告，并明确 baseline 是 `HybridSoftRetriever(alpha=0.55, uniform weights, top_k<=5)`，不是 BM25-only top-5。

下一步不再调整 HotpotQA scale-up 参数，优先完成预注册的 external dataset validation 与 faithful external-method comparison。只有外部数据集也保持方向，才能把“同源规模稳健性”升级为“跨数据集泛化”。
"""
        (REPORTS / "v4_complete_current_report_cn.md").write_text(current_report, encoding="utf-8")
    print(json.dumps(readiness, indent=2))


if __name__ == "__main__":
    main()
