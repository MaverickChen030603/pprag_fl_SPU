#!/usr/bin/env python3
from __future__ import annotations
from bm25_anchor_common import *


def table(methods: dict[str, Any]) -> str:
    lines = [
        "| method | answer_f1 | evidence_f1 | joint_f1 | answer_delta | evidence_delta | joint_delta | effective | fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in methods.items():
        lines.append(
            f"| {name} | {m.get('answer_f1', 0):.4f} | {m.get('evidence_f1', 0):.4f} | {m.get('joint_f1', 0):.4f} | "
            f"{m.get('answer_f1_delta_vs_bm25', 0):+.4f} | {m.get('evidence_f1_delta_vs_bm25', 0):+.4f} | "
            f"{m.get('joint_f1_delta_vs_bm25', 0):+.4f} | {m.get('selected_effective_action_rate', 0):.4f} | {m.get('fallback_rate', 0):.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    oracle = read_json(REPAIR / "outputs/oracle_gap_300/oracle_gap_summary.json")
    action = read_json(REPAIR / "outputs/action_table_300/action_table_summary.json")
    safety = read_json(REPAIR / "outputs/safety_predictor/safety_predictor_summary.json")
    smoke = read_json(REPAIR / "outputs/selector_smoke_300/summary.json")
    diag = read_json(REPAIR / "outputs/diagnostics/failure_summary.json")
    audit = read_json(REPAIR / "outputs/audit/no_leak_audit.json")
    methods = smoke["methods"]
    gate = smoke["gate"]
    if oracle["positive_vs_bm25_rate"] < 0.05:
        sentence = "On 2Wiki, the strong BM25 baseline already captures most available evidence in the current candidate pool, making additional action selection difficult."
        decision = "limitation_candidate_pool"
    elif gate["passed"]:
        sentence = "2Wiki provides preliminary external validation that answer-neutral action selection can improve beyond a strong lexical baseline when actions are anchored to BM25 evidence contexts."
        decision = "appendix_after_1000_gate"
    else:
        sentence = "2Wiki candidate actions contain positive opportunities beyond BM25, but current no-leak selector fails to identify them reliably."
        decision = "limitation_selector_failure"
    report = f"""# 2Wiki BM25-Anchor Repair Report

## 1. Purpose

This repair tests whether the previous 2Wiki selector failure was caused by action definitions that disrupted a strong BM25 context. The repair anchors all actions to BM25 top-5 and only permits minimal tail replacements.

## 2. Oracle Gap vs BM25

- num_queries: `{oracle['num_queries']}`
- positive_vs_bm25_rate: `{oracle['positive_vs_bm25_rate']:.4f}`
- oracle_best_answer_delta_vs_bm25: `{oracle['oracle_best_answer_delta_vs_bm25']:+.4f}`
- oracle_best_evidence_delta_vs_bm25: `{oracle['oracle_best_evidence_delta_vs_bm25']:+.4f}`
- oracle_best_joint_delta_vs_bm25: `{oracle['oracle_best_joint_delta_vs_bm25']:+.4f}`
- selector_recall_of_positive_vs_bm25: `{oracle['selector_recall_of_positive_vs_bm25']:.4f}`
- oracle decision: `{oracle['decision']}`

Oracle is diagnostic only.

## 3. BM25-Anchor Action Table

- num_actions: `{action['num_actions']}`
- effective_action_rate: `{action['effective_action_rate']:.4f}`
- bm25_top1_preserve_rate: `{action['bm25_top1_preserve_rate']:.4f}`
- bm25_top2_preserve_rate: `{action['bm25_top2_preserve_rate']:.4f}`
- bm25_top3_preserve_rate: `{action['bm25_top3_preserve_rate']:.4f}`
- hard_rule_violations: `{action['hard_rule_violations']}`

## 4. Real Safety Predictor

- answer_safe_auc: `{safety['answer_safe_auc']:.4f}`
- paper_positive_auc: `{safety['paper_positive_auc']:.4f}`
- false_safe_rate: `{safety['false_safe_rate']:.4f}`
- false_negative_rate: `{safety['false_negative_rate']:.4f}`

## 5. Selector Smoke 300

{table(methods)}

Gate:

```json
{json.dumps(gate, ensure_ascii=False, indent=2)}
```

## 6. Failure Diagnosis

```json
{json.dumps(diag.get('failure_distribution', {}), ensure_ascii=False, indent=2)}
```

## 7. No-Leak Audit

```json
{json.dumps(audit, ensure_ascii=False, indent=2)}
```

## 8. Paper Decision

Decision: `{decision}`

{sentence}

## 9. MuSiQue Decision

Do not start MuSiQue from this state. The bottleneck is selector reliability over a strong lexical baseline, not cross-dataset plumbing.
"""
    rec = f"""# 2Wiki External Validation Decision

Decision: `{decision}`

{sentence}

Current recommendation: keep 2Wiki in the paper as pipeline / lexical-routing validation or limitation, not as main selector generalization evidence.
"""
    for p, text in [
        (REPAIR / "reports/2wiki_bm25_anchor_repair_report.md", report),
        (REPAIR / "reports/2wiki_external_validation_decision.md", rec),
        (MIRROR / "2wiki_bm25_anchor_repair_report_latest.md", report),
        (MIRROR / "2wiki_external_validation_decision_latest.md", rec),
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "written", "decision": decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
