#!/usr/bin/env python3
from __future__ import annotations
from selector_alignment_common import *


def table(methods: dict[str, Any]) -> str:
    lines = [
        "| method | answer_f1 | evidence_f1 | joint_f1 | answer_delta_vs_bm25 | evidence_delta_vs_bm25 | joint_delta_vs_bm25 | effective_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in methods.items():
        lines.append(
            f"| {name} | {m.get('answer_f1', 0):.4f} | {m.get('evidence_f1', 0):.4f} | {m.get('joint_f1', 0):.4f} | "
            f"{m.get('answer_f1_delta_vs_bm25', 0):+.4f} | {m.get('evidence_f1_delta_vs_bm25', 0):+.4f} | "
            f"{m.get('joint_f1_delta_vs_bm25', 0):+.4f} | {m.get('selected_effective_action_rate', 0):.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    action = read_json(ALIGN / "outputs/action_table_300/action_table_summary.json")
    feat = read_json(ALIGN / "outputs/action_table_300/feature_summary.json")
    smoke = read_json(ALIGN / "outputs/selector_smoke_300/summary.json")
    audit = read_json(ALIGN / "outputs/audit/2wiki_no_leak_audit.json")
    diag = read_json(ALIGN / "outputs/diagnostics/failure_summary.json")
    crossfit = read_json(ALIGN / "outputs/selector_crossfit_1000/final_1000_summary.json")
    methods = smoke.get("methods", {})
    gate = smoke.get("gate", {})
    if gate.get("passed"):
        recommendation = "appendix_or_main_after_1000"
        paper_text = "2Wiki smoke passed; wait for formal 1000 before main-paper external validation claim."
    else:
        recommendation = "pipeline_validation_only"
        paper_text = "2Wiki results validate dataset transfer and lexical routing effectiveness, but do not yet establish selector-level generalization beyond a strong BM25 baseline."
    report = f"""# V7-HP-PAPER 2Wiki Selector Alignment Report

## 1. Purpose

This experiment upgrades the previous 2Wiki lexical/BM25 reader smoke into a selector-level validation attempt for `selector_v2.3_answer_neutral_positive_selector`.

## 2. What the Previous Smoke Proved

The previous 2Wiki dev-300 reader-backed smoke proved that the 2Wiki adapter and reader path work, and that lexical/BM25 routing is much stronger than raw context order. It did not prove HotpotQA v2.3 selector generalization.

## 3. Action Table and Feature Alignment

- queries: `{action['num_queries']}`
- actions: `{action['num_actions']}`
- effective action rate: `{action['effective_action_rate']:.4f}`
- prefix2 preserve rate: `{action['prefix2_preserve_rate']:.4f}`
- prefix3 preserve rate: `{action['prefix3_preserve_rate']:.4f}`
- dense feature available: `{action['dense_feature_available']}`
- safe answer mode: `{feat['safe_answer_prob_mode']}`

Aligned feature set:

```text
{chr(10).join(feat['aligned_features'])}
```

## 4. Selector Smoke 300 Results

Main baseline is `bm25_or_lexical_routing`.

{table(methods)}

Gate:

```json
{json.dumps(gate, ensure_ascii=False, indent=2)}
```

## 5. No-Leak Audit

- audit status: `{audit['status']}`
- query fold disjoint: `{audit['query_fold_disjoint']}`
- held-out outcome not used for inference: `{audit['held_out_outcome_not_used_for_inference']}`
- oracle separated: `{audit['oracle_separated_from_formal_method']}`

## 6. Failure Diagnosis

```json
{json.dumps(diag.get('failure_distribution', {}), ensure_ascii=False, indent=2)}
```

## 7. 1000 Decision

```json
{json.dumps(crossfit, ensure_ascii=False, indent=2)}
```

## 8. Paper Recommendation

Recommendation: `{recommendation}`

{paper_text}

## 9. MuSiQue Recommendation

Do not start MuSiQue as a broad stress test until the 2Wiki selector-level gap against BM25 is resolved or explicitly framed as a limitation. A MuSiQue run now would mostly test dataset plumbing rather than selector generalization.
"""
    rec = f"""# 2Wiki Paper Recommendation

Recommendation: `{recommendation}`

{paper_text}

Current status:

- 2Wiki adapter: ready
- dev-300 lexical/BM25 reader smoke: positive
- 2Wiki action-feature alignment: complete
- selector smoke 300 gate: `{gate.get('decision')}`
- formal 1000: `{crossfit.get('status')}`

Paper-safe sentence:

> 2WikiMultiHopQA validates the cross-dataset data and reader pipeline and confirms that lexical routing is a strong external baseline; however, selector-level generalization beyond BM25 requires additional action-feature adaptation.
"""
    for p, text in [
        (ALIGN / "reports/2wiki_selector_alignment_report.md", report),
        (ALIGN / "reports/2wiki_paper_recommendation.md", rec),
        (MIRROR_DIR / "2wiki_selector_alignment_report_latest.md", report),
        (MIRROR_DIR / "2wiki_paper_recommendation_latest.md", rec),
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "written", "recommendation": recommendation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
