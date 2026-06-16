from __future__ import annotations

import json
from pathlib import Path
import math
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
ANALYSIS = BASE / "analysis"
ANALYSIS.mkdir(exist_ok=True)


def load_strict() -> pd.DataFrame:
    frames = []
    for p in (ANALYSIS / "strict_runs").glob("*/hp1_strict_summary.csv"):
        df = pd.read_csv(p)
        df["strict_suite"] = p.parent.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(ANALYSIS / "strict_diagnostic_seed_level.csv", index=False)
    if out.empty:
        pd.DataFrame(columns=["method", "n"]).to_csv(ANALYSIS / "strict_diagnostic_summary.csv", index=False)
        return out
    summary = out.groupby(["method", "agent_profile"], as_index=False).agg(
        n=("seed", "count"),
        early_recall=("early_evidence_recall_hp1", "mean"),
        bridge_recall=("bridge_block_recall_hp1", "mean"),
        target_recall=("target_block_recall_hp1", "mean"),
        diversity=("selection_diversity_hp1", "mean"),
        hp1_score=("hp1_multihop_score", "mean"),
        avg_topk=("avg_budget_topk_hp1", "mean"),
        budget_std=("budget_std_hp1", "mean"),
    )
    summary.to_csv(ANALYSIS / "strict_diagnostic_summary.csv", index=False)
    summary[summary["method"].str.contains("agent_pm_dynamic", na=False)].to_csv(ANALYSIS / "memory_ablation_summary.csv", index=False)
    summary[summary["method"].str.contains("agent_fixed_slot|agent_dynamic", regex=True, na=False)].to_csv(ANALYSIS / "dynamic_ablation_summary.csv", index=False)
    summary[summary["method"].str.contains("agent_pm_bandit|agent_rule_v7_dynamic|agent_pm_dynamic_full", regex=True, na=False)].to_csv(ANALYSIS / "bandit_slot_summary.csv", index=False)
    return out


def load_official() -> pd.DataFrame:
    records = []
    for p in (BASE / "eval_outputs" / "official_fid_t5").rglob("official_metrics.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        rec = {"method": data.get("method"), "profile": data.get("profile"), "suite": data.get("suite_tag"), "seed": data.get("seed"), "path": str(p)}
        rec.update(data.get("metrics", {}))
        records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(ANALYSIS / "official_fid_t5_seed_level.csv", index=False)
    if df.empty:
        pd.DataFrame(columns=["method", "n"]).to_csv(ANALYSIS / "official_fid_t5_summary.csv", index=False)
        return df
    rows = []
    metrics = ["answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1", "support_title_recall_at_k"]
    for method, g in df.groupby("method"):
        row = {"method": method, "n": len(g)}
        for metric in metrics:
            vals = pd.to_numeric(g.get(metric), errors="coerce").dropna()
            row[f"{metric}_mean"] = vals.mean() if len(vals) else math.nan
            row[f"{metric}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(ANALYSIS / "official_fid_t5_summary.csv", index=False)
    return df


def statistical_tests(strict_df: pd.DataFrame) -> None:
    tests = []
    pairs = [
        ("agent_rule_v7_dynamic", "agent_pm_dynamic_full", "hp1_multihop_score"),
        ("agent_rule_v7_dynamic", "agent_pm_bandit_slot", "hp1_multihop_score"),
        ("agent_pm_dynamic_full", "agent_pm_dynamic_no_memory", "hp1_multihop_score"),
        ("agent_pm_dynamic_full", "agent_pm_dynamic_no_failure_memory", "hp1_multihop_score"),
        ("agent_dynamic_slot", "agent_fixed_slot_1", "hp1_multihop_score"),
        ("agent_dynamic_slot", "agent_fixed_slot_2", "hp1_multihop_score"),
    ]
    for a, b, metric in pairs:
        if strict_df.empty or metric not in strict_df:
            continue
        aa = strict_df[strict_df.method == a][["seed", metric]].rename(columns={metric: "a"})
        bb = strict_df[strict_df.method == b][["seed", metric]].rename(columns={metric: "b"})
        merged = aa.merge(bb, on="seed")
        if len(merged) >= 2:
            try:
                stat, pval = stats.wilcoxon(merged["a"], merged["b"])
            except Exception:
                stat, pval = math.nan, math.nan
            tests.append({"comparison": f"{a} vs {b}", "metric": metric, "n": len(merged), "mean_delta": (merged["a"] - merged["b"]).mean(), "wilcoxon_stat": stat, "p_value": pval})
    pd.DataFrame(tests).to_csv(ANALYSIS / "statistical_tests.csv", index=False)


def subgroup_and_behavior(strict_df: pd.DataFrame) -> None:
    subgroups = ["hard-query subset", "easy-query subset", "rare-domain subset", "common-domain subset", "hard-client subset", "normal-client subset", "early-evidence-needed subset", "bridge-heavy subset"]
    rows = []
    if not strict_df.empty:
        for method, g in strict_df.groupby("method"):
            for subgroup in subgroups:
                rows.append({"method": method, "subgroup": subgroup, "n": int(g["event_count"].sum()) if "event_count" in g else len(g), "early_recall": g["early_evidence_recall_hp1"].mean(), "bridge_recall": g["bridge_block_recall_hp1"].mean(), "support_title_recall": math.nan, "answer_f1": math.nan, "joint_f1": math.nan, "hp1_score": g["hp1_multihop_score"].mean(), "note": "proxy strict subgroup; query-level merge after official eval"})
    pd.DataFrame(rows).to_csv(ANALYSIS / "subgroup_analysis.csv", index=False)
    behavior_rows = []
    for p in (BASE / "eval_outputs" / "official_fid_t5").rglob("per_query_official.jsonl"):
        method = p.parent.name
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()[:50]):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            behavior_rows.append({"query_id": rec.get("example_id") or rec.get("id") or i, "question": rec.get("question", ""), "gold_answer": rec.get("gold_answer", ""), "gold_supporting_titles": rec.get("gold_supporting_titles", ""), "client_id": "", "round_id": "", "method": method, "query_hardness": rec.get("query_hardness", ""), "domain_rarity": "", "selected_blocks": "", "selected_block_types": "", "score_components": "", "replacement_reason": "", "early_hit": "", "bridge_hit": "", "support_title_hit": rec.get("support_title_recall_at_k", ""), "answer_prediction": rec.get("answer_pred", ""), "answer_correct": rec.get("answer_em", ""), "joint_f1": rec.get("joint_f1", "")})
    pd.DataFrame(behavior_rows).to_csv(ANALYSIS / "per_query_behavior.csv", index=False)
    lines = ["# V7-agent-PM Representative Cases", "", "Official per-query records are summarized here after FiD/T5 evaluation completes.", ""]
    for idx, row in enumerate(behavior_rows[:10], start=1):
        lines += [f"## Case {idx}: {row.get('method', 'unknown')}", f"- Query: {row.get('question', '')}", f"- Prediction: {row.get('answer_prediction', '')}", f"- Joint F1: {row.get('joint_f1', '')}", "- Interpretation: PM behavior explanation requires matching selection trace; current row is official eval evidence.", ""]
    (ANALYSIS / "representative_cases.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    strict_df = load_strict()
    load_official()
    statistical_tests(strict_df)
    subgroup_and_behavior(strict_df)
    print("analysis written to", ANALYSIS)


if __name__ == "__main__":
    main()
