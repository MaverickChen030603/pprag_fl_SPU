from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parent
REPORT_DIR = BASE / "实验分析报告" / "V7-agent-2"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
STRICT_FILES = [
    BASE / "outputs" / "hp1_strict_eval" / "hp1_budget_aligned" / "hp1_strict_summary.csv",
    BASE / "outputs" / "hp1_strict_eval" / "v7agent2_ablation" / "hp1_strict_summary.csv",
]
METHODS = ["hypernet_v6", "agent_rule_v7", "agent_rule_v7_no_prior", "agent_rule_v7_no_coverage", "agent_rule_v7_no_memory"]


def load_rows() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in STRICT_FILES if p.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            rows.append({"method": method, "n": 0})
            continue
        rows.append({
            "method": method,
            "n": len(sub),
            "early_recall_mean": sub["early_evidence_recall_hp1"].mean(),
            "early_recall_std": sub["early_evidence_recall_hp1"].std(ddof=0),
            "hp1_score_mean": sub["hp1_multihop_score"].mean(),
            "hp1_score_std": sub["hp1_multihop_score"].std(ddof=0),
            "bridge_recall_mean": sub["bridge_block_recall_hp1"].mean(),
            "avg_budget_topk": sub["avg_budget_topk_hp1"].mean(),
        })
    return pd.DataFrame(rows)


def paired_wilcoxon(df: pd.DataFrame, base_method: str, other_method: str) -> dict:
    a = df[df["method"] == base_method][["seed", "hp1_multihop_score"]].rename(columns={"hp1_multihop_score": "a"})
    b = df[df["method"] == other_method][["seed", "hp1_multihop_score"]].rename(columns={"hp1_multihop_score": "b"})
    merged = a.merge(b, on="seed")
    if len(merged) < 5:
        return {"comparison": f"{base_method} vs {other_method}", "n": len(merged), "p_value": "N/A (n<5)", "significant": False}
    stat, p = wilcoxon(merged["a"], merged["b"])
    return {"comparison": f"{base_method} vs {other_method}", "n": len(merged), "statistic": float(stat), "p_value": float(p), "significant": bool(p < 0.05)}


def main() -> int:
    df = load_rows()
    summary = summarize(df) if not df.empty else pd.DataFrame({"method": METHODS, "n": [0]*len(METHODS)})
    summary.to_csv(REPORT_DIR / "ablation_summary.csv", index=False)
    tests = [paired_wilcoxon(df, "agent_rule_v7", m) for m in ["agent_rule_v7_no_prior", "agent_rule_v7_no_coverage", "agent_rule_v7_no_memory"]] if not df.empty else []
    (REPORT_DIR / "ablation_wilcoxon.json").write_text(json.dumps(tests, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(tests, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
