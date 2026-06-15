from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

BASE = Path(__file__).resolve().parent
EVAL_DIR = BASE / "outputs" / "hotpot_official_eval" / "v7agent2_all"
REPORT_DIR = BASE / "实验分析报告" / "V7-agent-2"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
METRICS = ["answer_em", "answer_f1", "sp_em", "sp_f1", "joint_em", "joint_f1", "support_title_recall_at_k"]
METHODS = ["hypernet_v6", "adaptive_v6", "agent_bandit_v7", "agent_rule_v7"]


def method_from_name(name: str) -> str:
    for prefix, method in [("hypernet-v6", "hypernet_v6"), ("adaptive-v6", "adaptive_v6"), ("agent-bandit-v7", "agent_bandit_v7"), ("agent-rule-v7", "agent_rule_v7")]:
        if name.startswith(prefix):
            return method
    return name.split("_k", 1)[0].replace("-", "_")


def load_eval_records(method: str) -> pd.DataFrame:
    rows = []
    for metrics_path in EVAL_DIR.rglob("official_metrics.json"):
        if method_from_name(metrics_path.parent.name) != method:
            continue
        per_query = metrics_path.parent / "per_query_official.jsonl"
        if per_query.exists():
            for line in per_query.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                flat = {"method": method, "run_name": metrics_path.parent.name, "example_id": rec.get("example_id") or rec.get("id")}
                for k in METRICS:
                    flat[k] = rec.get(k, (rec.get("metrics") or {}).get(k))
                rows.append(flat)
        else:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            flat = {"method": method, "run_name": metrics_path.parent.name, "example_id": metrics_path.parent.name}
            for k in METRICS:
                flat[k] = (data.get("metrics") or {}).get(k)
            rows.append(flat)
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n: int = 1000) -> tuple[float, float, float]:
    values = values.astype(float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    means = [np.mean(np.random.choice(values, size=len(values), replace=True)) for _ in range(n)]
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_ttest(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> dict:
    merged = a[["example_id", metric]].merge(b[["example_id", metric]], on="example_id", suffixes=("_a", "_b"))
    if len(merged) < 10:
        return {"metric": metric, "p_value": "N/A", "n_pairs": len(merged), "significant": False}
    stat, p = ttest_rel(merged[f"{metric}_a"], merged[f"{metric}_b"])
    return {"metric": metric, "statistic": float(stat), "p_value": float(p), "n_pairs": len(merged), "significant": bool(p < 0.05)}


def main() -> int:
    dfs = {m: load_eval_records(m) for m in METHODS}
    rows = []
    for method, df in dfs.items():
        row = {"method": method, "n": len(df)}
        for metric in METRICS:
            if metric in df and len(df):
                mu, lo, hi = bootstrap_ci(df[metric].dropna().to_numpy())
                row[f"{metric}_mean"] = round(mu, 4)
                row[f"{metric}_ci95_lo"] = round(lo, 4)
                row[f"{metric}_ci95_hi"] = round(hi, 4)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(REPORT_DIR / "official_eval_summary_1000.csv", index=False)
    tests = []
    if not dfs["agent_rule_v7"].empty and not dfs["hypernet_v6"].empty:
        for metric in ["answer_f1", "sp_f1", "joint_f1"]:
            tests.append(paired_ttest(dfs["agent_rule_v7"], dfs["hypernet_v6"], metric))
    (REPORT_DIR / "official_eval_paired_ttest.json").write_text(json.dumps(tests, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(tests, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
