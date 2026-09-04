from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


READER = _load("V7-HP4/run_hp4_reader_counterfactual_eval.py", "hp4_reader")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def row_index(rows: list[dict[str, Any]], mode: str) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in rows if row.get("mode") == mode}


def doc_maps(validation: Path, sample_size: int) -> dict[str, dict[str, Any]]:
    out = {}
    for case, docs in READER.materialize_dev(validation, sample_size):
        out[str(case.get("id", case.get("_id", "")))] = {
            "case": case,
            "docs": {doc.doc_id: doc for doc in docs},
        }
    return out


def context_text(doc_ids: list[str], docs: dict[str, Any]) -> str:
    parts = []
    for doc_id in doc_ids:
        doc = docs.get(doc_id)
        if doc is not None:
            parts.append(f"{doc.title}. {doc.text}")
    return "\n".join(parts)


def doc_titles(doc_ids: list[str], docs: dict[str, Any]) -> list[str]:
    return [docs[doc_id].title if doc_id in docs else doc_id for doc_id in doc_ids]


def answer_access(answer: str, doc_ids: list[str], docs: dict[str, Any]) -> float:
    top_docs = [docs[doc_id] for doc_id in doc_ids if doc_id in docs]
    return float(READER.answer_in_context(answer, top_docs))


def classify_case(base: dict[str, Any], cand: dict[str, Any], case: dict[str, Any], docs: dict[str, Any]) -> tuple[str, list[str]]:
    base_ids = [str(x) for x in base.get("top_doc_ids", [])]
    cand_ids = [str(x) for x in cand.get("top_doc_ids", [])]
    base_set = set(base_ids)
    cand_set = set(cand_ids)
    overlap = len(base_set & cand_set) / max(len(base_set), 1)
    answer = str(case.get("answer", ""))
    base_access = answer_access(answer, base_ids, docs)
    cand_access = answer_access(answer, cand_ids, docs)
    d_support = float(cand.get("support_recall_at_k", 0.0)) - float(base.get("support_recall_at_k", 0.0))
    d_sp = float(cand.get("sp_f1", 0.0)) - float(base.get("sp_f1", 0.0))
    d_answer = float(cand.get("answer_f1", 0.0)) - float(base.get("answer_f1", 0.0))

    reasons = [
        f"d_answer={d_answer:+.4f}",
        f"d_support={d_support:+.4f}",
        f"d_sp={d_sp:+.4f}",
        f"overlap@5={overlap:.2f}",
        f"base_access={base_access:.0f}",
        f"cand_access={cand_access:.0f}",
    ]
    if cand_access < base_access or cand_access == 0.0:
        return "answer_text_missing", reasons
    if overlap >= 0.80 and cand_access >= base_access and abs(d_support) < 1e-9:
        return "evidence_order_issue", reasons
    if d_support > 1e-9 and cand_access >= base_access:
        return "support_gain_reader_interference", reasons
    if overlap < 0.80 and cand_access >= base_access:
        return "context_replacement_loss", reasons
    return "other_reader_sensitivity", reasons


def prediction_shift(base: dict[str, Any], cand: dict[str, Any]) -> str:
    b = str(base.get("prediction", "")).strip()
    c = str(cand.get("prediction", "")).strip()
    if not b and not c:
        return "both_empty"
    if READER.normalize_answer(b) == READER.normalize_answer(c):
        return "same_prediction"
    return "changed_prediction"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# V7-HP4 Phase 3 Reader-Aware Oracle Diagnostic",
        "",
        f"- baseline_mode: `{payload['baseline_mode']}`",
        f"- candidate_mode: `{payload['candidate_mode']}`",
        f"- compared_cases: {payload['compared_cases']}",
        f"- answer_f1_drop_cases: {payload['drop_cases']}",
        "",
        "## Drop-Case Categories",
        "",
        "| category | count | share |",
        "| --- | ---: | ---: |",
    ]
    total = max(payload["drop_cases"], 1)
    for category, count in payload["category_counts"].items():
        lines.append(f"| {category} | {count} | {count / total:.2%} |")
    lines.extend([
        "",
        "## Metric Deltas On Drop Cases",
        "",
        f"- mean_delta_answer_f1: {payload['drop_delta_means']['answer_f1']:+.4f}",
        f"- mean_delta_joint_f1: {payload['drop_delta_means']['joint_f1']:+.4f}",
        f"- mean_delta_support_recall: {payload['drop_delta_means']['support_recall_at_k']:+.4f}",
        f"- mean_delta_sp_f1: {payload['drop_delta_means']['sp_f1']:+.4f}",
        "",
        "## Interpretation",
        "",
    ])
    dominant = next(iter(payload["category_counts"]), None)
    if dominant == "answer_text_missing":
        lines.append("The main failure mode is answer-text loss: rerank improves support routing but removes or buries answer-bearing wording.")
    elif dominant == "support_gain_reader_interference":
        lines.append("The main failure mode is reader interference after support gain: extra support-like context changes the reasoning path even when answer text remains accessible.")
    elif dominant == "evidence_order_issue":
        lines.append("The main failure mode is ordering/prompt sensitivity: document sets are mostly preserved, but reader output changes after reordering.")
    else:
        lines.append("Failures are mixed; inspect case samples before changing the policy objective.")
    lines.extend([
        "",
        "## Representative Drop Cases",
        "",
        "| id | category | base_f1 | cand_f1 | base_pred | cand_pred | answer | base_top_titles | cand_top_titles |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ])
    for row in payload["representative_cases"][:12]:
        def esc(v: Any) -> str:
            return str(v).replace("|", "/").replace("\n", " ")[:180]
        lines.append(
            f"| {esc(row['id'])} | {esc(row['category'])} | {row['base_answer_f1']:.3f} | "
            f"{row['cand_answer_f1']:.3f} | {esc(row['base_prediction'])} | {esc(row['cand_prediction'])} | "
            f"{esc(row['answer'])} | {esc('; '.join(row['base_top_titles']))} | {esc('; '.join(row['cand_top_titles']))} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", default="V7-HP4/data/hotpot_validation_1000.json")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--baseline-rows", default="V7-HP4/outputs/hp4_phase3_100_batch/phase3_100_rows.json")
    parser.add_argument("--candidate-rows", default="V7-HP4/outputs/hp4_phase3_rerank_ablation/rerank_ablation_rows.json")
    parser.add_argument("--baseline-mode", default="A_phase2_baseline_tau1")
    parser.add_argument("--candidate-mode", default="A_tau0.7_gate4_bgpool16_balanced")
    parser.add_argument("--output-root", default="V7-HP4/outputs/hp4_phase3_reader_oracle_diagnostic")
    parser.add_argument("--report-dir", default="实验分析报告/V7-HP4")
    args = parser.parse_args()

    base_rows = row_index(json.loads(Path(args.baseline_rows).read_text(encoding="utf-8")), args.baseline_mode)
    cand_rows = row_index(json.loads(Path(args.candidate_rows).read_text(encoding="utf-8")), args.candidate_mode)
    cases = doc_maps(Path(args.validation), args.sample_size)

    compared = sorted(set(base_rows) & set(cand_rows) & set(cases))
    drops = []
    improvements = []
    unchanged = []
    for qid in compared:
        base = base_rows[qid]
        cand = cand_rows[qid]
        d_answer = float(cand.get("answer_f1", 0.0)) - float(base.get("answer_f1", 0.0))
        item = cases[qid]
        category, reasons = classify_case(base, cand, item["case"], item["docs"])
        record = {
            "id": qid,
            "category": category,
            "reasons": reasons,
            "question": item["case"].get("question", ""),
            "answer": item["case"].get("answer", ""),
            "base_prediction": base.get("prediction", ""),
            "cand_prediction": cand.get("prediction", ""),
            "prediction_shift": prediction_shift(base, cand),
            "base_answer_f1": float(base.get("answer_f1", 0.0)),
            "cand_answer_f1": float(cand.get("answer_f1", 0.0)),
            "delta_answer_f1": d_answer,
            "base_joint_f1": float(base.get("joint_f1", 0.0)),
            "cand_joint_f1": float(cand.get("joint_f1", 0.0)),
            "delta_joint_f1": float(cand.get("joint_f1", 0.0)) - float(base.get("joint_f1", 0.0)),
            "base_support_recall_at_k": float(base.get("support_recall_at_k", 0.0)),
            "cand_support_recall_at_k": float(cand.get("support_recall_at_k", 0.0)),
            "delta_support_recall_at_k": float(cand.get("support_recall_at_k", 0.0)) - float(base.get("support_recall_at_k", 0.0)),
            "base_sp_f1": float(base.get("sp_f1", 0.0)),
            "cand_sp_f1": float(cand.get("sp_f1", 0.0)),
            "delta_sp_f1": float(cand.get("sp_f1", 0.0)) - float(base.get("sp_f1", 0.0)),
            "base_top_doc_ids": base.get("top_doc_ids", []),
            "cand_top_doc_ids": cand.get("top_doc_ids", []),
            "base_top_titles": doc_titles([str(x) for x in base.get("top_doc_ids", [])], item["docs"]),
            "cand_top_titles": doc_titles([str(x) for x in cand.get("top_doc_ids", [])], item["docs"]),
        }
        if d_answer < -1e-9:
            drops.append(record)
        elif d_answer > 1e-9:
            improvements.append(record)
        else:
            unchanged.append(record)

    category_counts = Counter(row["category"] for row in drops)
    ordered_counts = dict(category_counts.most_common())
    def mean_delta(name: str) -> float:
        key = f"delta_{name}"
        return sum(float(row[key]) for row in drops) / len(drops) if drops else 0.0
    representatives = sorted(drops, key=lambda r: (r["category"], r["delta_answer_f1"]))[:20]
    payload = {
        "baseline_mode": args.baseline_mode,
        "candidate_mode": args.candidate_mode,
        "compared_cases": len(compared),
        "drop_cases": len(drops),
        "improved_cases": len(improvements),
        "unchanged_cases": len(unchanged),
        "category_counts": ordered_counts,
        "prediction_shift_counts": dict(Counter(row["prediction_shift"] for row in drops).most_common()),
        "drop_delta_means": {
            "answer_f1": mean_delta("answer_f1"),
            "joint_f1": mean_delta("joint_f1"),
            "support_recall_at_k": mean_delta("support_recall_at_k"),
            "sp_f1": mean_delta("sp_f1"),
        },
        "representative_cases": representatives,
    }
    out = Path(args.output_root)
    save_json(out / "reader_oracle_diagnostic_summary.json", payload)
    save_json(out / "reader_oracle_drop_cases.json", drops)
    save_json(out / "reader_oracle_improved_cases.json", improvements)
    report = Path(args.report_dir) / "v7_hp4_phase3_reader_oracle_diagnostic_latest.md"
    write_report(report, payload)
    print(json.dumps({**payload, "report_path": str(report), "output_root": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
