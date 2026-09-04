from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from src.v7_hp4.agent_continuous import ContinuousUploadPolicy
from src.v7_hp4.hybrid_retriever import HybridDocument, HybridSoftRetriever
from src.v7_hp4.policy_gradient import block_state_from_doc


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_READER = _load_module("V7-HP4/run_hp4_reader_counterfactual_eval.py", "v7_hp4_reader_helpers")
_FULLVAL = _load_module("V7-HP4/run_hp4_full_validation_eval.py", "v7_hp4_full_validation_helpers")

Reader = _READER.Reader
answer_in_context = _READER.answer_in_context
f1_score = _READER.f1_score
make_prompt = _READER.make_prompt
materialize_dev = _READER.materialize_dev
normalize_answer = _READER.normalize_answer
sp_metrics = _READER.sp_metrics
load_or_build_validation_split = _FULLVAL.load_or_build_validation_split
permutation_p_value = _FULLVAL.permutation_p_value


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sanitize_docs_for_policy(docs: list[HybridDocument]) -> list[HybridDocument]:
    """Remove support-derived routing hints while preserving labels for metrics."""
    sanitized = []
    for idx, doc in enumerate(docs):
        sanitized.append(replace(
            doc,
            client_id=f"client_{idx % 5}",
            support_role="unknown",
            bridge_entities=[],
            rare_tokens=[],
            dense_score_hint=1.0,
            soft_weight=1.0,
        ))
    return sanitized


def load_policy(path: Path, device: str) -> tuple[ContinuousUploadPolicy, torch.device]:
    runtime_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(path, map_location=runtime_device)
    hidden_dim = int(payload.get("hidden_dim", 48))
    policy = ContinuousUploadPolicy(hidden_dim=hidden_dim, init_bias=-0.4).to(runtime_device)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    return policy, runtime_device


def no_leak_policy_weights(
    question: str,
    docs: list[HybridDocument],
    policy: ContinuousUploadPolicy,
    device: torch.device,
    mode: str,
) -> dict[str, float]:
    if mode == "baseline_uniform":
        return {doc.doc_id: 1.0 for doc in docs}
    states = [block_state_from_doc(question, doc, mode="natural") for doc in docs]
    features = ContinuousUploadPolicy.tensor_from_states(states, device=str(device))
    with torch.no_grad():
        weights = policy(features).detach().cpu().tolist()
    return {doc.doc_id: float(max(0.0, min(1.0, w))) for doc, w in zip(docs, weights)}


def build_item(
    case: dict[str, Any],
    docs: list[HybridDocument],
    policy: ContinuousUploadPolicy,
    device: torch.device,
    mode: str,
    top_k: int,
    alpha: float,
) -> dict[str, Any]:
    question = str(case.get("question", ""))
    weights = no_leak_policy_weights(question, docs, policy, device, mode)
    ranked = HybridSoftRetriever(docs, alpha=alpha).rank(question, weights=weights, top_k=top_k)
    top_docs = [doc for doc, _ in ranked]
    gold_titles = {str(t) for t in case.get("supporting_titles", [])}
    pred_titles = {doc.title for doc in top_docs}
    support_recall, sp_f1 = sp_metrics(pred_titles, gold_titles)
    return {
        "case": case,
        "mode": mode,
        "top_docs": top_docs,
        "top_doc_ids": [doc.doc_id for doc in top_docs],
        "top_titles": [doc.title for doc in top_docs],
        "weights": weights,
        "answer_access_at_k": answer_in_context(case.get("answer", ""), top_docs),
        "support_recall_at_k": support_recall,
        "sp_f1": sp_f1,
        "prompt": make_prompt(question, top_docs),
    }


def evaluate_no_leak(
    examples: list[tuple[dict[str, Any], list[HybridDocument]]],
    reader: Reader,
    policy: ContinuousUploadPolicy,
    device: torch.device,
    top_k: int,
    alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retrieval_items = []
    for case, docs in examples:
        sanitized = sanitize_docs_for_policy(docs)
        retrieval_items.append(build_item(case, sanitized, policy, device, "baseline_uniform", top_k, alpha))
        retrieval_items.append(build_item(case, sanitized, policy, device, "hp4_no_leak_policy", top_k, alpha))

    predictions = reader.generate([item["prompt"] for item in retrieval_items])
    rows = []
    for item, pred in zip(retrieval_items, predictions):
        case = item["case"]
        answer = case.get("answer", "")
        answer_em = float(normalize_answer(pred) == normalize_answer(answer))
        answer_f1 = f1_score(pred, answer)
        rows.append({
            "id": str(case.get("id", case.get("_id", ""))),
            "mode": item["mode"],
            "prediction": pred,
            "answer": answer,
            "answer_em": answer_em,
            "answer_f1": answer_f1,
            "answer_access_at_k": item["answer_access_at_k"],
            "support_recall_at_k": item["support_recall_at_k"],
            "sp_f1": item["sp_f1"],
            "joint_f1": answer_f1 * float(item["sp_f1"]),
            "top_doc_ids": item["top_doc_ids"],
            "top_titles": item["top_titles"],
            "weights": item["weights"],
        })
    return rows, summarize(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append(row)
    out = {}
    for mode, items in grouped.items():
        out[mode] = {
            "n": len(items),
            "answer_access_at_k": sum(float(r["answer_access_at_k"]) for r in items) / len(items),
            "support_recall_at_k": sum(float(r["support_recall_at_k"]) for r in items) / len(items),
            "sp_f1": sum(float(r["sp_f1"]) for r in items) / len(items),
            "answer_em": sum(float(r["answer_em"]) for r in items) / len(items),
            "answer_f1": sum(float(r["answer_f1"]) for r in items) / len(items),
            "joint_f1": sum(float(r["joint_f1"]) for r in items) / len(items),
        }
    return out


def paired_diffs(rows: list[dict[str, Any]], metric: str) -> list[float]:
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_id[str(row["id"])][str(row["mode"])] = row
    diffs = []
    for modes in by_id.values():
        if "baseline_uniform" in modes and "hp4_no_leak_policy" in modes:
            diffs.append(float(modes["hp4_no_leak_policy"].get(metric, 0.0)) - float(modes["baseline_uniform"].get(metric, 0.0)))
    return diffs


def load_oracle_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_report(
    path: Path,
    summary: dict[str, Any],
    significance: dict[str, Any],
    oracle_summary: dict[str, Any],
    rows_path: Path,
    data_source: str,
    policy_path: Path,
) -> None:
    def metric(mode: str, name: str) -> float:
        return float(summary.get(mode, {}).get(name, 0.0))

    def oracle_metric(mode: str, name: str) -> float:
        return float(oracle_summary.get(mode, {}).get(name, 0.0))

    lines = [
        "# V7-HP4 No-Leak Learned Policy Validation",
        "",
        f"Data source: `{data_source}`",
        f"Policy checkpoint: `{policy_path}`",
        f"Rows: `{rows_path}`",
        "",
        "## No-Leak Metrics",
        "",
        "| mode | n | access@k | support_recall@k | sp_f1 | answer_em | answer_f1 | joint_f1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ["baseline_uniform", "hp4_no_leak_policy"]:
        m = summary.get(mode, {})
        lines.append(
            f"| {mode} | {m.get('n', 0)} | {m.get('answer_access_at_k', 0):.4f} | "
            f"{m.get('support_recall_at_k', 0):.4f} | {m.get('sp_f1', 0):.4f} | "
            f"{m.get('answer_em', 0):.4f} | {m.get('answer_f1', 0):.4f} | {m.get('joint_f1', 0):.4f} |"
        )
    lines.extend([
        "",
        "## Paired Significance",
        "",
        "| metric | n | mean_gap | p_value_two_sided | bootstrap_ci95 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for name in ["joint_f1", "sp_f1"]:
        s = significance[name]
        lines.append(
            f"| {name} | {s['n']} | {s['mean_diff']:+.4f} | {s['p_value_two_sided']:.6f} | "
            f"[{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}] |"
        )
    lines.extend([
        "",
        "## Oracle Upper Bound Context",
        "",
        "| metric | no-leak gap | oracle gap | retained_ratio |",
        "| --- | ---: | ---: | ---: |",
    ])
    for name in ["support_recall_at_k", "sp_f1", "answer_f1", "joint_f1"]:
        no_leak_gap = metric("hp4_no_leak_policy", name) - metric("baseline_uniform", name)
        oracle_gap = oracle_metric("hp4_soft_agent", name) - oracle_metric("baseline_uniform", name)
        ratio = no_leak_gap / oracle_gap if abs(oracle_gap) > 1e-12 else 0.0
        lines.append(f"| {name} | {no_leak_gap:+.4f} | {oracle_gap:+.4f} | {ratio:.4f} |")
    lines.extend([
        "",
        "## Leakage Control",
        "",
        "- Policy weights are produced by `hp4_policy_reinforce.pt`.",
        "- Weight features exclude `doc.is_support`, gold supporting titles, and answer presence.",
        "- Support-derived client IDs are sanitized to deterministic non-gold buckets before policy inference.",
        "- Gold supporting titles and answers are used only after retrieval for metric computation.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preferred-dev", default="V7-HP4/data/hotpot_validation_1000.json")
    parser.add_argument("--generated-dev", default="V7-HP4/data/hotpot_validation_1000.json")
    parser.add_argument("--fallback-dev", default="FedE/select_data_hotpot_train_5000.json")
    parser.add_argument("--policy", default="V7-HP4/outputs/hp4_policy_gradient/hp4_policy_reinforce.pt")
    parser.add_argument("--oracle-summary", default="V7-HP4/outputs/hp4_full_validation/full_validation_reader_summary.json")
    parser.add_argument("--output-root", default="V7-HP4/outputs/hp4_no_leak_validation")
    parser.add_argument("--report-dir", default="实验分析报告/V7-HP4")
    parser.add_argument("--reader-model", default="google/flan-t5-large")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--reader-batch-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--max-dev", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--permutation-rounds", type=int, default=10000)
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    dev_path, data_source, split_n = load_or_build_validation_split(
        Path(args.preferred_dev),
        Path(args.generated_dev),
        args.max_dev,
        args.seed,
        Path(args.fallback_dev),
    )
    examples = materialize_dev(dev_path, args.max_dev)
    if len(examples) < min(args.max_dev, split_n) * 0.8:
        raise RuntimeError(f"Only materialized {len(examples)} examples from {dev_path}")

    policy, policy_device = load_policy(Path(args.policy), args.device)
    reader = Reader(args.reader_model, args.device, args.reader_batch_size)
    rows, summary = evaluate_no_leak(examples, reader, policy, policy_device, args.top_k, args.alpha)
    rows_path = out / "no_leak_reader_rows.json"
    summary_path = out / "no_leak_reader_summary.json"
    sig_path = out / "no_leak_significance.json"
    _save_json(rows_path, rows)
    _save_json(summary_path, summary)
    significance = {
        "joint_f1": permutation_p_value(paired_diffs(rows, "joint_f1"), args.permutation_rounds, args.seed + 11),
        "sp_f1": permutation_p_value(paired_diffs(rows, "sp_f1"), args.permutation_rounds, args.seed + 12),
    }
    _save_json(sig_path, significance)
    report_path = Path(args.report_dir) / "v7_hp4_no_leak_validation_latest.md"
    write_report(
        report_path,
        summary,
        significance,
        load_oracle_summary(Path(args.oracle_summary)),
        rows_path,
        data_source,
        Path(args.policy),
    )
    print(json.dumps({
        "examples": len(examples),
        "data_source": data_source,
        "rows_path": str(rows_path),
        "summary_path": str(summary_path),
        "significance_path": str(sig_path),
        "report_path": str(report_path),
        "summary": summary,
        "significance": significance,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
