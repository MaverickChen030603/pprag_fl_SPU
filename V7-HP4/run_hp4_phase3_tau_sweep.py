from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from src.v7_hp4.agent_continuous import ContinuousUploadPolicy
from src.v7_hp4.hybrid_retriever import HybridDocument, HybridSoftRetriever
from src.v7_hp4.policy_gradient import block_state_from_doc


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


def sanitize_docs(docs: list[HybridDocument]) -> list[HybridDocument]:
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
    runtime = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(path, map_location=runtime)
    input_dim = len(payload.get("feature_names", ContinuousUploadPolicy.feature_names))
    model = ContinuousUploadPolicy(
        input_dim=input_dim,
        hidden_dim=int(payload.get("hidden_dim", 48)),
        init_bias=-0.4,
    ).to(runtime)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, runtime


def weights_for_docs(
    question: str,
    docs: list[HybridDocument],
    model: ContinuousUploadPolicy,
    device: torch.device,
) -> dict[str, float]:
    states = [block_state_from_doc(question, doc, mode="natural") for doc in docs]
    x = ContinuousUploadPolicy.tensor_from_states(states, device=str(device))
    with torch.no_grad():
        weights = model(x).detach().cpu().tolist()
    return {doc.doc_id: float(max(0.0, min(1.0, w))) for doc, w in zip(docs, weights)}


def apply_hardgate(weights: dict[str, float], gate: int | None, floor: float) -> dict[str, float]:
    if gate is None:
        return dict(weights)
    keep = {doc_id for doc_id, _ in sorted(weights.items(), key=lambda item: item[1], reverse=True)[:gate]}
    return {doc_id: (weight if doc_id in keep else weight * floor) for doc_id, weight in weights.items()}


def evaluate_retrieval(
    examples: list[tuple[dict[str, Any], list[HybridDocument]]],
    model: ContinuousUploadPolicy,
    device: torch.device,
    policy_name: str,
    tau: float,
    gate: int | None,
    top_k: int,
    alpha: float,
    floor: float,
) -> list[dict[str, Any]]:
    rows = []
    mode = f"{policy_name}_tau{tau:g}_gate{gate or 'none'}"
    for case, raw_docs in examples:
        docs = sanitize_docs(raw_docs)
        question = str(case.get("question", ""))
        raw_weights = weights_for_docs(question, docs, model, device)
        weights = apply_hardgate(raw_weights, gate, floor)
        retriever = HybridSoftRetriever(
            docs,
            alpha=alpha,
            dense_weight_mode="temperature",
            weight_temperature=tau,
        )
        ranked = retriever.rank(question, weights=weights, top_k=top_k)
        top_docs = [doc for doc, _ in ranked]
        pred_titles = {doc.title for doc in top_docs}
        gold_titles = {str(t) for t in case.get("supporting_titles", [])}
        support_recall, sp_f1 = READER.sp_metrics(pred_titles, gold_titles)
        weight_values = list(weights.values())
        raw_values = list(raw_weights.values())
        rows.append({
            "id": str(case.get("id", case.get("_id", ""))),
            "mode": mode,
            "policy": policy_name,
            "tau": tau,
            "gate": gate or "none",
            "answer_access_at_k": READER.answer_in_context(case.get("answer", ""), top_docs),
            "support_recall_at_k": support_recall,
            "sp_f1": sp_f1,
            "top_doc_ids": [doc.doc_id for doc in top_docs],
            "top_titles": [doc.title for doc in top_docs],
            "raw_weight_mean": sum(raw_values) / max(len(raw_values), 1),
            "raw_weight_variance": torch.tensor(raw_values).var(unbiased=False).item() if raw_values else 0.0,
            "gated_weight_mean": sum(weight_values) / max(len(weight_values), 1),
            "gated_weight_variance": torch.tensor(weight_values).var(unbiased=False).item() if weight_values else 0.0,
        })
    return rows


def summarize(rows: list[dict[str, Any]], baseline_mode: str) -> list[dict[str, Any]]:
    baseline_rows = {row["id"]: row for row in rows if row["mode"] == baseline_mode}
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    out = []
    for mode, items in sorted(by_mode.items()):
        n = len(items)
        overlap = 0.0
        support_delta = 0.0
        sp_delta = 0.0
        for row in items:
            base = baseline_rows.get(row["id"])
            if base:
                overlap += len(set(row["top_doc_ids"]) & set(base["top_doc_ids"])) / max(len(row["top_doc_ids"]), 1)
                support_delta += float(row["support_recall_at_k"]) - float(base["support_recall_at_k"])
                sp_delta += float(row["sp_f1"]) - float(base["sp_f1"])
        denom = max(len([r for r in items if r["id"] in baseline_rows]), 1)
        out.append({
            "mode": mode,
            "n": n,
            "answer_access_at_k": sum(float(r["answer_access_at_k"]) for r in items) / n,
            "support_recall_at_k": sum(float(r["support_recall_at_k"]) for r in items) / n,
            "sp_f1": sum(float(r["sp_f1"]) for r in items) / n,
            "delta_support_recall_vs_baseline": support_delta / denom,
            "delta_sp_f1_vs_baseline": sp_delta / denom,
            "overlap_at_k_vs_baseline": overlap / denom,
            "raw_weight_variance": sum(float(r["raw_weight_variance"]) for r in items) / n,
            "gated_weight_variance": sum(float(r["gated_weight_variance"]) for r in items) / n,
        })
    out.sort(key=lambda item: (item["support_recall_at_k"], item["sp_f1"], item["answer_access_at_k"]), reverse=True)
    return out


def write_report(path: Path, summary: list[dict[str, Any]], baseline_mode: str) -> None:
    lines = [
        "# V7-HP4 Phase 3 Tau/Hardgate Retrieval Sweep",
        "",
        f"- baseline_mode: `{baseline_mode}`",
        "- strict_no_leak: policy input excludes support labels, gold titles, and answer presence",
        "- hardgate: keep top-N policy weights per query, multiply the remaining weights by 0.01",
        "",
        "| rank | mode | n | access@5 | support_recall@5 | sp_f1 | delta_recall | delta_sp_f1 | overlap@5 | gated_var |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(summary, start=1):
        lines.append(
            f"| {idx} | {row['mode']} | {row['n']} | {row['answer_access_at_k']:.4f} | "
            f"{row['support_recall_at_k']:.4f} | {row['sp_f1']:.4f} | "
            f"{row['delta_support_recall_vs_baseline']:+.4f} | {row['delta_sp_f1_vs_baseline']:+.4f} | "
            f"{row['overlap_at_k_vs_baseline']:.4f} | {row['gated_weight_variance']:.6f} |"
        )
    best_safe = [r for r in summary if r["delta_support_recall_vs_baseline"] >= -1e-9]
    lines.extend(["", "## Recommendation", ""])
    if best_safe:
        best = best_safe[0]
        lines.append(
            f"Use `{best['mode']}` for the next reader check because it preserves support_recall "
            f"({best['delta_support_recall_vs_baseline']:+.4f}) while ranking highest among non-harmful settings."
        )
    else:
        best = summary[0]
        lines.append(
            f"No tau/hardgate setting preserved support_recall. The least damaging top setting is `{best['mode']}`; "
            "do not launch 1000 reader validation until the gate is relaxed or retrained."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", default="V7-HP4/data/hotpot_validation_1000.json")
    parser.add_argument("--output-root", default="V7-HP4/outputs/hp4_phase3_tau_sweep")
    parser.add_argument("--report-dir", default="实验分析报告/V7-HP4")
    parser.add_argument("--policy-a", default="V7-HP4/outputs/hp4_phase3_100_batch/policy_a/hp4_policy_reinforce.pt")
    parser.add_argument("--policy-c", default="V7-HP4/outputs/hp4_phase3_100_batch/policy_c_coroute/hp4_policy_reinforce.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--gate-floor", type=float, default=0.01)
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    examples = READER.materialize_dev(Path(args.validation), args.sample_size)
    policy_a, device_a = load_policy(Path(args.policy_a), args.device)
    policy_c, device_c = load_policy(Path(args.policy_c), args.device)
    rows = []
    # The baseline is the Phase-2 policy under tau=1/no hardgate, matching the
    # non-destructive routing setting used to decide whether hardening hurts.
    configs = []
    for policy_name in ["A", "C"]:
        for tau in [0.5, 0.7, 1.0]:
            for gate in [None, 3, 4]:
                configs.append((policy_name, tau, gate))
    for policy_name, tau, gate in configs:
        model, device = (policy_a, device_a) if policy_name == "A" else (policy_c, device_c)
        rows.extend(evaluate_retrieval(
            examples,
            model,
            device,
            policy_name,
            tau,
            gate,
            args.top_k,
            args.alpha,
            args.gate_floor,
        ))
    baseline_mode = "A_tau1_gateNone".replace("None", "none")
    summary = summarize(rows, baseline_mode)
    save_json(out / "tau_sweep_rows.json", rows)
    save_json(out / "tau_sweep_summary.json", summary)
    report = Path(args.report_dir) / "v7_hp4_phase3_tau_sweep_latest.md"
    write_report(report, summary, baseline_mode)
    print(json.dumps({
        "sample_size": len(examples),
        "baseline_mode": baseline_mode,
        "summary": summary,
        "report_path": str(report),
        "output_root": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
