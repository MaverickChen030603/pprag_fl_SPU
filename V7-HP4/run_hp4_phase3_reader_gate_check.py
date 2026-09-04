from __future__ import annotations

import argparse
import importlib.util
import json
import time
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


def log(message: str) -> None:
    print(f"[gate-check] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sanitize_docs(docs: list[HybridDocument]) -> list[HybridDocument]:
    out = []
    for idx, doc in enumerate(docs):
        out.append(replace(
            doc,
            client_id=f"client_{idx % 5}",
            support_role="unknown",
            bridge_entities=[],
            rare_tokens=[],
            dense_score_hint=1.0,
            soft_weight=1.0,
        ))
    return out


def load_policy(path: Path, device: str) -> tuple[ContinuousUploadPolicy, torch.device]:
    runtime = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(path, map_location=runtime)
    model = ContinuousUploadPolicy(
        input_dim=len(payload.get("feature_names", ContinuousUploadPolicy.feature_names)),
        hidden_dim=int(payload.get("hidden_dim", 48)),
        init_bias=-0.4,
    ).to(runtime)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, runtime


def weights_for_docs(question: str, docs: list[HybridDocument], model: ContinuousUploadPolicy, device: torch.device) -> dict[str, float]:
    states = [block_state_from_doc(question, doc, mode="natural") for doc in docs]
    x = ContinuousUploadPolicy.tensor_from_states(states, device=str(device))
    with torch.no_grad():
        weights = model(x).detach().cpu().tolist()
    return {doc.doc_id: float(max(0.0, min(1.0, w))) for doc, w in zip(docs, weights)}


def hardgate(weights: dict[str, float], gate: int, floor: float) -> dict[str, float]:
    keep = {doc_id for doc_id, _ in sorted(weights.items(), key=lambda item: item[1], reverse=True)[:gate]}
    return {doc_id: (weight if doc_id in keep else weight * floor) for doc_id, weight in weights.items()}


def build_items(
    examples: list[tuple[dict[str, Any], list[HybridDocument]]],
    model: ContinuousUploadPolicy,
    device: torch.device,
    mode: str,
    tau: float,
    gate: int | None,
    gate_floor: float,
    top_k: int,
    alpha: float,
) -> list[dict[str, Any]]:
    items = []
    for idx, (case, raw_docs) in enumerate(examples, start=1):
        docs = sanitize_docs(raw_docs)
        question = str(case.get("question", ""))
        weights = weights_for_docs(question, docs, model, device)
        if gate is not None:
            weights = hardgate(weights, gate, gate_floor)
        retriever = HybridSoftRetriever(docs, alpha=alpha, dense_weight_mode="temperature", weight_temperature=tau)
        ranked = retriever.rank(question, weights=weights, top_k=top_k)
        top_docs = [doc for doc, _ in ranked]
        pred_titles = {doc.title for doc in top_docs}
        gold_titles = {str(t) for t in case.get("supporting_titles", [])}
        support_recall, sp_f1 = READER.sp_metrics(pred_titles, gold_titles)
        items.append({
            "case": case,
            "mode": mode,
            "top_docs": top_docs,
            "top_doc_ids": [doc.doc_id for doc in top_docs],
            "top_titles": [doc.title for doc in top_docs],
            "answer_access_at_k": READER.answer_in_context(case.get("answer", ""), top_docs),
            "support_recall_at_k": support_recall,
            "sp_f1": sp_f1,
            "prompt": READER.make_prompt(question, top_docs),
        })
        if idx % 25 == 0 or idx == len(examples):
            log(f"{mode}: prepared {idx}/{len(examples)} prompts")
    return items


def score_items(items: list[dict[str, Any]], reader: Any) -> list[dict[str, Any]]:
    log(f"reader generation start for {len(items)} prompts")
    preds = reader.generate([item["prompt"] for item in items])
    log("reader generation complete")
    rows = []
    for item, pred in zip(items, preds):
        case = item["case"]
        answer = case.get("answer", "")
        answer_f1 = READER.f1_score(pred, answer)
        rows.append({
            "id": str(case.get("id", case.get("_id", ""))),
            "mode": item["mode"],
            "prediction": pred,
            "answer": answer,
            "answer_em": float(READER.normalize_answer(pred) == READER.normalize_answer(answer)),
            "answer_f1": answer_f1,
            "answer_access_at_k": item["answer_access_at_k"],
            "support_recall_at_k": item["support_recall_at_k"],
            "sp_f1": item["sp_f1"],
            "joint_f1": answer_f1 * float(item["sp_f1"]),
            "top_doc_ids": item["top_doc_ids"],
            "top_titles": item["top_titles"],
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out = {}
    for mode in sorted({row["mode"] for row in rows}):
        items = [row for row in rows if row["mode"] == mode]
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


def write_report(path: Path, summary: dict[str, Any], baseline: dict[str, float], candidate: str) -> None:
    cand = summary[candidate]
    lines = [
        "# V7-HP4 Phase 3 Reader Gate Check",
        "",
        f"- candidate: `{candidate}`",
        "- strict_no_leak: policy input excludes support labels, gold titles, and answer presence",
        "",
        "| mode | n | access@5 | support_recall@5 | sp_f1 | answer_em | answer_f1 | joint_f1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| prior_A_tau1_gatenone | {int(baseline.get('n', 0))} | {baseline.get('answer_access_at_k', 0):.4f} | "
            f"{baseline.get('support_recall_at_k', 0):.4f} | {baseline.get('sp_f1', 0):.4f} | "
            f"{baseline.get('answer_em', 0):.4f} | {baseline.get('answer_f1', 0):.4f} | "
            f"{baseline.get('joint_f1', 0):.4f} |"
        ),
        (
            f"| {candidate} | {int(cand.get('n', 0))} | {cand.get('answer_access_at_k', 0):.4f} | "
            f"{cand.get('support_recall_at_k', 0):.4f} | {cand.get('sp_f1', 0):.4f} | "
            f"{cand.get('answer_em', 0):.4f} | {cand.get('answer_f1', 0):.4f} | "
            f"{cand.get('joint_f1', 0):.4f} |"
        ),
        "",
        "## Deltas",
        "",
        f"- answer_f1_delta: {cand.get('answer_f1', 0) - baseline.get('answer_f1', 0):+.4f}",
        f"- joint_f1_delta: {cand.get('joint_f1', 0) - baseline.get('joint_f1', 0):+.4f}",
        f"- support_recall_delta: {cand.get('support_recall_at_k', 0) - baseline.get('support_recall_at_k', 0):+.4f}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", default="V7-HP4/data/hotpot_validation_1000.json")
    parser.add_argument("--output-root", default="V7-HP4/outputs/hp4_phase3_reader_gate_check")
    parser.add_argument("--report-dir", default="实验分析报告/V7-HP4")
    parser.add_argument("--policy-a", default="V7-HP4/outputs/hp4_phase3_100_batch/policy_a/hp4_policy_reinforce.pt")
    parser.add_argument("--prior-summary", default="V7-HP4/outputs/hp4_phase3_100_batch/phase3_100_summary.json")
    parser.add_argument("--reader-model", default="google/flan-t5-large")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reader-batch-size", type=int, default=2)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--gate", type=int, default=4)
    parser.add_argument("--gate-floor", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.55)
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    mode = f"A_tau{args.tau:g}_gate{args.gate}"
    prior = json.loads(Path(args.prior_summary).read_text(encoding="utf-8"))
    baseline = prior["A_phase2_baseline_tau1"]
    log(f"materializing {args.sample_size} validation cases")
    examples = READER.materialize_dev(Path(args.validation), args.sample_size)
    policy, policy_device = load_policy(Path(args.policy_a), args.device)
    items = build_items(examples, policy, policy_device, mode, args.tau, args.gate, args.gate_floor, args.top_k, args.alpha)
    log(f"loading reader {args.reader_model} on {args.device}")
    reader = READER.Reader(args.reader_model, args.device, args.reader_batch_size)
    rows = score_items(items, reader)
    summary = summarize(rows)
    payload = {
        "sample_size": len(examples),
        "candidate": mode,
        "baseline": baseline,
        "summary": summary,
        "deltas": {
            "answer_f1": summary[mode]["answer_f1"] - baseline["answer_f1"],
            "joint_f1": summary[mode]["joint_f1"] - baseline["joint_f1"],
            "support_recall_at_k": summary[mode]["support_recall_at_k"] - baseline["support_recall_at_k"],
            "sp_f1": summary[mode]["sp_f1"] - baseline["sp_f1"],
        },
        "reader_pass": (
            summary[mode]["answer_f1"] + 1e-12 >= baseline["answer_f1"]
            and summary[mode]["joint_f1"] + 1e-12 >= baseline["joint_f1"]
        ),
    }
    save_json(out / "reader_gate_rows.json", rows)
    save_json(out / "reader_gate_summary.json", payload)
    report = Path(args.report_dir) / "v7_hp4_phase3_reader_gate_check_latest.md"
    write_report(report, summary, baseline, mode)
    print(json.dumps({**payload, "report_path": str(report), "output_root": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
