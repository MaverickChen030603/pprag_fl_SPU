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
from src.v7_hp4.policy_gradient import (
    block_state_from_doc,
    load_counterfactual_examples,
    load_dev_docs,
    load_micro_docs,
    train_policy,
)


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


READER = _load("V7-HP4/run_hp4_reader_counterfactual_eval.py", "hp4_reader")
FULLVAL = _load("V7-HP4/run_hp4_full_validation_eval.py", "hp4_fullval")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def log(message: str) -> None:
    print(f"[phase3] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


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
    input_dim = len(payload.get("feature_names", ContinuousUploadPolicy.feature_names))
    model = ContinuousUploadPolicy(input_dim=input_dim, hidden_dim=int(payload.get("hidden_dim", 48)), init_bias=-0.4).to(runtime)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, runtime


def weights_for_docs(question: str, docs: list[HybridDocument], model: ContinuousUploadPolicy, device: torch.device) -> dict[str, float]:
    states = [block_state_from_doc(question, doc, mode="natural") for doc in docs]
    x = ContinuousUploadPolicy.tensor_from_states(states, device=str(device))
    with torch.no_grad():
        weights = model(x).detach().cpu().tolist()
    return {doc.doc_id: float(max(0.0, min(1.0, w))) for doc, w in zip(docs, weights)}


def evaluate_variant(
    examples: list[tuple[dict[str, Any], list[HybridDocument]]],
    reader: Any,
    model: ContinuousUploadPolicy,
    device: torch.device,
    mode: str,
    top_k: int,
    alpha: float,
    dense_weight_mode: str,
    temperature: float,
) -> list[dict[str, Any]]:
    log(f"{mode}: materializing retrieval prompts for {len(examples)} examples")
    items = []
    for idx, (case, raw_docs) in enumerate(examples, start=1):
        docs = sanitize_docs(raw_docs)
        question = str(case.get("question", ""))
        weights = weights_for_docs(question, docs, model, device)
        retriever = HybridSoftRetriever(
            docs,
            alpha=alpha,
            dense_weight_mode=dense_weight_mode,
            weight_temperature=temperature,
        )
        ranked = retriever.rank(question, weights=weights, top_k=top_k)
        top_docs = [doc for doc, _ in ranked]
        gold = {str(t) for t in case.get("supporting_titles", [])}
        pred_titles = {doc.title for doc in top_docs}
        support_recall, sp_f1 = READER.sp_metrics(pred_titles, gold)
        items.append({
            "case": case,
            "mode": mode,
            "top_docs": top_docs,
            "top_doc_ids": [doc.doc_id for doc in top_docs],
            "top_titles": [doc.title for doc in top_docs],
            "weight_mean": sum(weights.values()) / max(len(weights), 1),
            "weight_variance": torch.tensor(list(weights.values())).var(unbiased=False).item() if weights else 0.0,
            "answer_access_at_k": READER.answer_in_context(case.get("answer", ""), top_docs),
            "support_recall_at_k": support_recall,
            "sp_f1": sp_f1,
            "prompt": READER.make_prompt(question, top_docs),
        })
        if idx % 25 == 0 or idx == len(examples):
            log(f"{mode}: prepared {idx}/{len(examples)} prompts")
    log(f"{mode}: reader generation start for {len(items)} prompts")
    preds = reader.generate([item["prompt"] for item in items])
    log(f"{mode}: reader generation complete")
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
            "weight_mean": item["weight_mean"],
            "weight_variance": item["weight_variance"],
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out = {}
    modes = sorted({r["mode"] for r in rows})
    for mode in modes:
        items = [r for r in rows if r["mode"] == mode]
        out[mode] = {
            "n": len(items),
            "answer_access_at_k": sum(float(r["answer_access_at_k"]) for r in items) / len(items),
            "support_recall_at_k": sum(float(r["support_recall_at_k"]) for r in items) / len(items),
            "sp_f1": sum(float(r["sp_f1"]) for r in items) / len(items),
            "answer_em": sum(float(r["answer_em"]) for r in items) / len(items),
            "answer_f1": sum(float(r["answer_f1"]) for r in items) / len(items),
            "joint_f1": sum(float(r["joint_f1"]) for r in items) / len(items),
            "weight_variance": sum(float(r["weight_variance"]) for r in items) / len(items),
        }
    return out


def write_report(path: Path, train_a: dict[str, Any], train_c: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# V7-HP4 Phase 3 100-Sample Initial Batch",
        "",
        "## Training Diagnostics",
        "",
        f"- A initial_weight_variance: {train_a.get('initial_weight_variance', 0.0):.6f}",
        f"- A final_weight_variance: {train_a.get('final_weight_variance', 0.0):.6f}",
        f"- C co_route_weight: {train_c.get('co_route_weight', 0.0):.4f}",
        f"- C initial_weight_variance: {train_c.get('initial_weight_variance', 0.0):.6f}",
        f"- C final_weight_variance: {train_c.get('final_weight_variance', 0.0):.6f}",
        f"- C final_co_route_loss: {train_c.get('final_co_route_loss', 0.0):.6f}",
        "",
        "## Reader Metrics",
        "",
        "| variant | n | access@k | support_recall@k | sp_f1 | answer_em | answer_f1 | joint_f1 | weight_var |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ["A_phase2_baseline_tau1", "B_temperature_tau03", "C_temperature_coroute_tau03"]:
        m = summary.get(mode, {})
        lines.append(
            f"| {mode} | {m.get('n', 0)} | {m.get('answer_access_at_k', 0):.4f} | "
            f"{m.get('support_recall_at_k', 0):.4f} | {m.get('sp_f1', 0):.4f} | "
            f"{m.get('answer_em', 0):.4f} | {m.get('answer_f1', 0):.4f} | "
            f"{m.get('joint_f1', 0):.4f} | {m.get('weight_variance', 0):.6f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-data", default="data/v7_hp4_micro_benchmark.json")
    parser.add_argument("--dev-data", default="V7-HP3/data/hotpot_dev_stratified_300.json")
    parser.add_argument("--micro-rows", default="V7-HP4/outputs/hp4_reader_counterfactual/micro_reader_rows.json")
    parser.add_argument("--dev-rows", default="V7-HP4/outputs/hp4_reader_counterfactual/dev300_reader_rows.json")
    parser.add_argument("--validation", default="V7-HP4/data/hotpot_validation_1000.json")
    parser.add_argument("--output-root", default="V7-HP4/outputs/hp4_phase3_100_batch")
    parser.add_argument("--report-dir", default="实验分析报告/V7-HP4")
    parser.add_argument("--reader-model", default="google/flan-t5-large")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reader-batch-size", type=int, default=2)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--co-route-weight", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.55)
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    log("loading training materials")
    micro_docs = load_micro_docs(Path(args.micro_data))
    dev_docs = load_dev_docs(Path(args.dev_data), max_examples=300)
    train_examples = []
    train_examples.extend(load_counterfactual_examples(Path(args.micro_rows), micro_docs, "micro"))
    train_examples.extend(load_counterfactual_examples(Path(args.dev_rows), dev_docs, "dev300"))

    policy_a_dir = out / "policy_a"
    policy_c_dir = out / "policy_c_coroute"
    if (policy_a_dir / "hp4_policy_reinforce.pt").exists() and (policy_a_dir / "training_summary.json").exists():
        log("reusing phase2 baseline policy checkpoint")
        train_a = json.loads((policy_a_dir / "training_summary.json").read_text(encoding="utf-8"))
    else:
        log("training phase2 baseline policy checkpoint")
        train_a = train_policy(train_examples, policy_a_dir, epochs=args.epochs, hidden_dim=48, lr=3e-3, co_route_weight=0.0)
    if (policy_c_dir / "hp4_policy_reinforce.pt").exists() and (policy_c_dir / "training_summary.json").exists():
        log("reusing co-route policy checkpoint")
        train_c = json.loads((policy_c_dir / "training_summary.json").read_text(encoding="utf-8"))
    else:
        log("training co-route policy checkpoint")
        train_c = train_policy(train_examples, policy_c_dir, epochs=args.epochs, hidden_dim=48, lr=3e-3, co_route_weight=args.co_route_weight)

    log(f"materializing validation subset sample_size={args.sample_size}")
    examples = READER.materialize_dev(Path(args.validation), args.sample_size)
    log(f"loading reader {args.reader_model} on {args.device}")
    reader = READER.Reader(args.reader_model, args.device, args.reader_batch_size)
    policy_a, dev_a = load_policy(Path(train_a["model_path"]), args.device)
    policy_c, dev_c = load_policy(Path(train_c["model_path"]), args.device)
    rows = []
    rows.extend(evaluate_variant(examples, reader, policy_a, dev_a, "A_phase2_baseline_tau1", args.top_k, args.alpha, "identity", 1.0))
    rows.extend(evaluate_variant(examples, reader, policy_a, dev_a, "B_temperature_tau03", args.top_k, args.alpha, "temperature", args.temperature))
    rows.extend(evaluate_variant(examples, reader, policy_c, dev_c, "C_temperature_coroute_tau03", args.top_k, args.alpha, "temperature", args.temperature))
    summary = summarize(rows)
    save_json(out / "phase3_100_rows.json", rows)
    save_json(out / "phase3_100_summary.json", summary)
    save_json(out / "phase3_train_a_summary.json", train_a)
    save_json(out / "phase3_train_c_summary.json", train_c)
    report = Path(args.report_dir) / "v7_hp4_phase3_100_batch_latest.md"
    write_report(report, train_a, train_c, summary)
    print(json.dumps({
        "sample_size": len(examples),
        "train_a": train_a,
        "train_c": train_c,
        "summary": summary,
        "report_path": str(report),
        "output_root": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
