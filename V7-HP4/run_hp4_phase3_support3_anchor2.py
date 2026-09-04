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
from src.v7_hp4.hybrid_retriever import HybridDocument, HybridSoftRetriever, tokenize
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
    print(f"[support3-anchor2] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


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


def minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def novelty(doc: HybridDocument, selected: list[HybridDocument]) -> float:
    if not selected:
        return 1.0
    toks = set(tokenize(doc.content))
    return 1.0 - max(jaccard(toks, set(tokenize(other.content))) for other in selected)


def support3_anchor2_context(
    question: str,
    docs: list[HybridDocument],
    weights: dict[str, float],
    tau: float,
    gate_floor: float,
    alpha: float,
    anchor_pool: int,
    anchor_profile: str,
) -> tuple[list[HybridDocument], dict[str, Any]]:
    gated_weights = hardgate(weights, gate=3, floor=gate_floor)
    gated = HybridSoftRetriever(docs, alpha=alpha, dense_weight_mode="temperature", weight_temperature=tau)
    support_ranked = gated.rank(question, weights=gated_weights, top_k=5)
    support_docs = [doc for doc, _ in support_ranked[:3]]
    selected = {doc.doc_id for doc in support_docs}

    ungated = HybridSoftRetriever(docs, alpha=alpha, dense_weight_mode="temperature", weight_temperature=1.0)
    pool = ungated.rank(question, weights=weights, top_k=min(len(docs), anchor_pool))
    sparse = {doc.doc_id: float(score["sparse"]) for doc, score in pool}
    dense = {doc.doc_id: float(score["dense"]) for doc, score in pool}
    final = {doc.doc_id: float(score["final"]) for doc, score in pool}
    sparse_n, dense_n, final_n = minmax(sparse), minmax(dense), minmax(final)
    query_tokens = set(tokenize(question))

    if anchor_profile == "lexical":
        coeff = (0.55, 0.20, 0.15, 0.10)
    elif anchor_profile == "dense":
        coeff = (0.30, 0.40, 0.20, 0.10)
    else:
        coeff = (0.40, 0.30, 0.20, 0.10)

    candidates = []
    for doc, _score in pool:
        if doc.doc_id in selected:
            continue
        lexical_overlap = jaccard(query_tokens, set(tokenize(doc.content)))
        nov = novelty(doc, support_docs)
        anchor_score = (
            coeff[0] * sparse_n.get(doc.doc_id, 0.0)
            + coeff[1] * dense_n.get(doc.doc_id, 0.0)
            + coeff[2] * lexical_overlap
            + coeff[3] * nov
        )
        candidates.append((anchor_score, doc, lexical_overlap, nov))
    candidates.sort(key=lambda item: (item[0], item[2], item[1].doc_id), reverse=True)
    anchors = [doc for _, doc, _, _ in candidates[:2]]

    context = support_docs + anchors
    context_ids = {doc.doc_id for doc in context}
    # Reader order: strongest lexical anchor first, then interleave support evidence.
    order_scores = {}
    for idx, doc in enumerate(docs):
        if doc.doc_id not in context_ids:
            continue
        score = ungated.score(question, doc, idx, weights=weights)
        lex = jaccard(query_tokens, set(tokenize(doc.content)))
        is_anchor = 1.0 if doc in anchors else 0.0
        order_scores[doc.doc_id] = 0.45 * score["sparse"] + 0.25 * score["dense"] + 0.20 * lex + 0.10 * is_anchor
    ordered = sorted(context, key=lambda doc: (order_scores.get(doc.doc_id, 0.0), doc.doc_id), reverse=True)[:5]
    return ordered, {
        "support_doc_ids": [doc.doc_id for doc in support_docs],
        "anchor_doc_ids": [doc.doc_id for doc in anchors],
        "top_doc_ids": [doc.doc_id for doc in ordered],
        "top_titles": [doc.title for doc in ordered],
    }


def evaluate(
    examples: list[tuple[dict[str, Any], list[HybridDocument]]],
    model: ContinuousUploadPolicy,
    device: torch.device,
    reader: Any,
    mode: str,
    tau: float,
    gate_floor: float,
    alpha: float,
    anchor_pool: int,
    anchor_profile: str,
) -> list[dict[str, Any]]:
    items = []
    for idx, (case, raw_docs) in enumerate(examples, start=1):
        docs = sanitize_docs(raw_docs)
        question = str(case.get("question", ""))
        weights = weights_for_docs(question, docs, model, device)
        top_docs, audit = support3_anchor2_context(question, docs, weights, tau, gate_floor, alpha, anchor_pool, anchor_profile)
        gold = {str(t) for t in case.get("supporting_titles", [])}
        pred_titles = {doc.title for doc in top_docs}
        support_recall, sp_f1 = READER.sp_metrics(pred_titles, gold)
        items.append({
            "case": case,
            "mode": mode,
            "top_docs": top_docs,
            "answer_access_at_k": READER.answer_in_context(case.get("answer", ""), top_docs),
            "support_recall_at_k": support_recall,
            "sp_f1": sp_f1,
            "prompt": READER.make_prompt(question, top_docs),
            **audit,
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
            "support_doc_ids": item["support_doc_ids"],
            "anchor_doc_ids": item["anchor_doc_ids"],
            "top_doc_ids": item["top_doc_ids"],
            "top_titles": item["top_titles"],
        })
    return rows


def summarize(rows: list[dict[str, Any]], baseline: dict[str, float]) -> list[dict[str, Any]]:
    out = []
    for mode in sorted({row["mode"] for row in rows}):
        items = [row for row in rows if row["mode"] == mode]
        metrics = {
            "mode": mode,
            "n": len(items),
            "answer_access_at_k": sum(float(r["answer_access_at_k"]) for r in items) / len(items),
            "support_recall_at_k": sum(float(r["support_recall_at_k"]) for r in items) / len(items),
            "sp_f1": sum(float(r["sp_f1"]) for r in items) / len(items),
            "answer_em": sum(float(r["answer_em"]) for r in items) / len(items),
            "answer_f1": sum(float(r["answer_f1"]) for r in items) / len(items),
            "joint_f1": sum(float(r["joint_f1"]) for r in items) / len(items),
        }
        metrics.update({
            "delta_support_recall_at_k": metrics["support_recall_at_k"] - baseline["support_recall_at_k"],
            "delta_sp_f1": metrics["sp_f1"] - baseline["sp_f1"],
            "delta_answer_f1": metrics["answer_f1"] - baseline["answer_f1"],
            "delta_joint_f1": metrics["joint_f1"] - baseline["joint_f1"],
            "reader_pass": metrics["answer_f1"] + 1e-12 >= baseline["answer_f1"] and metrics["joint_f1"] + 1e-12 >= baseline["joint_f1"],
        })
        out.append(metrics)
    out.sort(key=lambda row: (row["reader_pass"], row["delta_answer_f1"], row["delta_joint_f1"], row["delta_support_recall_at_k"]), reverse=True)
    return out


def write_report(path: Path, baseline: dict[str, float], ranked: list[dict[str, Any]]) -> None:
    lines = [
        "# V7-HP4 Phase 3 Support3 + Anchor2 Check",
        "",
        "- context: top3 policy-routed support-like blocks + top2 no-leak query lexical anchors",
        "- no answer string, support label, or gold title is used for anchor selection",
        "",
        f"- baseline answer_f1={baseline['answer_f1']:.4f}, joint_f1={baseline['joint_f1']:.4f}, support_recall@5={baseline['support_recall_at_k']:.4f}",
        "",
        "| rank | mode | pass | access@5 | recall@5 | sp_f1 | answer_f1 | joint_f1 | d_answer | d_joint | d_recall |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(ranked, start=1):
        lines.append(
            f"| {idx} | {row['mode']} | {str(row['reader_pass']).lower()} | "
            f"{row['answer_access_at_k']:.4f} | {row['support_recall_at_k']:.4f} | {row['sp_f1']:.4f} | "
            f"{row['answer_f1']:.4f} | {row['joint_f1']:.4f} | "
            f"{row['delta_answer_f1']:+.4f} | {row['delta_joint_f1']:+.4f} | {row['delta_support_recall_at_k']:+.4f} |"
        )
    lines.extend(["", "## Decision", ""])
    passing = [row for row in ranked if row["reader_pass"]]
    if passing:
        lines.append(f"Launch 1000 validation with `{passing[0]['mode']}`.")
    else:
        lines.append(f"No config recovered answer_f1 to baseline. Best near-miss: `{ranked[0]['mode']}`.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", default="V7-HP4/data/hotpot_validation_1000.json")
    parser.add_argument("--output-root", default="V7-HP4/outputs/hp4_phase3_support3_anchor2")
    parser.add_argument("--report-dir", default="实验分析报告/V7-HP4")
    parser.add_argument("--policy-a", default="V7-HP4/outputs/hp4_phase3_100_batch/policy_a/hp4_policy_reinforce.pt")
    parser.add_argument("--prior-summary", default="V7-HP4/outputs/hp4_phase3_100_batch/phase3_100_summary.json")
    parser.add_argument("--reader-model", default="google/flan-t5-large")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reader-batch-size", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--gate-floor", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.55)
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(Path(args.prior_summary).read_text(encoding="utf-8"))["A_phase2_baseline_tau1"]
    log(f"materializing {args.sample_size} validation cases")
    examples = READER.materialize_dev(Path(args.validation), args.sample_size)
    policy, policy_device = load_policy(Path(args.policy_a), args.device)
    log(f"loading reader {args.reader_model} on {args.device}")
    reader = READER.Reader(args.reader_model, args.device, args.reader_batch_size)
    all_rows = []
    for anchor_pool in [8, 16, 24]:
        for profile in ["balanced", "lexical", "dense"]:
            mode = f"A_tau{args.tau:g}_support3_anchor2_pool{anchor_pool}_{profile}"
            rows = evaluate(examples, policy, policy_device, reader, mode, args.tau, args.gate_floor, args.alpha, anchor_pool, profile)
            all_rows.extend(rows)
            save_json(out / "support3_anchor2_partial_summary.json", summarize(all_rows, baseline))
    ranked = summarize(all_rows, baseline)
    payload = {
        "sample_size": len(examples),
        "baseline": baseline,
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
        "has_reader_pass": any(row["reader_pass"] for row in ranked),
    }
    save_json(out / "support3_anchor2_rows.json", all_rows)
    save_json(out / "support3_anchor2_summary.json", payload)
    report = Path(args.report_dir) / "v7_hp4_phase3_support3_anchor2_latest.md"
    write_report(report, baseline, ranked)
    print(json.dumps({**payload, "report_path": str(report), "output_root": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
