from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.v7_hp4.hybrid_retriever import BM25Scorer, HybridDocument, HybridSoftRetriever, entity_tokens, tokenize


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_READER = _load_module("V7-HP4/run_hp4_reader_counterfactual_eval.py", "v7_hp4_reader_helpers")
_FULLVAL = _load_module("V7-HP4/run_hp4_full_validation_eval.py", "v7_hp4_full_validation_helpers")
_FULL = _load_module("V7-HP4/run_hp4_full_experiment.py", "v7_hp4_full_helpers")

Reader = _READER.Reader
answer_in_context = _READER.answer_in_context
f1_score = _READER.f1_score
make_prompt = _READER.make_prompt
materialize_dev = _READER.materialize_dev
normalize_answer = _READER.normalize_answer
sp_metrics = _READER.sp_metrics
load_or_build_validation_split = _FULLVAL.load_or_build_validation_split
permutation_p_value = _FULLVAL.permutation_p_value
normalize_hotpot_item = _FULLVAL._normalize_hotpot_item
build_dev300_case = _FULL.build_dev300_case


FEATURE_NAMES = (
    "lexical_jaccard",
    "entity_jaccard",
    "rare_jaccard",
    "title_jaccard",
    "bm25_norm",
    "dense_norm",
    "bm25_rr",
    "dense_rr",
    "doc_len_norm",
    "query_len_norm",
    "title_entity_hit",
    "bridge_potential",
    "client_bucket_score",
    "reader_context_fit",
)


class NoLeakPolicyV2(nn.Module):
    def __init__(self, input_dim: int = len(FEATURE_NAMES), hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _rare(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_'-]{6,}", text or "")}


def sanitize_docs_for_policy(docs: list[HybridDocument]) -> list[HybridDocument]:
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


def dense_proxy(question: str, doc: HybridDocument) -> float:
    q_tokens = set(tokenize(question))
    d_tokens = set(tokenize(doc.content))
    q_ents = set(entity_tokens(question))
    d_ents = set(entity_tokens(doc.content))
    return 0.70 * _jaccard(q_tokens, d_tokens) + 0.30 * _jaccard(q_ents, d_ents)


def feature_rows(question: str, docs: list[HybridDocument], top_k: int = 5) -> list[list[float]]:
    bm25 = BM25Scorer(docs)
    bm25_scores = [bm25.score(question, idx) for idx, _ in enumerate(docs)]
    dense_scores = [dense_proxy(question, doc) for doc in docs]
    max_bm25 = max(bm25_scores) if bm25_scores else 1.0
    max_dense = max(dense_scores) if dense_scores else 1.0
    bm25_order = {idx: rank + 1 for rank, idx in enumerate(sorted(range(len(docs)), key=lambda i: bm25_scores[i], reverse=True))}
    dense_order = {idx: rank + 1 for rank, idx in enumerate(sorted(range(len(docs)), key=lambda i: dense_scores[i], reverse=True))}
    q_tokens = set(tokenize(question))
    q_ents = set(entity_tokens(question))
    q_rare = _rare(question)
    q_len = len(q_tokens)
    rows = []
    for idx, doc in enumerate(docs):
        d_tokens = set(tokenize(doc.content))
        title_tokens = set(tokenize(doc.title))
        d_ents = set(entity_tokens(doc.content))
        title_ents = set(entity_tokens(doc.title))
        d_rare = _rare(doc.content)
        doc_len = len(tokenize(doc.content))
        bm25_norm = bm25_scores[idx] / max(max_bm25, 1e-9)
        dense_norm = dense_scores[idx] / max(max_dense, 1e-9)
        title_hit = float(bool(q_ents & title_ents))
        bridge_potential = min(1.0, 0.50 * _jaccard(q_ents, d_ents) + 0.30 * _jaccard(q_rare, d_rare) + 0.20 * title_hit)
        reader_context_fit = 1.0 / (1.0 + math.exp((doc_len - 95.0) / 45.0))
        rows.append([
            _jaccard(q_tokens, d_tokens),
            _jaccard(q_ents, d_ents),
            _jaccard(q_rare, d_rare),
            _jaccard(q_tokens, title_tokens),
            bm25_norm,
            dense_norm,
            1.0 / bm25_order[idx],
            1.0 / dense_order[idx],
            min(1.0, doc_len / 180.0),
            min(1.0, q_len / 28.0),
            title_hit,
            bridge_potential,
            (idx % 5) / 4.0,
            reader_context_fit,
        ])
    return rows


def train_target(case: dict[str, Any], docs: list[HybridDocument], doc: HybridDocument) -> float:
    # Training labels may use Hotpot supervision; evaluation features may not.
    if doc.is_support:
        return 1.0
    answer = normalize_answer(case.get("answer", ""))
    if answer and answer not in {"yes", "no"} and answer in normalize_answer(doc.content):
        return 0.45
    return 0.02


def load_training_examples(path: Path, max_train: int, seed: int) -> list[tuple[dict[str, Any], list[HybridDocument]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    normalized = []
    for idx, item in enumerate(raw):
        built = normalize_hotpot_item(item, idx)
        if built is not None:
            normalized.append(built)
    rng = random.Random(seed)
    rng.shuffle(normalized)
    examples = []
    for idx, item in enumerate(normalized[:max_train]):
        built = build_dev300_case(item, idx)
        if built is not None:
            case, docs = built
            examples.append((case, sanitize_docs_for_policy(docs)))
    return examples


def train_policy_v2(
    train_examples: list[tuple[dict[str, Any], list[HybridDocument]]],
    output_dir: Path,
    epochs: int,
    lr: float,
    hidden_dim: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    pair_indices: list[tuple[list[int], list[int]]] = []
    cursor = 0
    for case, docs in train_examples:
        feats = feature_rows(str(case.get("question", "")), docs)
        labels = [train_target(case, docs, doc) for doc in docs]
        pos = [cursor + i for i, label in enumerate(labels) if label >= 0.99]
        neg = [cursor + i for i, label in enumerate(labels) if label < 0.5]
        if pos and neg:
            pair_indices.append((pos, neg))
        x_rows.extend(feats)
        y_rows.extend(labels)
        cursor += len(feats)
    x = torch.tensor(x_rows, dtype=torch.float32, device=device)
    y = torch.tensor(y_rows, dtype=torch.float32, device=device)
    model = NoLeakPolicyV2(input_dim=len(FEATURE_NAMES), hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(x).clamp(1e-5, 1.0 - 1e-5)
        bce = nn.functional.binary_cross_entropy(pred, y)
        pair_loss = torch.tensor(0.0, device=device)
        pair_count = 0
        for pos, neg in pair_indices:
            pos_scores = pred[torch.tensor(pos, dtype=torch.long, device=device)]
            neg_scores = pred[torch.tensor(neg, dtype=torch.long, device=device)]
            pair_loss = pair_loss + nn.functional.relu(0.18 - pos_scores[:, None] + neg_scores[None, :]).mean()
            pair_count += 1
        if pair_count:
            pair_loss = pair_loss / pair_count
        entropy = -(pred * torch.log(pred) + (1.0 - pred) * torch.log(1.0 - pred)).mean()
        loss = bce + 0.65 * pair_loss - 0.005 * entropy
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
        opt.step()
        with torch.no_grad():
            pos_mask = y >= 0.99
            neg_mask = y < 0.5
            logs.append({
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "bce": float(bce.detach().cpu()),
                "pair_loss": float(pair_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "grad_norm": grad_norm,
                "pos_weight": float(pred[pos_mask].mean().detach().cpu()),
                "neg_weight": float(pred[neg_mask].mean().detach().cpu()),
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "hp4_no_leak_policy_v2.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "feature_names": FEATURE_NAMES,
        "hidden_dim": hidden_dim,
        "train_examples": len(train_examples),
    }, model_path)
    _save_json(output_dir / "hp4_no_leak_policy_v2_training_log.json", logs)
    summary = {
        "model_path": str(model_path),
        "train_examples": len(train_examples),
        "train_blocks": len(x_rows),
        "initial_loss": logs[0]["loss"],
        "final_loss": logs[-1]["loss"],
        "loss_delta": logs[-1]["loss"] - logs[0]["loss"],
        "final_pos_weight": logs[-1]["pos_weight"],
        "final_neg_weight": logs[-1]["neg_weight"],
        "pos_neg_gap": logs[-1]["pos_weight"] - logs[-1]["neg_weight"],
        "max_grad_norm": max(row["grad_norm"] for row in logs),
    }
    _save_json(output_dir / "hp4_no_leak_policy_v2_training_summary.json", summary)
    return summary


def load_policy_v2(path: Path, device_name: str) -> tuple[NoLeakPolicyV2, torch.device]:
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(path, map_location=device)
    model = NoLeakPolicyV2(input_dim=len(FEATURE_NAMES), hidden_dim=int(payload.get("hidden_dim", 64))).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, device


def weights_for_docs(question: str, docs: list[HybridDocument], model: NoLeakPolicyV2, device: torch.device, mode: str) -> dict[str, float]:
    if mode == "baseline_uniform":
        return {doc.doc_id: 1.0 for doc in docs}
    x = torch.tensor(feature_rows(question, docs), dtype=torch.float32, device=device)
    with torch.no_grad():
        weights = model(x).detach().cpu().tolist()
    # Mild sharpening helps Top-K routing while keeping all blocks nonzero.
    return {doc.doc_id: float(max(0.05, min(1.0, w))) for doc, w in zip(docs, weights)}


def build_item(case: dict[str, Any], docs: list[HybridDocument], model: NoLeakPolicyV2, device: torch.device, mode: str, top_k: int, alpha: float) -> dict[str, Any]:
    question = str(case.get("question", ""))
    weights = weights_for_docs(question, docs, model, device, mode)
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


def evaluate_v2(examples: list[tuple[dict[str, Any], list[HybridDocument]]], reader: Reader, model: NoLeakPolicyV2, device: torch.device, top_k: int, alpha: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retrieval_items = []
    for case, docs in examples:
        sanitized = sanitize_docs_for_policy(docs)
        retrieval_items.append(build_item(case, sanitized, model, device, "baseline_uniform", top_k, alpha))
        retrieval_items.append(build_item(case, sanitized, model, device, "hp4_no_leak_policy_v2", top_k, alpha))
    predictions = reader.generate([item["prompt"] for item in retrieval_items])
    rows = []
    for item, pred in zip(retrieval_items, predictions):
        case = item["case"]
        answer = case.get("answer", "")
        answer_f1 = f1_score(pred, answer)
        rows.append({
            "id": str(case.get("id", case.get("_id", ""))),
            "mode": item["mode"],
            "prediction": pred,
            "answer": answer,
            "answer_em": float(normalize_answer(pred) == normalize_answer(answer)),
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


def paired_diffs(rows: list[dict[str, Any]], metric: str) -> list[float]:
    by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_id[str(row["id"])][str(row["mode"])] = row
    diffs = []
    for modes in by_id.values():
        if "baseline_uniform" in modes and "hp4_no_leak_policy_v2" in modes:
            diffs.append(float(modes["hp4_no_leak_policy_v2"].get(metric, 0.0)) - float(modes["baseline_uniform"].get(metric, 0.0)))
    return diffs


def write_report(path: Path, train_summary: dict[str, Any], summary: dict[str, Any], significance: dict[str, Any], v1_summary: dict[str, Any], oracle_summary: dict[str, Any]) -> None:
    def gap(payload: dict[str, Any], agent_mode: str, metric: str) -> float:
        return float(payload.get(agent_mode, {}).get(metric, 0.0)) - float(payload.get("baseline_uniform", {}).get(metric, 0.0))

    lines = [
        "# V7-HP4 No-Leak V2 Validation",
        "",
        "## Training",
        "",
        f"- train_examples: {train_summary['train_examples']}",
        f"- train_blocks: {train_summary['train_blocks']}",
        f"- loss_delta: {train_summary['loss_delta']:.6f}",
        f"- pos_neg_gap: {train_summary['pos_neg_gap']:.4f}",
        f"- max_grad_norm: {train_summary['max_grad_norm']:.4f}",
        "",
        "## Reader Metrics",
        "",
        "| mode | n | access@k | support_recall@k | sp_f1 | answer_em | answer_f1 | joint_f1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ["baseline_uniform", "hp4_no_leak_policy_v2"]:
        m = summary.get(mode, {})
        lines.append(
            f"| {mode} | {m.get('n', 0)} | {m.get('answer_access_at_k', 0):.4f} | {m.get('support_recall_at_k', 0):.4f} | "
            f"{m.get('sp_f1', 0):.4f} | {m.get('answer_em', 0):.4f} | {m.get('answer_f1', 0):.4f} | {m.get('joint_f1', 0):.4f} |"
        )
    lines.extend([
        "",
        "## Significance",
        "",
        "| metric | n | mean_gap | p_value_two_sided | ci95 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for metric in ["joint_f1", "sp_f1"]:
        s = significance[metric]
        lines.append(f"| {metric} | {s['n']} | {s['mean_diff']:+.4f} | {s['p_value_two_sided']:.6f} | [{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}] |")
    lines.extend([
        "",
        "## Gap Comparison",
        "",
        "| metric | v1 no-leak gap | v2 no-leak gap | oracle gap |",
        "| --- | ---: | ---: | ---: |",
    ])
    for metric in ["support_recall_at_k", "sp_f1", "answer_f1", "joint_f1"]:
        lines.append(
            f"| {metric} | {gap(v1_summary, 'hp4_no_leak_policy', metric):+.4f} | "
            f"{gap(summary, 'hp4_no_leak_policy_v2', metric):+.4f} | {gap(oracle_summary, 'hp4_soft_agent', metric):+.4f} |"
        )
    lines.extend([
        "",
        "## Leakage Control",
        "",
        "- Evaluation policy features exclude support labels, gold titles, and answer presence.",
        "- Training uses Hotpot train supervision only; the 1000 validation split is not used for labels.",
        "- Support-derived client IDs are sanitized before both training feature extraction and evaluation inference.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default="FedE/select_data_hotpot_train_5000.json")
    parser.add_argument("--preferred-dev", default="V7-HP4/data/hotpot_validation_1000.json")
    parser.add_argument("--generated-dev", default="V7-HP4/data/hotpot_validation_1000.json")
    parser.add_argument("--fallback-dev", default="FedE/select_data_hotpot_train_5000.json")
    parser.add_argument("--output-root", default="V7-HP4/outputs/hp4_no_leak_v2_validation")
    parser.add_argument("--report-dir", default="实验分析报告/V7-HP4")
    parser.add_argument("--reader-model", default="google/flan-t5-large")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--reader-batch-size", type=int, default=2)
    parser.add_argument("--max-train", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--max-dev", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--permutation-rounds", type=int, default=10000)
    args = parser.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    train_examples = load_training_examples(Path(args.train_data), args.max_train, args.seed)
    train_summary = train_policy_v2(train_examples, out, args.epochs, args.lr, args.hidden_dim, args.seed)
    dev_path, data_source, split_n = load_or_build_validation_split(Path(args.preferred_dev), Path(args.generated_dev), args.max_dev, args.seed, Path(args.fallback_dev))
    examples = materialize_dev(dev_path, args.max_dev)
    if len(examples) < min(args.max_dev, split_n) * 0.8:
        raise RuntimeError(f"Only materialized {len(examples)} examples from {dev_path}")
    model, model_device = load_policy_v2(Path(train_summary["model_path"]), args.device)
    reader = Reader(args.reader_model, args.device, args.reader_batch_size)
    rows, summary = evaluate_v2(examples, reader, model, model_device, args.top_k, args.alpha)
    rows_path = out / "no_leak_v2_reader_rows.json"
    summary_path = out / "no_leak_v2_reader_summary.json"
    sig_path = out / "no_leak_v2_significance.json"
    _save_json(rows_path, rows)
    _save_json(summary_path, summary)
    significance = {
        "joint_f1": permutation_p_value(paired_diffs(rows, "joint_f1"), args.permutation_rounds, args.seed + 1),
        "sp_f1": permutation_p_value(paired_diffs(rows, "sp_f1"), args.permutation_rounds, args.seed + 2),
    }
    _save_json(sig_path, significance)
    v1_summary = json.loads(Path("V7-HP4/outputs/hp4_no_leak_validation/no_leak_reader_summary.json").read_text(encoding="utf-8"))
    oracle_summary = json.loads(Path("V7-HP4/outputs/hp4_full_validation/full_validation_reader_summary.json").read_text(encoding="utf-8"))
    report_path = Path(args.report_dir) / "v7_hp4_no_leak_v2_validation_latest.md"
    write_report(report_path, train_summary, summary, significance, v1_summary, oracle_summary)
    print(json.dumps({
        "examples": len(examples),
        "data_source": data_source,
        "train_summary": train_summary,
        "rows_path": str(rows_path),
        "summary_path": str(summary_path),
        "significance_path": str(sig_path),
        "report_path": str(report_path),
        "summary": summary,
        "significance": significance,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
