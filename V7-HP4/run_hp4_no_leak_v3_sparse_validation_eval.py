from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V2 = _load("V7-HP4/run_hp4_no_leak_v2_validation_eval.py", "hp4_v2")
FULLVAL = _load("V7-HP4/run_hp4_full_validation_eval.py", "hp4_fullval")
READER = _load("V7-HP4/run_hp4_reader_counterfactual_eval.py", "hp4_reader")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sparse_weights(raw: torch.Tensor, budget: float = 3.0, temperature: float = 0.45) -> torch.Tensor:
    if raw.numel() == 0:
        return raw
    logits = torch.logit(raw.clamp(1e-5, 1.0 - 1e-5)) / temperature
    mass = torch.softmax(logits, dim=0) * min(float(budget), float(raw.numel()))
    return mass.clamp(0.02, 1.0)


def hard_negative_mask(case: dict[str, Any], docs: list[Any], top_n: int = 5) -> list[bool]:
    question = str(case.get("question", ""))
    bm25 = V2.BM25Scorer(docs)
    order = sorted(range(len(docs)), key=lambda i: bm25.score(question, i), reverse=True)
    hard = set(order[:top_n])
    return [(idx in hard and not bool(doc.is_support)) for idx, doc in enumerate(docs)]


def train_policy_v3(train_examples: list[tuple[dict[str, Any], list[Any]]], out: Path, epochs: int, lr: float, hidden_dim: int, seed: int, budget: float) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V2.NoLeakPolicyV2(input_dim=len(V2.FEATURE_NAMES), hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    packed = []
    for case, docs in train_examples:
        x = torch.tensor(V2.feature_rows(str(case.get("question", "")), docs), dtype=torch.float32, device=device)
        y = torch.tensor([V2.train_target(case, docs, doc) for doc in docs], dtype=torch.float32, device=device)
        hard = torch.tensor(hard_negative_mask(case, docs), dtype=torch.bool, device=device)
        packed.append((x, y, hard))

    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        total = torch.tensor(0.0, device=device)
        bce_total = torch.tensor(0.0, device=device)
        pair_total = torch.tensor(0.0, device=device)
        budget_total = torch.tensor(0.0, device=device)
        entropy_total = torch.tensor(0.0, device=device)
        pos_ws, neg_ws, hard_ws = [], [], []
        for x, y, hard in packed:
            raw = model(x).clamp(1e-5, 1.0 - 1e-5)
            w = sparse_weights(raw, budget=budget)
            sample_weight = torch.ones_like(y)
            sample_weight = torch.where(hard, torch.full_like(sample_weight, 4.0), sample_weight)
            bce = nn.functional.binary_cross_entropy(w, y.clamp(0.0, 1.0), weight=sample_weight)
            pos = y >= 0.99
            neg = y < 0.5
            pair = torch.tensor(0.0, device=device)
            if pos.any() and neg.any():
                margin = torch.where(hard[neg], torch.full_like(w[neg], 0.42), torch.full_like(w[neg], 0.22))
                pair = nn.functional.relu(margin[None, :] - w[pos][:, None] + w[neg][None, :]).mean()
            budget_loss = ((w.sum() - min(float(budget), float(w.numel()))) / max(float(w.numel()), 1.0)).pow(2)
            entropy = -(w * torch.log(w.clamp_min(1e-5)) + (1.0 - w).clamp_min(1e-5) * torch.log((1.0 - w).clamp_min(1e-5))).mean()
            loss = bce + 0.9 * pair + 0.8 * budget_loss + 0.03 * entropy
            total = total + loss
            bce_total = bce_total + bce.detach()
            pair_total = pair_total + pair.detach()
            budget_total = budget_total + budget_loss.detach()
            entropy_total = entropy_total + entropy.detach()
            if pos.any():
                pos_ws.append(w[pos].mean().detach())
            if neg.any():
                neg_ws.append(w[neg].mean().detach())
            if hard.any():
                hard_ws.append(w[hard].mean().detach())
        loss = total / max(len(packed), 1)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
        opt.step()
        logs.append({
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "bce": float((bce_total / len(packed)).cpu()),
            "pair": float((pair_total / len(packed)).cpu()),
            "budget": float((budget_total / len(packed)).cpu()),
            "entropy": float((entropy_total / len(packed)).cpu()),
            "grad_norm": grad_norm,
            "pos_weight": float(torch.stack(pos_ws).mean().cpu()) if pos_ws else 0.0,
            "neg_weight": float(torch.stack(neg_ws).mean().cpu()) if neg_ws else 0.0,
            "hard_neg_weight": float(torch.stack(hard_ws).mean().cpu()) if hard_ws else 0.0,
        })
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "hp4_no_leak_policy_v3_sparse.pt"
    torch.save({"state_dict": model.state_dict(), "feature_names": V2.FEATURE_NAMES, "hidden_dim": hidden_dim, "budget": budget}, model_path)
    save_json(out / "hp4_no_leak_policy_v3_training_log.json", logs)
    summary = {
        "model_path": str(model_path),
        "train_examples": len(train_examples),
        "initial_loss": logs[0]["loss"],
        "final_loss": logs[-1]["loss"],
        "loss_delta": logs[-1]["loss"] - logs[0]["loss"],
        "final_pos_weight": logs[-1]["pos_weight"],
        "final_neg_weight": logs[-1]["neg_weight"],
        "final_hard_neg_weight": logs[-1]["hard_neg_weight"],
        "pos_neg_gap": logs[-1]["pos_weight"] - logs[-1]["neg_weight"],
        "pos_hard_neg_gap": logs[-1]["pos_weight"] - logs[-1]["hard_neg_weight"],
        "max_grad_norm": max(r["grad_norm"] for r in logs),
    }
    save_json(out / "hp4_no_leak_policy_v3_training_summary.json", summary)
    return summary


def weights_for_docs(question: str, docs: list[Any], model: Any, device: torch.device, mode: str, budget: float) -> dict[str, float]:
    if mode == "baseline_uniform":
        return {doc.doc_id: 1.0 for doc in docs}
    x = torch.tensor(V2.feature_rows(question, docs), dtype=torch.float32, device=device)
    with torch.no_grad():
        raw = model(x).clamp(1e-5, 1.0 - 1e-5)
        w = sparse_weights(raw, budget=budget).detach().cpu().tolist()
    return {doc.doc_id: float(v) for doc, v in zip(docs, w)}


def evaluate(examples: list[tuple[dict[str, Any], list[Any]]], reader: Any, model: Any, device: torch.device, top_k: int, alpha: float, budget: float):
    retrieval_items = []
    for case, docs in examples:
        docs = V2.sanitize_docs_for_policy(docs)
        for mode in ["baseline_uniform", "hp4_no_leak_policy_v3_sparse"]:
            q = str(case.get("question", ""))
            weights = weights_for_docs(q, docs, model, device, mode, budget)
            ranked = V2.HybridSoftRetriever(docs, alpha=alpha).rank(q, weights=weights, top_k=top_k)
            top_docs = [doc for doc, _ in ranked]
            gold = {str(t) for t in case.get("supporting_titles", [])}
            pred_titles = {doc.title for doc in top_docs}
            support_recall, sp_f1 = V2.sp_metrics(pred_titles, gold)
            retrieval_items.append({
                "case": case,
                "mode": mode,
                "top_docs": top_docs,
                "top_doc_ids": [doc.doc_id for doc in top_docs],
                "top_titles": [doc.title for doc in top_docs],
                "weights": weights,
                "answer_access_at_k": V2.answer_in_context(case.get("answer", ""), top_docs),
                "support_recall_at_k": support_recall,
                "sp_f1": sp_f1,
                "prompt": V2.make_prompt(q, top_docs),
            })
    preds = reader.generate([x["prompt"] for x in retrieval_items])
    rows = []
    for item, pred in zip(retrieval_items, preds):
        case = item["case"]
        ans = case.get("answer", "")
        af1 = V2.f1_score(pred, ans)
        rows.append({
            "id": str(case.get("id", case.get("_id", ""))),
            "mode": item["mode"],
            "prediction": pred,
            "answer": ans,
            "answer_em": float(V2.normalize_answer(pred) == V2.normalize_answer(ans)),
            "answer_f1": af1,
            "answer_access_at_k": item["answer_access_at_k"],
            "support_recall_at_k": item["support_recall_at_k"],
            "sp_f1": item["sp_f1"],
            "joint_f1": af1 * float(item["sp_f1"]),
            "top_doc_ids": item["top_doc_ids"],
            "top_titles": item["top_titles"],
            "weights": item["weights"],
        })
    return rows, V2.summarize(rows)


def paired(rows: list[dict[str, Any]], metric: str) -> list[float]:
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by_id.setdefault(str(r["id"]), {})[str(r["mode"])] = r
    out = []
    for modes in by_id.values():
        if "baseline_uniform" in modes and "hp4_no_leak_policy_v3_sparse" in modes:
            out.append(float(modes["hp4_no_leak_policy_v3_sparse"].get(metric, 0.0)) - float(modes["baseline_uniform"].get(metric, 0.0)))
    return out


def report(path: Path, train: dict[str, Any], summary: dict[str, Any], sig: dict[str, Any], v1: dict[str, Any], v2: dict[str, Any], oracle: dict[str, Any]) -> None:
    def g(payload, mode, metric):
        return float(payload.get(mode, {}).get(metric, 0.0)) - float(payload.get("baseline_uniform", {}).get(metric, 0.0))
    lines = ["# V7-HP4 No-Leak V3 Sparse Policy", "", "## Training", ""]
    for k in ["train_examples", "loss_delta", "final_pos_weight", "final_neg_weight", "final_hard_neg_weight", "pos_neg_gap", "pos_hard_neg_gap", "max_grad_norm"]:
        lines.append(f"- {k}: {train.get(k)}")
    lines += ["", "## Reader Metrics", "", "| mode | n | access@k | support_recall@k | sp_f1 | answer_em | answer_f1 | joint_f1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode in ["baseline_uniform", "hp4_no_leak_policy_v3_sparse"]:
        m = summary.get(mode, {})
        lines.append(f"| {mode} | {m.get('n',0)} | {m.get('answer_access_at_k',0):.4f} | {m.get('support_recall_at_k',0):.4f} | {m.get('sp_f1',0):.4f} | {m.get('answer_em',0):.4f} | {m.get('answer_f1',0):.4f} | {m.get('joint_f1',0):.4f} |")
    lines += ["", "## Significance", "", "| metric | n | mean_gap | p_value_two_sided | ci95 |", "| --- | ---: | ---: | ---: | ---: |"]
    for metric in ["joint_f1", "sp_f1"]:
        s = sig[metric]
        lines.append(f"| {metric} | {s['n']} | {s['mean_diff']:+.4f} | {s['p_value_two_sided']:.6f} | [{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}] |")
    lines += ["", "## Gap Comparison", "", "| metric | v1 | v2 | v3 sparse | oracle |", "| --- | ---: | ---: | ---: | ---: |"]
    for metric in ["support_recall_at_k", "sp_f1", "answer_f1", "joint_f1"]:
        lines.append(f"| {metric} | {g(v1,'hp4_no_leak_policy',metric):+.4f} | {g(v2,'hp4_no_leak_policy_v2',metric):+.4f} | {g(summary,'hp4_no_leak_policy_v3_sparse',metric):+.4f} | {g(oracle,'hp4_soft_agent',metric):+.4f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", default="FedE/select_data_hotpot_train_5000.json")
    p.add_argument("--preferred-dev", default="V7-HP4/data/hotpot_validation_1000.json")
    p.add_argument("--generated-dev", default="V7-HP4/data/hotpot_validation_1000.json")
    p.add_argument("--fallback-dev", default="FedE/select_data_hotpot_train_5000.json")
    p.add_argument("--output-root", default="V7-HP4/outputs/hp4_no_leak_v3_sparse_validation")
    p.add_argument("--report-dir", default="实验分析报告/V7-HP4")
    p.add_argument("--reader-model", default="google/flan-t5-large")
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--reader-batch-size", type=int, default=2)
    p.add_argument("--max-train", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--budget", type=float, default=3.0)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--max-dev", type=int, default=1000)
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--permutation-rounds", type=int, default=10000)
    args = p.parse_args()
    out = Path(args.output_root)
    train_examples = V2.load_training_examples(Path(args.train_data), args.max_train, args.seed)
    train = train_policy_v3(train_examples, out, args.epochs, args.lr, args.hidden_dim, args.seed, args.budget)
    dev_path, data_source, split_n = FULLVAL.load_or_build_validation_split(Path(args.preferred_dev), Path(args.generated_dev), args.max_dev, args.seed, Path(args.fallback_dev))
    examples = V2.materialize_dev(dev_path, args.max_dev)
    model, device = V2.load_policy_v2(Path(train["model_path"]), args.device)
    reader = V2.Reader(args.reader_model, args.device, args.reader_batch_size)
    rows, summary = evaluate(examples, reader, model, device, args.top_k, args.alpha, args.budget)
    save_json(out / "no_leak_v3_reader_rows.json", rows)
    save_json(out / "no_leak_v3_reader_summary.json", summary)
    sig = {
        "joint_f1": FULLVAL.permutation_p_value(paired(rows, "joint_f1"), args.permutation_rounds, args.seed + 1),
        "sp_f1": FULLVAL.permutation_p_value(paired(rows, "sp_f1"), args.permutation_rounds, args.seed + 2),
    }
    save_json(out / "no_leak_v3_significance.json", sig)
    v1 = json.loads(Path("V7-HP4/outputs/hp4_no_leak_validation/no_leak_reader_summary.json").read_text(encoding="utf-8"))
    v2 = json.loads(Path("V7-HP4/outputs/hp4_no_leak_v2_validation/no_leak_v2_reader_summary.json").read_text(encoding="utf-8"))
    oracle = json.loads(Path("V7-HP4/outputs/hp4_full_validation/full_validation_reader_summary.json").read_text(encoding="utf-8"))
    report_path = Path(args.report_dir) / "v7_hp4_no_leak_v3_sparse_validation_latest.md"
    report(report_path, train, summary, sig, v1, v2, oracle)
    print(json.dumps({"examples": len(examples), "data_source": data_source, "train_summary": train, "summary": summary, "significance": sig, "report_path": str(report_path), "output_root": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
