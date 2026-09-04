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
V3 = _load("V7-HP4/run_hp4_no_leak_v3_sparse_validation_eval.py", "hp4_v3")
FULLVAL = _load("V7-HP4/run_hp4_full_validation_eval.py", "hp4_fullval")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")



def reader_aware_hard_negative_mask(case: dict[str, Any], docs: list[Any], top_k: int = 5, alpha: float = 0.55) -> list[bool]:
    """Hard negatives are non-support docs that enter baseline Top-K on failed/partial contexts.

    This is reader-aware in the available local-training sense: a negative is hard only
    when the baseline context either misses at least one supporting title or lacks answer
    access, so the non-support block plausibly consumes reader context budget.
    """
    question = str(case.get("question", ""))
    weights = {doc.doc_id: 1.0 for doc in docs}
    ranked = V2.HybridSoftRetriever(docs, alpha=alpha).rank(question, weights=weights, top_k=top_k)
    top_docs = [doc for doc, _ in ranked]
    gold = {str(t) for t in case.get("supporting_titles", [])}
    pred = {doc.title for doc in top_docs}
    support_recall, _ = V2.sp_metrics(pred, gold)
    answer_access = V2.answer_in_context(case.get("answer", ""), top_docs)
    failed_context = support_recall < 1.0 or answer_access < 1.0
    top_ids = {doc.doc_id for doc in top_docs}
    return [bool(failed_context and (doc.doc_id in top_ids) and not bool(doc.is_support)) for doc in docs]


def train_policy_v5(train_examples: list[tuple[dict[str, Any], list[Any]]], out: Path, epochs: int, lr: float, hidden_dim: int, seed: int, target_topk: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V2.NoLeakPolicyV2(input_dim=len(V2.FEATURE_NAMES), hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    packed = []
    for case, docs in train_examples:
        x = torch.tensor(V2.feature_rows(str(case.get("question", "")), docs), dtype=torch.float32, device=device)
        y = torch.tensor([V2.train_target(case, docs, doc) for doc in docs], dtype=torch.float32, device=device)
        hard = torch.tensor(reader_aware_hard_negative_mask(case, docs, top_k=5), dtype=torch.bool, device=device)
        packed.append((x, y, hard))

    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        total = torch.tensor(0.0, device=device)
        pos_ws, neg_ws, hard_ws, topk_hard_rates = [], [], [], []
        for x, y, hard in packed:
            raw = model(x).clamp(1e-5, 1.0 - 1e-5)
            pos = y >= 0.99
            neg = y < 0.5
            bce_weight = torch.where(pos, torch.full_like(y, 2.5), torch.ones_like(y))
            bce = nn.functional.binary_cross_entropy(raw, y.clamp(0.0, 1.0), weight=bce_weight)
            pair = torch.tensor(0.0, device=device)
            if pos.any() and neg.any():
                pair = nn.functional.relu(0.35 - raw[pos][:, None] + raw[neg][None, :]).mean()
            exclusion = torch.tensor(0.0, device=device)
            if hard.any():
                k = min(int(target_topk), int(raw.numel()))
                threshold = torch.topk(raw, k=k).values[-1].detach()
                # Penalize hard negatives that are competitive with the top-k boundary.
                exclusion = nn.functional.relu(raw[hard] - threshold + 0.08).mean()
            support_mass = raw[pos].sum() if pos.any() else torch.tensor(0.0, device=device)
            budget = ((raw.sum() - min(float(target_topk), float(raw.numel()))) / max(float(raw.numel()), 1.0)).pow(2)
            support_floor = nn.functional.relu(1.35 - support_mass)
            entropy = -(raw * torch.log(raw) + (1.0 - raw) * torch.log(1.0 - raw)).mean()
            loss = bce + 0.85 * pair + 2.25 * exclusion + 0.45 * budget + 0.75 * support_floor + 0.02 * entropy
            total = total + loss
            if pos.any():
                pos_ws.append(raw[pos].mean().detach())
            if neg.any():
                neg_ws.append(raw[neg].mean().detach())
            if hard.any():
                hard_ws.append(raw[hard].mean().detach())
                top_idx = set(torch.topk(raw.detach(), k=min(int(target_topk), int(raw.numel()))).indices.cpu().tolist())
                hard_idx = set(torch.where(hard.detach().cpu())[0].tolist())
                topk_hard_rates.append(torch.tensor(float(bool(top_idx & hard_idx)), device=device))
        loss = total / max(len(packed), 1)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu())
        opt.step()
        logs.append({
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "grad_norm": grad_norm,
            "pos_weight": float(torch.stack(pos_ws).mean().cpu()) if pos_ws else 0.0,
            "neg_weight": float(torch.stack(neg_ws).mean().cpu()) if neg_ws else 0.0,
            "hard_neg_weight": float(torch.stack(hard_ws).mean().cpu()) if hard_ws else 0.0,
            "topk_has_hard_negative_rate": float(torch.stack(topk_hard_rates).mean().cpu()) if topk_hard_rates else 0.0,
        })
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "hp4_no_leak_policy_v5_readeraware_hardgate.pt"
    torch.save({"state_dict": model.state_dict(), "feature_names": V2.FEATURE_NAMES, "hidden_dim": hidden_dim, "target_topk": target_topk}, model_path)
    save_json(out / "hp4_no_leak_policy_v5_training_log.json", logs)
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
        "topk_has_hard_negative_rate": logs[-1]["topk_has_hard_negative_rate"],
        "max_grad_norm": max(r["grad_norm"] for r in logs),
    }
    save_json(out / "hp4_no_leak_policy_v5_training_summary.json", summary)
    return summary


def hardgate_weights(raw: torch.Tensor, top_k: int, low_scale: float) -> torch.Tensor:
    if raw.numel() <= top_k:
        return raw.clamp(0.02, 1.0)
    idx = torch.topk(raw, k=top_k).indices
    gated = raw * float(low_scale)
    gated[idx] = raw[idx]
    return gated.clamp(0.001, 1.0)


def weights_for_docs(question: str, docs: list[Any], model: Any, device: torch.device, mode: str, top_gate: int, low_scale: float) -> dict[str, float]:
    if mode == "baseline_uniform":
        return {doc.doc_id: 1.0 for doc in docs}
    x = torch.tensor(V2.feature_rows(question, docs), dtype=torch.float32, device=device)
    with torch.no_grad():
        raw = model(x).clamp(1e-5, 1.0 - 1e-5)
        w = hardgate_weights(raw, top_gate, low_scale).detach().cpu().tolist()
    return {doc.doc_id: float(v) for doc, v in zip(docs, w)}


def evaluate(examples: list[tuple[dict[str, Any], list[Any]]], reader: Any, model: Any, device: torch.device, top_k: int, alpha: float, top_gate: int, low_scale: float):
    retrieval_items = []
    for case, docs in examples:
        docs = V2.sanitize_docs_for_policy(docs)
        for mode in ["baseline_uniform", "hp4_no_leak_policy_v5_readeraware_hardgate"]:
            q = str(case.get("question", ""))
            weights = weights_for_docs(q, docs, model, device, mode, top_gate, low_scale)
            ranked = V2.HybridSoftRetriever(docs, alpha=alpha).rank(q, weights=weights, top_k=top_k)
            top_docs = [doc for doc, _ in ranked]
            gold = {str(t) for t in case.get("supporting_titles", [])}
            support_recall, sp_f1 = V2.sp_metrics({doc.title for doc in top_docs}, gold)
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
    return [
        float(m["hp4_no_leak_policy_v5_readeraware_hardgate"].get(metric, 0.0)) - float(m["baseline_uniform"].get(metric, 0.0))
        for m in by_id.values()
        if "baseline_uniform" in m and "hp4_no_leak_policy_v5_readeraware_hardgate" in m
    ]


def write_report(path: Path, train: dict[str, Any], summary: dict[str, Any], sig: dict[str, Any], v1: dict[str, Any], v2: dict[str, Any], v3: dict[str, Any], v4s: dict[str, Any], oracle: dict[str, Any]) -> None:
    def g(payload, mode, metric):
        return float(payload.get(mode, {}).get(metric, 0.0)) - float(payload.get("baseline_uniform", {}).get(metric, 0.0))
    lines = ["# V7-HP4 No-Leak V5 Reader-Aware Hardgate Policy", "", "## Training", ""]
    for k in ["train_examples", "loss_delta", "final_pos_weight", "final_neg_weight", "final_hard_neg_weight", "pos_neg_gap", "pos_hard_neg_gap", "topk_has_hard_negative_rate", "max_grad_norm"]:
        lines.append(f"- {k}: {train.get(k)}")
    lines += ["", "## Reader Metrics", "", "| mode | n | access@k | support_recall@k | sp_f1 | answer_em | answer_f1 | joint_f1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode in ["baseline_uniform", "hp4_no_leak_policy_v5_readeraware_hardgate"]:
        m = summary.get(mode, {})
        lines.append(f"| {mode} | {m.get('n',0)} | {m.get('answer_access_at_k',0):.4f} | {m.get('support_recall_at_k',0):.4f} | {m.get('sp_f1',0):.4f} | {m.get('answer_em',0):.4f} | {m.get('answer_f1',0):.4f} | {m.get('joint_f1',0):.4f} |")
    lines += ["", "## Significance", "", "| metric | n | mean_gap | p_value_two_sided | ci95 |", "| --- | ---: | ---: | ---: | ---: |"]
    for metric in ["joint_f1", "sp_f1"]:
        s = sig[metric]
        lines.append(f"| {metric} | {s['n']} | {s['mean_diff']:+.4f} | {s['p_value_two_sided']:.6f} | [{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}] |")
    lines += ["", "## Gap Comparison", "", "| metric | v1 | v2 | v3 sparse | v4 hardgate | v5 reader-aware | oracle |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for metric in ["support_recall_at_k", "sp_f1", "answer_f1", "joint_f1"]:
        lines.append(f"| {metric} | {g(v1,'hp4_no_leak_policy',metric):+.4f} | {g(v2,'hp4_no_leak_policy_v2',metric):+.4f} | {g(v3,'hp4_no_leak_policy_v3_sparse',metric):+.4f} | {g(v4s,'hp4_no_leak_policy_v4_hardgate',metric):+.4f} | {g(summary,'hp4_no_leak_policy_v5_readeraware_hardgate',metric):+.4f} | {g(oracle,'hp4_soft_agent',metric):+.4f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", default="FedE/select_data_hotpot_train_5000.json")
    p.add_argument("--preferred-dev", default="V7-HP4/data/hotpot_validation_1000.json")
    p.add_argument("--generated-dev", default="V7-HP4/data/hotpot_validation_1000.json")
    p.add_argument("--fallback-dev", default="FedE/select_data_hotpot_train_5000.json")
    p.add_argument("--output-root", default="V7-HP4/outputs/hp4_no_leak_v5_readeraware_hardgate_validation")
    p.add_argument("--report-dir", default="实验分析报告/V7-HP4")
    p.add_argument("--reader-model", default="google/flan-t5-large")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reader-batch-size", type=int, default=2)
    p.add_argument("--max-train", type=int, default=1200)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--target-topk", type=int, default=3)
    p.add_argument("--low-scale", type=float, default=0.01)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--max-dev", type=int, default=1000)
    p.add_argument("--seed", type=int, default=88)
    p.add_argument("--permutation-rounds", type=int, default=10000)
    args = p.parse_args()
    out = Path(args.output_root)
    train_examples = V2.load_training_examples(Path(args.train_data), args.max_train, args.seed)
    train = train_policy_v5(train_examples, out, args.epochs, args.lr, args.hidden_dim, args.seed, args.target_topk)
    dev_path, _, _ = FULLVAL.load_or_build_validation_split(Path(args.preferred_dev), Path(args.generated_dev), args.max_dev, args.seed, Path(args.fallback_dev))
    examples = V2.materialize_dev(dev_path, args.max_dev)
    model, device = V2.load_policy_v2(Path(train["model_path"]), args.device)
    reader = V2.Reader(args.reader_model, args.device, args.reader_batch_size)
    rows, summary = evaluate(examples, reader, model, device, args.top_k, args.alpha, args.target_topk, args.low_scale)
    save_json(out / "no_leak_v4_reader_rows.json", rows)
    save_json(out / "no_leak_v4_reader_summary.json", summary)
    sig = {
        "joint_f1": FULLVAL.permutation_p_value(paired(rows, "joint_f1"), args.permutation_rounds, args.seed + 1),
        "sp_f1": FULLVAL.permutation_p_value(paired(rows, "sp_f1"), args.permutation_rounds, args.seed + 2),
    }
    save_json(out / "no_leak_v4_significance.json", sig)
    v1 = json.loads(Path("V7-HP4/outputs/hp4_no_leak_validation/no_leak_reader_summary.json").read_text(encoding="utf-8"))
    v2s = json.loads(Path("V7-HP4/outputs/hp4_no_leak_v2_validation/no_leak_v2_reader_summary.json").read_text(encoding="utf-8"))
    v3s = json.loads(Path("V7-HP4/outputs/hp4_no_leak_v3_sparse_fast_validation/no_leak_v3_reader_summary.json").read_text(encoding="utf-8"))
    v4s = json.loads(Path("V7-HP4/outputs/hp4_no_leak_v4_hardgate_validation/no_leak_v4_reader_summary.json").read_text(encoding="utf-8"))
    oracle = json.loads(Path("V7-HP4/outputs/hp4_full_validation/full_validation_reader_summary.json").read_text(encoding="utf-8"))
    report_path = Path(args.report_dir) / "v7_hp4_no_leak_v5_readeraware_hardgate_validation_latest.md"
    write_report(report_path, train, summary, sig, v1, v2s, v3s, v4s, oracle)
    print(json.dumps({"train_summary": train, "summary": summary, "significance": sig, "report_path": str(report_path), "output_root": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
