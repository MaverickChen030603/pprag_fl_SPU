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
V4 = _load("V7-HP4/run_hp4_no_leak_v4_hardgate_validation_eval.py", "hp4_v4")
FULLVAL = _load("V7-HP4/run_hp4_full_validation_eval.py", "hp4_fullval")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def rank_docs(case: dict[str, Any], docs: list[Any], weights: dict[str, float], top_k: int, alpha: float):
    return V2.HybridSoftRetriever(docs, alpha=alpha).rank(str(case.get("question", "")), weights=weights, top_k=top_k)


def item_prompt(case: dict[str, Any], top_docs: list[Any]) -> str:
    return V2.make_prompt(str(case.get("question", "")), top_docs)


def mine_causal_hard_negatives(
    examples: list[tuple[dict[str, Any], list[Any]]],
    reader: Any,
    out_path: Path,
    top_k: int,
    alpha: float,
    max_candidates_per_query: int,
    min_delta: float,
) -> dict[str, Any]:
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))

    prompts = []
    meta = []
    for case, raw_docs in examples:
        docs = V2.sanitize_docs_for_policy(raw_docs)
        base_weights = {doc.doc_id: 1.0 for doc in docs}
        base_ranked = rank_docs(case, docs, base_weights, top_k, alpha)
        base_docs = [doc for doc, _ in base_ranked]
        prompts.append(item_prompt(case, base_docs))
        meta.append({"kind": "baseline", "case": case, "doc_id": None, "top_doc_ids": [d.doc_id for d in base_docs]})
        candidates = [doc for doc in base_docs if not bool(doc.is_support)][:max_candidates_per_query]
        for cand in candidates:
            cf_weights = dict(base_weights)
            cf_weights[cand.doc_id] = 0.0
            cf_ranked = rank_docs(case, docs, cf_weights, top_k, alpha)
            cf_docs = [doc for doc, _ in cf_ranked]
            prompts.append(item_prompt(case, cf_docs))
            meta.append({"kind": "delete_negative", "case": case, "doc_id": cand.doc_id, "top_doc_ids": [d.doc_id for d in cf_docs]})

    preds = reader.generate(prompts)
    baseline_by_id: dict[str, float] = {}
    rows = []
    for m, pred in zip(meta, preds):
        case = m["case"]
        qid = str(case.get("id", case.get("_id", "")))
        answer_f1 = V2.f1_score(pred, case.get("answer", ""))
        row = {
            "id": qid,
            "kind": m["kind"],
            "doc_id": m["doc_id"],
            "prediction": pred,
            "answer_f1": answer_f1,
            "top_doc_ids": m["top_doc_ids"],
        }
        if m["kind"] == "baseline":
            baseline_by_id[qid] = answer_f1
        rows.append(row)

    harmful: dict[str, list[str]] = {}
    for row in rows:
        if row["kind"] != "delete_negative":
            continue
        base = baseline_by_id.get(str(row["id"]), 0.0)
        delta = float(row["answer_f1"]) - float(base)
        row["delta_answer_f1"] = delta
        if delta > min_delta:
            harmful.setdefault(str(row["id"]), []).append(str(row["doc_id"]))

    payload = {
        "min_delta": min_delta,
        "top_k": top_k,
        "alpha": alpha,
        "examples": len(examples),
        "prompts": len(prompts),
        "harmful_query_count": len(harmful),
        "harmful_block_count": sum(len(v) for v in harmful.values()),
        "harmful": harmful,
        "rows": rows,
    }
    save_json(out_path, payload)
    return payload


def train_policy_v6(train_examples: list[tuple[dict[str, Any], list[Any]]], harmful: dict[str, list[str]], out: Path, epochs: int, lr: float, hidden_dim: int, seed: int, target_topk: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V2.NoLeakPolicyV2(input_dim=len(V2.FEATURE_NAMES), hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)
    packed = []
    for case, docs in train_examples:
        docs = V2.sanitize_docs_for_policy(docs)
        qid = str(case.get("id", case.get("_id", "")))
        hard_ids = set(harmful.get(qid, []))
        x = torch.tensor(V2.feature_rows(str(case.get("question", "")), docs), dtype=torch.float32, device=device)
        y = torch.tensor([V2.train_target(case, docs, doc) for doc in docs], dtype=torch.float32, device=device)
        hard = torch.tensor([doc.doc_id in hard_ids for doc in docs], dtype=torch.bool, device=device)
        packed.append((x, y, hard))
    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        total = torch.tensor(0.0, device=device)
        pos_ws, neg_ws, hard_ws, rates = [], [], [], []
        for x, y, hard in packed:
            raw = model(x).clamp(1e-5, 1 - 1e-5)
            pos = y >= 0.99
            neg = y < 0.5
            bce_weight = torch.where(pos, torch.full_like(y, 2.0), torch.ones_like(y))
            bce_weight = torch.where(hard, torch.full_like(y, 7.0), bce_weight)
            target = torch.where(hard, torch.zeros_like(y), y.clamp(0, 1))
            bce = nn.functional.binary_cross_entropy(raw, target, weight=bce_weight)
            pair = torch.tensor(0.0, device=device)
            if pos.any() and neg.any():
                margin = torch.where(hard[neg], torch.full_like(raw[neg], 0.65), torch.full_like(raw[neg], 0.25))
                pair = nn.functional.relu(margin[None, :] - raw[pos][:, None] + raw[neg][None, :]).mean()
            exclusion = torch.tensor(0.0, device=device)
            if hard.any():
                k = min(target_topk, int(raw.numel()))
                threshold = torch.topk(raw, k=k).values[-1].detach()
                exclusion = nn.functional.relu(raw[hard] - threshold + 0.15).mean()
            support_mass = raw[pos].sum() if pos.any() else torch.tensor(0.0, device=device)
            support_floor = nn.functional.relu(1.35 - support_mass)
            budget = ((raw.sum() - min(float(target_topk), float(raw.numel()))) / max(float(raw.numel()), 1.0)).pow(2)
            loss = bce + 1.0 * pair + 3.5 * exclusion + 0.4 * budget + 0.7 * support_floor
            total = total + loss
            if pos.any():
                pos_ws.append(raw[pos].mean().detach())
            if neg.any():
                neg_ws.append(raw[neg].mean().detach())
            if hard.any():
                hard_ws.append(raw[hard].mean().detach())
                top = set(torch.topk(raw.detach(), k=min(target_topk, int(raw.numel()))).indices.cpu().tolist())
                hidx = set(torch.where(hard.detach().cpu())[0].tolist())
                rates.append(torch.tensor(float(bool(top & hidx)), device=device))
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
            "causal_hard_neg_weight": float(torch.stack(hard_ws).mean().cpu()) if hard_ws else 0.0,
            "topk_has_causal_hard_negative_rate": float(torch.stack(rates).mean().cpu()) if rates else 0.0,
        })
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "hp4_no_leak_policy_v6_causal_hardneg.pt"
    torch.save({"state_dict": model.state_dict(), "feature_names": V2.FEATURE_NAMES, "hidden_dim": hidden_dim, "target_topk": target_topk}, model_path)
    save_json(out / "hp4_no_leak_policy_v6_training_log.json", logs)
    summary = {
        "model_path": str(model_path),
        "train_examples": len(train_examples),
        "initial_loss": logs[0]["loss"],
        "final_loss": logs[-1]["loss"],
        "loss_delta": logs[-1]["loss"] - logs[0]["loss"],
        "final_pos_weight": logs[-1]["pos_weight"],
        "final_neg_weight": logs[-1]["neg_weight"],
        "final_causal_hard_neg_weight": logs[-1]["causal_hard_neg_weight"],
        "pos_neg_gap": logs[-1]["pos_weight"] - logs[-1]["neg_weight"],
        "pos_causal_hard_neg_gap": logs[-1]["pos_weight"] - logs[-1]["causal_hard_neg_weight"],
        "topk_has_causal_hard_negative_rate": logs[-1]["topk_has_causal_hard_negative_rate"],
        "max_grad_norm": max(r["grad_norm"] for r in logs),
    }
    save_json(out / "hp4_no_leak_policy_v6_training_summary.json", summary)
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
        for mode in ["baseline_uniform", "hp4_no_leak_policy_v6_causal_hardneg"]:
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
        float(m["hp4_no_leak_policy_v6_causal_hardneg"].get(metric, 0.0)) - float(m["baseline_uniform"].get(metric, 0.0))
        for m in by_id.values()
        if "baseline_uniform" in m and "hp4_no_leak_policy_v6_causal_hardneg" in m
    ]


def write_report(path: Path, train: dict[str, Any], mining: dict[str, Any], summary: dict[str, Any], sig: dict[str, Any], v4: dict[str, Any], oracle: dict[str, Any]) -> None:
    def g(payload, mode, metric):
        return float(payload.get(mode, {}).get(metric, 0.0)) - float(payload.get("baseline_uniform", {}).get(metric, 0.0))
    lines = [
        "# V7-HP4 No-Leak V6 Causal Harmful Negative Mining",
        "",
        "## Mining",
        "",
        f"- mining_examples: {mining.get('examples')}",
        f"- prompts: {mining.get('prompts')}",
        f"- harmful_query_count: {mining.get('harmful_query_count')}",
        f"- harmful_block_count: {mining.get('harmful_block_count')}",
        "",
        "## Training",
        "",
    ]
    for k in ["loss_delta", "final_pos_weight", "final_neg_weight", "final_causal_hard_neg_weight", "pos_causal_hard_neg_gap", "topk_has_causal_hard_negative_rate", "max_grad_norm"]:
        lines.append(f"- {k}: {train.get(k)}")
    lines += ["", "## Reader Metrics", "", "| mode | n | access@k | support_recall@k | sp_f1 | answer_em | answer_f1 | joint_f1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode in ["baseline_uniform", "hp4_no_leak_policy_v6_causal_hardneg"]:
        m = summary.get(mode, {})
        lines.append(f"| {mode} | {m.get('n',0)} | {m.get('answer_access_at_k',0):.4f} | {m.get('support_recall_at_k',0):.4f} | {m.get('sp_f1',0):.4f} | {m.get('answer_em',0):.4f} | {m.get('answer_f1',0):.4f} | {m.get('joint_f1',0):.4f} |")
    lines += ["", "## Significance", "", "| metric | n | mean_gap | p_value_two_sided | ci95 |", "| --- | ---: | ---: | ---: | ---: |"]
    for metric in ["joint_f1", "sp_f1"]:
        s = sig[metric]
        lines.append(f"| {metric} | {s['n']} | {s['mean_diff']:+.4f} | {s['p_value_two_sided']:.6f} | [{s['ci95_low']:+.4f}, {s['ci95_high']:+.4f}] |")
    lines += ["", "## Gap Comparison", "", "| metric | v4 hardgate | v6 causal | oracle |", "| --- | ---: | ---: | ---: |"]
    for metric in ["support_recall_at_k", "sp_f1", "answer_f1", "joint_f1"]:
        lines.append(f"| {metric} | {g(v4,'hp4_no_leak_policy_v4_hardgate',metric):+.4f} | {g(summary,'hp4_no_leak_policy_v6_causal_hardneg',metric):+.4f} | {g(oracle,'hp4_soft_agent',metric):+.4f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", default="FedE/select_data_hotpot_train_5000.json")
    p.add_argument("--preferred-dev", default="V7-HP4/data/hotpot_validation_1000.json")
    p.add_argument("--generated-dev", default="V7-HP4/data/hotpot_validation_1000.json")
    p.add_argument("--fallback-dev", default="FedE/select_data_hotpot_train_5000.json")
    p.add_argument("--output-root", default="V7-HP4/outputs/hp4_no_leak_v6_causal_hardneg_validation")
    p.add_argument("--report-dir", default="实验分析报告/V7-HP4")
    p.add_argument("--reader-model", default="google/flan-t5-large")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reader-batch-size", type=int, default=2)
    p.add_argument("--max-mine", type=int, default=240)
    p.add_argument("--max-train", type=int, default=1200)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--target-topk", type=int, default=3)
    p.add_argument("--low-scale", type=float, default=0.01)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--max-dev", type=int, default=1000)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--permutation-rounds", type=int, default=10000)
    args = p.parse_args()

    out = Path(args.output_root)
    train_examples = V2.load_training_examples(Path(args.train_data), args.max_train, args.seed)
    reader = V2.Reader(args.reader_model, args.device, args.reader_batch_size)
    mining = mine_causal_hard_negatives(
        train_examples[: args.max_mine],
        reader,
        out / "causal_hard_negative_mining.json",
        args.top_k,
        args.alpha,
        max_candidates_per_query=3,
        min_delta=0.05,
    )
    train = train_policy_v6(train_examples, mining.get("harmful", {}), out, args.epochs, args.lr, args.hidden_dim, args.seed, args.target_topk)
    dev_path, _, _ = FULLVAL.load_or_build_validation_split(Path(args.preferred_dev), Path(args.generated_dev), args.max_dev, args.seed, Path(args.fallback_dev))
    examples = V2.materialize_dev(dev_path, args.max_dev)
    model, device = V2.load_policy_v2(Path(train["model_path"]), args.device)
    rows, summary = evaluate(examples, reader, model, device, args.top_k, args.alpha, args.target_topk, args.low_scale)
    save_json(out / "no_leak_v6_reader_rows.json", rows)
    save_json(out / "no_leak_v6_reader_summary.json", summary)
    sig = {
        "joint_f1": FULLVAL.permutation_p_value(paired(rows, "joint_f1"), args.permutation_rounds, args.seed + 1),
        "sp_f1": FULLVAL.permutation_p_value(paired(rows, "sp_f1"), args.permutation_rounds, args.seed + 2),
    }
    save_json(out / "no_leak_v6_significance.json", sig)
    v4 = json.loads(Path("V7-HP4/outputs/hp4_no_leak_v4_hardgate_validation/no_leak_v4_reader_summary.json").read_text(encoding="utf-8"))
    oracle = json.loads(Path("V7-HP4/outputs/hp4_full_validation/full_validation_reader_summary.json").read_text(encoding="utf-8"))
    report_path = Path(args.report_dir) / "v7_hp4_no_leak_v6_causal_hardneg_validation_latest.md"
    write_report(report_path, train, mining, summary, sig, v4, oracle)
    print(json.dumps({"mining": {k: mining.get(k) for k in ["examples", "prompts", "harmful_query_count", "harmful_block_count"]}, "train_summary": train, "summary": summary, "significance": sig, "report_path": str(report_path), "output_root": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
