from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from src.v7_hp4.agent_continuous import BlockState, ContinuousUploadPolicy
from src.v7_hp4.hybrid_retriever import HybridDocument, docs_from_micro_case


def _load_hp4_helpers():
    helper_path = Path("V7-HP4/run_hp4_full_experiment.py")
    spec = importlib.util.spec_from_file_location("v7_hp4_full_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load HP4 helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HP4 = _load_hp4_helpers()
build_dev300_case = _HP4.build_dev300_case


def _clamp(value: float, lo: float = -10.0, hi: float = 10.0) -> float:
    if math.isnan(float(value)) or math.isinf(float(value)):
        return 0.0
    return max(lo, min(hi, float(value)))


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_'-]*", text or "")}


def _entity_tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in re.finditer(r"\b[A-Z][A-Za-z0-9_'-]{2,}\b", text or "")}


def _rare_tokens(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_'-]{6,}", text or "")}


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    aa, bb = set(a), set(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


@dataclass(frozen=True)
class PolicyExample:
    query_id: str
    doc_id: str
    client_id: str
    state: BlockState
    advantage: float
    is_counterfactual_positive: bool
    source: str


def block_state_from_doc(question: str, doc: HybridDocument, mode: str = "natural") -> BlockState:
    lexical = _jaccard(_tokens(question), _tokens(doc.content))
    query_entities = _entity_tokens(question)
    sparse_dictionary = set(doc.bridge_entities) | _entity_tokens(doc.content) | _tokens(doc.title)
    entity = _jaccard(query_entities, sparse_dictionary)
    rare = _jaccard(_rare_tokens(question), set(doc.rare_tokens) | _rare_tokens(doc.content))
    bridge_match = _jaccard(query_entities, sparse_dictionary | set(doc.rare_tokens))
    client_rarity = 0.65 if doc.client_id in {"client_x", "client_y"} else 0.25
    # No gold support flag is used here. Counterfactual reward supplies credit.
    return BlockState(
        local_utility=min(1.0, 0.25 + 1.50 * lexical),
        memory_utility=min(1.0, 0.20 + 1.20 * entity + 0.35 * rare),
        hard_query_alignment=min(1.0, 0.30 + 1.15 * entity),
        client_rarity_score=client_rarity,
        bridge_entity_overlap=min(1.0, entity),
        rare_token_overlap=min(1.0, rare),
        bridge_entity_match_ratio=min(1.0, bridge_match),
        diversity_bonus=0.45 if mode == "micro" and doc.client_id in {"client_x", "client_y"} else 0.25,
        instability_penalty=0.05,
    )


def load_micro_docs(path: Path) -> dict[str, tuple[dict[str, Any], list[HybridDocument]]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    return {str(case.get("id")): (case, docs_from_micro_case(case)) for case in cases}


def load_dev_docs(path: Path, max_examples: int = 300) -> dict[str, tuple[dict[str, Any], list[HybridDocument]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[dict[str, Any], list[HybridDocument]]] = {}
    for idx, item in enumerate(payload[:max_examples]):
        built = build_dev300_case(item, idx)
        if built is None:
            continue
        case, docs = built
        out[str(case.get("id"))] = (case, docs)
    return out


def load_counterfactual_examples(
    rows_path: Path,
    docs_by_id: Mapping[str, tuple[dict[str, Any], list[HybridDocument]]],
    source: str,
    include_zero_advantage_negatives: bool = True,
) -> list[PolicyExample]:
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    actual_by_id = {str(r["id"]): r for r in rows if r.get("mode") == "hp4_soft_agent"}
    cf_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("mode") == "hp4_counterfactual_zero":
            cf_by_id[str(row.get("id"))].append(row)

    examples: list[PolicyExample] = []
    positive_docs: set[str] = set()
    for qid, cf_rows in cf_by_id.items():
        if qid not in docs_by_id:
            continue
        case, docs = docs_by_id[qid]
        doc_by_id = {doc.doc_id: doc for doc in docs}
        actual = actual_by_id.get(qid, {})
        for cf in cf_rows:
            doc_id = str(cf.get("zero_doc_id"))
            doc = doc_by_id.get(doc_id)
            if doc is None:
                continue
            advantage = float(actual.get("joint_f1", 0.0)) - float(cf.get("joint_f1", 0.0))
            advantage = _clamp(advantage, -2.0, 2.0)
            positive_docs.add(f"{qid}::{doc_id}")
            examples.append(PolicyExample(
                query_id=qid,
                doc_id=doc_id,
                client_id=doc.client_id,
                state=block_state_from_doc(str(case.get("question", "")), doc, mode=source),
                advantage=advantage,
                is_counterfactual_positive=advantage > 1e-9,
                source=source,
            ))
        if include_zero_advantage_negatives:
            # Keep a few non-counterfactual docs as zero/negative stabilizers so
            # the policy does not trivially push every block to 1.0.
            for doc in docs[: min(len(docs), 8)]:
                key = f"{qid}::{doc.doc_id}"
                if key in positive_docs:
                    continue
                examples.append(PolicyExample(
                    query_id=qid,
                    doc_id=doc.doc_id,
                    client_id=doc.client_id,
                    state=block_state_from_doc(str(case.get("question", "")), doc, mode=source),
                    advantage=-0.05,
                    is_counterfactual_positive=False,
                    source=source,
                ))
    return examples


def normalize_advantages(examples: list[PolicyExample]) -> torch.Tensor:
    values = torch.tensor([ex.advantage for ex in examples], dtype=torch.float32)
    mean = values.mean()
    std = values.std(unbiased=False).clamp_min(1e-6)
    return (values - mean) / std


def reinforce_loss(
    policy: ContinuousUploadPolicy,
    features: torch.Tensor,
    advantages: torch.Tensor,
    examples: list[PolicyExample] | None = None,
    entropy_weight: float = 0.01,
    co_route_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = policy(features).clamp(1e-5, 1.0 - 1e-5)
    positive = (advantages >= 0).float()
    log_prob = positive * torch.log(weights) + (1.0 - positive) * torch.log(1.0 - weights)
    loss = -(advantages.abs().detach() * log_prob).mean()
    entropy = -(weights * torch.log(weights) + (1.0 - weights) * torch.log(1.0 - weights)).mean()
    co_route = torch.tensor(0.0, device=features.device)
    co_pairs = 0
    if examples is not None and co_route_weight > 0:
        by_query: dict[str, list[int]] = defaultdict(list)
        for idx, ex in enumerate(examples):
            if ex.advantage > 1e-9:
                by_query[ex.query_id].append(idx)
        penalties = []
        for indices in by_query.values():
            if len(indices) < 2:
                continue
            clients = {examples[i].client_id for i in indices}
            if len(clients) < 2:
                continue
            vals = weights[torch.tensor(indices, dtype=torch.long, device=features.device)]
            penalties.append(((vals - vals.mean()) ** 2).mean())
        if penalties:
            co_route = torch.stack(penalties).mean()
            co_pairs = len(penalties)
    total = loss - float(entropy_weight) * entropy + float(co_route_weight) * co_route
    return total, {
        "loss": float(total.detach().cpu()),
        "pg_loss": float(loss.detach().cpu()),
        "co_route_loss": float(co_route.detach().cpu()),
        "co_route_pairs": float(co_pairs),
        "entropy": float(entropy.detach().cpu()),
        "mean_weight": float(weights.mean().detach().cpu()),
        "min_weight": float(weights.min().detach().cpu()),
        "max_weight": float(weights.max().detach().cpu()),
        "weight_variance": float(weights.var(unbiased=False).detach().cpu()),
    }


def train_policy(
    examples: list[PolicyExample],
    output_dir: Path,
    epochs: int = 80,
    lr: float = 3e-3,
    hidden_dim: int = 48,
    grad_clip: float = 1.0,
    seed: int = 7,
    co_route_weight: float = 0.0,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("No policy examples to train on")
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ContinuousUploadPolicy(hidden_dim=hidden_dim, init_bias=-0.4).to(device)
    features = ContinuousUploadPolicy.tensor_from_states([ex.state for ex in examples], device=str(device))
    advantages = normalize_advantages(examples).to(device)
    raw_advantages = torch.tensor([ex.advantage for ex in examples], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    logs = []
    with torch.no_grad():
        initial_weights = policy(features).detach()
        initial_weight_variance = float(initial_weights.var(unbiased=False).cpu())
    for epoch in range(1, int(epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = reinforce_loss(policy, features, advantages, examples=examples, co_route_weight=co_route_weight)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip).detach().cpu())
        optimizer.step()
        with torch.no_grad():
            weights = policy(features).detach()
            pos_mask = raw_advantages > 1e-9
            neg_mask = raw_advantages <= 0.0
            metrics.update({
                "epoch": epoch,
                "grad_norm": grad_norm,
                "positive_mean_weight": float(weights[pos_mask].mean().cpu()) if pos_mask.any() else 0.0,
                "nonpositive_mean_weight": float(weights[neg_mask].mean().cpu()) if neg_mask.any() else 0.0,
                "raw_advantage_mean": float(raw_advantages.mean().detach().cpu()),
                "raw_advantage_positive_rate": float((raw_advantages > 1e-9).float().mean().detach().cpu()),
            })
        logs.append(metrics)

    torch.save({
        "state_dict": policy.state_dict(),
        "feature_names": ContinuousUploadPolicy.feature_names,
        "hidden_dim": hidden_dim,
        "examples": len(examples),
    }, output_dir / "hp4_policy_reinforce.pt")
    (output_dir / "training_log.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")

    final = logs[-1]
    first = logs[0]
    summary = {
        "examples": len(examples),
        "device": str(device),
        "epochs": epochs,
        "lr": lr,
        "co_route_weight": co_route_weight,
        "initial_weight_variance": initial_weight_variance,
        "initial_loss": first["loss"],
        "final_loss": final["loss"],
        "loss_delta": final["loss"] - first["loss"],
        "final_grad_norm": final["grad_norm"],
        "max_grad_norm": max(row["grad_norm"] for row in logs),
        "final_positive_mean_weight": final["positive_mean_weight"],
        "final_nonpositive_mean_weight": final["nonpositive_mean_weight"],
        "positive_minus_nonpositive_weight": final["positive_mean_weight"] - final["nonpositive_mean_weight"],
        "final_weight_variance": final["weight_variance"],
        "final_co_route_loss": final["co_route_loss"],
        "final_co_route_pairs": final["co_route_pairs"],
        "raw_advantage_mean": final["raw_advantage_mean"],
        "raw_advantage_positive_rate": final["raw_advantage_positive_rate"],
        "model_path": str(output_dir / "hp4_policy_reinforce.pt"),
        "log_path": str(output_dir / "training_log.json"),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_markdown(path: Path, summary: Mapping[str, Any], logs: list[Mapping[str, Any]]) -> None:
    head = logs[:5]
    tail = logs[-5:]

    def rows(items: list[Mapping[str, Any]]) -> str:
        lines = [
            "| epoch | loss | grad_norm | mean_w | pos_w | nonpos_w | entropy |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in items:
            lines.append(
                f"| {r['epoch']} | {r['loss']:.6f} | {r['grad_norm']:.6f} | {r['mean_weight']:.4f} | "
                f"{r['positive_mean_weight']:.4f} | {r['nonpositive_mean_weight']:.4f} | {r['entropy']:.4f} |"
            )
        return "\n".join(lines)

    text = f"""# V7-HP4 Phase 2 Policy Gradient Training Report

## Objective

Train the HP4 continuous upload policy with REINFORCE using real-reader counterfactual marginal rewards:

`A(a_ij) = R_actual - R_counter(j)`

Loss:

`L_policy = - sum A(a_ij) log pi_theta(w_ij | s_ij)`

## Summary

- examples: {summary['examples']}
- device: {summary['device']}
- epochs: {summary['epochs']}
- initial_loss: {summary['initial_loss']:.6f}
- final_loss: {summary['final_loss']:.6f}
- loss_delta: {summary['loss_delta']:.6f}
- co_route_weight: {summary.get('co_route_weight', 0.0):.4f}
- initial_weight_variance: {summary.get('initial_weight_variance', 0.0):.6f}
- final_weight_variance: {summary.get('final_weight_variance', 0.0):.6f}
- final_co_route_loss: {summary.get('final_co_route_loss', 0.0):.6f}
- max_grad_norm: {summary['max_grad_norm']:.6f}
- final_grad_norm: {summary['final_grad_norm']:.6f}
- final_positive_mean_weight: {summary['final_positive_mean_weight']:.4f}
- final_nonpositive_mean_weight: {summary['final_nonpositive_mean_weight']:.4f}
- positive_minus_nonpositive_weight: {summary['positive_minus_nonpositive_weight']:.4f}

## First Epochs

{rows(head)}

## Final Epochs

{rows(tail)}

## Stability Judgment

Gradient clipping was enabled. Training is considered stable if max_grad_norm remains finite and the final positive weights separate from non-positive weights without NaN/Inf.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-data", default="data/v7_hp4_micro_benchmark.json")
    parser.add_argument("--dev-data", default="V7-HP3/data/hotpot_dev_stratified_300.json")
    parser.add_argument("--micro-rows", default="V7-HP4/outputs/hp4_reader_counterfactual/micro_reader_rows.json")
    parser.add_argument("--dev-rows", default="V7-HP4/outputs/hp4_reader_counterfactual/dev300_reader_rows.json")
    parser.add_argument("--output-dir", default="V7-HP4/outputs/hp4_policy_gradient")
    parser.add_argument("--report-dir", default="实验分析报告/V7-HP4")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--max-dev", type=int, default=300)
    parser.add_argument("--co-route-weight", type=float, default=0.0)
    args = parser.parse_args()

    micro_docs = load_micro_docs(Path(args.micro_data))
    dev_docs = load_dev_docs(Path(args.dev_data), max_examples=args.max_dev)
    examples = []
    examples.extend(load_counterfactual_examples(Path(args.micro_rows), micro_docs, "micro"))
    examples.extend(load_counterfactual_examples(Path(args.dev_rows), dev_docs, "dev300"))
    out = Path(args.output_dir)
    summary = train_policy(examples, out, epochs=args.epochs, lr=args.lr, hidden_dim=args.hidden_dim, co_route_weight=args.co_route_weight)
    logs = json.loads((out / "training_log.json").read_text(encoding="utf-8"))
    report = Path(args.report_dir) / "v7_hp4_phase2_policy_gradient_latest.md"
    write_markdown(report, summary, logs)
    print(json.dumps({"summary": summary, "report_path": str(report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
