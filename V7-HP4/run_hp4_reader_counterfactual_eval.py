from __future__ import annotations

import argparse
import json
import re
import string
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.v7_hp4.hybrid_retriever import HybridSoftRetriever, docs_from_micro_case


ARTICLES = {"a", "an", "the"}


def _load_hp4_helpers():
    helper_path = Path("V7-HP4/run_hp4_full_experiment.py")
    spec = importlib.util.spec_from_file_location("v7_hp4_full_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load HP4 helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HP4 = _load_hp4_helpers()
answer_in_context = _HP4.answer_in_context
build_dev300_case = _HP4.build_dev300_case
f1_score = _HP4.f1_score
hp4_weights_for_docs = _HP4.hp4_weights_for_docs
normalize_answer = _HP4.normalize_answer


def make_prompt(question: str, top_docs: list[Any], max_chars: int = 3200) -> str:
    context_parts = []
    for idx, doc in enumerate(top_docs, start=1):
        context_parts.append(f"[{idx}] {doc.title}: {doc.text}")
    context = "\n".join(context_parts)
    if len(context) > max_chars:
        context = context[:max_chars]
    return (
        "Answer the question using only the context. "
        "Return a short answer.\n\n"
        f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    )


class Reader:
    def __init__(self, model_name: str, device: str, batch_size: int = 2, max_new_tokens: int = 32) -> None:
        self.model_name = model_name
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = int(batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate(self, prompts: list[str]) -> list[str]:
        outputs: list[str] = []
        with torch.no_grad():
            for start in range(0, len(prompts), self.batch_size):
                batch = prompts[start:start + self.batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(self.device)
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )
                outputs.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
        return [o.strip() for o in outputs]


def sp_metrics(pred_titles: set[str], gold_titles: set[str]) -> tuple[float, float]:
    if not gold_titles:
        return 0.0, 0.0
    tp = len(pred_titles & gold_titles)
    precision = tp / len(pred_titles) if pred_titles else 0.0
    recall = tp / len(gold_titles)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if gold_titles <= pred_titles:
        f1 = max(f1, 1.0)
    return recall, f1


def build_retrieval_item(case: dict[str, Any], docs: list[Any], mode: str, top_k: int, alpha: float, zero_doc_id: str | None = None) -> dict[str, Any]:
    question = str(case.get("question", ""))
    weights = hp4_weights_for_docs(question, docs, mode)
    if mode == "baseline_uniform" and "baseline_weights" in case:
        weights = {str(k): float(v) for k, v in case["baseline_weights"].items()}
    if zero_doc_id is not None:
        weights[zero_doc_id] = 0.0
    ranked = HybridSoftRetriever(docs, alpha=alpha).rank(question, weights=weights, top_k=top_k)
    top_docs = [doc for doc, _ in ranked]
    gold_titles = {str(t) for t in case.get("supporting_titles", [])}
    pred_titles = {doc.title for doc in top_docs}
    support_recall, sp_f1 = sp_metrics(pred_titles, gold_titles)
    return {
        "case": case,
        "mode": mode,
        "zero_doc_id": zero_doc_id,
        "top_docs": top_docs,
        "top_doc_ids": [doc.doc_id for doc in top_docs],
        "top_titles": [doc.title for doc in top_docs],
        "answer_access_at_k": answer_in_context(case.get("answer", ""), top_docs),
        "support_recall_at_k": support_recall,
        "sp_f1": sp_f1,
        "prompt": make_prompt(question, top_docs),
    }


def materialize_micro(path: Path) -> list[tuple[dict[str, Any], list[Any]]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    return [(case, docs_from_micro_case(case)) for case in cases]


def materialize_dev(path: Path, max_examples: int) -> list[tuple[dict[str, Any], list[Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for idx, item in enumerate(payload[:max_examples]):
        built = build_dev300_case(item, idx)
        if built is not None:
            out.append(built)
    return out


def evaluate_dataset(
    examples: list[tuple[dict[str, Any], list[Any]]],
    reader: Reader,
    top_k: int,
    alpha: float,
    counterfactual: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    retrieval_items = []
    for case, docs in examples:
        retrieval_items.append(build_retrieval_item(case, docs, "baseline_uniform", top_k, alpha))
        actual = build_retrieval_item(case, docs, "hp4_soft_agent", top_k, alpha)
        retrieval_items.append(actual)
        if counterfactual:
            support_ids = [doc.doc_id for doc in docs if doc.is_support]
            for doc_id in support_ids[:2]:
                retrieval_items.append(build_retrieval_item(case, docs, "hp4_counterfactual_zero", top_k, alpha, zero_doc_id=doc_id))

    predictions = reader.generate([item["prompt"] for item in retrieval_items])
    rows = []
    for item, pred in zip(retrieval_items, predictions):
        case = item["case"]
        answer = case.get("answer", "")
        answer_em = float(normalize_answer(pred) == normalize_answer(answer))
        answer_f1 = f1_score(pred, answer)
        joint_f1 = answer_f1 * float(item["sp_f1"])
        rows.append({
            "id": str(case.get("id", case.get("_id", ""))),
            "mode": item["mode"],
            "zero_doc_id": item["zero_doc_id"],
            "prediction": pred,
            "answer": answer,
            "answer_em": answer_em,
            "answer_f1": answer_f1,
            "answer_access_at_k": item["answer_access_at_k"],
            "support_recall_at_k": item["support_recall_at_k"],
            "sp_f1": item["sp_f1"],
            "joint_f1": joint_f1,
            "top_doc_ids": item["top_doc_ids"],
            "top_titles": item["top_titles"],
        })

    actual_by_id = {r["id"]: r for r in rows if r["mode"] == "hp4_soft_agent"}
    cf_rewards = []
    for row in rows:
        if row["mode"] != "hp4_counterfactual_zero":
            continue
        actual = actual_by_id.get(row["id"])
        if not actual:
            continue
        reward = float(actual["joint_f1"]) - float(row["joint_f1"])
        row["counterfactual_reward_joint_f1"] = reward
        cf_rewards.append(reward)

    summary = summarize(rows)
    summary["_counterfactual"] = {
        "n": len(cf_rewards),
        "avg_marginal_joint_f1": sum(cf_rewards) / len(cf_rewards) if cf_rewards else 0.0,
        "positive_rate": sum(1 for r in cf_rewards if r > 1e-9) / len(cf_rewards) if cf_rewards else 0.0,
    }
    return rows, summary


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped = defaultdict(list)
    for row in rows:
        if row["mode"] == "hp4_counterfactual_zero":
            continue
        grouped[row["mode"]].append(row)
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


def write_report(path: Path, micro: dict[str, Any], dev: dict[str, Any], model_name: str) -> None:
    def table(summary: dict[str, Any]) -> str:
        lines = [
            "| mode | n | access@k | support_recall@k | sp_f1 | answer_em | answer_f1 | joint_f1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for mode in ["baseline_uniform", "hp4_soft_agent"]:
            if mode not in summary:
                continue
            m = summary[mode]
            lines.append(
                f"| {mode} | {m['n']} | {m['answer_access_at_k']:.4f} | {m['support_recall_at_k']:.4f} | "
                f"{m['sp_f1']:.4f} | {m['answer_em']:.4f} | {m['answer_f1']:.4f} | {m['joint_f1']:.4f} |"
            )
        return "\n".join(lines)

    def gap(summary: dict[str, Any], metric: str) -> float:
        return summary.get("hp4_soft_agent", {}).get(metric, 0.0) - summary.get("baseline_uniform", {}).get(metric, 0.0)

    text = f"""# V7-HP4 Real Reader + Online Counterfactual Reward 报告

Reader model: `{model_name}`

## Micro-Benchmark Real Reader

{table(micro)}

Micro reader joint_f1 gap: {gap(micro, "joint_f1"):+.4f}

Counterfactual marginal reward:

- n: {micro.get("_counterfactual", {}).get("n", 0)}
- avg_marginal_joint_f1: {micro.get("_counterfactual", {}).get("avg_marginal_joint_f1", 0.0):.4f}
- positive_rate: {micro.get("_counterfactual", {}).get("positive_rate", 0.0):.4f}

## Dev300 Real Reader

{table(dev)}

Dev reader joint_f1 gap: {gap(dev, "joint_f1"):+.4f}

Counterfactual marginal reward:

- n: {dev.get("_counterfactual", {}).get("n", 0)}
- avg_marginal_joint_f1: {dev.get("_counterfactual", {}).get("avg_marginal_joint_f1", 0.0):.4f}
- positive_rate: {dev.get("_counterfactual", {}).get("positive_rate", 0.0):.4f}

## 判断

- 若 real-reader joint_f1 gap 为正，说明 HP4 的 soft routing 不仅改变 Top-K context，也能向真实生成式 QA 指标转化。
- counterfactual marginal reward 为正，说明 support block 的 soft weight 对 reader joint_f1 有可观测因果贡献，可作为下一步 online policy learning 的训练信号。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro", default="data/v7_hp4_micro_benchmark.json")
    parser.add_argument("--dev300", default="V7-HP3/data/hotpot_dev_stratified_300.json")
    parser.add_argument("--output-root", default="V7-HP4/outputs/hp4_reader_counterfactual")
    parser.add_argument("--report-dir", default="实验分析报告/V7-HP4")
    parser.add_argument("--reader-model", default="google/flan-t5-large")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--reader-batch-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--max-dev", type=int, default=300)
    parser.add_argument("--no-counterfactual", action="store_true")
    args = parser.parse_args()

    reader = Reader(args.reader_model, args.device, args.reader_batch_size)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    micro_rows, micro_summary = evaluate_dataset(
        materialize_micro(Path(args.micro)), reader, args.top_k, args.alpha, not args.no_counterfactual
    )
    dev_rows, dev_summary = evaluate_dataset(
        materialize_dev(Path(args.dev300), args.max_dev), reader, args.top_k, args.alpha, not args.no_counterfactual
    )
    (out / "micro_reader_rows.json").write_text(json.dumps(micro_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "micro_reader_summary.json").write_text(json.dumps(micro_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "dev300_reader_rows.json").write_text(json.dumps(dev_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "dev300_reader_summary.json").write_text(json.dumps(dev_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = Path(args.report_dir) / "v7_hp4_real_reader_counterfactual_latest.md"
    write_report(report, micro_summary, dev_summary, args.reader_model)
    print(json.dumps({
        "micro_summary": micro_summary,
        "dev300_summary": dev_summary,
        "report_path": str(report),
        "output_root": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
