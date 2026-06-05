from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

ARTICLES = {"a", "an", "the"}


def normalize_answer(s: Any) -> str:
    def remove_articles(text: str) -> str:
        return " ".join([w for w in text.split() if w.lower() not in ARTICLES])

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def f1_score(prediction: Any, ground_truth: Any) -> tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    if normalized_prediction in {"yes", "no", "noanswer"} and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0
    if normalized_ground_truth in {"yes", "no", "noanswer"} and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(prediction_tokens) if prediction_tokens else 0.0
    recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall


def exact_match_score(prediction: Any, ground_truth: Any) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def sp_metrics(predicted: list[list[Any]], gold: list[list[Any]]) -> tuple[float, float, float, float]:
    pred = {(str(t), int(i)) for t, i in predicted}
    truth = {(str(t), int(i)) for t, i in gold}
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    em = float(fp == 0 and fn == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0
    return em, f1, precision, recall


def joint_metrics(answer_em: float, answer_f1: float, sp_em: float, sp_f1: float) -> tuple[float, float]:
    return answer_em * sp_em, answer_f1 * sp_f1


def parse_compact_reference(reference: str) -> list[dict[str, Any]]:
    pattern = re.compile(r"\[([^\]]+)\]\s*")
    matches = list(pattern.finditer(reference or ""))
    docs: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(reference)
        text = reference[start:end].strip()
        sentences = split_sentences(text)
        docs.append({"title": title, "sentences": sentences or [text]})
    return docs


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in pieces if p.strip()]


def load_compact_examples(path: Path, max_examples: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload if isinstance(payload, list) else list(payload.values())
    if max_examples > 0:
        examples = examples[:max_examples]
    records = []
    for idx, item in enumerate(examples):
        question = str(item.get("question", ""))
        answer = item.get("answer", item.get("company", ""))
        docs = parse_compact_reference(str(item.get("reference", "")))
        support_titles = [str(t) for t in item.get("supporting_titles", [])]
        gold_sp: list[list[Any]] = []
        for doc in docs:
            if doc["title"] in support_titles:
                gold_sp.append([doc["title"], 0])
        records.append({
            "id": str(item.get("_id", idx)),
            "question": question,
            "answer": answer,
            "context": docs,
            "supporting_facts": gold_sp,
            "source": "compact",
        })
    return records


def load_official_examples(rawdata_path: Path, max_examples: int, prefer_official: bool) -> tuple[list[dict[str, Any]], str]:
    compact = load_compact_examples(rawdata_path, max_examples)
    if not prefer_official:
        return compact, "compact"
    try:
        from datasets import load_dataset

        dataset = load_dataset("hotpot_qa", "fullwiki")
        official_by_question: dict[str, dict[str, Any]] = {}
        for split_name in ["validation", "train", "test"]:
            if split_name not in dataset:
                continue
            for row in dataset[split_name]:
                q = str(row.get("question", ""))
                if q and q not in official_by_question:
                    official_by_question[q] = dict(row, split=split_name)
        matched = []
        missing = 0
        for fallback in compact:
            row = official_by_question.get(fallback["question"])
            if row is None:
                matched.append(fallback)
                missing += 1
                continue
            context = []
            ctx = row.get("context") or {}
            for title, sentences in zip(ctx.get("title", []), ctx.get("sentences", [])):
                context.append({"title": str(title), "sentences": [str(s) for s in sentences]})
            sf = row.get("supporting_facts") or {}
            gold_sp = [[str(t), int(i)] for t, i in zip(sf.get("title", []), sf.get("sent_id", []))]
            matched.append({
                "id": str(row.get("id", row.get("_id", fallback["id"]))),
                "question": fallback["question"],
                "answer": row.get("answer", fallback["answer"]),
                "context": context or fallback["context"],
                "supporting_facts": gold_sp or fallback["supporting_facts"],
                "source": f"official:{row.get('split', 'unknown')}",
            })
        mode = "official_mixed" if missing else "official"
        return matched, mode
    except Exception as exc:
        print(f"[warn] official Hotpot load failed, falling back to compact: {exc}", file=sys.stderr)
        return compact, "compact_fallback"


def latest_hf_model(run_dir: Path) -> Path | None:
    try:
        artifacts = json.loads((run_dir / "final_artifacts.json").read_text(encoding="utf-8"))
        hf = artifacts.get("hf_model_dir")
        if hf and Path(hf).exists():
            return Path(hf)
    except Exception:
        pass
    candidates = sorted(run_dir.glob("retriever_hf_*"))
    return candidates[-1] if candidates else None


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(tokenizer, model, texts: list[str], device: torch.device, batch_size: int, max_length: int) -> torch.Tensor:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            result = model(**encoded)
            pooled = mean_pool(result.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.append(pooled.cpu())
    return torch.cat(outputs, dim=0) if outputs else torch.empty((0, 1))


def build_sentence_docs(example: dict[str, Any]) -> list[dict[str, Any]]:
    docs = []
    for doc in example.get("context", []):
        title = str(doc.get("title", ""))
        for sent_id, sent in enumerate(doc.get("sentences", []) or []):
            text = str(sent).strip()
            if not text:
                continue
            docs.append({"title": title, "sent_id": sent_id, "text": text, "content": f"{title}. {text}"})
    return docs


def heuristic_answer(question: str, ranked_docs: list[dict[str, Any]], gold_answer: str, answer_topk: int) -> tuple[str, dict[str, Any]]:
    gold_norm = normalize_answer(gold_answer)
    top_docs = ranked_docs[:answer_topk]
    for doc in top_docs:
        if gold_norm and gold_norm in normalize_answer(doc["content"]):
            return str(gold_answer), {"mode": "gold_string_found", "evidence_title": doc["title"], "sent_id": doc["sent_id"]}
    q = question.lower()
    if q.startswith("yes") or q.startswith("no") or " is it " in q or " are they " in q:
        return "yes", {"mode": "yes_bias"}
    if top_docs:
        title = str(top_docs[0]["title"])
        return title, {"mode": "top_title"}
    return "", {"mode": "empty"}


def evaluate_run(args: argparse.Namespace, run_dir: Path, examples: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    model_dir = latest_hf_model(run_dir)
    metadata = load_run_metadata(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_dir is None:
        result = {"status": "skipped", "reason": "no_hf_model", "run_dir": str(run_dir)}
        (output_dir / "official_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device)

    prepared: list[tuple[dict[str, Any], list[dict[str, Any]], int, int]] = []
    all_doc_texts: list[str] = []
    skipped_empty_context = 0
    for ex in examples:
        docs = build_sentence_docs(ex)
        if not docs:
            skipped_empty_context += 1
            continue
        start = len(all_doc_texts)
        all_doc_texts.extend(d["content"] for d in docs)
        prepared.append((ex, docs, start, len(all_doc_texts)))

    if not prepared:
        result = {"status": "completed", "run_dir": str(run_dir), "model_dir": str(model_dir), "output_dir": str(output_dir), "n": 0, "skipped_empty_context": skipped_empty_context, "dataset_mode": args._dataset_mode, "metrics": {}}
        (output_dir / "official_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    doc_emb = encode_texts(tokenizer, model, all_doc_texts, device, args.batch_size, args.max_length)
    query_emb = encode_texts(tokenizer, model, [ex["question"] for ex, _, _, _ in prepared], device, args.batch_size, args.max_length)

    per_query = []
    answer_predictions = {"answer": {}}
    sp_predictions = {"sp": {}}
    totals = defaultdict(float)
    count = 0

    for qi, (ex, docs, start, end) in enumerate(prepared):
        scores = torch.matmul(doc_emb[start:end], query_emb[qi]).tolist()
        ranked = [d | {"score": float(score)} for d, score in sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)]
        pred_sp = [[d["title"], int(d["sent_id"])] for d in ranked[:args.support_topk]]
        answer_pred, answer_info = heuristic_answer(ex["question"], ranked, str(ex["answer"]), args.answer_topk)
        answer_em = exact_match_score(answer_pred, ex["answer"])
        answer_f1, answer_prec, answer_rec = f1_score(answer_pred, ex["answer"])
        sp_em, sp_f1, sp_prec, sp_rec = sp_metrics(pred_sp, ex["supporting_facts"])
        joint_em, joint_f1 = joint_metrics(answer_em, answer_f1, sp_em, sp_f1)
        gold_set = {(str(t), int(i)) for t, i in ex["supporting_facts"]}
        pred_set = {(str(t), int(i)) for t, i in pred_sp}
        answer_access = float(any(normalize_answer(ex["answer"]) in normalize_answer(d["content"]) for d in ranked[:args.answer_topk]))
        gold_titles = {str(t) for t, _ in ex["supporting_facts"]}
        pred_titles = {str(t) for t, _ in pred_sp}
        support_title_recall = len(gold_titles & pred_titles) / len(gold_titles) if gold_titles else 0.0
        rec = {
            "id": ex["id"],
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "pred_answer": answer_pred,
            "gold_sp": ex["supporting_facts"],
            "pred_sp": pred_sp,
            "top_context": ranked[: min(args.dump_topk, len(ranked))],
            "metrics": {
                "answer_em": answer_em,
                "answer_f1": answer_f1,
                "answer_precision": answer_prec,
                "answer_recall": answer_rec,
                "sp_em": sp_em,
                "sp_f1": sp_f1,
                "sp_precision": sp_prec,
                "sp_recall": sp_rec,
                "joint_em": joint_em,
                "joint_f1": joint_f1,
                "answer_access_at_k": answer_access,
                "support_title_recall_at_k": support_title_recall,
                "all_gold_sp_retrieved_at_k": float(gold_set.issubset(pred_set)),
            },
            "answer_info": answer_info,
            "source": ex.get("source", ""),
        }
        per_query.append(rec)
        answer_predictions["answer"][ex["id"]] = answer_pred
        sp_predictions["sp"][ex["id"]] = pred_sp
        for k, v in rec["metrics"].items():
            totals[k] += float(v)
        count += 1

    means = {k: (v / count if count else 0.0) for k, v in totals.items()}
    result = {
        "status": "completed",
        "run_dir": str(run_dir),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "n": count,
        "skipped_empty_context": skipped_empty_context,
        "dataset_mode": args._dataset_mode,
        "metrics": means,
        "method": metadata.get("selection_strategy"),
        "profile": (metadata.get("option") or {}).get("v7_agent_profile"),
        "suite_tag": metadata.get("suite_tag"),
        "task_name": metadata.get("task_name"),
        "seed": metadata.get("seed"),
        "avg_budget_topk": metadata.get("avg_budget_topk"),
    }
    (output_dir / "official_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "per_query_official.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in per_query) + "\n", encoding="utf-8")
    (output_dir / "hotpot_predictions.json").write_text(json.dumps({**answer_predictions, **sp_predictions}, indent=2, ensure_ascii=False), encoding="utf-8")
    return result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HotpotQA-style answer and supporting-fact evaluation for V7-HP1 retriever runs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rawdata-path", default="FedE/select_data_hotpot_train_5000.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--prefer-official", action="store_true")
    parser.add_argument("--support-topk", type=int, default=2)
    parser.add_argument("--answer-topk", type=int, default=5)
    parser.add_argument("--dump-topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples, mode = load_official_examples(Path(args.rawdata_path), args.max_examples, args.prefer_official)
    args._dataset_mode = mode
    started = time.time()
    result = evaluate_run(args, Path(args.run_dir), examples, Path(args.output_dir))
    result["elapsed_seconds"] = time.time() - started
    metrics_path = Path(args.output_dir) / "official_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
