from __future__ import annotations

import hashlib
import re
import string
from collections import Counter
from typing import Any


ARTICLES = {"a", "an", "the"}


def normalize_answer(value: str) -> str:
    value = "".join(character for character in str(value).lower() if character not in string.punctuation)
    return " ".join(token for token in value.split() if token not in ARTICLES)


def answer_metrics(prediction: str, golds: list[str]) -> dict[str, float]:
    scores = []
    for gold in golds or [""]:
        predicted, truth = normalize_answer(prediction).split(), normalize_answer(gold).split()
        overlap = sum((Counter(predicted) & Counter(truth)).values())
        precision = overlap / len(predicted) if predicted else float(predicted == truth)
        recall = overlap / len(truth) if truth else float(predicted == truth)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append({"em": float(normalize_answer(prediction) == normalize_answer(gold)), "precision": precision, "recall": recall, "f1": f1})
    return max(scores, key=lambda row: (row["f1"], row["em"]))


def normalize_title(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def words(value: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9]+", str(value).lower()) if len(token) > 1]


def document_id(dataset: str, title: str, text: str = "") -> str:
    identity = normalize_title(title) if dataset != "musique" else normalize_title(title) + "\n" + normalize_title(text)
    return f"{dataset}:{hashlib.sha1(identity.encode('utf-8')).hexdigest()[:20]}"


def source_documents(row: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    if dataset == "musique":
        return [
            {"doc_id": document_id(dataset, str(value.get("title", "")), str(value.get("paragraph_text", ""))), "title": str(value.get("title", "")), "text": str(value.get("paragraph_text", "")), "paragraph_idx": int(value.get("idx", index)), "source_rank": index}
            for index, value in enumerate(row.get("paragraphs", []))
        ]
    context, pairs = row.get("context", []), []
    if isinstance(context, dict):
        titles, sentences = context.get("title", []), context.get("sentences", [])
        pairs = [(title, sentences[index] if index < len(sentences) else []) for index, title in enumerate(titles)]
    else:
        pairs = [value[:2] for value in context if isinstance(value, (list, tuple)) and len(value) >= 2]
    output = []
    for index, (title, sentences) in enumerate(pairs):
        values = list(map(str, sentences)) if isinstance(sentences, list) else [str(sentences)]
        text = " ".join(values)
        output.append({"doc_id": document_id(dataset, str(title)), "title": str(title), "text": text, "sentences": values, "source_rank": index})
    return output


def gold_support(row: dict[str, Any], dataset: str) -> set[tuple[str, int]]:
    if dataset == "musique":
        return {("paragraph", int(value.get("idx", index))) for index, value in enumerate(row.get("paragraphs", [])) if value.get("is_supporting", False)}
    facts = row.get("supporting_facts", [])
    if isinstance(facts, dict):
        return {(normalize_title(title), int(sent_id)) for title, sent_id in zip(facts.get("title", []), facts.get("sent_id", []))}
    return {(normalize_title(value[0]), int(value[1])) for value in facts if isinstance(value, (list, tuple)) and len(value) >= 2}


def support_metrics(prediction: set[tuple[str, int]], gold: set[tuple[str, int]]) -> dict[str, float]:
    true_positive = len(prediction & gold)
    precision = true_positive / len(prediction) if prediction else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"em": float(prediction == gold), "precision": precision, "recall": recall, "f1": f1}


def official_metrics(prediction: str, row: dict[str, Any], support_prediction: set[tuple[str, int]], dataset: str) -> dict[str, Any]:
    golds = [str(row.get("answer", ""))] + [str(value) for value in row.get("answer_aliases", [])]
    answer = answer_metrics(prediction, golds)
    support = support_metrics(support_prediction, gold_support(row, dataset))
    joint_precision = answer["precision"] * support["precision"]
    joint_recall = answer["recall"] * support["recall"]
    joint_f1 = 2 * joint_precision * joint_recall / (joint_precision + joint_recall) if joint_precision + joint_recall else 0.0
    return {
        "answer_em": answer["em"], "answer_f1": answer["f1"], "sp_em": support["em"], "sp_f1": support["f1"],
        "joint_em": answer["em"] * support["em"], "joint_f1": joint_f1,
        "joint_metric_kind": "official_hotpot_style" if dataset in {"hotpotqa", "2wikimultihopqa"} else "constructed_answer_support_composite",
    }


def unit_features(question: str, title: str, text: str, doc_rank: int, unit_rank: int, unit_count: int) -> list[float]:
    question_tokens, text_tokens, title_tokens = set(words(question)), set(words(text)), set(words(title))
    union = question_tokens | text_tokens
    capitals_q = set(re.findall(r"\b[A-Z][A-Za-z0-9'-]+", question))
    capitals_d = set(re.findall(r"\b[A-Z][A-Za-z0-9'-]+", f"{title} {text}"))
    return [
        len(question_tokens & text_tokens) / len(question_tokens) if question_tokens else 0.0,
        len(question_tokens & title_tokens) / len(question_tokens) if question_tokens else 0.0,
        len(question_tokens & text_tokens) / len(union) if union else 0.0,
        len(capitals_q & capitals_d) / len(capitals_q | capitals_d) if capitals_q and capitals_d else 0.0,
        float(unit_rank == 0), unit_rank / max(1, unit_count - 1), min(doc_rank, 4) / 4.0,
        min(len(text_tokens), 120) / 120.0,
        len(title_tokens & text_tokens) / len(title_tokens) if title_tokens else 0.0,
    ]
