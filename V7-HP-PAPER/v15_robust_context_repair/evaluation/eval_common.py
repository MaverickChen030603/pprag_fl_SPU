from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


ARTICLES = {"a", "an", "the"}


def normalize_answer(value: str) -> str:
    value = str(value).lower()
    value = "".join(character for character in value if character not in string.punctuation)
    return " ".join(token for token in value.split() if token not in ARTICLES)


def answer_metrics(prediction: str, gold: str) -> dict[str, float]:
    predicted, truth = normalize_answer(prediction).split(), normalize_answer(gold).split()
    common = Counter(predicted) & Counter(truth)
    overlap = sum(common.values())
    precision = overlap / len(predicted) if predicted else float(predicted == truth)
    recall = overlap / len(truth) if truth else float(predicted == truth)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"em": float(normalize_answer(prediction) == normalize_answer(gold)), "precision": precision, "recall": recall, "f1": f1}


def normalize_title(value: str) -> str:
    return " ".join(str(value).lower().split())


def gold_support(row: dict[str, Any]) -> set[tuple[str, int]]:
    facts = row.get("supporting_facts", [])
    if isinstance(facts, dict):
        return {(normalize_title(title), int(sent_id)) for title, sent_id in zip(facts.get("title", []), facts.get("sent_id", []))}
    return {(normalize_title(item[0]), int(item[1])) for item in facts if isinstance(item, (list, tuple)) and len(item) >= 2}


def support_metrics(prediction: set[tuple[str, int]], gold: set[tuple[str, int]]) -> dict[str, float]:
    true_positive = len(prediction & gold)
    precision = true_positive / len(prediction) if prediction else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"em": float(prediction == gold), "precision": precision, "recall": recall, "f1": f1}


def official_metrics(prediction: str, answer: str, support_prediction: set[tuple[str, int]], support_gold: set[tuple[str, int]]) -> dict[str, float]:
    answer_score = answer_metrics(prediction, answer)
    support_score = support_metrics(support_prediction, support_gold)
    joint_precision = answer_score["precision"] * support_score["precision"]
    joint_recall = answer_score["recall"] * support_score["recall"]
    joint_f1 = 2 * joint_precision * joint_recall / (joint_precision + joint_recall) if joint_precision + joint_recall else 0.0
    return {"answer_em": answer_score["em"], "answer_f1": answer_score["f1"], "sp_em": support_score["em"], "sp_f1": support_score["f1"], "joint_em": answer_score["em"] * support_score["em"], "joint_f1": joint_f1}


def words(value: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9]+", str(value).lower()) if len(token) > 1]


def overlap(left: str, right: str) -> float:
    a, b = set(words(left)), set(words(right))
    return len(a & b) / len(a) if a else 0.0


def sentence_features(question: str, title: str, sentence: str, doc_rank: int, sent_id: int, count: int) -> list[float]:
    question_tokens, sentence_tokens, title_tokens = set(words(question)), set(words(sentence)), set(words(title))
    union = question_tokens | sentence_tokens
    capitals_q = set(re.findall(r"\b[A-Z][A-Za-z0-9'-]+", question))
    capitals_s = set(re.findall(r"\b[A-Z][A-Za-z0-9'-]+", f"{title} {sentence}"))
    return [
        len(question_tokens & sentence_tokens) / len(question_tokens) if question_tokens else 0.0,
        len(question_tokens & title_tokens) / len(question_tokens) if question_tokens else 0.0,
        len(question_tokens & sentence_tokens) / len(union) if union else 0.0,
        len(capitals_q & capitals_s) / len(capitals_q | capitals_s) if capitals_q and capitals_s else 0.0,
        float(sent_id == 0), sent_id / max(1, count - 1), doc_rank / 4.0,
        min(len(sentence_tokens), 80) / 80.0,
        len(title_tokens & sentence_tokens) / len(title_tokens) if title_tokens else 0.0,
    ]

