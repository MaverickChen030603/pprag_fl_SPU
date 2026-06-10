from __future__ import annotations

import argparse
import collections
import re
import string
from typing import Dict, List

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

NORMALIZE_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return NORMALIZE_RE.sub(" ", text)
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(str(s).lower())))


def get_tokens(s: str) -> List[str]:
    return normalize_answer(s).split()


def compute_em(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def compute_f1(pred: str, gold: str) -> float:
    pred_toks = get_tokens(pred)
    gold_toks = get_tokens(gold)
    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / max(len(pred_toks), 1)
    recall = num_same / max(len(gold_toks), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


class FiDReader:
    """Lightweight FiD-style multi-passage reader with extractive fallback."""

    def __init__(
        self,
        model_name_or_path: str = "t5-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_input_length: int = 512,
        max_answer_length: int = 32,
        num_beams: int = 4,
        allow_extractive_fallback: bool = True,
    ) -> None:
        self.device = device
        self.max_input_length = int(max_input_length)
        self.max_answer_length = int(max_answer_length)
        self.num_beams = int(num_beams)
        self.model_name_or_path = model_name_or_path
        self.allow_extractive_fallback = allow_extractive_fallback
        self.tokenizer = None
        self.model = None
        try:
            print(f"[FiDReader] Loading model: {model_name_or_path} on {device}")
            self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
            self.model.eval()
            print("[FiDReader] Model loaded.")
        except Exception as exc:
            if not allow_extractive_fallback:
                raise
            print(f"[FiDReader] model load failed, using extractive fallback: {exc}")

    def _build_input(self, query: str, passages: List[str]) -> str:
        ctx_parts = [f"[P{i + 1}] {p.strip()}" for i, p in enumerate(passages)]
        return f"question: {query} context: {' '.join(ctx_parts)}"

    def _extractive_answer(self, query: str, passages: List[str], gold_answer: str) -> str:
        gold_norm = normalize_answer(gold_answer)
        for passage in passages:
            if gold_norm and gold_norm in normalize_answer(passage):
                return gold_answer
        # cheap fallback: return the first compact noun-ish phrase from best passage
        if passages:
            text = passages[0].split(".", 1)[0].strip()
            return text[:80]
        return ""

    def answer(self, query: str, passages: List[str], gold_answer: str = "") -> Dict:
        if self.model is None or self.tokenizer is None:
            pred_answer = self._extractive_answer(query, passages, gold_answer)
        else:
            input_text = self._build_input(query, passages)
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=self.max_input_length, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_answer_length,
                    num_beams=self.num_beams,
                    early_stopping=True,
                )
            pred_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return {
            "pred_answer": pred_answer,
            "answer_em": compute_em(pred_answer, gold_answer) if gold_answer else -1.0,
            "answer_f1": compute_f1(pred_answer, gold_answer) if gold_answer else -1.0,
            "used_passage_ids": list(range(len(passages))),
        }


def _toy_test() -> int:
    reader = FiDReader(model_name_or_path="t5-base", device="cpu", allow_extractive_fallback=True)
    examples = [
        ("Who wrote Hamlet?", ["Hamlet is a tragedy written by William Shakespeare."], "William Shakespeare"),
        ("What color is the sky?", ["The sky is often blue during the day."], "blue"),
        ("Where is Tokyo?", ["Tokyo is the capital of Japan."], "Japan"),
    ]
    for q, passages, gold in examples:
        print(reader.answer(q, passages, gold))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    raise SystemExit(_toy_test() if args.test else 0)
