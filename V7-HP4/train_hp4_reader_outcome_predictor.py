from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


READER = _load("V7-HP4/run_hp4_reader_counterfactual_eval.py", "hp4_reader")
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for reader outcome predictor") from exc


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def tokenize(text: str) -> set[str]:
    return {t.lower() for t in READER.re.findall(r"[A-Za-z0-9][A-Za-z0-9_'-]*", text or "")}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows.extend(payload)
    return rows


def doc_maps(validation: Path, sample_size: int) -> dict[str, dict[str, Any]]:
    out = {}
    for case, docs in READER.materialize_dev(validation, sample_size):
        out[str(case.get("id", case.get("_id", "")))] = {
            "case": case,
            "docs": {doc.doc_id: doc for doc in docs},
        }
    return out


def text_for_ids(ids: list[str], docs: dict[str, Any]) -> list[str]:
    texts = []
    for doc_id in ids:
        doc = docs.get(doc_id)
        if doc is not None:
            texts.append(f"{doc.title}. {doc.text}")
    return texts


def lexical_stats(question: str, ids: list[str], docs: dict[str, Any]) -> dict[str, float]:
    q = tokenize(question)
    vals = [jaccard(q, tokenize(text)) for text in text_for_ids(ids, docs)]
    if not vals:
        return {"mean": 0.0, "max": 0.0, "first": 0.0}
    return {"mean": sum(vals) / len(vals), "max": max(vals), "first": vals[0]}


def displacement(base_ids: list[str], cand_ids: list[str]) -> float:
    pos = {doc_id: idx for idx, doc_id in enumerate(base_ids)}
    moves = []
    for idx, doc_id in enumerate(cand_ids):
        if doc_id in pos:
            moves.append(abs(idx - pos[doc_id]) / max(len(base_ids) - 1, 1))
    return sum(moves) / len(moves) if moves else 1.0


def make_features(base: dict[str, Any], cand: dict[str, Any], meta: dict[str, Any]) -> dict[str, float]:
    docs = meta["docs"]
    question = str(meta["case"].get("question", ""))
    base_ids = [str(x) for x in base.get("top_doc_ids", [])]
    cand_ids = [str(x) for x in cand.get("top_doc_ids", [])]
    base_set, cand_set = set(base_ids), set(cand_ids)
    base_lex = lexical_stats(question, base_ids, docs)
    cand_lex = lexical_stats(question, cand_ids, docs)
    prefix2_same = float(base_ids[:2] == cand_ids[:2])
    prefix3_same = float(base_ids[:3] == cand_ids[:3])
    added = [doc_id for doc_id in cand_ids if doc_id not in base_set]
    removed = [doc_id for doc_id in base_ids if doc_id not in cand_set]
    added_lex = lexical_stats(question, added, docs)
    removed_lex = lexical_stats(question, removed, docs)
    overlap = len(base_set & cand_set) / max(len(base_set), 1)
    return {
        "overlap_at5": overlap,
        "num_added": float(len(added)),
        "num_removed": float(len(removed)),
        "prefix2_same": prefix2_same,
        "prefix3_same": prefix3_same,
        "avg_displacement": displacement(base_ids, cand_ids),
        "base_lex_mean": base_lex["mean"],
        "base_lex_max": base_lex["max"],
        "base_lex_first": base_lex["first"],
        "cand_lex_mean": cand_lex["mean"],
        "cand_lex_max": cand_lex["max"],
        "cand_lex_first": cand_lex["first"],
        "delta_lex_mean": cand_lex["mean"] - base_lex["mean"],
        "delta_lex_max": cand_lex["max"] - base_lex["max"],
        "delta_lex_first": cand_lex["first"] - base_lex["first"],
        "added_lex_mean": added_lex["mean"],
        "added_lex_max": added_lex["max"],
        "removed_lex_mean": removed_lex["mean"],
        "removed_lex_max": removed_lex["max"],
    }


def build_dataset(
    rows: list[dict[str, Any]],
    baseline_mode: str,
    metas: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    by_mode_id = {(str(r.get("mode")), str(r.get("id"))): r for r in rows}
    baseline_by_id = {qid: r for (mode, qid), r in by_mode_id.items() if mode == baseline_mode}
    feature_names: list[str] = []
    samples = []
    for (mode, qid), cand in by_mode_id.items():
        if mode == baseline_mode or qid not in baseline_by_id or qid not in metas:
            continue
        if "answer_f1" not in cand:
            continue
        base = baseline_by_id[qid]
        features = make_features(base, cand, metas[qid])
        if not feature_names:
            feature_names = list(features)
        d_answer = float(cand["answer_f1"]) - float(base["answer_f1"])
        d_joint = float(cand["joint_f1"]) - float(base["joint_f1"])
        samples.append({
            "id": qid,
            "mode": mode,
            "features": features,
            "label_safe_answer": int(d_answer >= -1e-12),
            "label_safe_joint": int(d_joint >= -1e-12),
            "delta_answer_f1": d_answer,
            "delta_joint_f1": d_joint,
            "delta_support_recall_at_k": float(cand.get("support_recall_at_k", 0.0)) - float(base.get("support_recall_at_k", 0.0)),
            "delta_sp_f1": float(cand.get("sp_f1", 0.0)) - float(base.get("sp_f1", 0.0)),
            "candidate_metrics": cand,
            "baseline_metrics": base,
        })
    return samples, feature_names


def vectorize(samples: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.array([[float(s["features"].get(name, 0.0)) for name in feature_names] for s in samples], dtype=np.float32)


def train_model(train: list[dict[str, Any]], feature_names: list[str], label: str):
    y = np.array([int(s[label]) for s in train], dtype=np.int64)
    if len(set(y.tolist())) < 2:
        return None, None
    x = vectorize(train, feature_names)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    # RF handles feature interactions like "same prefix but new tail doc"; LR
    # remains useful for feature-sign audit, so keep both and prefer RF.
    model = RandomForestClassifier(n_estimators=160, max_depth=5, min_samples_leaf=3, random_state=7, class_weight="balanced")
    model.fit(x_scaled, y)
    return scaler, model


def predict_safe_leave_one_query(samples: list[dict[str, Any]], feature_names: list[str]) -> list[dict[str, Any]]:
    out = []
    qids = sorted({s["id"] for s in samples})
    for qid in qids:
        train = [s for s in samples if s["id"] != qid]
        test = [s for s in samples if s["id"] == qid]
        scaler, model = train_model(train, feature_names, "label_safe_answer")
        if model is None:
            continue
        x = scaler.transform(vectorize(test, feature_names))
        probs = model.predict_proba(x)[:, 1]
        for sample, prob in zip(test, probs):
            row = dict(sample)
            row["pred_safe_prob"] = float(prob)
            row["pred_safe"] = bool(prob >= 0.55)
            out.append(row)
    return out


def simulate_guard(preds: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in preds:
        by_q[row["id"]].append(row)
    chosen = []
    for qid, rows in by_q.items():
        safe = [r for r in rows if r["pred_safe_prob"] >= threshold]
        if not safe:
            base = rows[0]["baseline_metrics"]
            chosen.append({
                "id": qid,
                "mode": "baseline_fallback",
                "used_candidate": False,
                "pred_safe_prob": 0.0,
                "answer_f1": float(base["answer_f1"]),
                "joint_f1": float(base["joint_f1"]),
                "support_recall_at_k": float(base["support_recall_at_k"]),
                "sp_f1": float(base["sp_f1"]),
                "delta_answer_f1": 0.0,
                "delta_joint_f1": 0.0,
                "delta_support_recall_at_k": 0.0,
                "delta_sp_f1": 0.0,
            })
            continue
        # Prefer predicted-safe candidates with stronger support/joint proxy, but
        # still avoid low safety confidence.
        best = max(safe, key=lambda r: (r["pred_safe_prob"], r["delta_sp_f1"], r["delta_support_recall_at_k"]))
        cand = best["candidate_metrics"]
        chosen.append({
            "id": qid,
            "mode": best["mode"],
            "used_candidate": True,
            "pred_safe_prob": best["pred_safe_prob"],
            "answer_f1": float(cand["answer_f1"]),
            "joint_f1": float(cand["joint_f1"]),
            "support_recall_at_k": float(cand["support_recall_at_k"]),
            "sp_f1": float(cand["sp_f1"]),
            "delta_answer_f1": best["delta_answer_f1"],
            "delta_joint_f1": best["delta_joint_f1"],
            "delta_support_recall_at_k": best["delta_support_recall_at_k"],
            "delta_sp_f1": best["delta_sp_f1"],
        })
    n = len(chosen)
    return {
        "threshold": threshold,
        "n": n,
        "used_candidate_rate": sum(1 for r in chosen if r["used_candidate"]) / max(n, 1),
        "answer_f1": sum(r["answer_f1"] for r in chosen) / max(n, 1),
        "joint_f1": sum(r["joint_f1"] for r in chosen) / max(n, 1),
        "support_recall_at_k": sum(r["support_recall_at_k"] for r in chosen) / max(n, 1),
        "sp_f1": sum(r["sp_f1"] for r in chosen) / max(n, 1),
        "delta_answer_f1": sum(r["delta_answer_f1"] for r in chosen) / max(n, 1),
        "delta_joint_f1": sum(r["delta_joint_f1"] for r in chosen) / max(n, 1),
        "delta_support_recall_at_k": sum(r["delta_support_recall_at_k"] for r in chosen) / max(n, 1),
        "delta_sp_f1": sum(r["delta_sp_f1"] for r in chosen) / max(n, 1),
        "chosen": chosen,
    }


def model_quality(preds: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    y = np.array([int(r["label_safe_answer"]) for r in preds])
    yhat = np.array([int(r["pred_safe_prob"] >= threshold) for r in preds])
    precision, recall, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, yhat)),
        "precision_safe": float(precision),
        "recall_safe": float(recall),
        "f1_safe": float(f1),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "pred_positive_rate": float(yhat.mean()) if len(yhat) else 0.0,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# V7-HP4 Reader Outcome Predictor / Rerank Classifier",
        "",
        "- target: predict whether a candidate rerank will not reduce answer_f1 versus baseline",
        "- training signal: existing 100-sample reader outcomes across Phase 3 configs",
        "- no-leak features: context overlap, order displacement, prefix stability, and query-doc lexical stats",
        "- policy: accept predicted-safe rerank candidates, otherwise fallback to baseline context",
        "",
        "## Dataset",
        "",
        f"- samples: {payload['dataset']['samples']}",
        f"- queries: {payload['dataset']['queries']}",
        f"- modes: {payload['dataset']['modes']}",
        f"- safe_answer_rate: {payload['dataset']['safe_answer_rate']:.4f}",
        "",
        "## Classifier",
        "",
        "| threshold | accuracy | precision_safe | recall_safe | f1_safe | pred_safe_rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for q in payload["quality"]:
        lines.append(
            f"| {q['threshold']:.2f} | {q['accuracy']:.4f} | {q['precision_safe']:.4f} | "
            f"{q['recall_safe']:.4f} | {q['f1_safe']:.4f} | {q['pred_positive_rate']:.4f} |"
        )
    lines.extend([
        "",
        "## Guarded Rerank Simulation",
        "",
        "| threshold | used_candidate | answer_f1 | joint_f1 | recall@5 | sp_f1 | d_answer | d_joint | d_recall |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for sim in payload["simulations"]:
        lines.append(
            f"| {sim['threshold']:.2f} | {sim['used_candidate_rate']:.4f} | {sim['answer_f1']:.4f} | "
            f"{sim['joint_f1']:.4f} | {sim['support_recall_at_k']:.4f} | {sim['sp_f1']:.4f} | "
            f"{sim['delta_answer_f1']:+.4f} | {sim['delta_joint_f1']:+.4f} | {sim['delta_support_recall_at_k']:+.4f} |"
        )
    lines.extend(["", "## Decision", ""])
    best = payload["best_simulation"]
    if best["delta_answer_f1"] >= -1e-12 and best["delta_joint_f1"] >= -1e-12:
        lines.append(f"Proceed to 1000 validation with guarded rerank threshold {best['threshold']:.2f}.")
    else:
        lines.append(
            f"Do not launch 1000 yet. Best guarded simulation still has "
            f"d_answer={best['delta_answer_f1']:+.4f}, d_joint={best['delta_joint_f1']:+.4f}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", default="V7-HP4/data/hotpot_validation_1000.json")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--output-root", default="V7-HP4/outputs/hp4_phase3_reader_outcome_predictor")
    parser.add_argument("--report-dir", default="实验分析报告/V7-HP4")
    parser.add_argument("--baseline-mode", default="A_phase2_baseline_tau1")
    parser.add_argument("--row-path", action="append", default=[
        "V7-HP4/outputs/hp4_phase3_100_batch/phase3_100_rows.json",
        "V7-HP4/outputs/hp4_phase3_reader_gate_check/reader_gate_rows.json",
        "V7-HP4/outputs/hp4_phase3_reader_gate_check_tau1_gate4/reader_gate_rows.json",
        "V7-HP4/outputs/hp4_phase3_reader_aware_rerank/reader_aware_rerank_rows.json",
        "V7-HP4/outputs/hp4_phase3_rerank_ablation/rerank_ablation_rows.json",
        "V7-HP4/outputs/hp4_phase3_support3_anchor2/support3_anchor2_rows.json",
        "V7-HP4/outputs/hp4_phase3_anchor_preserve_orderlock/anchor_preserve_rows.json",
    ])
    args = parser.parse_args()

    rows = load_rows([Path(p) for p in args.row_path])
    metas = doc_maps(Path(args.validation), args.sample_size)
    samples, feature_names = build_dataset(rows, args.baseline_mode, metas)
    preds = predict_safe_leave_one_query(samples, feature_names)
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    quality = [model_quality(preds, t) for t in thresholds]
    simulations = [simulate_guard(preds, t) for t in thresholds]
    # Prioritize answer safety first, then joint/support.
    best = max(simulations, key=lambda s: (s["delta_answer_f1"] >= -1e-12, s["delta_joint_f1"], s["delta_support_recall_at_k"], -s["threshold"]))
    payload = {
        "dataset": {
            "samples": len(samples),
            "queries": len({s["id"] for s in samples}),
            "modes": len({s["mode"] for s in samples}),
            "safe_answer_rate": sum(s["label_safe_answer"] for s in samples) / max(len(samples), 1),
            "feature_names": feature_names,
        },
        "quality": quality,
        "simulations": [{k: v for k, v in sim.items() if k != "chosen"} for sim in simulations],
        "best_simulation": {k: v for k, v in best.items() if k != "chosen"},
        "best_chosen": best["chosen"],
    }
    out = Path(args.output_root)
    save_json(out / "reader_outcome_samples.json", samples)
    save_json(out / "reader_outcome_predictions.json", preds)
    save_json(out / "reader_outcome_predictor_summary.json", payload)
    report = Path(args.report_dir) / "v7_hp4_phase3_reader_outcome_predictor_latest.md"
    write_report(report, payload)
    print(json.dumps({**payload, "report_path": str(report), "output_root": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
