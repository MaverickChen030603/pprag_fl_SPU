from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path
from typing import Any


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FULL = _load_module("V7-HP4/run_hp4_full_experiment.py", "v7_hp4_full_helpers")
_READER = _load_module("V7-HP4/run_hp4_reader_counterfactual_eval.py", "v7_hp4_reader_helpers")

build_retrieval_item = _READER.build_retrieval_item
materialize_dev = _READER.materialize_dev
HybridSoftRetriever = _READER.HybridSoftRetriever
hp4_weights_for_docs = _FULL.hp4_weights_for_docs


def load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def paired(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        mode = row.get("mode")
        if mode in {"baseline_uniform", "hp4_soft_agent"}:
            out.setdefault(str(row.get("id")), {})[str(mode)] = row
    return out


def bridge_entities(question: str, title: str, text: str) -> str:
    q_ents = {m.group(0).lower() for m in re.finditer(r"\b[A-Z][A-Za-z0-9_'-]{2,}\b", question or "")}
    d_ents = {m.group(0) for m in re.finditer(r"\b[A-Z][A-Za-z0-9_'-]{2,}\b", f"{title} {text}")}
    hits = [ent for ent in d_ents if ent.lower() in q_ents]
    return "; ".join(sorted(hits))


def export_routing_dataframe(examples: list[tuple[dict[str, Any], list[Any]]], out_csv: Path, top_k: int, alpha: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case, docs in examples:
        question = str(case.get("question", ""))
        weights = hp4_weights_for_docs(question, docs, "hp4_soft_agent")
        retriever = HybridSoftRetriever(docs, alpha=alpha)
        baseline = build_retrieval_item(case, docs, "baseline_uniform", top_k, alpha)
        agent = build_retrieval_item(case, docs, "hp4_soft_agent", top_k, alpha)
        baseline_set = set(baseline["top_doc_ids"])
        agent_set = set(agent["top_doc_ids"])
        for idx, doc in enumerate(docs):
            score = retriever.score(question, doc, idx, weights=weights)
            rows.append({
                "query_id": str(case.get("id", case.get("_id", ""))),
                "query": question,
                "answer": str(case.get("answer", "")),
                "gold_supporting_facts": "; ".join(str(t) for t in case.get("supporting_titles", [])),
                "client_id": doc.client_id,
                "doc_id": doc.doc_id,
                "title": doc.title,
                "is_support": int(bool(doc.is_support)),
                "soft_weight": float(weights.get(doc.doc_id, doc.soft_weight)),
                "dense_score": float(score["dense"]),
                "bm25_score": float(score["sparse"]),
                "final_score": float(score["final"]),
                "bridge_entities": bridge_entities(question, doc.title, doc.text),
                "in_baseline_topk": int(doc.doc_id in baseline_set),
                "in_agent_topk": int(doc.doc_id in agent_set),
                "agent_promoted": int(doc.doc_id not in baseline_set and doc.doc_id in agent_set),
            })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_case_studies(
    examples: list[tuple[dict[str, Any], list[Any]]],
    eval_rows: list[dict[str, Any]],
    routing_rows: list[dict[str, Any]],
    out_path: Path,
    limit: int,
) -> list[dict[str, Any]]:
    by_id = paired(eval_rows)
    route_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in routing_rows:
        route_by_id.setdefault(str(row["query_id"]), []).append(row)

    cases = []
    case_map = {str(case.get("id", case.get("_id", ""))): case for case, _ in examples}
    for qid, modes in by_id.items():
        b = modes.get("baseline_uniform")
        a = modes.get("hp4_soft_agent")
        if not b or not a:
            continue
        b_joint = float(b.get("joint_f1", 0.0))
        a_joint = float(a.get("joint_f1", 0.0))
        b_sp = float(b.get("sp_f1", 0.0))
        a_sp = float(a.get("sp_f1", 0.0))
        if not (b_joint < 1e-9 and a_joint > b_joint and a_sp > b_sp):
            continue
        promoted = [
            r for r in route_by_id.get(qid, [])
            if int(r["is_support"]) == 1 and int(r["agent_promoted"]) == 1
        ]
        if not promoted:
            continue
        promoted.sort(key=lambda r: float(r["soft_weight"]), reverse=True)
        case = case_map.get(qid, {})
        cases.append({
            "query_id": qid,
            "question": case.get("question", ""),
            "answer": case.get("answer", ""),
            "gold_supporting_facts": case.get("supporting_titles", []),
            "baseline_top_titles": b.get("top_titles", []),
            "agent_top_titles": a.get("top_titles", []),
            "baseline_joint_f1": b_joint,
            "agent_joint_f1": a_joint,
            "baseline_sp_f1": b_sp,
            "agent_sp_f1": a_sp,
            "promoted_support_blocks": promoted[:3],
        })
        if len(cases) >= limit:
            break

    lines = ["# V7-HP4 Routing Behavior Case Studies", ""]
    for idx, case in enumerate(cases, start=1):
        lines.extend([
            f"## Case {idx}: {case['query_id']}",
            "",
            f"Question: {case['question']}",
            "",
            f"Answer: {case['answer']}",
            "",
            f"Baseline joint_f1/sp_f1: {case['baseline_joint_f1']:.4f}/{case['baseline_sp_f1']:.4f}",
            "",
            f"Soft-Agent joint_f1/sp_f1: {case['agent_joint_f1']:.4f}/{case['agent_sp_f1']:.4f}",
            "",
            f"Gold supporting facts: {case['gold_supporting_facts']}",
            "",
            f"Baseline Top titles: {case['baseline_top_titles']}",
            "",
            f"Soft-Agent Top titles: {case['agent_top_titles']}",
            "",
            "| promoted title | client | soft_weight | bm25 | dense | bridge_entities |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ])
        for block in case["promoted_support_blocks"]:
            lines.append(
                f"| {block['title']} | {block['client_id']} | {float(block['soft_weight']):.4f} | "
                f"{float(block['bm25_score']):.4f} | {float(block['dense_score']):.4f} | {block['bridge_entities']} |"
            )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="V7-HP4/data/hotpot_validation_1000.json")
    parser.add_argument("--eval-rows", default="V7-HP4/outputs/hp4_full_validation/full_validation_reader_rows.json")
    parser.add_argument("--output-root", default="V7-HP4/outputs/hp4_routing_visualization")
    parser.add_argument("--report-dir", default="实验分析报告/V7-HP4")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--max-dev", type=int, default=1000)
    parser.add_argument("--case-limit", type=int, default=5)
    args = parser.parse_args()

    examples = materialize_dev(Path(args.dev), args.max_dev)
    out = Path(args.output_root)
    routing_csv = out / "hp4_routing_dataframe.csv"
    routing_rows = export_routing_dataframe(examples, routing_csv, args.top_k, args.alpha)
    eval_rows = load_rows(Path(args.eval_rows))
    report_path = Path(args.report_dir) / "v7_hp4_phase2_routing_case_studies_latest.md"
    cases = write_case_studies(examples, eval_rows, routing_rows, report_path, args.case_limit)
    (out / "hp4_case_studies.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "routing_csv": str(routing_csv),
        "case_studies_json": str(out / "hp4_case_studies.json"),
        "case_report": str(report_path),
        "cases": len(cases),
        "routing_rows": len(routing_rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
