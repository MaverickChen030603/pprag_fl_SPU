#!/usr/bin/env python3
"""Build Recoverable-Hard 100 from completed V7-HP2 per-query scorecards.

Strict target: baseline joint_f1 == 0 and at least one agent/reader-aware run has
positive joint_f1/sp_f1/answer_access. If strict examples are fewer than the
requested size, fill with medium-hard items where baseline is not perfect and
some run has partial evidence. The metadata records both counts so the paper
analysis can separate strict recoverable from fallback medium-hard cases.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)


def profile_kind(path: Path) -> str:
    name = path.parent.name.lower()
    if "baseline" in name or name.startswith("hypernet") or name.startswith("adaptive"):
        return "baseline"
    return "agent"


def collect_scores(root: Path) -> Dict[str, Dict[str, List[float]]]:
    scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for file in root.glob("**/per_query_*.jsonl"):
        kind = profile_kind(file)
        for rec in iter_jsonl(file):
            m = rec.get("metrics") or {}
            qid = str(rec.get("id"))
            joint = float(m.get("joint_f1", 0.0) or 0.0)
            sp = float(m.get("sp_f1", 0.0) or 0.0)
            access = float(m.get("answer_access_at_k", 0.0) or 0.0)
            answer_f1 = float(m.get("answer_f1", 0.0) or 0.0)
            # A recoverable signal can arrive through official joint, support,
            # or answer accessibility; keep all channels visible.
            scores[qid][f"{kind}_joint"].append(joint)
            scores[qid][f"{kind}_sp"].append(sp)
            scores[qid][f"{kind}_access"].append(access)
            scores[qid][f"{kind}_answer_f1"].append(answer_f1)
    return scores


def load_examples(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for ex in data:
        qid = str(ex.get("id") or ex.get("_id"))
        ex.setdefault("id", qid)
        out[qid] = ex
    return out


def avg(vals: List[float]) -> float:
    return mean(vals) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hp2-official-root", type=Path, default=Path("V7-HP2/outputs/hotpot_official_fullwiki_dev300"))
    ap.add_argument("--hp2-reader-root", type=Path, default=Path("V7-HP2/outputs/hotpot_reader_fullwiki_t5small_dev300"))
    ap.add_argument("--source-dev", type=Path, default=Path("V7-HP2/data/hotpot_dev_stratified_300.json"))
    ap.add_argument("--output", type=Path, default=Path("V7-HP3/data/hotpot_recoverable_hard100.json"))
    ap.add_argument("--meta-output", type=Path, default=Path("V7-HP3/data/hotpot_recoverable_hard100.meta.json"))
    ap.add_argument("--target-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260609)
    args = ap.parse_args()

    official = collect_scores(args.hp2_official_root)
    reader = collect_scores(args.hp2_reader_root)
    ids = set(official) | set(reader)
    examples = load_examples(args.source_dev)

    ranked = []
    strict = []
    fallback = []
    easy = impossible = 0
    for qid in ids:
        o = official.get(qid, {})
        r = reader.get(qid, {})
        baseline_joint = max(avg(o.get("baseline_joint", [])), avg(r.get("baseline_joint", [])))
        agent_joint = max(avg(o.get("agent_joint", [])), avg(r.get("agent_joint", [])))
        baseline_sp = max(avg(o.get("baseline_sp", [])), avg(r.get("baseline_sp", [])))
        agent_sp = max(avg(o.get("agent_sp", [])), avg(r.get("agent_sp", [])))
        baseline_access = max(avg(o.get("baseline_access", [])), avg(r.get("baseline_access", [])))
        agent_access = max(avg(o.get("agent_access", [])), avg(r.get("agent_access", [])))
        best_any = max(agent_joint, agent_sp, agent_access, baseline_joint, baseline_sp, baseline_access)
        if baseline_joint >= 0.99 and baseline_sp >= 0.99:
            easy += 1
            continue
        if best_any <= 0.0:
            impossible += 1
            continue
        gain = max(agent_joint - baseline_joint, agent_sp - baseline_sp, agent_access - baseline_access)
        hardness = (1.0 - baseline_joint) + (1.0 - min(baseline_sp, 1.0)) + 0.5 * (1.0 - min(baseline_access, 1.0))
        item = {
            "id": qid,
            "baseline_joint": baseline_joint,
            "agent_joint": agent_joint,
            "baseline_sp": baseline_sp,
            "agent_sp": agent_sp,
            "baseline_access": baseline_access,
            "agent_access": agent_access,
            "gain": gain,
            "hardness": hardness,
            "score": gain * 5.0 + hardness,
        }
        if baseline_joint <= 1e-9 and (agent_joint > 0.0 or agent_sp > baseline_sp or agent_access > baseline_access):
            strict.append(item)
        elif baseline_joint < 0.5 and best_any > 0.0:
            fallback.append(item)
        ranked.append(item)

    strict.sort(key=lambda x: (x["gain"], x["score"]), reverse=True)
    fallback.sort(key=lambda x: (x["score"], x["gain"]), reverse=True)
    selected = strict[:args.target_size]
    if len(selected) < args.target_size:
        selected_ids = {x["id"] for x in selected}
        selected.extend([x for x in fallback if x["id"] not in selected_ids][: args.target_size - len(selected)])
    if len(selected) < args.target_size:
        selected_ids = {x["id"] for x in selected}
        rest = sorted([x for x in ranked if x["id"] not in selected_ids], key=lambda x: x["score"], reverse=True)
        selected.extend(rest[: args.target_size - len(selected)])

    rng = random.Random(args.seed)
    selected_examples = [examples[x["id"]] for x in selected if x["id"] in examples]
    rng.shuffle(selected_examples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected_examples, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "source": "V7-HP2 official+reader per-query reverse filter",
        "target_size": args.target_size,
        "actual_size": len(selected_examples),
        "strict_recoverable_count": len(strict),
        "selected_strict_count": sum(1 for x in selected if x in strict),
        "fallback_medium_hard_count": len(fallback),
        "easy_removed": easy,
        "impossible_removed": impossible,
        "selection_preview": selected[:20],
        "rule": "baseline joint_f1 zero plus agent partial recovery preferred; then medium-hard fallback if strict<target",
    }
    args.meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
