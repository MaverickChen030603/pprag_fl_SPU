#!/usr/bin/env python3
"""Freeze R2 train/dev/calibration/holdout splits before router development."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any, Iterable

SPECS = {"router_train": ("train", 0, 5000), "router_dev": ("calibration", 0, 300),
         "router_calibration": ("calibration", 300, 200), "router_holdout": ("development", 400, 500)}
def rows(path: Path) -> Iterable[dict[str,Any]]:
    with path.open(encoding="utf-8") as h:
        for line in h:
            if line.strip(): yield json.loads(line)
def qid(row: dict[str,Any]) -> str: return str(row.get("query_id",row.get("_id",row.get("id"))))
def digest(ids: list[str]) -> str: return hashlib.sha256("\n".join(ids).encode()).hexdigest()
def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--dataset",choices=("2wikimultihopqa","musique"),required=True); p.add_argument("--data-root",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); args=p.parse_args()
    payload={"dataset":args.dataset,"final_test_accessed":False,"u1_analysis_development_indices_zero_based":[100,400],"splits":{}}
    args.output_root.mkdir(parents=True,exist_ok=True)
    for name,(source,start,count) in SPECS.items():
        selected=list(rows(args.data_root/args.dataset/f"{source}.jsonl"))[start:start+count]
        if len(selected)!=count: raise ValueError(f"{name}: {len(selected)} != {count}")
        out=args.output_root/f"{name}.jsonl"
        with out.open("w",encoding="utf-8") as h:
            for row in selected: h.write(json.dumps(row,ensure_ascii=False)+"\n")
        ids=[qid(row) for row in selected]
        payload["splits"][name]={"source_split":source,"start_index_zero_based":start,"count":count,"query_ids":ids,"query_id_sha256":digest(ids),"path":str(out.resolve())}
    hold=set(payload["splits"]["router_holdout"]["query_ids"]); u1={qid(x) for x in list(rows(args.data_root/args.dataset/"development.jsonl"))[100:400]}
    payload["u1_analysis_overlap_with_router_holdout"]=len(hold & u1)
    if payload["u1_analysis_overlap_with_router_holdout"]: raise AssertionError("holdout overlaps U1 analysis")
    (args.output_root/"router_split_manifest.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"dataset":args.dataset,"holdout_overlap":0,"splits":{k:v["count"] for k,v in payload["splits"].items()}},indent=2))
if __name__=="__main__": main()
