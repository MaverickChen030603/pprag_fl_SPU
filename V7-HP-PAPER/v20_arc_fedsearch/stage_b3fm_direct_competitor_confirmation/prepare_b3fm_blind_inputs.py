#!/usr/bin/env python3
"""Materialize question-only B3F-M inputs after the freshness freeze."""
from __future__ import annotations
import argparse, json
from pathlib import Path

SOURCES={"hotpotqa":"hf_hotpot/hotpot_train_v1.json","2wikimultihopqa":"2wikimultihop_train.json","musique":"musique_ans_v1.0_train.jsonl"}
def rows(path):
    if path.suffix==".jsonl":
        with path.open(encoding="utf-8") as h:
            for line in h:
                if line.strip(): yield json.loads(line)
    else: yield from json.loads(path.read_text(encoding="utf-8"))
def qid(row): return str(row.get("query_id",row.get("_id",row.get("id"))))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--experiment-root",type=Path,required=True); p.add_argument("--stage",type=Path,required=True); a=p.parse_args()
    manifest=a.stage/"splits/b3fm_fresh_split_manifest.json"; output=a.stage/"inputs"
    if output.exists(): raise FileExistsError(output)
    frozen=json.loads(manifest.read_text(encoding="utf-8"))
    if frozen["status"]!="frozen_before_retrieval": raise ValueError("freshness freeze failed")
    output.mkdir(); summary={"status":"complete_question_only_blind_inputs","labels_loaded":False,"datasets":{}}
    for dataset,entries in frozen["datasets"].items():
        source={qid(row):row for row in rows(a.experiment_root/"public_training_sources"/SOURCES[dataset])}
        values=[{"dataset":dataset,"query_id":str(e["query_id"]),"question":str(source[str(e["query_id"])]["question"])} for e in entries]
        if len(values)!=500 or len({x["query_id"] for x in values})!=500: raise ValueError(f"{dataset}: invalid frozen split")
        path=output/f"{dataset}_blind_n500.jsonl"; path.write_text("".join(json.dumps(x)+"\n" for x in values),encoding="utf-8")
        summary["datasets"][dataset]={"rows":len(values),"path":str(path)}
    (output/"blind_input_manifest.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
