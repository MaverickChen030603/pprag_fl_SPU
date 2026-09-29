#!/usr/bin/env python3
"""Create B3F-Fresh question-only records from the frozen split manifest."""
from __future__ import annotations
import argparse,json
from pathlib import Path

SOURCES={"hotpotqa":"hf_hotpot/hotpot_train_v1.json","2wikimultihopqa":"2wikimultihop_train.json","musique":"musique_ans_v1.0_train.jsonl"}
def load(path):
    if path.suffix==".jsonl": return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return json.loads(path.read_text())
def qid(row): return str(row.get("query_id",row.get("_id",row.get("id"))))
def main():
    p=argparse.ArgumentParser();p.add_argument("--base",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    m=json.loads(a.manifest.read_text()); a.output.mkdir(parents=True)
    summary={"status":"complete_question_only_blind_inputs","labels_loaded":False,"datasets":{}}
    for dataset,entries in m["datasets"].items():
        source={qid(x):x for x in load(a.base/"public_training_sources"/SOURCES[dataset])}; out=[]
        for e in entries:
            row=source[str(e["query_id"])]
            out.append({"dataset":dataset,"query_id":qid(row),"question":str(row["question"])})
        if len(out)!=500 or len({x["query_id"] for x in out})!=500:raise ValueError(f"{dataset}: invalid frozen blind input")
        path=a.output/f"{dataset}_blind_n500.jsonl";path.write_text("".join(json.dumps(x)+"\n" for x in out));summary["datasets"][dataset]={"rows":500,"path":str(path)}
    (a.output/"blind_input_manifest.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
