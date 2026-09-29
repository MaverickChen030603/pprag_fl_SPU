#!/usr/bin/env python3
"""Create source-document-only Reader inputs; remove all answer/support labels."""
from __future__ import annotations
import argparse,json
from pathlib import Path
S={"hotpotqa":"hf_hotpot/hotpot_train_v1.json","2wikimultihopqa":"2wikimultihop_train.json","musique":"musique_ans_v1.0_train.jsonl"}
DROP={"answer","answers","supporting_facts","supporting_paragraphs","is_supporting","is_support"}
def rows(p):
 if p.suffix==".jsonl":
  with p.open() as h:
   for x in h:
    if x.strip():yield json.loads(x)
 else:yield from json.loads(p.read_text())
def q(r):return str(r.get("query_id",r.get("_id",r.get("id"))))
def scrub(x):
 if isinstance(x,dict):return {k:scrub(v) for k,v in x.items() if k not in DROP}
 if isinstance(x,list):return [scrub(v) for v in x]
 return x
def main():
 p=argparse.ArgumentParser();p.add_argument("--experiment-root",type=Path,required=True);p.add_argument("--stage",type=Path,required=True);a=p.parse_args();out=a.stage/"reader_inputs"
 if out.exists():raise FileExistsError(out)
 out.mkdir();m=json.loads((a.stage/"splits/b3fm_fresh_split_manifest.json").read_text())
 for d,entries in m["datasets"].items():
  want={str(x["query_id"]) for x in entries};found={q(r):scrub(r) for r in rows(a.experiment_root/"public_training_sources"/S[d]) if q(r) in want}
  if set(found)!=want:raise ValueError(d)
  (out/f"{d}_reader_inputs_n500.jsonl").write_text("".join(json.dumps(found[str(x["query_id"])])+"\n" for x in entries))
 (out/"manifest.json").write_text(json.dumps({"labels_loaded":False,"labels_removed":sorted(DROP),"datasets":list(m["datasets"])}))
if __name__=="__main__":main()
