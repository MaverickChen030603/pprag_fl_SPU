#!/usr/bin/env python3
"""Irreversible C1-D evaluation after the recorded human label unseal."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
READERS = ("flan", "unifiedqa")
METRICS = ("client_complete_at_3", "local_complete_at_10", "transmitted_complete_at_15", "merged_complete_at_10", "reader_context_complete_at_5", "answer_f1", "sp_f1", "joint_f1")
PRIMARY = ("reader_context_complete_at_5", "joint_f1")


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): yield json.loads(line)


def qid(row: dict[str, Any]) -> str: return str(row.get("query_id", row.get("_id", row.get("id"))))


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    fields=sorted({key for row in values for key in row}); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(values)


def load_raw(dataset: str, root: Path, selected: set[str]) -> dict[str, dict[str, Any]]:
    filename={"hotpotqa":"hf_hotpot/hotpot_train_v1.json","2wikimultihopqa":"2wikimultihop_train.json","musique":"musique_ans_v1.0_train.jsonl"}[dataset]
    path=root/filename
    if path.suffix==".jsonl": source=(json.loads(line) for line in path.open() if line.strip())
    else: source=iter(json.loads(path.read_text()))
    found={qid(row):row for row in source if qid(row) in selected}
    if set(found)!=selected: raise ValueError(f"{dataset}: unsealed labels do not match frozen C1 IDs")
    return found


def bootstrap(delta: np.ndarray, seed: int) -> tuple[float,float,float]:
    rng=np.random.default_rng(seed); n=len(delta)
    samples=np.asarray([delta[rng.integers(0,n,n)].mean() for _ in range(5000)])
    p=min(1.,2*(min((samples<=0).sum(),(samples>=0).sum())+1)/(len(samples)+1))
    return float(delta.mean()),float(np.quantile(samples,.025)),float(np.quantile(samples,.975)),float(p)


def randomization(delta: np.ndarray, seed: int) -> float:
    rng=np.random.default_rng(seed); observed=abs(delta.mean()); n=len(delta)
    draws=np.asarray([abs((delta*rng.choice((-1.,1.),size=n)).mean()) for _ in range(5000)])
    return float(((draws>=observed).sum()+1)/(len(draws)+1))


def holm(rows_: list[dict[str, Any]]) -> None:
    ordered=sorted(enumerate(rows_),key=lambda item:item[1]["randomization_p"]); m=len(ordered); running=0.
    for rank,(index,row) in enumerate(ordered):
        running=max(running,min(1.,row["randomization_p"]*(m-rank))); rows_[index]["holm_adjusted_p"]=running


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--repo",type=Path,required=True); parser.add_argument("--base",type=Path,required=True); parser.add_argument("--stage",type=Path,required=True); args=parser.parse_args()
    if not (args.stage/"protocol/c1_label_unseal_record.json").is_file(): raise ValueError("explicit label unseal record is required")
    sys.path.insert(0,str(args.base/"inputs/v16_evaluation")); from eval_common import document_id,official_metrics
    contexts=list(rows(args.stage/"retrieval/blind_retrieval_outputs.jsonl")); cmap={(r["dataset"],r["query_id"],r["method"]):r for r in contexts}
    if len(cmap)!=37500: raise ValueError("frozen retrieval artifact is incomplete")
    methods=sorted({r["method"] for r in contexts}); predictions={}
    for reader in READERS:
        values=list(rows(args.stage/f"readers/{reader}_unscored_predictions.jsonl")); marker=json.loads((args.stage/f"readers/{reader}_unscored_predictions.completed.json").read_text())
        if len(values)!=37500 or marker["prediction_sha256"]!=sha256(args.stage/f"readers/{reader}_unscored_predictions.jsonl"): raise ValueError(f"{reader}: frozen prediction check failed")
        predictions.update({(reader,r["dataset"],r["query_id"],r["method"]):r for r in values})
    selected={d:{r["query_id"] for r in rows(args.stage/f"inputs/{d}_c1_blind_n500.jsonl")} for d in DATASETS}
    gold={d:load_raw(d,args.base/"public_training_sources",selected[d]) for d in DATASETS}
    assignments={d:{str(r["doc_id"]):int(r["client_id"]) for r in rows(args.base/f"inputs/v17/partitions/assignments/{d}/topic_silo_m20.jsonl")} for d in DATASETS}
    packets={d:{str(r["query_id"]):r for r in rows(args.stage/f"retrieval/{d}_probe_packets.jsonl")} for d in DATASETS}
    out=[]
    for d in DATASETS:
        for query_id,source in gold[d].items():
            if d=="musique":
                support={document_id(d,str(x.get("title","")),str(x.get("paragraph_text",""))) for x in source.get("paragraphs",[]) if x.get("is_supporting",x.get("is_support",False))}
            else:
                facts=source.get("supporting_facts",[]); titles=facts.get("title",[]) if isinstance(facts,dict) else [x[0] for x in facts]
                support={document_id(d,str(title)) for title in titles}
            support_clients={assignments[d][x] for x in support if x in assignments[d]}
            for method in methods:
                context=cmap[(d,query_id,method)]; selected_clients=set(map(int,context["selected_client_ids"])); local={str(x["doc_id"]) for client in selected_clients for x in packets[d][query_id]["local_dense_docs_top10"][str(client)][:10]}
                complete=lambda docs:int(bool(support) and support.issubset(docs))
                chain={"client_complete_at_3":int(bool(support_clients) and support_clients.issubset(selected_clients)),"local_complete_at_10":complete(local),"transmitted_complete_at_15":complete(set(context["transmitted_doc_ids"])),"merged_complete_at_10":complete(set(context["merged_top10_doc_ids"])),"reader_context_complete_at_5":complete(set(context["reader_top5_doc_ids"]))}
                for reader in READERS:
                    prediction=predictions[(reader,d,query_id,method)]
                    official=official_metrics(prediction["predicted_answer"],source,{tuple(x) for x in prediction["predicted_support"]},d)
                    out.append({"dataset":d,"reader":reader,"query_id":query_id,"method":method,**chain,"answer_f1":official["answer_f1"],"sp_f1":official["sp_f1"],"joint_f1":official["joint_f1"]})
    output=args.stage/"statistics"; output.mkdir(exist_ok=True)
    write_csv(output/"per_query_results.csv",out)
    summary=[]
    for d in DATASETS:
        for reader in READERS:
            for method in methods:
                values=[r for r in out if r["dataset"]==d and r["reader"]==reader and r["method"]==method]
                summary.append({"dataset":d,"reader":reader,"method":method,"queries":len(values),**{metric:float(np.mean([r[metric] for r in values])) for metric in METRICS}})
    write_csv(output/"method_results.csv",summary)
    comparisons=[]
    for d in DATASETS:
        order=sorted(selected[d])
        for reader in READERS:
            lookup={(r["method"],r["query_id"]):r for r in out if r["dataset"]==d and r["reader"]==reader}
            for baseline in ("b0_static_top3","b3_ragroute_style_mlp","b6_dense_centroid_top3"):
                for metric in PRIMARY:
                    delta=np.asarray([lookup[("m2_logistic_proberoute",q)][metric]-lookup[(baseline,q)][metric] for q in order],dtype=float)
                    seed=int(hashlib.sha256(f"{d}|{reader}|{baseline}|{metric}".encode()).hexdigest()[:8],16); mean,low,high,boot=bootstrap(delta,seed)
                    comparisons.append({"dataset":d,"reader":reader,"comparison":f"m2_minus_{baseline}","metric":metric,"queries":len(delta),"absolute_delta":mean,"ci_low":low,"ci_high":high,"bootstrap_p":boot,"randomization_p":randomization(delta,seed+1)})
    holm(comparisons); write_csv(output/"primary_comparisons.csv",comparisons)
    manifest={"status":"complete_c1_post_unseal_evaluation","label_unseal_record_sha256":sha256(args.stage/"protocol/c1_label_unseal_record.json"),"rows":len(out),"methods":methods,"metrics":list(METRICS),"primary_comparisons":len(comparisons),"retrieval_sha256":sha256(args.stage/"retrieval/blind_retrieval_outputs.jsonl")}
    (output/"evaluation_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n"); print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
