#!/usr/bin/env python3
"""Offline U1 audit: factor frozen routing from frozen local retrieval quality."""
from __future__ import annotations

import argparse, csv, hashlib, json, re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

RANKERS = ("L0_dense", "L1_bm25", "L2_hybrid", "L3_rrf")
ROUTES = ("R0_actual", "R1_oracle_bc3")
TOKEN = re.compile(r"[A-Za-z0-9]+")

def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as h:
        for line in h:
            if line.strip(): yield json.loads(line)
def qid(row: dict[str, Any]) -> str: return str(row.get("query_id", row.get("_id", row.get("id"))))
def norm(value: str) -> str: return " ".join(str(value).lower().split())
def docid(dataset: str, title: str, text: str="") -> str:
    key = norm(title) if dataset != "musique" else norm(title) + "\n" + norm(text)
    return f"{dataset}:{hashlib.sha1(key.encode()).hexdigest()[:20]}"
def supports(row: dict[str, Any], dataset: str) -> tuple[set[str], list[str]]:
    if dataset == "musique":
        values = [(docid(dataset, x.get("title", ""), x.get("paragraph_text", "")), str(x.get("title", ""))) for x in row.get("paragraphs", []) if x.get("is_supporting", x.get("is_support", False))]
    else:
        facts = row.get("supporting_facts", {})
        titles = facts.get("title", []) if isinstance(facts, dict) else [x[0] for x in facts if x]
        values = [(docid(dataset, title), str(title)) for title in titles]
    return {x[0] for x in values}, [x[1] for x in values]
def assignment(path: Path) -> dict[str, int]: return {str(x["doc_id"]):int(x["client_id"]) for x in rows(path)}
def complete(gold: set[str], docs: list[dict[str, Any]]) -> int: return int(bool(gold) and gold <= {str(d["doc_id"]) for d in docs})
def recall(gold: set[str], docs: list[dict[str, Any]]) -> float: return len(gold & {str(d["doc_id"]) for d in docs}) / max(1,len(gold))
def mean(xs: list[float]) -> float: return float(sum(xs)/len(xs)) if xs else 0.0
def write_csv(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(values[0])); w.writeheader(); w.writerows(values)
def route_oracle(gold_clients: list[int], actual: list[int]) -> list[int]:
    selected = list(gold_clients[:3])
    for client in actual + list(range(20)):
        if len(selected) == 3: break
        if client not in selected: selected.append(client)
    return selected
def local_lists(pool: dict[str, Any], selected: list[int], ranker: str) -> dict[int,list[dict[str,Any]]]:
    return {c:list(pool["rankers"][str(c)][ranker]) for c in selected}
def transmitted(local: dict[int,list[dict[str,Any]]]) -> list[dict[str,Any]]: return [d for c in local for d in local[c][:5]]
def merged(docs: list[dict[str,Any]], kind: str) -> list[dict[str,Any]]:
    if kind == "raw": return sorted(docs, key=lambda d:(-float(d["local_score"]),-float(d["dense_score"]),str(d["doc_id"])))
    if kind == "percentile": return sorted(docs, key=lambda d:(int(d["local_rank"]),-float(d["dense_score"]),str(d["doc_id"])))
    return sorted(docs, key=lambda d:(-(1/(61+int(d["local_rank"]))),-float(d["dense_score"]),str(d["doc_id"])))
def rank_values(gold: set[str], docs: list[dict[str,Any]]) -> tuple[Any,Any,Any]:
    found = [int(d["local_rank"]) for d in docs if str(d["doc_id"]) in gold]
    found.sort()
    return (found[0] if found else None, found[1] if len(found)>1 else None, max(found) if found else None)
def question_type(question: str, titles: list[str]) -> tuple[str,float]:
    q = {x.lower() for x in TOKEN.findall(question)}; t = {x.lower() for title in titles for x in TOKEN.findall(title)}
    overlap = len(q&t)/max(1,len(t))
    low = question.lower()
    if any(x in low for x in ("than", "more", "less", "same", "both", "either")): typ="comparison"
    elif overlap >= .25: typ="lexical_entity_heavy"
    else: typ="bridge_or_indirect"
    return typ,overlap
def ci(delta: np.ndarray, seed: int=20260804) -> tuple[float,float]:
    rng=np.random.default_rng(seed); n=len(delta); samples=np.empty(5000)
    for i in range(5000): samples[i]=delta[rng.integers(0,n,n)].mean()
    return float(np.quantile(samples,.025)),float(np.quantile(samples,.975))

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--dataset",choices=("2wikimultihopqa","musique"),required=True); p.add_argument("--split",type=Path,required=True); p.add_argument("--pool",type=Path,required=True); p.add_argument("--actual-route",type=Path,required=True); p.add_argument("--assignment",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True); args=p.parse_args()
    data={qid(x):x for x in rows(args.split)}; pools={str(x["query_id"]):x for x in rows(args.pool)}; actual={str(x["query_id"]):x for x in rows(args.actual_route)}; client_for=assignment(args.assignment)
    ids=sorted(set(data)&set(pools)&set(actual)); assert len(ids)==300
    oracle_rows=[]; perq=[]; matrix=[]; comp=[]
    for query in ids:
        gold,titles=supports(data[query],args.dataset); counts=Counter(client_for[x] for x in gold if x in client_for); gold_clients=sorted(counts,key=lambda c:(-counts[c],c)); actual_clients=[int(x) for x in actual[query]["selected_clients"]]; oracle_clients=route_oracle(gold_clients,actual_clients)
        if len(actual_clients)!=3 or len(oracle_clients)!=3: raise AssertionError(query)
        actual_cov=int(set(gold_clients)<=set(actual_clients)); oracle_cov=int(set(gold_clients)<=set(oracle_clients)); qtype, overlap=question_type(str(data[query]["question"]),titles)
        oracle_rows.append({"query_id":query,"actual_selected_clients":json.dumps(actual_clients),"oracle_selected_clients":json.dumps(oracle_clients),"gold_clients":json.dumps(gold_clients),"minimum_gold_client_count":len(gold_clients),"oracle_coverage_at_1":int(len(gold_clients)<=1),"oracle_coverage_at_2":int(len(gold_clients)<=2),"oracle_coverage_at_3":oracle_cov,"actual_coverage_at_3":actual_cov,"routing_recoverable_gap":oracle_cov-actual_cov})
        record={"query_id":query,"question_type_offline_analysis":qtype,"support_title_query_lexical_overlap":overlap,"gold_clients":gold_clients,"conditions":{}}
        for route_name, clients, covered in ((ROUTES[0],actual_clients,actual_cov),(ROUTES[1],oracle_clients,oracle_cov)):
            for ranker in RANKERS:
                local=local_lists(pools[query],clients,ranker); docs5=[d for c in clients for d in local[c][:5]]; docs10=[d for c in clients for d in local[c][:10]]; tx=transmitted(local)
                first,second,worst=rank_values(gold,docs10); condition={"route_all_gold_covered":covered,"local_complete_at_5":complete(gold,docs5),"local_complete_at_10":complete(gold,docs10),"support_recall_at_5":recall(gold,docs5),"support_recall_at_10":recall(gold,docs10),"first_support_rank":first,"second_support_rank":second,"worst_support_rank":worst,"transmitted_complete_at_15":complete(gold,tx),"merges":{}}
                for merge_name in ("raw","percentile","rrf"):
                    top=merged(tx,merge_name)[:10]; condition["merges"][merge_name]={"complete_at_10":complete(gold,top),"support_recall_at_10":recall(gold,top)}
                    matrix.append({"query_id":query,"route":route_name,"local_retriever":ranker,"merge":merge_name,"route_all_gold_covered":covered,"local_complete_at_5":condition["local_complete_at_5"],"local_complete_at_10":condition["local_complete_at_10"],"support_recall_at_5":condition["support_recall_at_5"],"support_recall_at_10":condition["support_recall_at_10"],"transmitted_complete_at_15":condition["transmitted_complete_at_15"],"complete_at_10":complete(gold,top),"support_recall_merged_at_10":recall(gold,top),"support_lost_by_transmission":int(condition["local_complete_at_10"] and not condition["transmitted_complete_at_15"]),"support_lost_by_merge":int(condition["transmitted_complete_at_15"] and not complete(gold,top))})
                record["conditions"][f"{route_name}_{ranker}"]=condition
        # Complementarity is computed only in each route's covered subset.
        for route_name in ROUTES:
            cond={r:record["conditions"][f"{route_name}_{r}"] for r in RANKERS}; covered=cond["L0_dense"]["route_all_gold_covered"]
            comp.append({"query_id":query,"route":route_name,"covered_subset":covered,"question_type":qtype,"support_title_query_lexical_overlap":overlap,"dense_success":cond["L0_dense"]["local_complete_at_10"],"bm25_success":cond["L1_bm25"]["local_complete_at_10"],"hybrid_success":cond["L2_hybrid"]["local_complete_at_10"],"rrf_success":cond["L3_rrf"]["local_complete_at_10"],"dense_only_success":int(cond["L0_dense"]["local_complete_at_10"] and not cond["L1_bm25"]["local_complete_at_10"]),"bm25_only_success":int(cond["L1_bm25"]["local_complete_at_10"] and not cond["L0_dense"]["local_complete_at_10"]),"hybrid_rescue":int(not cond["L0_dense"]["local_complete_at_10"] and cond["L2_hybrid"]["local_complete_at_10"]),"hybrid_harm":int(cond["L0_dense"]["local_complete_at_10"] and not cond["L2_hybrid"]["local_complete_at_10"])})
        perq.append(record)
    out=args.output_root; (out/"oracle_routing").mkdir(parents=True,exist_ok=True); (out/"local_retrievers").mkdir(exist_ok=True); (out/"factorial_matrix").mkdir(exist_ok=True); (out/"error_analysis").mkdir(exist_ok=True); (out/"reports").mkdir(exist_ok=True)
    write_csv(out/"oracle_routing/oracle_coverage.csv",oracle_rows); write_csv(out/"factorial_matrix/per_query_matrix.csv",matrix); write_csv(out/"error_analysis/dense_bm25_complementarity.csv",comp)
    with (out/"factorial_matrix/per_query_results.jsonl").open("w",encoding="utf-8") as h:
        for x in perq: h.write(json.dumps(x,ensure_ascii=False)+"\n")
    # Aggregate R x L for the requested local and merged metrics.
    summary=[]
    for route in ROUTES:
      for ranker in RANKERS:
        values=[x for x in matrix if x["route"]==route and x["local_retriever"]==ranker]
        base=[x for x in values if x["merge"]=="raw"]
        covered=[x for x in base if x["route_all_gold_covered"]]
        summary.append({"route":route,"local_retriever":ranker,"queries":len(base),"local_complete_at_5":mean([x["local_complete_at_5"] for x in base]),"local_complete_at_10":mean([x["local_complete_at_10"] for x in base]),"support_recall_at_5":mean([x["support_recall_at_5"] for x in base]),"support_recall_at_10":mean([x["support_recall_at_10"] for x in base]),"conditional_queries":len(covered),"conditional_complete_support_at_10":mean([x["local_complete_at_10"] for x in covered]),"conditional_support_recall_at_10":mean([x["support_recall_at_10"] for x in covered]),"transmitted_complete_at_15":mean([x["transmitted_complete_at_15"] for x in base]),"raw_complete_at_10":mean([x["complete_at_10"] for x in base]),"percentile_complete_at_10":mean([x["complete_at_10"] for x in values if x["merge"]=="percentile"]),"rrf_complete_at_10":mean([x["complete_at_10"] for x in values if x["merge"]=="rrf"]),"support_lost_by_transmission":sum(x["support_lost_by_transmission"] for x in base),"support_lost_by_raw_merge":sum(x["support_lost_by_merge"] for x in base)})
    write_csv(out/"local_retrievers/retriever_results.csv",summary); write_csv(out/"factorial_matrix/routing_local_matrix.csv",summary)
    # Contributions use one globally best frozen local ranker per stated arm, never per-query selection.
    def vec(route,ranker): return np.array([x["conditions"][f"{route}_{ranker}"]["local_complete_at_10"] for x in perq],float)
    r0dense=vec("R0_actual","L0_dense"); r1dense=vec("R1_oracle_bc3","L0_dense")
    best_local=max(RANKERS[1:],key=lambda r:vec("R0_actual",r).mean()); best_joint=max(RANKERS[1:],key=lambda r:vec("R1_oracle_bc3",r).mean())
    routing=r1dense-r0dense; local=vec("R0_actual",best_local)-r0dense; joint=vec("R1_oracle_bc3",best_joint)-r0dense; inter=joint-routing-local
    gains=[]
    for name,values,choice in (("RoutingGain",routing,"R1L0"),("LocalGain",local,f"R0{best_local}"),("JointGain",joint,f"R1{best_joint}"),("Interaction",inter,"derived")):
        lo,hi=ci(values); gains.append({"dataset":args.dataset,"component":name,"fixed_comparator":choice,"gain":float(values.mean()),"ci95_low":lo,"ci95_high":hi,"bootstrap_replicates":5000})
    routing_gain,local_gain=float(routing.mean()),float(local.mean())
    if routing_gain>=.05 and local_gain>=.05: status="routing_local_joint_method_required"
    elif routing_gain>=.05 and routing_gain>local_gain: status="routing_primary_bottleneck_confirmed"
    elif local_gain>=.05 and local_gain>routing_gain: status="local_retrieval_primary_bottleneck_confirmed"
    elif routing_gain<.03 and local_gain<.03: status="partition_or_candidate_contract_bottleneck"
    else: status="v20_upstream_opportunity_insufficient"
    merge_reactivated=any(row["transmitted_complete_at_15"]-row["raw_complete_at_10"]>.05 for row in summary)
    report="# V20 U1 Factorized Audit\n\n"+f"Dataset: `{args.dataset}`. Final status: `{status}`. Reader is `blocked_before_reader`.\n\n"+"| Component | Gain | 95% bootstrap CI | Fixed comparator |\n|---|---:|---:|---|\n"+"\n".join(f"| {x['component']} | {x['gain']:+.3f} | [{x['ci95_low']:+.3f}, {x['ci95_high']:+.3f}] | {x['fixed_comparator']} |" for x in gains)+f"\n\nMerge residual reactivated: `{merge_reactivated}`. Oracle is audit-only and uses at most three clients.\n"
    (out/"reports/factorized_bottleneck_report.md").write_text(report,encoding="utf-8")
    decision={"dataset":args.dataset,"status":status,"merge_residual_reactivated":merge_reactivated,"reader_start_decision":"blocked_before_reader","reader_started":False,"gold_used_only_for":"oracle_audit_and_final_evaluation"}
    (out/"reports/next_method_decision.json").write_text(json.dumps(decision,indent=2)+"\n",encoding="utf-8"); (out/"reports/reader_start_decision.json").write_text(json.dumps({"status":"blocked_before_reader","reader_started":False},indent=2)+"\n",encoding="utf-8")
    print(json.dumps(decision,indent=2))
if __name__=="__main__": main()
