#!/usr/bin/env python3
"""R2-A.5: audit candidate-to-Bc=3 compression without training or profile edits."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,re
from collections import Counter
from pathlib import Path
from typing import Any,Iterable
import numpy as np
TOKEN=re.compile(r"[A-Za-z][A-Za-z0-9'-]+")
def rows(p:Path)->Iterable[dict[str,Any]]:
 with p.open(encoding="utf-8") as h:
  for l in h:
   if l.strip():yield json.loads(l)
def qid(x):return str(x.get("query_id",x.get("_id",x.get("id"))))
def norm(x):return " ".join(str(x).lower().split())
def did(ds,t,txt=""):
 k=norm(t) if ds!="musique" else norm(t)+"\n"+norm(txt);return f"{ds}:{hashlib.sha1(k.encode()).hexdigest()[:20]}"
def support_seq(row,ds):
 if ds=="musique":return [did(ds,x.get("title",""),x.get("paragraph_text","")) for x in row.get("paragraphs",[]) if x.get("is_supporting",x.get("is_support",False))]
 f=row.get("supporting_facts",{});ts=f.get("title",[]) if isinstance(f,dict) else [x[0] for x in f if x];return [did(ds,x) for x in ts]
def terms(x):return {t.lower() for t in TOKEN.findall(x) if len(t)>2}
def view_q0(q):return q
def local_complete(gold,docs):return int(bool(gold) and gold<={d["doc_id"] for d in docs})
def recall(gold,docs):return len(gold&{d["doc_id"] for d in docs})/max(1,len(gold))
def write(path,values):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",encoding="utf-8",newline="") as h:
  w=csv.DictWriter(h,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
def bootstrap(a,b):
 if not a or not b:return (None,None,None)
 x=np.asarray(a);y=np.asarray(b);rng=np.random.default_rng(20260804);d=[]
 for _ in range(5000):
  ia=rng.integers(0,len(x),len(x));ib=rng.integers(0,len(y),len(y));d.append(x[ia].mean()-y[ib].mean())
 return (float(x.mean()-y.mean()),float(np.quantile(d,.025)),float(np.quantile(d,.975)))
def main():
 p=argparse.ArgumentParser();p.add_argument("--dataset",choices=("2wikimultihopqa","musique"),required=True);p.add_argument("--split",type=Path,required=True);p.add_argument("--profiles",type=Path,required=True);p.add_argument("--assignment",type=Path,required=True);p.add_argument("--local-pool",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--best-p",type=int,required=True);args=p.parse_args()
 from sentence_transformers import SentenceTransformer
 prof=json.loads(args.profiles.read_text())["profiles"];data={qid(x):x for x in rows(args.split)};assign={str(x["doc_id"]):int(x["client_id"]) for x in rows(args.assignment)};pool={str(x["query_id"]):x for x in rows(args.local_pool)};model=SentenceTransformer("BAAI/bge-base-en-v1.5",device="cuda")
 cent=np.asarray([x["p0_single_centroid"] for x in prof]); sketch=[x["p2_lexical_sketch"] for x in prof];df={t:sum(t in s["term_counts"] for s in sketch) for s in sketch for t in s["term_counts"]}; proto=[np.asarray([z["centroid"] for z in x["p1_multi_prototypes"][str(args.best_p)]]) for x in prof]
 methods=("P0","P1","P2","P3") ; metric=[];oracle=[];omitted=[];redund=[];probe=[]
 for query,row in data.items():
  gold_docs=set(support_seq(row,args.dataset));seq=[assign[x] for x in support_seq(row,args.dataset) if x in assign];gold=list(dict.fromkeys(seq));goldset=set(gold);q=str(row["question"]);e=model.encode([view_q0(q)],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)[0]; qt=terms(q)
  s0=(cent@e).astype(float);s1=np.asarray([float((e@x.T).max()) for x in proto]);s2=np.asarray([sum(math.log1p(s["term_counts"][t])*math.log(21/(1+df.get(t,0))) for t in qt if t in s["term_counts"]) for s in sketch]);r0=np.argsort(-s0);r1=np.argsort(-s1);r2=np.argsort(-s2);rank0={int(c):i for i,c in enumerate(r0)};rank2={int(c):i for i,c in enumerate(r2)};s3=np.asarray([1/(60+rank0[c])+1/(60+rank2[c]) for c in range(20)])
  scores={"P0":s0,"P1":s1,"P2":s2,"P3":s3}; ranked={name:[int(x) for x in np.argsort(-v)] for name,v in scores.items()}; cap3=int(len(goldset)<=3);oraclecov=cap3
  for name in methods:
   rank=ranked[name];top3=set(rank[:3]);
   for k in (3,5,8):
    chosen=set(rank[:k]);metric.append({"dataset":args.dataset,"query_id":query,"profile":name,"k":k,"gold_client_count":len(goldset),"candidate_gold_client_recall":len(chosen&goldset)/max(1,len(goldset)),"candidate_complete_set_recall":int(goldset<=chosen),"independent_top3_coverage":int(goldset<=top3),"gold_used_only_offline":True})
   o5=int(goldset<=set(rank[:5]) and cap3);o8=int(goldset<=set(rank[:8]) and cap3);ind=int(goldset<=top3);oracle.append({"dataset":args.dataset,"query_id":query,"profile":name,"gold_clients":json.dumps(gold),"oracle_coverage_at_3":oraclecov,"oracle_subset_at_3_within_candidate5":o5,"oracle_subset_at_3_within_candidate8":o8,"independent_top3_coverage":ind,"compression_gap_at_5":o5-ind,"compression_gap_at_8":o8-ind,"candidate_absence_loss":oraclecov-o8,"selection_compression_loss":o5-ind})
  # Probes operate on frozen P0 candidates/scores; S2 uses P1 specialization score only.
  cand=ranked["P0"][:8];pnorm=(s0-s0.min())/max(1e-8,s0.max()-s0.min());sim=cent@cent.T
  selections={"S0_independent_top3":cand[:3]}
  for lam in (.05,.10,.20):
   chosen=[]
   while len(chosen)<3:
    choices=[c for c in cand if c not in chosen];chosen.append(max(choices,key=lambda c:float(pnorm[c])-lam*(max([sim[c,x] for x in chosen],default=0))))
   selections[f"S1_redundancy_greedy_lambda_{lam:.2f}"]=chosen
  chosen=[]
  while len(chosen)<3:
   choices=[c for c in cand if c not in chosen];chosen.append(max(choices,key=lambda c:float(pnorm[c])+.10*max(0,float(s1[c]-s0[c]))-.05*max([sim[c,x] for x in chosen],default=0)))
  selections["S2_prototype_coverage_greedy"]=chosen
  dense3=cand[:3];lex3=ranked["P2"][:3];j=len(set(dense3)&set(lex3))/3
  s3sel=list(dense3)
  if j<.5:
   for c in lex3:
    if c not in s3sel:s3sel[-1]=c;break
  selections["S3_disagreement_diversity_gated"]=s3sel
  local_dense=pool[query]["rankers"]
  for name,chosen in selections.items():
   lists=[local_dense[str(c)]["L0_dense"] for c in chosen];docs10=[d for l in lists for d in l[:10]];tx=[d for l in lists for d in l[:5]];raw=sorted(tx,key=lambda d:(-float(d["local_score"]),-float(d["dense_score"])))[:10];pct=sorted(tx,key=lambda d:(int(d["local_rank"]),-float(d["dense_score"])))[:10]
   probe.append({"dataset":args.dataset,"query_id":query,"probe":name,"selected_clients":json.dumps(chosen),"complete_client_set_coverage_at_3":int(goldset<=set(chosen)),"gold_client_recall_at_3":len(goldset&set(chosen))/max(1,len(goldset)),"client_precision_at_3":len(goldset&set(chosen))/3,"local_complete_at_10":local_complete(gold_docs,docs10),"transmitted_complete_at_15":local_complete(gold_docs,tx),"raw_merged_complete_at_10":local_complete(gold_docs,raw),"percentile_merged_complete_at_10":local_complete(gold_docs,pct)})
  ranks={c:(ranked["P0"].index(c)+1 if c in ranked["P0"] else None) for c in gold};omit=[c for c in gold if c not in selections["S0_independent_top3"]];omitted.append({"query_id":query,"gold_clients":json.dumps(gold),"top5_candidates":json.dumps(cand[:5]),"top5_scores":json.dumps([float(s0[c]) for c in cand[:5]]),"selected_top3":json.dumps(selections["S0_independent_top3"]),"omitted_gold_clients":json.dumps(omit),"omitted_gold_rank":json.dumps({str(c):ranks[c] for c in omit}),"support_client_sequence":json.dumps(seq)})
  pairs=[(a,b) for i,a in enumerate(selections["S0_independent_top3"]) for b in selections["S0_independent_top3"][i+1:]];cs=[float(sim[a,b]) for a,b in pairs];eo=[len(set(sketch[a]["top_terms"])&set(sketch[b]["top_terms"]))/max(1,len(set(sketch[a]["top_terms"])|set(sketch[b]["top_terms"]))) for a,b in pairs];redund.append({"query_id":query,"p0_compression_failure":int(goldset<=set(cand[:5]) and not goldset<=set(cand[:3]) and cap3),"top3_success":int(goldset<=set(cand[:3])) ,"mean_pairwise_profile_cosine":float(np.mean(cs)),"max_pairwise_profile_cosine":float(np.max(cs)),"mean_entity_sketch_overlap":float(np.mean(eo))})
 out=args.output_root;write(out/"candidate_metrics/candidate_recall_by_k.csv",metric);write(out/"oracle_subset/oracle_subset_within_candidates.csv",oracle);write(out/"hop_analysis/omitted_gold_client_analysis.csv",omitted);write(out/"redundancy_analysis/top3_redundancy.csv",redund);write(out/"selector_probes/probe_results_per_query.csv",probe)
 # aggregate probes and candidate metrics
 agg=[]
 for name in methods:
  for k in (3,5,8):
   xs=[x for x in metric if x["profile"]==name and x["k"]==k];agg.append({"dataset":args.dataset,"profile":name,"k":k,"candidate_gold_client_recall":np.mean([x["candidate_gold_client_recall"] for x in xs]),"candidate_complete_set_recall":np.mean([x["candidate_complete_set_recall"] for x in xs]),"independent_top3_coverage":np.mean([x["independent_top3_coverage"] for x in xs])})
 write(out/"candidate_metrics/candidate_recall_summary.csv",agg)
 pagg=[]
 for name in sorted({x["probe"] for x in probe}):
  xs=[x for x in probe if x["probe"]==name];pagg.append({"dataset":args.dataset,"probe":name,"queries":len(xs),**{k:float(np.mean([x[k] for x in xs])) for k in ("complete_client_set_coverage_at_3","gold_client_recall_at_3","client_precision_at_3","local_complete_at_10","transmitted_complete_at_15","raw_merged_complete_at_10","percentile_merged_complete_at_10")}})
 write(out/"selector_probes/probe_results.csv",pagg)
 s0=[x["complete_client_set_coverage_at_3"] for x in probe if x["probe"]=="S0_independent_top3"]; best=max(pagg[1:],key=lambda x:x["complete_client_set_coverage_at_3"]); bd=[x["complete_client_set_coverage_at_3"] for x in probe if x["probe"]==best["probe"]];gain,lo,hi=bootstrap(bd,s0); comp5=[x for x in oracle if x["profile"]=="P0"]
 cgap=float(np.mean([x["compression_gap_at_5"] for x in comp5]));absence=float(np.mean([x["candidate_absence_loss"] for x in comp5]));status=("candidate_compression_bottleneck_confirmed" if cgap>=.08 and gain>=.05 and best["local_complete_at_10"]>=next(x for x in pagg if x["probe"]=="S0_independent_top3")["local_complete_at_10"]+.03 else "candidate_representation_bottleneck_confirmed" if cgap<.03 or absence>.05 else "set_selection_signal_not_captured")
 rf=[x["mean_pairwise_profile_cosine"] for x in redund if x["p0_compression_failure"]];rs=[x["mean_pairwise_profile_cosine"] for x in redund if x["top3_success"]];rd,rlo,rhi=bootstrap(rf,rs)
 report=f"# R2-A.5 Candidate Compression Audit\n\nDataset `{args.dataset}`. P0 CompressionGap@5={cgap:.3f}; CandidateAbsenceLoss={absence:.3f}. Best permitted probe `{best['probe']}` coverage gain={gain:+.3f}, CI=[{lo:+.3f},{hi:+.3f}]. Redundancy difference (compression failure - success)={rd:+.3f}, CI=[{rlo:+.3f},{rhi:+.3f}].\n\nFinal state: `{status}`. Reader remains `blocked_before_reader`.\n"
 (out/"reports/r2a5_compression_go_no_go.md").parent.mkdir(parents=True,exist_ok=True);(out/"reports/r2a5_compression_go_no_go.md").write_text(report);dec={"dataset":args.dataset,"status":status,"reader_start_decision":"blocked_before_reader","best_probe":best["probe"],"compression_gap_at_5":cgap,"candidate_absence_loss":absence,"best_probe_coverage_gain":gain,"best_probe_coverage_ci95":[lo,hi]};(out/"reports/next_method_decision.json").write_text(json.dumps(dec,indent=2)+"\n");(out/"reports/reader_start_decision.json").write_text(json.dumps({"status":"blocked_before_reader","reader_started":False},indent=2)+"\n");print(json.dumps(dec,indent=2))
if __name__=="__main__":main()
