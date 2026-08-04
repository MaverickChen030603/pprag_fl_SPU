#!/usr/bin/env python3
"""R2-A candidate recall smoke; profiles/query views only at inference."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, re
from pathlib import Path
from typing import Any, Iterable
import numpy as np
TOKEN=re.compile(r"[A-Za-z][A-Za-z0-9'-]+")
CAP=re.compile(r"(?:[A-Z][A-Za-z0-9'-]*(?:\s+|$)){1,5}")
def rows(path:Path)->Iterable[dict[str,Any]]:
 with path.open(encoding="utf-8") as h:
  for line in h:
   if line.strip():yield json.loads(line)
def qid(x:dict[str,Any])->str:return str(x.get("query_id",x.get("_id",x.get("id"))))
def norm(x:str)->str:return " ".join(x.lower().split())
def did(ds:str,t:str,txt:str="")->str:
 k=norm(t) if ds!="musique" else norm(t)+"\n"+norm(txt);return f"{ds}:{hashlib.sha1(k.encode()).hexdigest()[:20]}"
def support(row:dict[str,Any],ds:str)->set[str]:
 if ds=="musique":return {did(ds,x.get("title",""),x.get("paragraph_text","")) for x in row.get("paragraphs",[]) if x.get("is_supporting",x.get("is_support",False))}
 f=row.get("supporting_facts",{});ts=f.get("title",[]) if isinstance(f,dict) else [x[0] for x in f if x];return {did(ds,t) for t in ts}
def views(question:str)->dict[str,Any]:
 entity=[x.strip() for x in CAP.findall(question) if x.strip()]
 clauses=[x.strip() for x in re.split(r"\b(?:and|or|but|than|while|that|which|who|where)\b|[;,?]",question,flags=re.I) if len(x.strip())>5]
 relation=[x.strip() for x in re.findall(r"(?:of|in|by|from|with|for|between)\s+([A-Za-z][A-Za-z0-9' -]{2,40})",question,re.I)]
 return {"full_query":question,"entity_views":entity,"clause_views":clauses,"relation_views":relation,"view_count":1+len(entity)+len(clauses)+len(relation),"fallback_reason":None if (entity or clauses or relation) else "full_query_only"}
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--dataset",choices=("2wikimultihopqa","musique"),required=True);p.add_argument("--split",type=Path,required=True);p.add_argument("--profiles",type=Path,required=True);p.add_argument("--assignment",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--L",default="5,8");p.add_argument("--encoder",default="BAAI/bge-base-en-v1.5");p.add_argument("--device",default="cuda");args=p.parse_args()
 from sentence_transformers import SentenceTransformer
 profiles=json.loads(args.profiles.read_text())["profiles"]; assign={str(x["doc_id"]):int(x["client_id"]) for x in rows(args.assignment)}; data=list(rows(args.split)); model=SentenceTransformer(args.encoder,device=args.device); ls=[int(x) for x in args.L.split(",")]
 all_terms=[set(x["p2_lexical_sketch"]["term_counts"]) for x in profiles]; df={t:sum(t in s for s in all_terms) for s in all_terms for t in s}; result=[]; qouts=[]
 for row in data:
  query=qid(row);v=views(str(row["question"])); strings=[v["full_query"]]+v["entity_views"]+v["clause_views"]+v["relation_views"]; em=model.encode(strings,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False); gold=support(row,args.dataset); goldc=sorted({assign[x] for x in gold if x in assign})
  scores={"P0_Q0_single_centroid":[],"P2_lexical":[]}
  for k in (4,8,16):
   scores[f"P1_P{k}_Q0"]=[]
   scores[f"P1_P{k}_multiview"]=[]
  for client,prof in enumerate(profiles):
   scores["P0_Q0_single_centroid"].append(float(em[0]@np.asarray(prof["p0_single_centroid"])))
   for k in (4,8,16):
    cent=np.asarray([x["centroid"] for x in prof["p1_multi_prototypes"][str(k)]])
    scores[f"P1_P{k}_Q0"].append(float((em[0]@cent.T).max()))
    scores[f"P1_P{k}_multiview"].append(float((em@cent.T).max()))
   tc=prof["p2_lexical_sketch"]["term_counts"]; qt={x.lower() for s in strings for x in TOKEN.findall(s)}; scores["P2_lexical"].append(sum(math.log1p(tc[t])*math.log((21)/(1+df.get(t,0))) for t in qt if t in tc))
  candidates={name:[int(x) for x in np.argsort(-np.asarray(ss))] for name,ss in scores.items()}
  # Rank fusion exposes both cards at each L; concatenation would silently make
  # the first branch occupy every top-L position.
  for mode in ("Q0", "multiview"):
   dense=candidates[f"P1_P8_{mode}"]; lexical=candidates["P2_lexical"]
   dr={client:rank for rank,client in enumerate(dense)}; lr={client:rank for rank,client in enumerate(lexical)}
   candidates[f"P3_P8_{mode}_lexical_rrf"] = sorted(range(20),key=lambda c:(-(1/(60+dr[c])+1/(60+lr[c])),c))
  qouts.append({"query_id":query,**v})
  for name,ranked in candidates.items():
   for L in ls:
    chosen=ranked[:L];result.append({"dataset":args.dataset,"query_id":query,"method":name,"L":L,"candidate_clients":json.dumps(chosen),"gold_clients_offline_only":json.dumps(goldc),"gold_client_recall_at_L":len(set(chosen)&set(goldc))/max(1,len(goldc)),"complete_client_set_recall_at_L":int(set(goldc)<=set(chosen)),"gold_or_answer_used_for_ranking":False})
 args.output_root.mkdir(parents=True,exist_ok=True)
 with (args.output_root/"query_views.jsonl").open("w",encoding="utf-8") as h:
  for x in qouts:h.write(json.dumps(x,ensure_ascii=False)+"\n")
 with (args.output_root/"candidate_recall_per_query.csv").open("w",encoding="utf-8",newline="") as h:
  w=csv.DictWriter(h,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
 agg=[]
 for method in sorted({x["method"] for x in result}):
  for L in ls:
   xs=[x for x in result if x["method"]==method and x["L"]==L];agg.append({"dataset":args.dataset,"method":method,"L":L,"queries":len(xs),"candidate_gold_client_recall_at_L":sum(float(x["gold_client_recall_at_L"]) for x in xs)/len(xs),"candidate_complete_client_set_recall_at_L":sum(int(x["complete_client_set_recall_at_L"]) for x in xs)/len(xs),"gold_used_only_for_offline_metrics":True})
 with (args.output_root/"candidate_recall.csv").open("w",encoding="utf-8",newline="") as h:
  w=csv.DictWriter(h,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
 print(json.dumps({"dataset":args.dataset,"queries":len(data),"methods":sorted({x["method"] for x in result}),"reader_started":False},indent=2))
if __name__=="__main__":main()
