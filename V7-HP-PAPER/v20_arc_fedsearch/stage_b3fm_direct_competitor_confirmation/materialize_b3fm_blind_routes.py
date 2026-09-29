#!/usr/bin/env python3
"""Materialize the five frozen B3F-M routes without opening labels."""
from __future__ import annotations
import argparse,csv,hashlib,json,pickle,sqlite3,sys
from pathlib import Path
from typing import Any
import numpy as np

DATASETS=("hotpotqa","2wikimultihopqa","musique")
FEATURES=("dense_top1_score","dense_top3_mean","dense_top1_top2_margin","dense_score_std","dense_score_entropy","dense_local_rank_percentile","bm25_top1_score","bm25_top3_mean","bm25_top1_top2_margin","dense_bm25_top1_same","dense_bm25_top3_overlap","dense_sparse_rank_correlation","matched_query_entity_count","matched_query_token_count","matched_title_token_count","query_title_embedding_similarity","top3_title_diversity","top3_entity_diversity")
METHODS=("b0_static_top3","ragroute_fixed3","m2_logistic_proberoute","b4a_all_candidate_top8_high_cost_reference","ragroute_original_threshold")
def rows(p):
  with p.open(encoding="utf-8") as h:
    for x in h:
      if x.strip(): yield json.loads(x)
def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def enc(device):
  from transformers import AutoModel,AutoTokenizer
  n="BAAI/bge-base-en-v1.5";r="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
  return AutoTokenizer.from_pretrained(n,revision=r,local_files_only=True),AutoModel.from_pretrained(n,revision=r,local_files_only=True).to(device).eval()
def embed(tok,model,questions,device):
  import torch
  out=[]
  for i in range(0,len(questions),64):
    b=tok(questions[i:i+64],padding=True,truncation=True,max_length=512,return_tensors="pt");b={k:v.to(device) for k,v in b.items()}
    with torch.inference_mode(): out.append(torch.nn.functional.normalize(model(**b).last_hidden_state[:,0],p=2,dim=1).float().cpu().numpy())
  return np.concatenate(out).astype(np.float32)
def rag_models(stage,dataset,device):
  import torch
  sys.path.insert(0,str(stage));from train_b3f_ragroute import B3FRouter
  z=np.load(stage/f"models/{dataset}/scaler.npz");models=[]
  for s in (0,1,2):
    c=torch.load(stage/f"models/{dataset}/seed_{s}.pt",map_location=device,weights_only=True);m=B3FRouter(int(c["input_dim"])).to(device);m.load_state_dict(c["state_dict"]);m.eval();models.append(m)
  return models,z["mean"].astype(np.float32),z["scale"].astype(np.float32)
def rag_score(models,x,mean,scale,device):
  import torch
  x=(x-mean)/np.where(scale==0,1,scale)
  with torch.inference_mode(): return np.mean(np.stack([torch.sigmoid(m(torch.from_numpy(x.astype(np.float32)).to(device))).cpu().numpy() for m in models]),axis=0)
def merge(packet,clients):
  docs=[dict(x) for c in clients for x in packet["local_dense_docs_top10"][str(c)][:5]]
  if len(docs)!=5*len(clients):raise ValueError("incomplete local payload")
  return docs,sorted(docs,key=lambda x:(-float(x["dense_score"]),str(x["doc_id"])))[:10]
def lookup(db,ids):
  ans=[]
  for i in ids:
    x=db.execute("SELECT doc_id,title,text FROM docs WHERE doc_id=?",(i,)).fetchone()
    if x is None:raise KeyError(i)
    ans.append({"doc_id":str(x[0]),"title":str(x[1]),"text":str(x[2])})
  return ans
def main():
 p=argparse.ArgumentParser();p.add_argument("--experiment-root",type=Path,required=True);p.add_argument("--b3f-stage",type=Path,required=True);p.add_argument("--stage",type=Path,required=True);p.add_argument("--device",default="cuda");a=p.parse_args()
 out=a.stage/"contexts/all_contexts_unscored.jsonl"; route=a.stage/"routing"
 if out.exists() or route.exists():raise FileExistsError("refusing to overwrite blind B3F-M routes")
 route.mkdir(parents=True);out.parent.mkdir(parents=True,exist_ok=True);tok,bge=enc(a.device);all_routes={m:[] for m in METHODS};n=0
 with out.open("x",encoding="utf-8") as h:
  for d in DATASETS:
   blind=list(rows(a.stage/f"inputs/{d}_blind_n500.jsonl"));packets={str(x["query_id"]):x for x in rows(a.stage/f"retrieval/{d}_probe_packets.jsonl")}
   if len(blind)!=500 or set(map(lambda x:str(x["query_id"]),blind))!=set(packets):raise ValueError(f"{d}: packet mismatch")
   em=embed(tok,bge,[str(x["question"]) for x in blind],a.device);cent=np.load(a.experiment_root/f"runs/ragroute_b3_r5_posthoc_20260916/centroids/{d}/source_centroids.npy").astype(np.float32);models,mean,scale=rag_models(a.b3f_stage,d,a.device)
   with (a.experiment_root/f"inputs/models/{d}/logistic_seed_20260807.pkl").open("rb") as f:m2=pickle.load(f)
   db=sqlite3.connect(f"file:{a.experiment_root/f'inputs/indexes/{d}.sqlite'}?mode=ro",uri=True)
   try:
    for pos,src in enumerate(blind):
     q=str(src["query_id"]);question=str(src["question"]);packet=packets[q];records=list(packet["p0_candidate_records"]);cand=[int(x["client_id"]) for x in records]
     if len(cand)!=8 or len(set(cand))!=8:raise ValueError(f"{d}/{q}: invalid P0")
     fx=np.asarray([[float(x["static_score"]),*[float(x[k]) for k in FEATURES]] for x in records],dtype=np.float32);m2s=m2["model"].predict_proba(m2["scaler"].transform(fx))[:,1]
     rx=np.concatenate((np.repeat(em[pos:pos+1],8,axis=0),cent[cand],np.eye(cent.shape[0],dtype=np.float32)[cand]),axis=1);rs=rag_score(models,rx,mean,scale,a.device)
     routes={"b0_static_top3":[int(x["client_id"]) for x in sorted(records,key=lambda x:(int(x["static_candidate_rank"]),-float(x["static_score"]),int(x["client_id"])))[:3]],"ragroute_fixed3":sorted(cand,key=lambda c:(-float(rs[cand.index(c)]),c))[:3],"m2_logistic_proberoute":[cand[i] for i in np.argsort(-m2s,kind="stable")[:3]],"b4a_all_candidate_top8_high_cost_reference":cand,"ragroute_original_threshold":[c for i,c in enumerate(cand) if float(rs[i])>0.5]}
     scores={"static":{str(x["client_id"]):float(x["static_score"]) for x in records},"m2":{str(c):float(m2s[i]) for i,c in enumerate(cand)},"ragroute":{str(c):float(rs[i]) for i,c in enumerate(cand)}}
     for method,sel in routes.items():
      sent,top=merge(packet,sel);docs=lookup(db,[str(x["doc_id"]) for x in top[:5]]);ctx=digest({"question":question,"docs":docs})
      payload={"dataset":d,"query_id":q,"question":question,"method":method,"candidate_client_ids":cand,"selected_client_ids":sel,"routing_scores":scores,"local_doc_ids":{str(c):[str(x["doc_id"]) for x in packet["local_dense_docs_top10"][str(c)][:10]] for c in sel},"transmitted_doc_ids":[str(x["doc_id"]) for x in sent],"merged_top10_doc_ids":[str(x["doc_id"]) for x in top],"reader_top5_doc_ids":[str(x["doc_id"]) for x in top[:5]],"reader_context_docs":docs,"context_hash":ctx,"candidate_clients":8,"client_budget":len(sel),"local_depth":10,"documents_per_client":5,"transmitted_documents":len(sent),"gold_or_answer_used":False,"labels_loaded":False,"metrics_computed":False,"reader_started":False}
      h.write(json.dumps(payload,ensure_ascii=False)+"\n");all_routes[method].append({"dataset":d,"query_id":q,"candidate_top8":json.dumps(cand),"selected_clients":json.dumps(sel),"static_scores":json.dumps(scores["static"]),"m2_scores":json.dumps(scores["m2"]),"ragroute_scores":json.dumps(scores["ragroute"])});n+=1
   finally:db.close()
 for m,values in all_routes.items():
  with (route/f"{m}_scores.csv").open("x",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
 if n!=7500:raise ValueError(n)
 manifest={"status":"complete_b3fm_blind_routing_context_materialization","labels_loaded":False,"metrics_computed":False,"rows":n,"methods":list(METHODS),"output_sha256":hashlib.sha256(out.read_bytes()).hexdigest()};(a.stage/"contexts/materialization_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
