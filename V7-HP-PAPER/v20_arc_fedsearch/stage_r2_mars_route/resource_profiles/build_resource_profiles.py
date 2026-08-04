#!/usr/bin/env python3
"""Build client-local P0/P1/P2/P3 resource cards from documents only."""
from __future__ import annotations
import argparse, collections, json, re, sqlite3
from pathlib import Path
from typing import Any
import numpy as np

TOKEN=re.compile(r"[A-Za-z][A-Za-z0-9'-]+")
def terms(text: str) -> list[str]: return [x.lower() for x in TOKEN.findall(text) if len(x)>2]
def entities(title: str) -> list[str]: return [" ".join(x) for x in re.findall(r"(?:[A-Z][A-Za-z0-9'-]*\s*)+", title)]
def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument("--dataset",choices=("2wikimultihopqa","musique"),required=True); p.add_argument("--local-index-root",type=Path,required=True); p.add_argument("--p0-centroids",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--prototype-counts",default="4,8,16"); p.add_argument("--encoder",default="BAAI/bge-base-en-v1.5"); p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=20260804); p.add_argument("--batch-size",type=int,default=256); args=p.parse_args()
 from sentence_transformers import SentenceTransformer
 from sklearn.cluster import MiniBatchKMeans
 counts=[int(x) for x in args.prototype_counts.split(",")]; p0=np.load(args.p0_centroids); model=SentenceTransformer(args.encoder,device=args.device); profiles=[]; stats=[]
 for client in range(20):
  con=sqlite3.connect(args.local_index_root/f"client_{client:02d}.sqlite"); docs=[dict(zip(("title","text"),x)) for x in con.execute("select title,text from docs")]; con.close()
  texts=[f"{d['title']}. {d['text']}" for d in docs]; emb=model.encode(texts,normalize_embeddings=True,convert_to_numpy=True,batch_size=args.batch_size,show_progress_bar=False).astype("float32")
  tf=collections.Counter(t for d in docs for t in terms(d["title"]+" "+d["text"])); ef=collections.Counter(e for d in docs for e in entities(d["title"])); cards={}
  for k in counts:
   km=MiniBatchKMeans(n_clusters=k,random_state=args.seed+client,n_init=3,batch_size=min(2048,max(256,len(docs)))).fit(emb); labels=km.labels_; centers=km.cluster_centers_; centers/=np.linalg.norm(centers,axis=1,keepdims=True)
   proto=[]
   for cluster in range(k):
    ix=np.where(labels==cluster)[0]; sims=emb[ix]@centers[cluster]; closest=ix[np.argsort(-sims)[:3]]
    proto.append({"centroid":centers[cluster].astype(float).tolist(),"cluster_size":int(len(ix)),"intra_cluster_variance":float(np.mean(1-sims)),"representative_titles":[docs[int(i)]["title"] for i in closest],"representative_entities":[e for e,_ in collections.Counter(e for i in ix[:min(len(ix),1000)] for e in entities(docs[int(i)]["title"])).most_common(8)]})
   cards[str(k)]=proto
   stats.extend({"dataset":args.dataset,"client_id":client,"prototype_count":k,"cluster_id":i,"cluster_size":x["cluster_size"],"intra_cluster_variance":x["intra_cluster_variance"]} for i,x in enumerate(proto))
  profiles.append({"dataset":args.dataset,"client_id":client,"collection_size":len(docs),"p0_single_centroid":p0[client].astype(float).tolist(),"p1_multi_prototypes":cards,"p2_lexical_sketch":{"top_terms":[x for x,_ in tf.most_common(2000)],"term_counts":dict(tf.most_common(2000)),"entity_frequency_sketch":dict(ef.most_common(500)),"representative_titles":[d["title"] for d in docs[:10]]},"p3_combined":"P0+P1+P2+collection_size"})
  print(json.dumps({"status":"profiles_complete","client":client,"docs":len(docs)}),flush=True)
 args.output_dir.mkdir(parents=True,exist_ok=True); (args.output_dir/"client_profiles.json").write_text(json.dumps({"dataset":args.dataset,"seed":args.seed,"encoder":args.encoder,"profiles":profiles,"gold_or_development_fields_used":False},ensure_ascii=False)+"\n",encoding="utf-8")
 import csv
 with (args.output_dir/"prototype_statistics.csv").open("w",encoding="utf-8",newline="") as h:
  w=csv.DictWriter(h,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
 (args.output_dir/"profile_manifest.json").write_text(json.dumps({"status":"complete","dataset":args.dataset,"clients":20,"prototype_counts":counts,"profile_documents":"all local shard documents","gold_or_development_fields_used":False},indent=2)+"\n",encoding="utf-8")
if __name__=="__main__":main()
