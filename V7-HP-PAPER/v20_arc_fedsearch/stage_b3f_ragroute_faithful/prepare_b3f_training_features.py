#!/usr/bin/env python3
"""Materialize the frozen B3F RAGRoute training rows from historical IDs only."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

DATASETS=("hotpotqa","2wikimultihopqa","musique")

def read_json_rows(path):
    if path.suffix==".jsonl":
        with path.open() as f:
            return [json.loads(x) for x in f if x.strip()]
    return json.loads(path.read_text())

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for x in iter(lambda:f.read(1<<20),b""): h.update(x)
    return h.hexdigest()

def qid(row): return str(row.get("query_id",row.get("_id",row.get("id"))))

def support_docs(row,dataset,document_id):
    if dataset=="musique":
        return {document_id(dataset,str(x.get("title","")),str(x.get("paragraph_text",""))) for x in row.get("paragraphs",[]) if x.get("is_supporting",x.get("is_support",False))}
    facts=row.get("supporting_facts",{})
    titles=facts.get("title",[]) if isinstance(facts,dict) else [x[0] for x in facts if x]
    return {document_id(dataset,str(t)) for t in titles}

def encode(tokenizer,model,texts,device,batch):
    import torch
    out=[]
    for i in range(0,len(texts),batch):
        x=tokenizer(texts[i:i+batch],padding=True,truncation=True,max_length=512,return_tensors="pt")
        x={k:v.to(device) for k,v in x.items()}
        with torch.inference_mode(): out.append(torch.nn.functional.normalize(model(**x).last_hidden_state[:,0],p=2,dim=1).cpu().float().numpy())
    return np.concatenate(out)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",choices=DATASETS,required=True); p.add_argument("--source",type=Path,required=True); p.add_argument("--historical-manifest",type=Path,required=True); p.add_argument("--p0-profiles",type=Path,required=True); p.add_argument("--feature-centroids",type=Path,required=True); p.add_argument("--assignment",type=Path,required=True); p.add_argument("--v16-eval",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--model",default="BAAI/bge-base-en-v1.5"); p.add_argument("--revision",required=True); p.add_argument("--device",default="cpu"); p.add_argument("--batch-size",type=int,default=32); a=p.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    ids=set(map(str,json.loads(a.historical_manifest.read_text())["query_ids"]))
    source={qid(x):x for x in read_json_rows(a.source)}
    if len(ids)!=5000 or not ids <= set(source): raise ValueError("historical IDs missing or not exactly 5000")
    ordered=sorted((source[x] for x in ids),key=lambda x:qid(x))
    profile_rows=json.loads(a.p0_profiles.read_text())["profiles"]
    profile={int(row["client_id"]): row for row in profile_rows}
    if set(profile) != set(range(20)):
        raise ValueError("P0 profile does not contain exactly global clients 0..19")
    p0=np.asarray([profile[c]["p0_single_centroid"] for c in range(20)],dtype=np.float32)
    centroids=np.load(a.feature_centroids).astype(np.float32)
    if p0.shape!=(20,768) or centroids.shape!=(20,768): raise ValueError("expected 20x768 profiles/centroids")
    assignment={str(x["doc_id"]):int(x["client_id"]) for x in read_json_rows(a.assignment)}
    sys.path.insert(0,str(a.v16_eval)); from eval_common import document_id
    from transformers import AutoTokenizer,AutoModel
    tokenizer=AutoTokenizer.from_pretrained(a.model,revision=a.revision); model=AutoModel.from_pretrained(a.model,revision=a.revision).to(a.device).eval()
    qemb=encode(tokenizer,model,[str(x["question"]) for x in ordered],a.device,a.batch_size)
    candidate=np.argsort(-(qemb@p0.T),axis=1,kind="stable")[:,:8]
    features=[]; labels=[]; query_ids=[]; clients=[]
    onehot=np.eye(20,dtype=np.float32)
    for row,emb,cands in zip(ordered,qemb,candidate):
        supported={assignment[d] for d in support_docs(row,a.dataset,document_id) if d in assignment}
        for c in cands:
            features.append(np.concatenate((emb,centroids[c],onehot[c])))
            labels.append(int(c in supported)); query_ids.append(qid(row)); clients.append(int(c))
    a.output_dir.mkdir(parents=True)
    np.savez_compressed(a.output_dir/"candidate_rows.npz",features=np.asarray(features,dtype=np.float32),labels=np.asarray(labels,dtype=np.int8),query_ids=np.asarray(query_ids),clients=np.asarray(clients,dtype=np.int8))
    manifest={"status":"complete_historical_candidate_rows","dataset":a.dataset,"queries":len(ordered),"candidate_rows":len(labels),"candidate_rule":"recovered_frozen_p0_top8","query_encoder":a.model,"encoder_revision":a.revision,"feature_dim":1556,"feature_contract":"BGE_query + local_corpus_centroid + global_client_onehot","positive_rows":int(sum(labels)),"source_sha256":sha(a.source),"historical_manifest_sha256":sha(a.historical_manifest),"p0_profiles_sha256":sha(a.p0_profiles),"feature_centroids_sha256":sha(a.feature_centroids),"assignment_sha256":sha(a.assignment),"labels_from":"historical_public_training_rows_only","fresh_labels_opened":False}
    (a.output_dir/"feature_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n"); print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
