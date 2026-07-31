#!/usr/bin/env python3
"""PC-1: only replace implicit negatives with fixed train-only hard negatives."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np,torch
from transformers import AutoTokenizer
V19=Path(__file__).resolve().parents[1];sys.path.insert(0,str(V19/'stage0_full_upload'));sys.path.insert(0,str(V19/'model'))
from run_stage0_viability import evaluate_pool, make_model, pooled
from lora_blocks import adapter_state,load_adapter_state,state_bytes

def rows(path):
 with path.open(encoding='utf-8') as h:
  return [json.loads(x) for x in h if x.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--frozen-checkpoint',type=Path,required=True);p.add_argument('--development',type=Path,required=True);p.add_argument('--pool',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--steps',type=int,default=24);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--seed',type=int,default=20260731);p.add_argument('--model',default='BAAI/bge-base-en-v1.5');p.add_argument('--device',default='cuda');a=p.parse_args()
 torch.manual_seed(a.seed);device=torch.device(a.device);data=rows(a.manifest);tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=True);model=make_model(a.model,device,8,16.0);initial=torch.load(a.frozen_checkpoint,map_location='cpu')['adapter_state'];load_adapter_state(model,initial,device);opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=a.lr);logs=[]
 for step in range(a.steps):
  batch=[data[(step*a.batch_size+i)%len(data)] for i in range(a.batch_size)];q=tok(['Represent this sentence for searching relevant passages: '+x['query'] for x in batch],padding=True,truncation=True,max_length=128,return_tensors='pt').to(device);docs=[]
  for x in batch:docs += [x['positive'],*x['negatives']]
  d=tok(docs,padding=True,truncation=True,max_length=256,return_tensors='pt').to(device);model.train();qe=pooled(model,q);de=pooled(model,d).reshape(len(batch),5,-1);loss=torch.nn.functional.cross_entropy((qe.unsqueeze(1)*de).sum(-1)*20,torch.zeros(len(batch),dtype=torch.long,device=device));opt.zero_grad(set_to_none=True);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],1.0));opt.step();logs.append({'step':step,'loss':float(loss.detach().cpu()),'gradient_norm':gn})
 state=adapter_state(model);a.output_dir.mkdir(parents=True,exist_ok=True);torch.save({'adapter_state':state,'method':'centralized_pc1_hard_negative','seed':a.seed},a.output_dir/'adapter.pt');metrics=evaluate_pool(model,tok,state,a.development,a.pool,'hotpotqa',a.output_dir/'adapted_pool.jsonl',a.output_dir/'contexts.jsonl',100,device,32)
 with (a.output_dir/'pc_training_log.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=logs[0].keys());w.writeheader();w.writerows(logs)
 result={'status':'complete','control':'PC-1','single_changed_factor':'explicit_train_only_hard_negatives','steps':a.steps,'lr':a.lr,'adapter_bytes':state_bytes(state),'loss_first':logs[0]['loss'],'loss_last':logs[-1]['loss'],**metrics};(a.output_dir/'pc_training_log.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
