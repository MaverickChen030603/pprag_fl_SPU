#!/usr/bin/env python3
"""Train the frozen paper-era RAGRoute router ensemble on B3F historical rows."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
from torch import nn

class B3FRouter(nn.Module):
    """Paper-era RAGRoute MLP shared by frozen training and blind inference."""
    def __init__(self, dim):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(dim,256),nn.LayerNorm(256),nn.ReLU(),nn.Dropout(.4),nn.Linear(256,128),nn.LayerNorm(128),nn.ReLU(),nn.Dropout(.4),nn.Linear(128,1))
    def forward(self,v): return self.network(v).squeeze(-1)

def split_indices(query_ids, dataset):
    unique=sorted(set(map(str,query_ids)))
    if len(unique)!=5000: raise ValueError(f"expected 5,000 queries, found {len(unique)}")
    dev=set(sorted(unique,key=lambda q:hashlib.sha256(f"v20-b3f-router-dev-20260928-v1|{dataset}|{q}".encode()).hexdigest())[:500])
    mask=np.fromiter((str(q) in dev for q in query_ids),dtype=bool,count=len(query_ids))
    return np.flatnonzero(~mask),np.flatnonzero(mask)

def set_seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",choices=("hotpotqa","2wikimultihopqa","musique"),required=True); p.add_argument("--features",type=Path,required=True); p.add_argument("--contract",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",default="cpu"); p.add_argument("--batch-size",type=int,default=512); a=p.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    import torch
    from sklearn.metrics import average_precision_score,roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader,TensorDataset
    contract=json.loads(a.contract.read_text())
    if contract["model"]["seeds"]!=[0,1,2] or contract["model"]["epochs"]!=150: raise ValueError("frozen contract mismatch")
    z=np.load(a.features); x=z["features"].astype(np.float32); y=z["labels"].astype(np.float32)
    train,valid=split_indices(z["query_ids"],a.dataset)
    if len(train)!=36000 or len(valid)!=4000: raise ValueError("invalid query-level Router-Dev split")
    scaler=StandardScaler().fit(x[train]); tx=scaler.transform(x[train]).astype(np.float32); vx=scaler.transform(x[valid]).astype(np.float32); ty,vy=y[train],y[valid]
    pos,neg=float(ty.sum()),float(len(ty)-ty.sum())
    if not pos or not neg: raise ValueError("one-class historical labels")
    a.output_dir.mkdir(parents=True); np.savez_compressed(a.output_dir/"scaler.npz",mean=scaler.mean_.astype(np.float32),scale=scaler.scale_.astype(np.float32))
    device=torch.device(a.device); summaries=[]
    for seed in (0,1,2):
        set_seed(seed); model=B3FRouter(x.shape[1]).to(device); loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg/pos,device=device)); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5); cyclic=torch.optim.lr_scheduler.CyclicLR(opt,base_lr=1e-3,max_lr=5e-3,step_size_up=10,mode="triangular2",cycle_momentum=False); step=torch.optim.lr_scheduler.StepLR(opt,step_size=50,gamma=.05)
        loader=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(ty)),batch_size=a.batch_size,shuffle=True,num_workers=0); best_acc,best_epoch,best_state=-1.,-1,None
        for epoch in range(150):
            model.train()
            for bx,by in loader:
                opt.zero_grad(set_to_none=True); loss=loss_fn(model(bx.to(device)),by.to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
                if epoch<115: cyclic.step()
            if epoch>=115: step.step()
            model.eval()
            with torch.inference_mode(): prob=torch.sigmoid(model(torch.from_numpy(vx).to(device))).cpu().numpy()
            acc=float(((prob>=.5)==vy).mean())
            if acc>best_acc: best_acc,best_epoch,best_state=acc,epoch+1,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        ckpt=a.output_dir/f"seed_{seed}.pt"; torch.save({"state_dict":best_state,"input_dim":int(x.shape[1]),"seed":seed,"best_epoch":best_epoch},ckpt); model.load_state_dict(best_state); model.eval()
        with torch.inference_mode(): prob=torch.sigmoid(model(torch.from_numpy(vx).to(device))).cpu().numpy()
        summaries.append({"seed":seed,"best_validation_accuracy":best_acc,"best_epoch":best_epoch,"validation_auroc":float(roc_auc_score(vy,prob)),"validation_auprc":float(average_precision_score(vy,prob)),"checkpoint":ckpt.name,"checkpoint_sha256":digest(ckpt)})
    m={"status":"complete_three_seed_historical_training","dataset":a.dataset,"features":str(a.features),"feature_sha256":digest(a.features),"contract_sha256":digest(a.contract),"train_rows":36000,"validation_rows":4000,"train_queries":4500,"validation_queries":500,"selection_metric":"validation accuracy","model":"1556->256->128->1 with LayerNorm/ReLU/Dropout(0.4)","seeds":summaries,"ensemble":"mean sigmoid probabilities across seeds 0,1,2","fresh_labels_opened":False}
    (a.output_dir/"training_manifest.json").write_text(json.dumps(m,indent=2)+"\n"); print(json.dumps(m,indent=2))
if __name__=="__main__": main()
