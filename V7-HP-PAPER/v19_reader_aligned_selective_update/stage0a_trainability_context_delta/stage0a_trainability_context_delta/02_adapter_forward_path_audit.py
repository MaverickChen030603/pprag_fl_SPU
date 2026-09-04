#!/usr/bin/env python3
"""Prove trained LoRA tensors are in the actual BGE retrieval forward path."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from audit_common import LoRALinear, encode, model_and_tokenizer, query_prefix, rows, support_titles, tensor_hash


def set_enabled(model: torch.nn.Module, enabled: bool) -> list[float]:
    old = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            old.append(module.scale); module.scale = module.scale if enabled else 0.0
    return old


def restore(model: torch.nn.Module, scales: list[float]) -> None:
    index = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.scale = scales[index]; index += 1


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--smoke-root", type=Path, required=True); p.add_argument("--development", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True); p.add_argument("--model", default="BAAI/bge-base-en-v1.5"); p.add_argument("--device", default="cuda")
    args = p.parse_args(); args.output_root.mkdir(parents=True, exist_ok=True); device=torch.device(args.device)
    source={str(x['query_id']):x for x in rows(args.development)}
    probe=[]
    for pool in rows(args.pool):
        row=source[str(pool['query_id'])]; gold=support_titles(row)
        support=next((d for d in pool['pool'] if ' '.join(d['title'].lower().split()) in gold), None)
        negatives=[d for d in pool['pool'] if ' '.join(d['title'].lower().split()) not in gold][:2]
        if support and len(negatives)==2: probe += [query_prefix(row['question']), support['title']+': '+support['text'], *[x['title']+': '+x['text'] for x in negatives]]
        if len(probe)>=40: break
    states={p.parent.name:torch.load(p,map_location='cpu')['adapter_state'] for p in sorted(args.smoke_root.glob('*/adapter.pt'))}
    base_state=states['frozen']; data=[]; hook_trace={}
    base_model, tok=model_and_tokenizer(args.model, base_state, device); base=encode(base_model,tok,probe,device)
    for condition,state in states.items():
        model,tok=model_and_tokenizer(args.model,state,device); calls=[]
        hooks=[m.register_forward_hook(lambda _m,_i,_o,calls=calls: calls.append(1)) for m in model.modules() if isinstance(m,LoRALinear)]
        enabled=encode(model,tok,probe,device); scales=set_enabled(model,False); disabled=encode(model,tok,probe,device); restore(model,scales)
        for h in hooks: h.remove()
        reloaded,_=model_and_tokenizer(args.model,state,device); reload_output=encode(reloaded,tok,probe,device)
        hook_trace[condition]={"target_module_match_count":len(scales),"forward_hook_call_count":len(calls),"scales_nonzero":all(x!=0 for x in scales)}
        for label,value in (("enabled",enabled),("disabled",disabled),("reloaded",reload_output)):
            data.append({"condition":condition,"mode":label,"pooled_embedding_hash":tensor_hash(value),"l2_vs_base":float((value-base).norm(dim=1).mean()),"max_abs_vs_base":float((value-base).abs().max()),"l2_vs_enabled":float((value-enabled).norm(dim=1).mean())})
    with (args.output_root/'adapter_forward_outputs.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
    (args.output_root/'forward_hook_trace.json').write_text(json.dumps(hook_trace,indent=2)+'\n',encoding='utf-8')
    enabled_rows=[x for x in data if x['mode']=='enabled']; changed={x['condition']:x['l2_vs_base']>1e-8 for x in enabled_rows}
    text=['# Adapter Forward-path Audit','',f'Probe texts: {len(probe)}; cache: bypassed.','', '| Condition | enabled L2 vs base | disabled L2 vs base | hooks |','|---|---:|---:|---:|']
    for condition in states:
        e=next(x for x in data if x['condition']==condition and x['mode']=='enabled');d=next(x for x in data if x['condition']==condition and x['mode']=='disabled')
        text.append(f"| {condition} | {e['l2_vs_base']:.8f} | {d['l2_vs_base']:.8f} | {hook_trace[condition]['forward_hook_call_count']} |")
    text += ['',f'Forward delta detected: `{changed}`. Disabled outputs should match frozen within numerical tolerance; reload outputs are hashed separately. The LoRA analytic merge is equivalent to the module residual by construction; no persistent merge operation is used in V19.']
    (args.output_root/'adapter_forward_path_audit.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({'status':'complete','forward_delta_detected':changed},indent=2))

if __name__=='__main__': main()
