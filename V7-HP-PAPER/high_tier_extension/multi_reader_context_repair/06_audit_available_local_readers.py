#!/usr/bin/env python3
import json, os, subprocess, traceback
from pathlib import Path

ROOT=Path('/home/iiserver31/projects/FedE4RAG-main')
OUT=ROOT/'V7-HP-PAPER/high_tier_extension/multi_reader_context_repair'
AUDIT=OUT/'outputs/audit'
REPORTS=OUT/'reports'
AUDIT.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)

CANDIDATES=[
 'google/flan-t5-base','google/flan-t5-large','google/flan-t5-xl','t5-base','t5-large','allenai/unifiedqa-t5-base','allenai/unifiedqa-t5-large'
]
SEARCH_DIRS=[
 Path.home()/'.cache/huggingface/hub', Path.home()/'.cache/huggingface/transformers',
 Path('/home/iiserver31/.cache/huggingface/hub'), Path('/home/iiserver31/.cache/huggingface/transformers'),
 ROOT/'models', ROOT/'cache', ROOT/'V7-HP-PAPER/cache'
]

def find_local_path(model_name):
    safe='models--'+model_name.replace('/','--')
    hits=[]
    for d in SEARCH_DIRS:
        if not d.exists(): continue
        p=d/safe
        if p.exists(): hits.append(str(p))
        # direct named dirs for t5-base style
        q=d/model_name.replace('/','__')
        if q.exists(): hits.append(str(q))
        q=d/model_name.split('/')[-1]
        if q.exists(): hits.append(str(q))
    return hits[0] if hits else None

def inspect_path(p):
    if not p: return {'has_config':False,'has_tokenizer':False,'has_model_weights':False}
    pp=Path(p)
    files=[]
    if pp.exists():
        for f in pp.rglob('*'):
            if f.is_file(): files.append(f.name)
    names=set(files)
    return {
        'has_config': 'config.json' in names,
        'has_tokenizer': any(x in names for x in ['tokenizer.json','spiece.model','tokenizer_config.json','vocab.json']),
        'has_model_weights': any(x.endswith(('.bin','.safetensors')) or x=='pytorch_model.bin' for x in names),
        'num_files_seen': len(files),
    }

def try_load(model_name):
    out={'can_load_tokenizer_local_files_only':False,'can_load_model_local_files_only':False,'load_error':None}
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok=AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        out['can_load_tokenizer_local_files_only']=True
        model=AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        out['can_load_model_local_files_only']=True
        out['parameter_count']=sum(p.numel() for p in model.parameters())
        del model, tok
    except Exception as e:
        out['load_error']=repr(e)
        out['traceback_tail']=traceback.format_exc()[-2000:]
    return out

def gpu_state():
    try:
        txt=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'], text=True, timeout=20)
        rows=[]
        for line in txt.strip().splitlines():
            parts=[x.strip() for x in line.split(',')]
            if len(parts)>=5:
                used=int(parts[2].split()[0]); total=int(parts[3].split()[0])
                rows.append({'index':parts[0],'name':parts[1],'memory_used_mib':used,'memory_total_mib':total,'memory_free_mib':total-used,'utilization_gpu':parts[4]})
        return rows
    except Exception as e:
        return [{'error':repr(e)}]

readers=[]
for m in CANDIDATES:
    lp=find_local_path(m)
    info={'model_name':m,'local_path':lp,'estimated_gpu_memory':'~1-2GB base, ~3-6GB large, >10GB xl depending dtype','recommended_device':'gpu if free memory >=12GB else cpu smoke only'}
    info.update(inspect_path(lp))
    info.update(try_load(m))
    readers.append(info)

gpus=gpu_state()
usable=[r for r in readers if r.get('can_load_tokenizer_local_files_only') and r.get('can_load_model_local_files_only')]
summary={
 'status':'complete',
 'search_dirs':[str(x) for x in SEARCH_DIRS],
 'readers':readers,
 'usable_readers':[r['model_name'] for r in usable],
 'gpu_state':gpus,
 'decision':'reader_available' if usable else 'no_reader_available',
}
(AUDIT/'available_local_readers.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
lines=['# Available Local Readers Report','',f"Decision: **{summary['decision']}**",'','| model | local_path | config | tokenizer | weights | can_load | note |','| --- | --- | --- | --- | --- | --- | --- |']
for r in readers:
    can=r.get('can_load_model_local_files_only')
    note='ok' if can else (r.get('load_error') or '')[:120].replace('\n',' ')
    lines.append(f"| `{r['model_name']}` | `{r.get('local_path')}` | {r.get('has_config')} | {r.get('has_tokenizer')} | {r.get('has_model_weights')} | {can} | {note} |")
lines += ['','## GPU Snapshot','']
for g in gpus: lines.append(f"- {g}")
(REPORTS/'available_local_readers_report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'decision':summary['decision'],'usable_readers':summary['usable_readers']},ensure_ascii=False))
