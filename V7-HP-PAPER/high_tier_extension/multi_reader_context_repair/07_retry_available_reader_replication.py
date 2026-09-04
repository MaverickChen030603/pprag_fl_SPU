#!/usr/bin/env python3
import json, os, subprocess, traceback, time, re, math, random
from pathlib import Path

ROOT=Path('/home/iiserver31/projects/FedE4RAG-main')
OUT=ROOT/'V7-HP-PAPER/high_tier_extension/multi_reader_context_repair'
AUDIT=OUT/'outputs/audit'; AUDIT.mkdir(parents=True, exist_ok=True)
SNAP=OUT/'outputs/context_snapshots/final1000_baseline_selected_contexts.jsonl'

def read_json(p, default=None):
    p=Path(p)
    return json.loads(p.read_text()) if p.exists() else ({} if default is None else default)

def iter_jsonl(p):
    if not Path(p).exists(): return []
    with Path(p).open() as f:
        return [json.loads(x) for x in f if x.strip()]

def write_json(p,o):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')

def write_jsonl(p,rows):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')

def gpu_state():
    try:
        txt=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'], text=True, timeout=20)
        rows=[]
        for line in txt.strip().splitlines():
            parts=[x.strip() for x in line.split(',')]
            used=int(parts[2].split()[0]); total=int(parts[3].split()[0])
            rows.append({'index':int(parts[0]),'name':parts[1],'memory_used_mib':used,'memory_total_mib':total,'memory_free_mib':total-used,'utilization_gpu':parts[4]})
        return rows
    except Exception as e:
        return [{'error':repr(e)}]

def current_processes():
    try:
        txt=subprocess.check_output(['bash','-lc',"ps -eo pid,etime,stat,pcpu,pmem,cmd | grep -E 'V7-HP-PAPER|multi_reader_context_repair|flan|t5' | grep -v grep || true"], text=True, timeout=20)
        return txt.strip().splitlines()
    except Exception as e:
        return [repr(e)]

def prompt(question, docs):
    parts=[]
    for d in docs:
        text=' '.join(d.get('sentences',[])[:6])
        parts.append(f"[{d.get('title','')}] {text}")
    return 'Answer the question using the provided context.\n\nContext:\n'+"\n".join(parts)+f"\n\nQuestion: {question}\n\nAnswer:"

readers=read_json(AUDIT/'available_local_readers.json',{}).get('readers',[])
usable=[r for r in readers if r.get('can_load_model_local_files_only')]
gpus=gpu_state(); max_free=max([g.get('memory_free_mib',0) for g in gpus] or [0])
feas={'gpu_state':gpus,'current_v7_hp_paper_processes':current_processes(),'max_gpu_free_mib':max_free}
if max_free>=12000:
    feas.update({'decision':'gpu_1000_allowed','recommended_device':'cuda','max_examples':1000})
else:
    feas.update({'decision':'cpu_smoke_or_stop','recommended_device':'cpu','max_examples':50})
write_json(AUDIT/'reader_runtime_feasibility.json',feas)

priority=['google/flan-t5-base','t5-base','allenai/unifiedqa-t5-base','google/flan-t5-large','t5-large','allenai/unifiedqa-t5-large','google/flan-t5-xl']
chosen=None
for name in priority:
    for r in usable:
        if r['model_name']==name: chosen=r; break
    if chosen: break

if not chosen:
    d=OUT/'outputs/reader_outputs/no_available_reader'; d.mkdir(parents=True,exist_ok=True)
    s={'reader_name':'none','status':'not_executed','reason':'local_model_unavailable_or_runtime_blocked','error_type':'no_local_reader','error_message':'No candidate reader can load with local_files_only=True.','device':None,'gpu_memory':gpus,'recommendation':'Cache google/flan-t5-base or provide a local model path; keep multi-reader as limitation.'}
    write_json(d/'reader_run_summary.json',s); write_jsonl(d/'baseline_predictions.jsonl',[]); write_jsonl(d/'selected_predictions.jsonl',[])
    print(json.dumps(s,ensure_ascii=False)); raise SystemExit(0)

reader=chosen['model_name']; safe=reader.replace('/','__')
device='cuda' if feas['recommended_device']=='cuda' and os.environ.get('V7_HP_FORCE_CPU','0')!='1' else 'cpu'
max_examples=feas['max_examples']
# If only large/xl is available and GPU not available, do not run; avoid huge CPU job.
if device=='cpu' and any(x in reader for x in ['large','xl']):
    d=OUT/'outputs/reader_outputs'/safe; d.mkdir(parents=True,exist_ok=True)
    s={'reader_name':reader,'status':'not_executed','reason':'local_model_unavailable_or_runtime_blocked','error_type':'runtime_blocked','error_message':'Only a large/xl reader is locally available, but GPU free memory is below 12GB; CPU run is not appropriate.','local_model_path':chosen.get('local_path'),'device':device,'gpu_memory':gpus,'recommendation':'Wait for GPU or use cached flan-t5-base.'}
    write_json(d/'reader_run_summary.json',s); write_jsonl(d/'baseline_predictions.jsonl',[]); write_jsonl(d/'selected_predictions.jsonl',[])
    print(json.dumps(s,ensure_ascii=False)); raise SystemExit(0)

snap=iter_jsonl(SNAP)[:max_examples]
d=OUT/'outputs/reader_outputs'/safe; d.mkdir(parents=True,exist_ok=True)
base_path=d/'baseline_predictions.jsonl'; sel_path=d/'selected_predictions.jsonl'
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok=AutoTokenizer.from_pretrained(reader, local_files_only=True)
    model=AutoModelForSeq2SeqLM.from_pretrained(reader, local_files_only=True)
    model.to(device); model.eval()
    def gen(p):
        inputs=tok(p, return_tensors='pt', truncation=True, max_length=1024).to(device)
        with torch.no_grad(): ids=model.generate(**inputs, max_new_tokens=32, num_beams=1)
        return tok.decode(ids[0], skip_special_tokens=True).strip()
    bp=[]; sp=[]; t0=time.time()
    for i,x in enumerate(snap):
        bp.append({'query_id':x['query_id'],'prediction':gen(prompt(x['question'],x['baseline_context'])),'answer':x['answer']})
        sp.append({'query_id':x['query_id'],'prediction':gen(prompt(x['question'],x['selected_context'])),'answer':x['answer']})
        if (i+1)%10==0:
            write_jsonl(base_path,bp); write_jsonl(sel_path,sp)
    write_jsonl(base_path,bp); write_jsonl(sel_path,sp)
    s={'reader_name':reader,'status':'completed','num_examples':len(snap),'local_model_path':chosen.get('local_path'),'device':device,'runtime_decision':feas['decision'],'elapsed_sec':time.time()-t0,'gpu_memory':gpu_state(),'recommendation':'Evaluate outputs and keep as appendix robustness/smoke, not main result.'}
except Exception as e:
    s={'reader_name':reader,'status':'failed','reason':'local_model_unavailable_or_runtime_blocked','error_type':type(e).__name__,'error_message':repr(e),'traceback':traceback.format_exc()[-4000:],'local_model_path':chosen.get('local_path'),'device':device,'gpu_memory':gpu_state(),'recommendation':'Do not alter v2.3; keep multi-reader as limitation unless model/runtime is fixed.'}
    write_jsonl(base_path,[]); write_jsonl(sel_path,[])
write_json(d/'reader_run_summary.json',s)
print(json.dumps(s,ensure_ascii=False))
