#!/usr/bin/env python3
import json, re, math, random, statistics, time, subprocess, traceback, os
from collections import defaultdict
from pathlib import Path

ROOT=Path('/home/iiserver31/projects/FedE4RAG-main')
PAPER=ROOT/'V7-HP-PAPER'
V23=PAPER/'selector_v2_3'
V22=PAPER/'selector_v2_2'
HP4=ROOT/'V7-HP4/data/hotpot_validation_1000.json'
OUT=PAPER/'high_tier_extension/multi_reader_context_repair'


def ensure_dirs():
    for rel in ['outputs/context_snapshots','outputs/reader_outputs','outputs/metrics','outputs/tables','outputs/audit','reports']:
        (OUT/rel).mkdir(parents=True, exist_ok=True)

def read_json(p, default=None):
    p=Path(p)
    if not p.exists(): return {} if default is None else default
    return json.loads(p.read_text())

def iter_jsonl(p):
    p=Path(p)
    if not p.exists(): return
    with p.open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def write_json(p,o):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')

def write_jsonl(p, rows):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')

def md_table(headers, rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in r)+' |' for r in rows)+'\n'

def fmt(x, nd=4, signed=False):
    if x is None: return 'NA'
    if isinstance(x,str): return x
    try:
        x=float(x)
        if math.isnan(x): return 'NA'
        s=f'{x:.{nd}f}'
        return '+'+s if signed and x>0 else s
    except Exception:
        return str(x)

def parse_reference(ref):
    # Reference format: [Title] sentence text [Next Title] ...
    docs=[]
    matches=list(re.finditer(r'\[([^\]]+)\]\s*', ref or ''))
    for i,m in enumerate(matches):
        title=m.group(1).strip()
        start=m.end(); end=matches[i+1].start() if i+1<len(matches) else len(ref)
        text=(ref[start:end] or '').strip()
        sents=split_sentences(text)
        docs.append({'title':title,'sentences':sents,'text':text})
    return docs

def split_sentences(text):
    text=' '.join((text or '').split())
    if not text: return []
    parts=re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def normalize_title(t):
    return re.sub(r'\s+',' ',(t or '').replace('&amp;','&')).strip().lower()

def get_doc_map(item):
    docs=parse_reference(item.get('reference',''))
    mp={normalize_title(d['title']):d for d in docs}
    return docs, mp

def source_inventory():
    ensure_dirs()
    files={
        'v23_per_example_delta': V23/'outputs/final_1000/per_example_delta.jsonl',
        'v23_action_labels': V23/'outputs/labels/action_labels.jsonl',
        'v22_effective_action_table': V22/'outputs/action_table/effective_action_table.jsonl',
        'hp4_hotpot_validation_1000': HP4,
        'v23_summary': V23/'outputs/final_1000/final_1000_crossfit_summary.json',
    }
    per=list(iter_jsonl(files['v23_per_example_delta']))
    hp4=read_json(HP4, [])
    hp4_ids={x.get('_id') for x in hp4}
    sample=per[:50]
    inv={
        'files':{k:{'path':str(v),'exists':v.exists(),'size':v.stat().st_size if v.exists() else 0} for k,v in files.items()},
        'num_v23_examples':len(per),
        'num_hp4_examples':len(hp4),
        'sample_query_id_match_rate':sum(1 for r in sample if r.get('query_id') in hp4_ids)/max(1,len(sample)),
        'query_id_recoverable': bool(per),
        'question_recoverable': all((next((x for x in hp4 if x.get('_id')==r.get('query_id')),{}) or {}).get('question') for r in sample),
        'answer_recoverable': all((next((x for x in hp4 if x.get('_id')==r.get('query_id')),{}) or {}).get('answer') for r in sample),
        'baseline_titles_recoverable': all(r.get('baseline_titles') for r in sample if not r.get('fallback')) or any(r.get('baseline_titles') for r in sample),
        'selected_titles_recoverable': any(r.get('candidate_titles') or r.get('selected_titles') for r in sample),
        'baseline_context_text_recoverable': HP4.exists(),
        'selected_context_text_recoverable': HP4.exists(),
        'supporting_facts_recoverable': False,
        'supporting_titles_recoverable': all((next((x for x in hp4 if x.get('_id')==r.get('query_id')),{}) or {}).get('supporting_titles') for r in sample),
        'metrics_labels_recoverable': files['v23_per_example_delta'].exists(),
        'context_source': str(HP4),
        'notes': ['HotpotQA context text is recovered from V7-HP4/data/hotpot_validation_1000.json reference field.', 'supporting_titles are available; sentence-level supporting_facts are not present in HP4 converted file.'],
    }
    write_json(OUT/'outputs/audit/context_source_inventory.json', inv)
    report=f"""# Context Source Inventory\n\nStatus: **{'usable' if inv['baseline_context_text_recoverable'] else 'not usable'}**.\n\n- v2.3 final examples: {len(per)}\n- HP4 Hotpot validation examples: {len(hp4)}\n- sample query id match rate: {inv['sample_query_id_match_rate']:.3f}\n- context source: `{HP4}`\n\nRecoverability:\n\n- query_id: {inv['query_id_recoverable']}\n- question: {inv['question_recoverable']}\n- answer: {inv['answer_recoverable']}\n- baseline_titles: {inv['baseline_titles_recoverable']}\n- selected_titles: {inv['selected_titles_recoverable']}\n- baseline_context_text: {inv['baseline_context_text_recoverable']}\n- selected_context_text: {inv['selected_context_text_recoverable']}\n- supporting_titles: {inv['supporting_titles_recoverable']}\n- sentence-level supporting_facts: {inv['supporting_facts_recoverable']}\n- metrics labels: {inv['metrics_labels_recoverable']}\n"""
    (OUT/'reports/context_source_inventory.md').write_text(report)
    return inv

def selected_rows():
    rows=list(iter_jsonl(V23/'outputs/final_1000/per_example_delta.jsonl'))
    return rows

def materialize_contexts():
    ensure_dirs()
    inv=source_inventory()
    hp4={x.get('_id'):x for x in read_json(HP4, [])}
    rows=selected_rows()
    label_lookup={}
    for lr in iter_jsonl(V23/'outputs/labels/action_labels.jsonl'):
        q=lr.get('query_id')
        if q and q not in label_lookup:
            label_lookup[q]=lr
    out=[]; missing=[]
    for r in rows:
        qid=r.get('query_id') or r.get('id')
        item=hp4.get(qid,{})
        docs,mp=get_doc_map(item)
        lookup=label_lookup.get(qid,{})
        base_titles=r.get('baseline_titles') or lookup.get('baseline_titles') or []
        sel_titles=r.get('candidate_titles') or r.get('selected_titles') or (base_titles if r.get('fallback') else lookup.get('candidate_titles')) or base_titles
        def docs_for(titles):
            ds=[]; miss=[]
            for t in titles:
                d=mp.get(normalize_title(t))
                if d:
                    ds.append({'title':d['title'],'sentences':d['sentences']})
                else:
                    miss.append(t)
            return ds,miss
        bctx,bmiss=docs_for(base_titles)
        sctx,smiss=docs_for(sel_titles)
        rec={
            'query_id':qid,
            'question':item.get('question') or r.get('question') or '',
            'answer':item.get('answer',''),
            'baseline_titles':base_titles,
            'selected_titles':sel_titles,
            'baseline_context':bctx,
            'selected_context':sctx,
            'supporting_facts': item.get('supporting_facts', []),
            'supporting_titles': item.get('supporting_titles', []),
            'selected_action_name': r.get('candidate_name') or r.get('selected_mode') or ('baseline_fallback' if r.get('fallback') else ''),
            'candidate_family': r.get('candidate_family') or ('baseline' if r.get('fallback') else ''),
            'source':'hotpotqa_final_1000_v2_3_frozen_context_repaired_from_hp4_validation_1000',
            'frozen_metrics': {k:r.get(k) for k in ['answer_f1','baseline_answer_f1','answer_f1_delta','joint_f1','baseline_joint_f1','joint_f1_delta','sp_f1','baseline_sp_f1','sp_f1_delta','support_recall_at_k','baseline_support_recall_at_k','support_recall_at_k_delta'] if k in r},
            'missing_baseline_titles':bmiss,
            'missing_selected_titles':smiss,
        }
        if bmiss or smiss or not bctx or not sctx or not item.get('answer'):
            missing.append({'query_id':qid,'baseline_missing':bmiss,'selected_missing':smiss,'has_answer':bool(item.get('answer'))})
        out.append(rec)
    write_jsonl(OUT/'outputs/context_snapshots/final1000_baseline_selected_contexts.jsonl', out)
    n=len(out)
    summary={
        'num_examples':n,
        'num_with_question':sum(1 for x in out if x['question']),
        'num_with_answer':sum(1 for x in out if x['answer']),
        'num_with_baseline_context':sum(1 for x in out if x['baseline_context']),
        'num_with_selected_context':sum(1 for x in out if x['selected_context']),
        'num_with_support_labels':sum(1 for x in out if x['supporting_titles']),
        'avg_baseline_docs':sum(len(x['baseline_context']) for x in out)/max(1,n),
        'avg_selected_docs':sum(len(x['selected_context']) for x in out)/max(1,n),
        'num_missing_context':sum(1 for x in out if not x['baseline_context'] or not x['selected_context'] or x['missing_baseline_titles'] or x['missing_selected_titles']),
        'num_missing_answer':sum(1 for x in out if not x['answer']),
        'num_missing_support':sum(1 for x in out if not x['supporting_titles']),
        'missing_examples_sample':missing[:20],
    }
    write_json(OUT/'outputs/context_snapshots/context_snapshot_summary.json', summary)
    return summary

def validate_context_snapshots():
    ensure_dirs()
    snap=list(iter_jsonl(OUT/'outputs/context_snapshots/final1000_baseline_selected_contexts.jsonl'))
    summary=read_json(OUT/'outputs/context_snapshots/context_snapshot_summary.json')
    exact_same_effective=0; dup_base=0; dup_sel=0; empty=[]
    for x in snap:
        bt=[d['title'] for d in x['baseline_context']]
        st=[d['title'] for d in x['selected_context']]
        if len(bt)!=len(set(bt)): dup_base+=1
        if len(st)!=len(set(st)): dup_sel+=1
        if not bt or not st: empty.append(x['query_id'])
        if bt==st and x.get('candidate_family') not in {'baseline',''} and x.get('selected_action_name')!='baseline_fallback': exact_same_effective+=1
    passed=summary.get('num_missing_context',999)==0 and summary.get('num_with_selected_context',0)>=990 and summary.get('num_with_baseline_context',0)>=990
    audit={
        'status':'pass' if passed else 'fail',
        'num_examples':len(snap),
        'num_missing_context':summary.get('num_missing_context'),
        'num_with_selected_context':summary.get('num_with_selected_context'),
        'num_with_baseline_context':summary.get('num_with_baseline_context'),
        'duplicate_baseline_title_examples':dup_base,
        'duplicate_selected_title_examples':dup_sel,
        'empty_context_examples':empty[:50],
        'selected_equals_baseline_but_effective_count':exact_same_effective,
        'can_run_reader_replication':passed,
        'main_result_reproduction_note':'This audit materializes text only; original flan-t5-large metrics are preserved from frozen v2.3 outputs and are not overwritten.',
    }
    write_json(OUT/'outputs/audit/context_snapshot_audit.json', audit)
    report=f"""# Context Materialization Report\n\nStatus: **{audit['status']}**.\n\n- examples: {len(snap)}\n- with baseline context: {summary.get('num_with_baseline_context')}\n- with selected context: {summary.get('num_with_selected_context')}\n- missing context: {summary.get('num_missing_context')}\n- with answer: {summary.get('num_with_answer')}\n- with supporting titles: {summary.get('num_with_support_labels')}\n- duplicate baseline title examples: {dup_base}\n- duplicate selected title examples: {dup_sel}\n\nReader replication allowed: **{passed}**.\n\nThe contexts were reconstructed from `V7-HP4/data/hotpot_validation_1000.json` using v2.3 frozen baseline/selected titles. The frozen v2.3 metrics are kept as the main result and are not modified by this repair.\n"""
    (OUT/'reports/context_materialization_report.md').write_text(report)
    return audit

def normalize_answer(s):
    import string
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b',' ',text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc((s or '').lower())))

def f1_score(pred, gold):
    p=normalize_answer(pred).split(); g=normalize_answer(gold).split()
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    common={}
    for t in p: common[t]=common.get(t,0)+1
    num=0
    for t in g:
        if common.get(t,0)>0:
            num+=1; common[t]-=1
    if num==0: return 0.0
    prec=num/len(p); rec=num/len(g)
    return 2*prec*rec/(prec+rec)

def em_score(pred,gold): return 1.0 if normalize_answer(pred)==normalize_answer(gold) else 0.0

def context_prompt(question, docs):
    parts=[]
    for d in docs:
        text=' '.join(d.get('sentences',[])[:6])
        parts.append(f"[{d.get('title','')}] {text}")
    ctx='\n'.join(parts)
    return f"Answer the question using the provided context.\n\nContext:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"

def gpu_state():
    try:
        out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader'], text=True, timeout=20)
        return out.strip()
    except Exception as e:
        return repr(e)

def run_reader_replication():
    ensure_dirs()
    audit=read_json(OUT/'outputs/audit/context_snapshot_audit.json')
    snap=list(iter_jsonl(OUT/'outputs/context_snapshots/final1000_baseline_selected_contexts.jsonl'))
    readers=['google/flan-t5-base']
    summaries={}
    if audit.get('status')!='pass':
        for reader in readers:
            d=OUT/'outputs/reader_outputs'/reader.replace('/','__'); d.mkdir(parents=True,exist_ok=True)
            s={'reader_name':reader,'status':'not_executed','error':'context snapshot audit failed','gpu_memory':gpu_state(),'recommendation':'fix context materialization first'}
            write_json(d/'reader_run_summary.json',s); summaries[reader]=s
        return summaries
    for reader in readers:
        safe=reader.replace('/','__'); d=OUT/'outputs/reader_outputs'/safe; d.mkdir(parents=True,exist_ok=True)
        base_path=d/'baseline_predictions.jsonl'; sel_path=d/'selected_predictions.jsonl'
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            device='cpu'
            # GPUs are often shared; use CPU unless explicitly enabled.
            if os.environ.get('V7_HP_USE_GPU','0')=='1' and torch.cuda.is_available(): device='cuda'
            tok=AutoTokenizer.from_pretrained(reader, local_files_only=True)
            model=AutoModelForSeq2SeqLM.from_pretrained(reader, local_files_only=True)
            model.to(device); model.eval()
            def gen(prompt):
                inputs=tok(prompt, return_tensors='pt', truncation=True, max_length=1024).to(device)
                with torch.no_grad():
                    ids=model.generate(**inputs, max_new_tokens=32, num_beams=1)
                return tok.decode(ids[0], skip_special_tokens=True).strip()
            bpred=[]; spred=[]
            t0=time.time()
            for i,x in enumerate(snap):
                bp=gen(context_prompt(x['question'], x['baseline_context']))
                sp=gen(context_prompt(x['question'], x['selected_context']))
                bpred.append({'query_id':x['query_id'],'prediction':bp,'answer':x['answer']})
                spred.append({'query_id':x['query_id'],'prediction':sp,'answer':x['answer']})
                if (i+1)%50==0:
                    write_jsonl(base_path,bpred); write_jsonl(sel_path,spred)
            write_jsonl(base_path,bpred); write_jsonl(sel_path,spred)
            s={'reader_name':reader,'status':'completed','num_examples':len(snap),'device':device,'elapsed_sec':time.time()-t0,'gpu_memory':gpu_state()}
        except Exception as e:
            s={'reader_name':reader,'status':'failed','error':repr(e),'traceback':traceback.format_exc()[-4000:],'gpu_memory':gpu_state(),'recommendation':'Ensure model is cached locally or rerun with network/model availability; do not alter v2.3 main result.'}
            # create empty prediction files for downstream evaluator
            write_jsonl(base_path,[]); write_jsonl(sel_path,[])
        write_json(d/'reader_run_summary.json',s); summaries[reader]=s
    return summaries

def support_metrics(titles, supporting_titles):
    st={normalize_title(t) for t in supporting_titles}
    got={normalize_title(t) for t in titles}
    if not st: return 0.0, 0.0
    tp=len(st & got); recall=tp/len(st)
    prec=tp/len(got) if got else 0.0
    spf1=0.0 if prec+recall==0 else 2*prec*recall/(prec+recall)
    return recall, spf1

def bootstrap(vals, n=1000, seed=13):
    if not vals: return {'mean_delta':None,'ci95':[None,None],'p_value':None}
    rng=random.Random(seed); means=[]; m=len(vals)
    for _ in range(n): means.append(sum(vals[rng.randrange(m)] for _ in range(m))/m)
    means.sort(); mean=sum(vals)/m
    p=sum(1 for x in means if x<=0)/n if mean>=0 else sum(1 for x in means if x>=0)/n
    return {'mean_delta':mean,'ci95':[means[int(0.025*n)],means[int(0.975*n)-1]],'p_value':p}

def evaluate_multi_reader_outputs():
    ensure_dirs()
    snap={x['query_id']:x for x in iter_jsonl(OUT/'outputs/context_snapshots/final1000_baseline_selected_contexts.jsonl')}
    readers=[]
    for d in (OUT/'outputs/reader_outputs').glob('*'):
        if d.is_dir(): readers.append(d)
    metrics={}; sig={}; per=[]; table_rows=[]
    for d in readers:
        reader=d.name.replace('__','/')
        run=read_json(d/'reader_run_summary.json')
        b=list(iter_jsonl(d/'baseline_predictions.jsonl'))
        s=list(iter_jsonl(d/'selected_predictions.jsonl'))
        if run.get('status')!='completed' or not b or not s:
            metrics[reader]={'status':run.get('status','missing'),'error':run.get('error'),'run_summary':run}
            table_rows.append([reader,'NA','NA','NA','NA','NA','NA',run.get('status','not completed')])
            continue
        bm={x['query_id']:x for x in b}; sm={x['query_id']:x for x in s}
        rows=[]
        for qid,x in snap.items():
            if qid not in bm or qid not in sm: continue
            gold=x['answer']
            b_f1=f1_score(bm[qid]['prediction'], gold); s_f1=f1_score(sm[qid]['prediction'], gold)
            b_em=em_score(bm[qid]['prediction'], gold); s_em=em_score(sm[qid]['prediction'], gold)
            b_sup,b_sp=support_metrics(x['baseline_titles'],x['supporting_titles'])
            s_sup,s_sp=support_metrics(x['selected_titles'],x['supporting_titles'])
            b_joint=b_f1*b_sp; s_joint=s_f1*s_sp
            row={'reader':reader,'query_id':qid,'baseline_answer_f1':b_f1,'selected_answer_f1':s_f1,'answer_f1_delta':s_f1-b_f1,'baseline_answer_em':b_em,'selected_answer_em':s_em,'baseline_support_recall@5':b_sup,'selected_support_recall@5':s_sup,'support_recall_delta':s_sup-b_sup,'baseline_sp_f1':b_sp,'selected_sp_f1':s_sp,'sp_f1_delta':s_sp-b_sp,'baseline_joint_f1':b_joint,'selected_joint_f1':s_joint,'joint_f1_delta':s_joint-b_joint,'baseline_prediction':bm[qid]['prediction'],'selected_prediction':sm[qid]['prediction']}
            rows.append(row); per.append(row)
        def avg(k): return sum(r[k] for r in rows)/max(1,len(rows))
        m={'status':'completed','n':len(rows),'answer_em':avg('selected_answer_em'),'answer_f1':avg('selected_answer_f1'),'support_recall@5':avg('selected_support_recall@5'),'sp_f1':avg('selected_sp_f1'),'joint_f1':avg('selected_joint_f1'),'baseline_answer_em':avg('baseline_answer_em'),'baseline_answer_f1':avg('baseline_answer_f1'),'baseline_support_recall@5':avg('baseline_support_recall@5'),'baseline_sp_f1':avg('baseline_sp_f1'),'baseline_joint_f1':avg('baseline_joint_f1'),'answer_f1_delta':avg('answer_f1_delta'),'joint_f1_delta':avg('joint_f1_delta'),'support_recall_delta':avg('support_recall_delta'),'sp_f1_delta':avg('sp_f1_delta'),'wins':sum(1 for r in rows if r['joint_f1_delta']>0),'losses':sum(1 for r in rows if r['joint_f1_delta']<0),'ties':sum(1 for r in rows if r['joint_f1_delta']==0)}
        metrics[reader]=m
        sig[reader]={k:bootstrap([r[k] for r in rows]) for k in ['answer_f1_delta','joint_f1_delta','support_recall_delta','sp_f1_delta']}
        table_rows.append([reader,fmt(m['answer_f1_delta'],signed=True),fmt(m['joint_f1_delta'],signed=True),fmt(m['support_recall_delta'],signed=True),fmt(m['sp_f1_delta'],signed=True),fmt(sig[reader]['answer_f1_delta']['p_value']),fmt(sig[reader]['joint_f1_delta']['p_value']),'completed'])
    # include frozen main reader for reference
    summ=read_json(V23/'outputs/final_1000/final_1000_crossfit_summary.json')
    sigr=read_json(V23/'outputs/final_1000/significance_report.json')
    metrics['google/flan-t5-large_frozen_main']={'status':'completed_existing_frozen','answer_f1_delta':summ.get('answer_f1_delta'),'joint_f1_delta':summ.get('joint_f1_delta'),'support_recall_delta':summ.get('support_recall_delta'),'sp_f1_delta':summ.get('sp_f1_delta'),'note':'Original frozen v2.3 main reader result; not rerun.'}
    sig['google/flan-t5-large_frozen_main']=sigr.get('metrics',{})
    table_rows.insert(0,['google/flan-t5-large_frozen_main',fmt(summ.get('answer_f1_delta'),signed=True),fmt(summ.get('joint_f1_delta'),signed=True),fmt(summ.get('support_recall_delta'),signed=True),fmt(summ.get('sp_f1_delta'),signed=True),fmt(sigr.get('metrics',{}).get('answer_f1',{}).get('p_value')),fmt(sigr.get('metrics',{}).get('joint_f1',{}).get('p_value')),'frozen main reader; not rerun'])
    write_json(OUT/'outputs/metrics/multi_reader_metrics.json',metrics)
    write_json(OUT/'outputs/metrics/multi_reader_significance.json',sig)
    write_jsonl(OUT/'outputs/metrics/per_example_multi_reader_delta.jsonl',per)
    headers=['reader','answer_f1_delta','joint_f1_delta','support_recall@5_delta','sp_f1_delta','answer_f1_p','joint_f1_p','conclusion']
    (OUT/'outputs/tables/multi_reader_replication_table.md').write_text(md_table(headers,table_rows))
    return metrics

def write_multi_reader_report():
    ensure_dirs()
    summary=read_json(OUT/'outputs/context_snapshots/context_snapshot_summary.json')
    audit=read_json(OUT/'outputs/audit/context_snapshot_audit.json')
    metrics=read_json(OUT/'outputs/metrics/multi_reader_metrics.json')
    base=metrics.get('google/flan-t5-base',{})
    frozen=metrics.get('google/flan-t5-large_frozen_main',{})
    base_done=base.get('status')=='completed'
    report=f"""# Multi-Reader Replication Repair Report\n\n## 1. Context snapshots\n\nMaterialization status: **{audit.get('status')}**.\n\n- num_examples: {summary.get('num_examples')}\n- num_with_question: {summary.get('num_with_question')}\n- num_with_answer: {summary.get('num_with_answer')}\n- num_with_baseline_context: {summary.get('num_with_baseline_context')}\n- num_with_selected_context: {summary.get('num_with_selected_context')}\n- num_missing_context: {summary.get('num_missing_context')}\n\n## 2. Reader replication\n\n- flan-t5-base: **{base.get('status','missing')}**\n- flan-t5-large: frozen main-reader result retained, not overwritten.\n\nflan-t5-base result: answer_f1_delta={fmt(base.get('answer_f1_delta'),signed=True)}, joint_f1_delta={fmt(base.get('joint_f1_delta'),signed=True)}, support_recall_delta={fmt(base.get('support_recall_delta'),signed=True)}, sp_f1_delta={fmt(base.get('sp_f1_delta'),signed=True)}.\n\nFrozen flan-t5-large result: answer_f1_delta={fmt(frozen.get('answer_f1_delta'),signed=True)}, joint_f1_delta={fmt(frozen.get('joint_f1_delta'),signed=True)}, support_recall_delta={fmt(frozen.get('support_recall_delta'),signed=True)}, sp_f1_delta={fmt(frozen.get('sp_f1_delta'),signed=True)}.\n\n## 3. Interpretation\n\n{'The additional reader completed. Use the table to decide whether joint/support improvements are reader-consistent; answer_f1 must still be reported as reader-sensitive unless significant across readers.' if base_done else 'Additional reader replication remains inconclusive because flan-t5-base did not complete. Keep multi-reader replication as a limitation rather than a strengthened claim.'}\n\n## 4. Submission recommendation\n\nMulti-reader should be placed in the appendix/robustness section, not used to overwrite the HotpotQA v2.3 main result. Main-conference positioning improves only if the completed additional reader shows positive joint/support deltas; otherwise keep Findings/COLING as the safer target.\n"""
    (OUT/'reports/multi_reader_replication_report.md').write_text(report)
    boundary="""# Multi-Reader Claim Boundary\n\nAllowed if additional reader succeeds with positive joint/support deltas:\n\n> The frozen v2.3 selected contexts show consistent joint/support-side improvements across readers, while answer_f1 remains reader-sensitive.\n\nAllowed if only the frozen main reader is available or flan-t5-base fails:\n\n> Multi-reader replication remains inconclusive due to missing or failed additional reader runs.\n\nDo not write:\n\n- universally improves all readers\n- significantly improves answer_f1 across readers\n- solves reader sensitivity\n\nThe original HotpotQA v2.3 result remains frozen and should remain the main table.\n"""
    (OUT/'reports/multi_reader_claim_boundary.md').write_text(boundary)

def run_all():
    ensure_dirs(); source_inventory(); materialize_contexts(); validate_context_snapshots(); run_reader_replication(); evaluate_multi_reader_outputs(); write_multi_reader_report()

if __name__=='__main__': run_all()
