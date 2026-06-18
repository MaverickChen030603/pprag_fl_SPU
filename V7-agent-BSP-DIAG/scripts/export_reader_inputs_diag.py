from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path
from statistics import mean, median
from typing import Any
import torch
from transformers import AutoModel, AutoTokenizer

BASE=Path(__file__).resolve().parents[1]
BSP=Path('/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP')
sys.path.insert(0,str(BASE))
from run_hotpot_official_eval import load_official_examples, build_sentence_docs, latest_hf_model, load_run_metadata, encode_texts

METHOD_PATTERNS={
 'hypernet_v6':'hypernet-v6',
 'agent_rule_v7_dynamic':'agent-rule-v7-dynamic',
 'agent_pm_bandit_slot':'agent-pm-bandit-slot',
 'agent_bsp_memory_bandit_retrieval':'agent-bsp-memory-bandit-retrieval',
 'agent_bsp_hf_bandit_strict':'agent-bsp-hf-bandit-strict',
 'agent_bsp_hf_bandit_retrieval':'agent-bsp-hf-bandit-retrieval',
}

def h(x): return hashlib.sha1(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:16]
def find_runs(root:Path, exp:str, method:str):
    pat=METHOD_PATTERNS[method]
    return sorted((root/'outputs'/exp).glob(f'**/{pat}_k3*/final_artifacts.json'))
def run_seed(meta):
    return int(meta.get('seed') if meta.get('seed') is not None else (meta.get('option') or {}).get('seed',0))
def order_docs(ordering, ranked, gold_titles):
    if ordering=='gold_oracle_debug':
        return sorted(ranked,key=lambda d:(str(d['title']) in gold_titles,float(d.get('score',0))),reverse=True)
    if ordering=='agent_priority':
        # Diagnostic surrogate: preserve retrieval score but rotate equal title groups by title.
        # If hashes remain identical, the report will flag that agent priority is not wired.
        return sorted(ranked,key=lambda d:(round(float(d.get('score',0)),6),str(d['title'])),reverse=True)
    return ranked

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-root',default=str(BSP)); ap.add_argument('--source-exp',default='pprag_fl_v7agentbsp')
    ap.add_argument('--diag-root',default=str(BASE)); ap.add_argument('--diag-exp',default='pprag_fl_v7agentbspdiag')
    ap.add_argument('--rawdata-path',default='/home/iiserver31/projects/FedE4RAG-main/FedE/select_data_hotpot_train_5000.json')
    ap.add_argument('--methods',default='hypernet_v6,agent_rule_v7_dynamic,agent_pm_bandit_slot,agent_bsp_memory_bandit_retrieval,agent_bsp_hf_bandit_retrieval')
    ap.add_argument('--orderings',default='retrieval_score,agent_priority,gold_oracle_debug')
    ap.add_argument('--seeds',default='0,1,2,3,4'); ap.add_argument('--sample-size',type=int,default=50)
    ap.add_argument('--device',default='cpu'); ap.add_argument('--max-length',type=int,default=256); ap.add_argument('--answer-topk',type=int,default=5)
    args=ap.parse_args()
    methods=[m for m in args.methods.split(',') if m]; orderings=[o for o in args.orderings.split(',') if o]; seeds={int(s) for s in args.seeds.split(',') if s}
    examples,_=load_official_examples(Path(args.rawdata_path),1000,False)
    rng=random.Random(20260618); idxs=sorted(rng.sample(range(len(examples)),min(args.sample_size,len(examples))))
    rows=[]; outbase=BASE/'debug_reader_inputs'; outbase.mkdir(parents=True,exist_ok=True)
    run_roots=[(Path(args.source_root),args.source_exp),(Path(args.diag_root),args.diag_exp)]
    for method in methods:
        runs=[]
        for root,exp in run_roots: runs += find_runs(root,exp,method)
        byseed={}
        for art in runs:
            meta=load_run_metadata(art.parent); sd=run_seed(meta)
            if sd in seeds and sd not in byseed: byseed[sd]=art.parent
        for sd,run in sorted(byseed.items()):
            model_dir=latest_hf_model(run)
            if not model_dir: continue
            tok=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True)
            model=AutoModel.from_pretrained(str(model_dir),local_files_only=True).to(args.device)
            selected=[examples[i] for i in idxs]
            all_docs=[]; prep=[]
            for ex in selected:
                docs=build_sentence_docs(ex); st=len(all_docs); all_docs.extend(d['content'] for d in docs); prep.append((ex,docs,st,len(all_docs)))
            doc_emb=encode_texts(tok,model,all_docs,torch.device(args.device),16,args.max_length)
            query_emb=encode_texts(tok,model,[p[0]['question'] for p in prep],torch.device(args.device),16,args.max_length)
            for ordering in orderings:
                odir=outbase/method/ordering; odir.mkdir(parents=True,exist_ok=True)
                fp=(odir/f'seed_{sd}_reader_inputs.jsonl').open('w',encoding='utf-8')
                for qi,(ex,docs,st,en) in enumerate(prep):
                    scores=torch.matmul(doc_emb[st:en],query_emb[qi]).tolist()
                    ranked=[d|{'score':float(sc),'pid':f"{d['title']}::{d['sent_id']}"} for d,sc in sorted(zip(docs,scores),key=lambda p:p[1],reverse=True)]
                    gold_titles={str(t) for t,_ in ex.get('supporting_facts',[])}
                    ordered=order_docs(ordering,ranked,gold_titles)
                    top=ordered[:args.answer_topk]; text=' \n '.join(d['content'] for d in top)
                    poss=[i for i,d in enumerate(ordered,1) if str(d['title']) in gold_titles]
                    rec={'query_id':ex['id'],'question':ex['question'],'gold_answer':ex['answer'],'gold_supporting_titles':sorted(gold_titles),'method':method,'seed':sd,'passage_ordering':ordering,'reader_input_text':text,'input_token_length':len(tok.encode(text,truncation=False)),'passage_ids_in_order':[d['pid'] for d in top],'passage_titles_in_order':[d['title'] for d in top],'passage_scores_in_order':[d['score'] for d in top],'selected_blocks':None,'slot_allocation':None,'supporting_title_positions':poss,'gold_title_in_top1':bool(poss and min(poss)<=1),'gold_title_in_top3':bool(poss and min(poss)<=3),'gold_title_in_top5':bool(poss and min(poss)<=5),'gold_title_in_input':bool(poss and min(poss)<=args.answer_topk)}
                    rec['reader_input_hash']=h(rec['reader_input_text']); rec['passage_order_hash']=h(rec['passage_ids_in_order']); rec['top5_passage_titles']='|'.join(rec['passage_titles_in_order']); rec['top5_passage_ids']='|'.join(rec['passage_ids_in_order'])
                    fp.write(json.dumps(rec,ensure_ascii=False)+'\n'); rows.append(rec)
                fp.close()
            del model
            torch.cuda.empty_cache()
    dfrows=[]
    import pandas as pd
    df=pd.DataFrame(rows)
    if not df.empty:
        for keys,g in df.groupby(['method','seed','passage_ordering']):
            pos=[p for arr in g.supporting_title_positions for p in (arr if isinstance(arr,list) else [])]
            dfrows.append({'method':keys[0],'seed':keys[1],'passage_ordering':keys[2],'n':len(g),'reader_input_hash_unique':g.reader_input_hash.nunique(),'passage_order_hash_unique':g.passage_order_hash.nunique(),'gold_support_position_mean':mean(pos) if pos else None,'gold_support_position_median':median(pos) if pos else None,'gold_support_position_min':min(pos) if pos else None,'gold_support_position_max':max(pos) if pos else None,'gold_title_in_top1_rate':g.gold_title_in_top1.mean(),'gold_title_in_top3_rate':g.gold_title_in_top3.mean(),'gold_title_in_top5_rate':g.gold_title_in_top5.mean(),'gold_title_in_input_rate':g.gold_title_in_input.mean()})
        # diff rates against retrieval_score per method/seed/query
        base=df[df.passage_ordering=='retrieval_score'][['method','seed','query_id','reader_input_hash','passage_order_hash']].rename(columns={'reader_input_hash':'base_input','passage_order_hash':'base_order'})
        m=df.merge(base,on=['method','seed','query_id'],how='left')
        diff=m.groupby(['method','seed','passage_ordering']).apply(lambda x: __import__('pandas').Series({'reader_input_hash_diff_rate':(x.reader_input_hash!=x.base_input).mean(),'passage_order_hash_diff_rate':(x.passage_order_hash!=x.base_order).mean()})).reset_index()
        out=pd.DataFrame(dfrows).merge(diff,on=['method','seed','passage_ordering'],how='left')
    else: out=pd.DataFrame()
    (BASE/'analysis').mkdir(exist_ok=True)
    out.to_csv(BASE/'analysis/reader_input_ordering_verification.csv',index=False)
    print(BASE/'analysis/reader_input_ordering_verification.csv', 'rows', len(out))
if __name__=='__main__': main()
