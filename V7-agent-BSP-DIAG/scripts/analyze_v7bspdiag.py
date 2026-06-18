from __future__ import annotations
import json, math
from pathlib import Path
import pandas as pd
BASE=Path(__file__).resolve().parents[1]; BSP=Path('/home/iiserver31/projects/FedE4RAG-main/V7-agent-BSP'); A=BASE/'analysis'; A.mkdir(exist_ok=True)

def read(p):
    p=Path(p)
    try: return pd.read_csv(p) if p.exists() and p.stat().st_size else pd.DataFrame()
    except Exception: return pd.DataFrame()
def j(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return {}
def collect_strict():
    rows=[]
    for root,label in [(BSP,'bsp'),(BASE,'diag')]:
        for p in (root/'analysis/strict_runs').glob('**/hp1_strict_metrics.json'):
            d=j(p); method=(d.get('method') or d.get('selection_strategy') or p.parent.name.split('_k3_')[0]).replace('-','_')
            r={'source':label,'method':method,'path':str(p)}
            for k,v in d.items():
                if isinstance(v,(int,float,str,bool)): r[k]=v
            r.setdefault('avg_topk',r.get('avg_budget_topk_hp1',3.0)); r.setdefault('budget_std',r.get('budget_std_hp1',0.0)); rows.append(r)
    df=pd.DataFrame(rows); df.to_csv(A/'strict_diagnostic_diag_seed_level.csv',index=False)
    if not df.empty:
        nums=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        out=df.groupby('method')[nums].mean(numeric_only=True).reset_index(); out.insert(1,'n',df.groupby('method').size().reindex(out.method).to_numpy()); out.to_csv(A/'strict_diagnostic_diag_summary.csv',index=False)
    return df
def collect_official():
    rows=[]
    for root,label in [(BSP,'bsp'),(BASE,'diag')]:
        for p in (root/'eval_outputs/official_fid_t5').glob('**/official_metrics.json'):
            d=j(p); m=d.get('metrics') or {}; method=(d.get('method') or Path(d.get('run_dir','')).name.split('_k3_')[0]).replace('-','_')
            rows.append({'source':label,'method':method,'seed':d.get('seed'),'n_examples':d.get('n'),'answer_F1':m.get('answer_f1'),'support_F1':m.get('sp_f1'),'joint_F1':m.get('joint_f1'),'support_title_recall':m.get('support_title_recall_at_k'),'answer_EM':m.get('answer_em'),'support_EM':m.get('sp_em'),'joint_EM':m.get('joint_em'),'avg_topk':d.get('avg_budget_topk') or 3.0,'budget_std':d.get('budget_std',0.0),'reader_model':d.get('reader_model') or d.get('fid_model'),'beam_size':d.get('beam_size'),'max_input_length':d.get('max_input_length'),'passage_ordering':d.get('passage_ordering','retrieval_score'),'path':str(p)})
    df=pd.DataFrame(rows); df.to_csv(A/'official_fid_t5_diag_all_runs.csv',index=False); return df
def sensitivity_final():
    rows=[]
    for p in (BSP/'eval_outputs/reader_sensitivity').glob('**/official_metrics.json'):
        d=j(p); m=d.get('metrics') or {}; rows.append({'method':(d.get('method') or '').replace('-','_'),'seed':d.get('seed'),'beam_size':d.get('beam_size'),'max_input_length':d.get('max_input_length'),'passage_ordering':d.get('passage_ordering'),'n_examples':d.get('n'),'answer_EM':m.get('answer_em'),'answer_F1':m.get('answer_f1'),'support_EM':m.get('sp_em'),'support_F1':m.get('sp_f1'),'joint_EM':m.get('joint_em'),'joint_F1':m.get('joint_f1'),'support_title_recall':m.get('support_title_recall_at_k'),'avg_topk':d.get('avg_budget_topk') or 3.0,'budget_std':d.get('budget_std',0.0)})
    df=pd.DataFrame(rows)
    ver=read(A/'reader_input_ordering_verification.csv')
    if not df.empty and not ver.empty:
        v=ver.groupby(['method','seed','passage_ordering'])[['reader_input_hash_diff_rate','gold_support_position_mean','gold_support_position_median']].mean(numeric_only=True).reset_index()
        df=df.merge(v,on=['method','seed','passage_ordering'],how='left')
    df.to_csv(A/'reader_sensitivity_summary_final.csv',index=False); return df
def cache_audit(sens,ver):
    lines=['# Cache Reuse Audit','']
    dirs=sorted([p.name for p in (BSP/'eval_outputs/reader_sensitivity').glob('*') if p.is_dir()])
    lines += [f'- sensitivity directories: {len(dirs)}', f'- examples: {", ".join(dirs[:12])}', '']
    if not ver.empty:
        same=ver[ver.passage_ordering!='retrieval_score'].copy()
        zero=float((same.reader_input_hash_diff_rate.fillna(0)==0).mean()) if len(same) else 0
        lines += [f'- non-retrieval ordering rows: {len(same)}', f'- zero reader-input diff rate share: {zero:.3f}', '']
        if zero>0.9: lines += ['Finding: most ordering variants produce identical reader input hashes; ordering path is likely ineffective or not connected to reader input.', '']
    if not sens.empty:
        keys=sens.groupby(['beam_size','max_input_length','passage_ordering']).size().reset_index(name='n')
        lines += ['## sensitivity grid observed', keys.to_csv(index=False)]
    (A/'cache_reuse_audit.md').write_text('\n'.join(lines),encoding='utf-8')
def oracle_effect(sens):
    if sens.empty: pd.DataFrame().to_csv(A/'gold_oracle_debug_effect.csv',index=False); return
    norm=sens[sens.passage_ordering=='retrieval_score']; ora=sens[sens.passage_ordering=='gold_oracle_debug']
    m=norm.merge(ora,on=['method','seed','beam_size','max_input_length'],suffixes=('_normal','_gold_oracle'))
    rows=[]
    for _,r in m.iterrows():
        rows.append({'method':r.method,'seed':r.seed,'beam_size':r.beam_size,'max_input_length':r.max_input_length,'answer_F1_normal':r.answer_F1_normal,'answer_F1_gold_oracle':r.answer_F1_gold_oracle,'support_F1_normal':r.support_F1_normal,'support_F1_gold_oracle':r.support_F1_gold_oracle,'joint_F1_normal':r.joint_F1_normal,'joint_F1_gold_oracle':r.joint_F1_gold_oracle,'support_title_recall_normal':r.support_title_recall_normal,'support_title_recall_gold_oracle':r.support_title_recall_gold_oracle,'delta_answer_F1':r.answer_F1_gold_oracle-r.answer_F1_normal,'delta_support_F1':r.support_F1_gold_oracle-r.support_F1_normal,'delta_joint_F1':r.joint_F1_gold_oracle-r.joint_F1_normal})
    pd.DataFrame(rows).to_csv(A/'gold_oracle_debug_effect.csv',index=False)
def per_query():
    rows=[]; base={}
    for root,label in [(BSP,'bsp'),(BASE,'diag')]:
        for p in (root/'eval_outputs/official_fid_t5').glob('**/per_query_official.jsonl'):
            meta=j(p.parent/'official_metrics.json'); method=(meta.get('method') or '').replace('-','_'); seed=meta.get('seed')
            for line in p.read_text(encoding='utf-8').splitlines():
                if not line: continue
                d=json.loads(line); m=d.get('metrics') or {}; q=d.get('id') or d.get('example_id')
                row={'query_id':q,'question':d.get('question'),'gold_answer':d.get('gold_answer'),'gold_supporting_titles':';'.join(sorted({str(t) for t,_ in d.get('gold_sp',[])})),'method':method,'seed':seed,'answer_prediction':d.get('pred_answer'),'answer_F1':m.get('answer_f1'),'support_F1':m.get('sp_f1'),'joint_F1':m.get('joint_f1'),'support_title_hit':m.get('support_title_recall_at_k'),'avg_topk':meta.get('avg_budget_topk') or 3.0,'budget_std':meta.get('budget_std',0.0)}
                if method=='hypernet_v6': base[(seed,q)]=(row['answer_F1'],row['support_F1'],row['joint_F1'])
                rows.append(row)
    df=pd.DataFrame(rows)
    for i,r in df.iterrows():
        b=base.get((r.get('seed'),r.get('query_id')),(None,None,None)); df.loc[i,'baseline_answer_F1']=b[0]; df.loc[i,'baseline_support_F1']=b[1]; df.loc[i,'baseline_joint_F1']=b[2]
    if not df.empty:
        df['delta_answer_F1']=df.answer_F1-df.baseline_answer_F1; df['delta_support_F1']=df.support_F1-df.baseline_support_F1; df['delta_joint_F1']=df.joint_F1-df.baseline_joint_F1
    df.to_csv(A/'per_query_alignment_final.csv',index=False)
    corr=[]
    if not df.empty:
        for metric in ['answer_F1','support_F1','joint_F1','support_title_hit']:
            corr.append({'signal':'support_title_hit','metric':metric,'corr':df[['support_title_hit',metric]].corr().iloc[0,1] if df.support_title_hit.nunique()>1 else None})
    pd.DataFrame(corr).to_csv(A/'selection_to_qa_correlation.csv',index=False)
    return df
def subgroup(df):
    rows=[]
    if not df.empty:
        for method,g in df.groupby('method'):
            for name,sub in {'all':g,'hard_query':g[g.support_title_hit.fillna(0)<0.75],'easy_query':g[g.support_title_hit.fillna(0)>=0.75],'bandit_helped':g[g.delta_joint_F1.fillna(0)>0],'bandit_hurt':g[g.delta_joint_F1.fillna(0)<0],'answer_failed_baseline':g[g.baseline_answer_F1.fillna(1)<0.5],'support_failed_baseline':g[g.baseline_support_F1.fillna(1)<0.5]}.items():
                if len(sub): rows.append({'method':method,'subgroup':name,'n':len(sub),'answer_F1':sub.answer_F1.mean(),'support_F1':sub.support_F1.mean(),'joint_F1':sub.joint_F1.mean(),'support_title_recall':sub.support_title_hit.mean(),'avg_topk':sub.avg_topk.mean(),'budget_std':sub.budget_std.mean()})
    pd.DataFrame(rows).to_csv(A/'true_subgroup_analysis_final.csv',index=False)
def stats(strict,off,sens):
    pairs=[('agent_bsp_hf_bandit_strict','agent_pm_bandit_slot','hp1_multihop_score'),('agent_bsp_hf_bandit_retrieval','agent_pm_bandit_slot','hp1_multihop_score'),('agent_bsp_hf_bandit_retrieval','agent_bsp_memory_bandit_retrieval','hp1_multihop_score'),('agent_bsp_memory_bandit_retrieval','agent_bsp_memory_bandit_no_history_state','hp1_multihop_score'),('agent_bsp_memory_bandit_retrieval','agent_bsp_memory_bandit_no_failure_state','hp1_multihop_score')]
    rows=[]
    for a,b,m in pairs:
        df=strict if m.startswith('hp1') else off
        if df.empty or m not in df: rows.append({'method_a':a,'method_b':b,'metric':m,'n':0}); continue
        key='seed' if 'seed' in df else 'path'; ma=df[df.method==a][[key,m]].dropna(); mb=df[df.method==b][[key,m]].dropna(); mm=ma.merge(mb,on=key,suffixes=('_a','_b'))
        delta=(mm[f'{m}_a']-mm[f'{m}_b']).tolist(); rows.append({'method_a':a,'method_b':b,'metric':m,'n':len(delta),'mean_delta':sum(delta)/len(delta) if delta else None})
    pd.DataFrame(rows).to_csv(A/'statistical_tests_diag.csv',index=False)
def cases(df):
    lines=['# Representative Cases DIAG','']
    titles=['agent_pm_bandit_slot succeeds while dynamic fails','agent_bsp_hf_bandit succeeds while pm_bandit fails','history state helps','failure state helps','rarity state neutral or misleading','instability state conservative failure','strict improves but QA flat','gold_oracle_debug improves QA','gold_oracle_debug still flat','reader input unchanged case']
    for t in titles: lines += [f'## {t}','TBD from per-query/hash evidence after DIAG runs complete.','']
    (A/'representative_cases_diag.md').write_text('\n'.join(lines),encoding='utf-8')
def main():
    strict=collect_strict(); off=collect_official(); sens=sensitivity_final(); ver=read(A/'reader_input_ordering_verification.csv')
    cache_audit(sens,ver); oracle_effect(sens); pq=per_query(); subgroup(pq); stats(strict,off,sens); cases(pq)
    print('diag analysis written',A)
if __name__=='__main__': main()
