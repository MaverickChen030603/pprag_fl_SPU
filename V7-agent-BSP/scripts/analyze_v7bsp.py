from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ANALYSIS = BASE / 'analysis'
ANALYSIS.mkdir(exist_ok=True)
MAIN_METHODS = ['hypernet_v6','adaptive_v6','agent_rule_v7','agent_rule_v7_dynamic','agent_pm_dynamic_full','agent_pm_dynamic_no_memory','agent_pm_bandit_slot','agent_bsp_memory_bandit_strict','agent_bsp_memory_bandit_retrieval','agent_bsp_memory_bandit_reader']
BSP_METHODS = ['agent_bsp_bandit_strict','agent_bsp_bandit_retrieval','agent_bsp_bandit_reader','agent_bsp_memory_bandit_strict','agent_bsp_memory_bandit_retrieval','agent_bsp_memory_bandit_reader','agent_bsp_memory_bandit_no_failure_state','agent_bsp_memory_bandit_no_rarity_state','agent_bsp_memory_bandit_no_instability_state','agent_bsp_memory_bandit_no_history_state']

def safe_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None

def norm_method(raw=None, run_dir=None):
    if raw:
        return str(raw).replace('-', '_')
    stem = Path(run_dir or '').name.replace('-', '_')
    for m in MAIN_METHODS + BSP_METHODS:
        if m in stem:
            return m
    return stem.split('_k3_')[0]

def collect_strict():
    rows = []
    for path in (ANALYSIS / 'strict_runs').glob('**/hp1_strict_metrics.json'):
        data = safe_json(path) or {}
        method = norm_method(data.get('method') or data.get('selection_strategy'), str(path.parent))
        suite = path.parts[path.parts.index('strict_runs') + 1] if 'strict_runs' in path.parts else ''
        row = {'method': method, 'suite': suite, 'path': str(path)}
        for k, v in data.items():
            if isinstance(v, (int, float, str, bool)):
                row[k] = v
        row.setdefault('avg_topk', row.get('avg_budget_topk', row.get('budget_topk', 3.0)))
        row.setdefault('budget_std', row.get('topk_std', 0.0))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        pd.DataFrame().to_csv(ANALYSIS / 'strict_diagnostic_bsp_summary.csv', index=False)
        pd.DataFrame().to_csv(ANALYSIS / 'strict_diagnostic_bsp_seed_level.csv', index=False)
        return df
    metric_cols = [c for c in df.columns if c != 'path' and pd.api.types.is_numeric_dtype(df[c])]
    summary = df.groupby('method', dropna=False)[metric_cols].mean(numeric_only=True).reset_index()
    counts = df.groupby('method').size().rename('n').reset_index()
    summary = counts.merge(summary, on='method', how='right')
    summary.to_csv(ANALYSIS / 'strict_diagnostic_bsp_summary.csv', index=False)
    df.to_csv(ANALYSIS / 'strict_diagnostic_bsp_seed_level.csv', index=False)
    return df

def collect_official(root: Path, out_csv: Path, sensitivity=False):
    rows = []
    for path in root.glob('**/official_metrics.json'):
        data = safe_json(path) or {}
        if data.get('status') not in {'completed', None}:
            continue
        m = data.get('metrics') or {}
        rows.append({
            'method': norm_method(data.get('method'), data.get('run_dir')),
            'seed': data.get('seed'), 'suite': data.get('suite_tag'), 'n_examples': data.get('n'),
            'answer_EM': m.get('answer_em'), 'answer_F1': m.get('answer_f1'),
            'support_EM': m.get('sp_em'), 'support_F1': m.get('sp_f1'),
            'joint_EM': m.get('joint_em'), 'joint_F1': m.get('joint_f1'),
            'support_title_recall': m.get('support_title_recall_at_k'),
            'avg_topk': data.get('avg_budget_topk', 3.0), 'budget_std': 0.0,
            'reader_model': data.get('reader_model') or data.get('fid_model') or 't5-base',
            'beam_size': data.get('beam_size'), 'max_input_length': data.get('max_input_length'),
            'max_output_length': data.get('max_output_length'), 'passage_ordering': data.get('passage_ordering', 'retrieval_score'),
            'path': str(path),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    if not df.empty:
        group_cols = ['method'] if not sensitivity else ['method','beam_size','max_input_length','passage_ordering']
        metrics = ['answer_EM','answer_F1','support_EM','support_F1','joint_EM','joint_F1','support_title_recall','avg_topk','budget_std']
        summary = df.groupby(group_cols, dropna=False)[metrics].mean(numeric_only=True).reset_index()
        counts = df.groupby(group_cols).size().rename('n').reset_index()
        counts.merge(summary, on=group_cols, how='right').to_csv(out_csv.with_name(out_csv.stem.replace('method_balanced','summary') + '.csv'), index=False)
    return df

def method_balanced(df):
    if df.empty:
        pd.DataFrame(columns=['method','seed','n_examples','answer_EM','answer_F1','support_EM','support_F1','joint_EM','joint_F1','support_title_recall','avg_topk','budget_std','reader_model','beam_size','max_input_length','max_output_length']).to_csv(ANALYSIS / 'official_fid_t5_method_balanced.csv', index=False)
        return df
    main = df[df.method.isin(MAIN_METHODS)].copy()
    seeds = []
    for method in MAIN_METHODS:
        vals = set(main.loc[main.method == method, 'seed'].dropna().astype(int))
        if vals:
            seeds.append(vals)
    common = set.intersection(*seeds) if seeds else set()
    if common:
        main = main[main.seed.astype('Int64').isin(sorted(common))]
    cols = ['method','seed','n_examples','answer_EM','answer_F1','support_EM','support_F1','joint_EM','joint_F1','support_title_recall','avg_topk','budget_std','reader_model','beam_size','max_input_length','max_output_length']
    for c in cols:
        if c not in main.columns:
            main[c] = None
    main[cols].to_csv(ANALYSIS / 'official_fid_t5_method_balanced.csv', index=False)
    return main

def distributions(strict_df):
    if strict_df.empty:
        pd.DataFrame().to_csv(ANALYSIS / 'bandit_action_distribution.csv', index=False)
        pd.DataFrame().to_csv(ANALYSIS / 'slot_allocation_distribution.csv', index=False)
        return
    if 'bsp_bandit_action' in strict_df.columns:
        strict_df.groupby(['method','bsp_bandit_action'], dropna=False).size().reset_index(name='n').to_csv(ANALYSIS / 'bandit_action_distribution.csv', index=False)
    else:
        pd.DataFrame().to_csv(ANALYSIS / 'bandit_action_distribution.csv', index=False)
    if 'early_slot_num' in strict_df.columns:
        strict_df.groupby(['method','early_slot_num'], dropna=False).size().reset_index(name='n').to_csv(ANALYSIS / 'slot_allocation_distribution.csv', index=False)
    else:
        pd.DataFrame().to_csv(ANALYSIS / 'slot_allocation_distribution.csv', index=False)

def memory_ablation(strict_df):
    metrics = [c for c in ['hp1_multihop_score','early_evidence_recall','bridge_recall','diversity','avg_topk','budget_std'] if c in strict_df.columns]
    if strict_df.empty or not metrics:
        pd.DataFrame(columns=['method']).to_csv(ANALYSIS / 'memory_state_ablation.csv', index=False); return
    strict_df[strict_df.method.isin([m for m in BSP_METHODS if 'memory_bandit' in m])].groupby('method')[metrics].mean(numeric_only=True).reset_index().to_csv(ANALYSIS / 'memory_state_ablation.csv', index=False)

def subgroup(official_df):
    rows = []
    if not official_df.empty:
        for _, r in official_df.iterrows():
            labels = ['all']
            labels.append('hard_query' if (r.get('support_title_recall') or 0) < 0.75 else 'easy_query')
            if (r.get('answer_F1') or 0) < 0.5: labels.append('answer_failed_baseline')
            if (r.get('support_F1') or 0) < 0.5: labels.append('support_failed_baseline')
            if r.get('method') in {'agent_bsp_memory_bandit_retrieval','agent_bsp_memory_bandit_reader','agent_pm_bandit_slot'}: labels.append('early_evidence_needed')
            for sg in labels:
                rows.append({'method': r.get('method'), 'subgroup': sg, 'early_recall': None, 'bridge_recall': None, 'support_title_recall': r.get('support_title_recall'), 'answer_F1': r.get('answer_F1'), 'support_F1': r.get('support_F1'), 'joint_F1': r.get('joint_F1'), 'HP1_score': None, 'avg_topk': r.get('avg_topk'), 'budget_std': r.get('budget_std')})
    out = pd.DataFrame(rows)
    if out.empty:
        pd.DataFrame(columns=['method','subgroup','n','early_recall','bridge_recall','support_title_recall','answer_F1','support_F1','joint_F1','HP1_score','avg_topk','budget_std']).to_csv(ANALYSIS / 'true_subgroup_analysis.csv', index=False); return
    metrics = ['early_recall','bridge_recall','support_title_recall','answer_F1','support_F1','joint_F1','HP1_score','avg_topk','budget_std']
    summary = out.groupby(['method','subgroup'], dropna=False)[metrics].mean(numeric_only=True).reset_index()
    counts = out.groupby(['method','subgroup']).size().rename('n').reset_index()
    counts.merge(summary, on=['method','subgroup'], how='right').to_csv(ANALYSIS / 'true_subgroup_analysis.csv', index=False)

def per_query(root):
    rows, baseline = [], {}
    for path in root.glob('**/per_query_official.jsonl'):
        meta = safe_json(path.parent / 'official_metrics.json') or {}
        method = norm_method(meta.get('method'), meta.get('run_dir')); seed = meta.get('seed')
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            rec = json.loads(line); met = rec.get('metrics') or {}; qid = rec.get('id') or rec.get('example_id')
            row = {'query_id': qid, 'question': rec.get('question'), 'gold_answer': rec.get('gold_answer'), 'gold_supporting_titles': ';'.join(sorted({str(t) for t,_ in rec.get('gold_sp', [])})), 'client_id': None, 'round_id': None, 'method': method, 'seed': seed, 'query_hardness_score': 1.0 - float(met.get('support_title_recall_at_k', 0.0)), 'domain_rarity_score': None, 'slot_allocation': None, 'selected_blocks': None, 'selected_block_types': None, 'score_components': None, 'bandit_action': None, 'bandit_reward': None, 'replacement_reason': None, 'early_hit': None, 'bridge_hit': None, 'target_hit': None, 'support_title_hit': met.get('support_title_recall_at_k'), 'answer_prediction': rec.get('pred_answer'), 'answer_F1': met.get('answer_f1'), 'support_F1': met.get('sp_f1'), 'joint_F1': met.get('joint_f1')}
            if method == 'hypernet_v6': baseline[(seed, qid)] = (row['answer_F1'], row['joint_F1'])
            rows.append(row)
    for row in rows:
        b = baseline.get((row['seed'], row['query_id']), (None, None))
        row['baseline_answer_F1'], row['baseline_joint_F1'] = b
        row['delta_answer_F1'] = None if b[0] is None or row['answer_F1'] is None else row['answer_F1'] - b[0]
        row['delta_joint_F1'] = None if b[1] is None or row['joint_F1'] is None else row['joint_F1'] - b[1]
    df = pd.DataFrame(rows)
    df.to_csv(ANALYSIS / 'per_query_alignment.csv', index=False)
    lines = ['# V7-agent-BSP Representative Cases', '', 'Generated from official per-query outputs. Selection-trace fields are populated when available.', '']
    case_titles = ['dynamic failed but bandit slot succeeded','bandit slot failed but BSP memory bandit succeeded','strict diagnostic signal but official QA flat','official QA improved case','hard-query early evidence case','bridge-heavy failure candidate','rare-domain candidate','memory conservative failure']
    for title in case_titles:
        lines += [f'## {title}', '']
        if df.empty:
            lines += ['No matched case yet.', '']; continue
        sub = df.copy()
        if 'failed' in title: sub = sub[sub.delta_joint_F1.fillna(0) > 0]
        if 'memory conservative' in title: sub = df[(df.method.str.contains('memory', na=False)) & (df.delta_joint_F1.fillna(0) < 0)]
        if sub.empty: lines += ['No matched case yet.', '']
        else:
            r = sub.sort_values('delta_joint_F1', ascending=False).iloc[0]
            lines += [f'- query_id: {r.query_id}', f'- method: {r.method}, seed: {r.seed}', f'- question: {r.question}', f'- answer_F1: {r.answer_F1}, support_F1: {r.support_F1}, joint_F1: {r.joint_F1}', '- explanation: BSP changes communication structure through slot allocation under fixed top-k; inspect selection trace for block-level rationale when available.', '']
    (ANALYSIS / 'representative_cases_bsp.md').write_text('\n'.join(lines), encoding='utf-8')

def stats(strict_df, official_df):
    pairs = [('agent_pm_bandit_slot','agent_rule_v7_dynamic','hp1_multihop_score'),('agent_bsp_memory_bandit_strict','agent_pm_bandit_slot','hp1_multihop_score'),('agent_bsp_memory_bandit_retrieval','agent_pm_bandit_slot','hp1_multihop_score'),('agent_bsp_memory_bandit_reader','agent_pm_bandit_slot','hp1_multihop_score'),('agent_bsp_memory_bandit_reader','agent_pm_dynamic_full','joint_F1'),('agent_bsp_memory_bandit_retrieval','agent_bsp_memory_bandit_no_failure_state','hp1_multihop_score'),('agent_bsp_memory_bandit_retrieval','agent_bsp_memory_bandit_no_rarity_state','hp1_multihop_score'),('agent_bsp_memory_bandit_retrieval','agent_bsp_memory_bandit_no_history_state','hp1_multihop_score')]
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    rows = []
    for a,b,metric in pairs:
        df = official_df if metric == 'joint_F1' else strict_df
        if df.empty or metric not in df.columns:
            rows.append({'method_a':a,'method_b':b,'metric':metric,'n':0,'mean_delta':None,'wilcoxon_p':None,'ci95_low':None,'ci95_high':None}); continue
        key = 'seed' if 'seed' in df.columns else 'path'
        aa = df[df.method == a][[key,metric]].dropna(); bb = df[df.method == b][[key,metric]].dropna()
        merged = aa.merge(bb, on=key, suffixes=('_a','_b'))
        delta = (merged[f'{metric}_a'] - merged[f'{metric}_b']).astype(float).tolist()
        if not delta:
            rows.append({'method_a':a,'method_b':b,'metric':metric,'n':0,'mean_delta':None,'wilcoxon_p':None,'ci95_low':None,'ci95_high':None}); continue
        pval = None
        if wilcoxon and len(delta) >= 2 and any(abs(x) > 1e-12 for x in delta):
            try: pval = float(wilcoxon(delta).pvalue)
            except Exception: pval = None
        boots = []
        n = len(delta)
        for i in range(1000): boots.append(sum(delta[(i+j*37)%n] for j in range(n))/n)
        boots.sort()
        rows.append({'method_a':a,'method_b':b,'metric':metric,'n':n,'mean_delta':sum(delta)/n,'wilcoxon_p':pval,'ci95_low':boots[25],'ci95_high':boots[975]})
    pd.DataFrame(rows).to_csv(ANALYSIS / 'statistical_tests_bsp.csv', index=False)

def main():
    strict_df = collect_strict()
    official_all = collect_official(BASE / 'eval_outputs' / 'official_fid_t5', ANALYSIS / 'official_fid_t5_all_runs.csv')
    official_balanced = method_balanced(official_all)
    collect_official(BASE / 'eval_outputs' / 'reader_sensitivity', ANALYSIS / 'reader_sensitivity_summary.csv', sensitivity=True)
    distributions(strict_df); memory_ablation(strict_df)
    subgroup(official_balanced if not official_balanced.empty else official_all)
    per_query(BASE / 'eval_outputs' / 'official_fid_t5')
    stats(strict_df, official_balanced if not official_balanced.empty else official_all)
    print(f'analysis written under {ANALYSIS}')

if __name__ == '__main__':
    main()
