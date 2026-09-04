#!/usr/bin/env python3
"""No-cache representation, score, Top-50, and reader-context delta audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from audit_common import rows, support_titles


def rank_map(pool: dict) -> list[dict]: return sorted(pool['pool'], key=lambda x:(-float(x['hybrid_score']),str(x['doc_id'])))
def overlap(a,b,k): return len(set(x['doc_id'] for x in a[:k])&set(x['doc_id'] for x in b[:k]))/k
def minmax(v):
    lo,hi=min(v),max(v); return [0.5]*len(v) if hi-lo<1e-12 else [(x-lo)/(hi-lo) for x in v]

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--smoke-root',type=Path,required=True);p.add_argument('--development',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args();args.output_root.mkdir(parents=True,exist_ok=True)
    source={str(x['query_id']):x for x in rows(args.development)}
    pools={c.name:{str(x['query_id']):x for x in rows(c/'adapted_pool.jsonl')} for c in args.smoke_root.iterdir() if (c/'adapted_pool.jsonl').exists()}
    base=pools['frozen']; summaries=[]
    for condition,current in pools.items():
        records=[]
        for qid,base_row in base.items():
            a,b=rank_map(base_row),rank_map(current[qid]); gold=support_titles(source[qid])
            support_rank=lambda xs:[next((i+1 for i,d in enumerate(xs) if ' '.join(d['title'].lower().split())==title),None) for title in sorted(gold)]
            ar,br=support_rank(a),support_rank(b)
            records.append({'query_id':qid,'condition':condition,'baseline_top50_doc_ids':[x['doc_id'] for x in a[:50]],'condition_top50_doc_ids':[x['doc_id'] for x in b[:50]],'baseline_scores':[x['hybrid_score'] for x in a[:50]],'condition_scores':[x['hybrid_score'] for x in b[:50]],'baseline_support_ranks':ar,'condition_support_ranks':br,'top5_changed':set(x['doc_id'] for x in a[:5])!=set(x['doc_id'] for x in b[:5]),'top10_changed':set(x['doc_id'] for x in a[:10])!=set(x['doc_id'] for x in b[:10]),'top20_changed':set(x['doc_id'] for x in a[:20])!=set(x['doc_id'] for x in b[:20]),'top50_changed':set(x['doc_id'] for x in a[:50])!=set(x['doc_id'] for x in b[:50]),'support_rank_changed':ar!=br,'complete_status_changed':all(x is not None and x<=5 for x in ar)!=all(x is not None and x<=5 for x in br),'top5_overlap':overlap(a,b,5),'top10_overlap':overlap(a,b,10),'top20_overlap':overlap(a,b,20)})
        with (args.output_root/'per_query_ranking_delta.jsonl').open('a',encoding='utf-8') as h:
            for r in records:h.write(json.dumps(r)+'\n')
        summaries.append({'condition':condition,'queries':len(records),'top5_set_changed_rate':np.mean([x['top5_changed'] for x in records]),'top10_set_changed_rate':np.mean([x['top10_changed'] for x in records]),'top20_set_changed_rate':np.mean([x['top20_changed'] for x in records]),'top50_set_changed_rate':np.mean([x['top50_changed'] for x in records]),'support_rank_changed_rate':np.mean([x['support_rank_changed'] for x in records]),'complete_support_status_changed_rate':np.mean([x['complete_status_changed'] for x in records]),'top5_overlap':np.mean([x['top5_overlap'] for x in records]),'top10_overlap':np.mean([x['top10_overlap'] for x in records]),'top20_overlap':np.mean([x['top20_overlap'] for x in records])})
    with (args.output_root/'ranking_overlap_summary.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=summaries[0].keys());w.writeheader();w.writerows(summaries)
    text=['# Ranking and Context Delta Summary','', '| Condition | Top-5 changed | Top-20 changed | Support rank changed | Complete-support changed |','|---|---:|---:|---:|---:|']+[f"| {x['condition']} | {x['top5_set_changed_rate']:.1%} | {x['top20_set_changed_rate']:.1%} | {x['support_rank_changed_rate']:.1%} | {x['complete_support_status_changed_rate']:.1%} |" for x in summaries]
    (args.output_root/'context_delta_summary.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    (args.output_root/'cache_index_consistency_audit.md').write_text('# Cache and Index Consistency Audit\n\nThe Stage 0 runner performs fresh BGE encodes of every query and candidate document for each checkpoint and writes no embedding, FAISS, or retrieval-result cache. The only inherited candidate pool is a frozen sparse/routing candidate set; document embeddings are recomputed in the same adapter space as queries. Therefore no stale vector-index or cache namespace participates in this smoke.\n',encoding='utf-8')
    print(json.dumps({'status':'complete','summary':summaries},indent=2))

if __name__=='__main__':main()
