#!/usr/bin/env python3
"""Audit the actual Stage 0 smoke supervision rather than assuming it is hard."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

V19=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(V19/'stage0_full_upload'))
from run_stage0_viability import build_pairs


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--assignment',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--dataset',default='hotpotqa');p.add_argument('--limit',type=int,default=1000);p.add_argument('--clients',type=int,default=20);p.add_argument('--seed',type=int,default=20260731)
    a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=True); pairs=build_pairs(a.train,a.assignment,a.dataset,a.limit,a.clients,a.seed)
    records=[]
    for client in range(a.clients):
        values=pairs.get(client,[]); records.append({'client_id':client,'number_of_queries':len(values),'number_of_positive_pairs':len(values),'number_of_negative_pairs_explicit':0,'in_batch_negatives_per_step':7,'unique_positive_documents':len(set(x[1] for x in values)),'duplicate_pair_rate':1-len(set(values))/max(1,len(values)),'clients_with_no_effective_data':not bool(values)})
    with (a.output_root/'client_data_statistics.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    samples=['# Pair Quality Samples','', 'The initial smoke used one gold support document per query as the positive and other batch positives as implicit negatives. It has **no explicit retrieval, entity-overlap, partial-hop, or cross-client hard negatives**. This is a confirmed objective limitation, not an implementation failure.','', '| Client | Sample count | Unique positive documents |','|---|---:|---:|']+[f"| {r['client_id']} | {r['number_of_queries']} | {r['unique_positive_documents']} |" for r in records]
    (a.output_root/'pair_quality_samples.md').write_text('\n'.join(samples)+'\n',encoding='utf-8')
    report=['# Training Signal Audit','',f'Train pairs: {sum(len(x) for x in pairs.values())}; active clients: {sum(bool(x) for x in pairs.values())}/{a.clients}.','', 'The loss is in-batch contrastive with batch size 8. Its only negative source is another query\'s positive passage. Thus the initial smoke can establish adapter trainability but cannot test the stated hard-negative objective. The next legal single-factor positive control is PC-1: retain model, rank, learning rate, and steps while replacing implicit-only negatives with a precomputed train-only hard-negative manifest.']
    (a.output_root/'training_signal_audit.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({'status':'complete','pairs':sum(map(len,pairs.values())),'active_clients':sum(bool(x) for x in pairs.values()),'explicit_hard_negatives':False},indent=2))

if __name__=='__main__':main()
