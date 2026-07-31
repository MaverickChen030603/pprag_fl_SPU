#!/usr/bin/env python3
"""Create a fixed union changed-query reader diagnostic, never a main evaluation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def rows(p):
 with p.open(encoding='utf-8') as h:
  for x in h:
   if x.strip():yield json.loads(x)
def write(p,items):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8') as h:
  for x in items:h.write(json.dumps(x,ensure_ascii=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--ranking',type=Path,required=True);p.add_argument('--smoke-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--conditions',nargs='+',default=['centralized','fedavg','fedprox']);a=p.parse_args()
 changed={str(x['query_id']) for x in rows(a.ranking) if x['condition'] in a.conditions and x['top5_changed']}
 all_conditions=['frozen',*a.conditions]
 for c in all_conditions:
  source=a.smoke_root/c; pools=[x for x in rows(source/'adapted_pool.jsonl') if str(x['query_id']) in changed]; contexts=[x for x in rows(source/'contexts.jsonl') if str(x['query_id']) in changed]
  write(a.output_root/c/'pool.jsonl',pools);write(a.output_root/c/'contexts.jsonl',contexts)
 manifest={'status':'complete','kind':'exploratory_changed_context_diagnostic','reader_status':'not_started','conditions':all_conditions,'query_count':len(changed),'query_ids':sorted(changed),'selection_rule':'union of top5_changed queries from pre-existing Stage 0A ranking audit; no reader outcome used.'}
 a.output_root/'manifest.json';(a.output_root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
