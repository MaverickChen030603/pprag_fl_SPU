#!/usr/bin/env python3
"""Build train-only lexical/entity/partial-hop hard negatives for PC-1."""

from __future__ import annotations

import argparse, json, re
from collections import Counter
from pathlib import Path

def rows(path):
    with path.open(encoding='utf-8') as h:
        for line in h:
            if line.strip(): yield json.loads(line)
def tok(x): return {w.lower() for w in re.findall(r'[A-Za-z0-9]+',str(x)) if len(w)>2}
def norm(x): return ' '.join(str(x).lower().split())

def main():
 p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=1000);p.add_argument('--per-query',type=int,default=4);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
 count=Counter();written=0
 with a.output.open('w',encoding='utf-8') as out:
  for row in rows(a.train):
   context=row.get('context',{}); titles=context.get('title',[]) if isinstance(context,dict) else []; sents=context.get('sentences',[]) if isinstance(context,dict) else []
   gold={norm(x) for x in row.get('supporting_facts',{}).get('title',[])}; q=tok(row['question']); docs=[]
   for title,ss in zip(titles,sents):
    text=' '.join(str(x).strip() for x in ss if str(x).strip()); title_norm=norm(title)
    if title_norm in gold: docs.append(('positive',str(title),text))
    else:
     d=tok(str(title)+' '+text); overlap=len(q&d); entity=sum(x[0].isupper() for x in str(title).split()); docs.append(('negative',str(title),text,overlap,entity))
   positives=[x for x in docs if x[0]=='positive']; negatives=sorted([x for x in docs if x[0]=='negative'],key=lambda x:(x[3],x[4],len(x[2])),reverse=True)
   if positives and len(negatives)>=a.per_query:
    pos=positives[0]; chosen=negatives[:a.per_query]
    payload={'query_id':str(row['query_id']),'query':row['question'],'positive':pos[1]+': '+pos[2],'negatives':[x[1]+': '+x[2] for x in chosen], 'negative_provenance':['entity_overlap' if x[4] else 'lexical_overlap' for x in chosen], 'gold_title_count':len(gold)}
    out.write(json.dumps(payload,ensure_ascii=False)+'\n');written+=1;count.update(payload['negative_provenance'])
   if written>=a.limit:break
 stats={'status':'complete','queries':written,'negatives_per_query':a.per_query,'provenance':dict(count),'source':'train_only'}
 a.output.with_name('hard_negative_statistics.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8');print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
