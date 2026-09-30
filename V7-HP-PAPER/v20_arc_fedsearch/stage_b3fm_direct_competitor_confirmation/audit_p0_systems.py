#!/usr/bin/env python3
"""Read-only P0 application-payload, storage, and recorded-latency audit."""
from __future__ import annotations
import argparse,csv,json,sqlite3,platform,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
DATASETS=("hotpotqa","2wikimultihopqa","musique")
def rows(p):
 with p.open() as h:
  for x in h:
   if x.strip():yield json.loads(x)
def header(**x):return len(json.dumps(x,sort_keys=True,separators=(",",":")).encode())
def write(p,v):
 with p.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=sorted({k for x in v for k in x}));w.writeheader();w.writerows(v)
def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment-root',type=Path,required=True);p.add_argument('--stage',type=Path,required=True);a=p.parse_args();cost=a.stage/'cost';cost.mkdir(exist_ok=True)
 docs={};db={d:sqlite3.connect(f"file:{a.experiment_root/f'inputs/indexes/{d}.sqlite'}?mode=ro",uri=True) for d in DATASETS}
 def size(d,i):
  k=(d,i)
  if k not in docs:
   r=db[d].execute('select title,text from docs where doc_id=?',(i,)).fetchone();docs[k]=len((str(r[0])+'\n'+str(r[1])).encode())
  return docs[k]
 out=[];packets={d:{str(x['query_id']):x for x in rows(a.stage/f'retrieval/{d}_probe_packets.jsonl')} for d in DATASETS}
 for x in rows(a.stage/'contexts/all_contexts_unscored.jsonl'):
  d,q,m=x['dataset'],str(x['query_id']),x['method'];sel=list(x['selected_client_ids']);question=len(str(x['question']).encode());probe=m=='m2_logistic_proberoute'; req=8 if probe else len(sel); probe_resp=576 if probe else 0
  req_bytes=question*req; req_head=sum(header(dataset=d,query_id=q,client_id=c,message_type='probe_request' if probe else 'deep_request',payload_length=question) for c in (x['candidate_client_ids'] if probe else sel))
  resp_head=sum(header(dataset=d,query_id=q,client_id=c,message_type='probe_response',payload_length=72) for c in x['candidate_client_ids']) if probe else 0
  deep_incremental=0 # Frozen implementation reuses M2's client request for selected clients.
  docids=list(x['transmitted_doc_ids']);docbytes=sum(size(d,i) for i in docids);deep_head=sum(header(dataset=d,query_id=q,client_id='document_return',message_type='document',payload_length=size(d,i)) for i in docids)
  out.append({'dataset':d,'query_id':q,'method':m,'selected_clients':len(sel),'documents':len(docids),'query_request_bytes':req_bytes,'probe_float32_bytes':probe_resp,'probe_protocol_header_bytes':resp_head,'request_protocol_header_bytes':req_head,'deep_request_incremental_bytes':deep_incremental,'document_utf8_payload_bytes':docbytes,'deep_return_header_bytes':deep_head,'application_total_bytes':req_bytes+probe_resp+resp_head+req_head+docbytes+deep_head,'request_reuse':probe,'wire_level_measurement':'unavailable_shared_memory_filesystem_simulation'})
 for v in db.values():v.close()
 write(cost/'per_query_communication_compute_audit.csv',out);summary=[]
 groups=defaultdict(list)
 for row in out: groups[(row['dataset'],row['method'])].append(row)
 for k,z in sorted(groups.items()):
  summary.append({'dataset':k[0],'method':k[1],'queries':len(z),**{f'mean_{f}':float(np.mean([x[f] for x in z])) for f in ('selected_clients','documents','query_request_bytes','probe_float32_bytes','document_utf8_payload_bytes','application_total_bytes')}})
 write(cost/'communication_compute_audit.csv',summary)
 lat=[]
 for d in DATASETS:
  vals=np.asarray([float(x['probe_materialization_latency_ms']) for x in packets[d].values()]);lat.append({'dataset':d,'component':'eight_candidate_probe_materialization','measurement':'recorded_warm_cache_simulation','mean_ms':float(vals.mean()),'p50_ms':float(np.median(vals)),'p95_ms':float(np.quantile(vals,.95)),'note':'Packet materialization time; not real network latency and not a complete service critical path.'})
 write(cost/'recorded_latency_audit.csv',lat)
 offline=[]
 for d in DATASETS:
  for pth in [a.experiment_root/f'inputs/models/{d}/logistic_seed_20260807.pkl',a.experiment_root/f'runs/ragroute_b3_r5_posthoc_20260916/centroids/{d}/source_centroids.npy',a.stage.parent/'stage_b3f_ragroute_faithful'/f'models/{d}/seed_0.pt']:
   offline.append({'dataset':d,'artifact':pth.name,'bytes':pth.stat().st_size,'category':'offline_persistent_not_per_query'})
 write(cost/'offline_storage_audit.csv',offline)
 (cost/'environment_manifest.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'wire_level_measurement':'unavailable_shared_memory_filesystem_simulation','cache_contract':'existing packet materialization timings; warm/cold deep retrieval benchmark not yet available','latency_scope':'recorded probe materialization only; no fabricated deep/Reader critical-path latency'},indent=2)+'\n')
if __name__=='__main__':main()
