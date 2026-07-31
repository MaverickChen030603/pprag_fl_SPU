#!/usr/bin/env python3
"""Apply the pre-registered context-delta gate without reader calls."""

from __future__ import annotations

import argparse,csv,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--forward',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=True)
 summary=list(csv.DictReader(a.summary.open(encoding='utf-8'))); forward=list(csv.DictReader(a.forward.open(encoding='utf-8')))
 trainable=[x for x in summary if x['condition']!='frozen']; fwd={x['condition']:float(x['l2_vs_base'])>1e-8 for x in forward if x['mode']=='enabled'}
 # The quantitative alternatives do not replace the prior engineering
 # requirement that a reader-relevant complete-support status changes at all.
 complete_status_exists=any(float(x['complete_support_status_changed_rate'])>0 for x in trainable)
 passes=[x for x in trainable if complete_status_exists and float(x['top5_set_changed_rate'])>=.05 and (float(x['support_rank_changed_rate'])>=.10 or float(x['complete_support_status_changed_rate'])>=.03 or float(x['top10_set_changed_rate'])>=.10) and fwd.get(x['condition'],False)]
 status='pass' if passes else 'blocked_before_reader'
 payload={'status':status,'reader_evaluation_started':False,'positive_context_conditions':[x['condition'] for x in passes],'reason':'Top-5/ranking changes exist, but no condition changes complete-support status; the pre-registered all-criteria gate is not met.' if not passes else 'At least one condition meets the context-delta engineering gate.'}
 (a.output_root/'reader_start_decision.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
 (a.output_root/'stage0a_context_delta_gate.md').write_text('# Stage 0A Context Delta Gate\n\n'+json.dumps(payload,indent=2)+'\n\nReader labeling is forbidden while this status is `blocked_before_reader`.\n',encoding='utf-8');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
