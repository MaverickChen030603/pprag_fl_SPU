#!/usr/bin/env python3
import json, math
from pathlib import Path
ROOT=Path('/home/iiserver31/projects/FedE4RAG-main')
OUT=ROOT/'V7-HP-PAPER/high_tier_extension/multi_reader_context_repair'

def read_json(p, default=None):
    p=Path(p); return json.loads(p.read_text()) if p.exists() else ({} if default is None else default)

def fmt(x, nd=4, signed=False):
    if x is None: return 'NA'
    try:
        x=float(x); s=f'{x:.{nd}f}'; return '+'+s if signed and x>0 else s
    except Exception: return str(x)

def md_table(headers, rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(c) for c in r)+' |' for r in rows)+'\n'
metrics=read_json(OUT/'outputs/metrics/multi_reader_metrics.json',{})
sig=read_json(OUT/'outputs/metrics/multi_reader_significance.json',{})
aud=read_json(OUT/'outputs/audit/available_local_readers.json',{})
feas=read_json(OUT/'outputs/audit/reader_runtime_feasibility.json',{})
ctx=read_json(OUT/'outputs/context_snapshots/context_snapshot_summary.json',{})
ca=read_json(OUT/'outputs/audit/context_snapshot_audit.json',{})
rows=[]
# frozen main
fm=metrics.get('google/flan-t5-large_frozen_main',{})
fs=sig.get('google/flan-t5-large_frozen_main',{})
rows.append(['google/flan-t5-large_frozen_main','1000',fmt(fm.get('answer_f1_delta'),signed=True),fmt(fm.get('joint_f1_delta'),signed=True),fmt(fm.get('support_recall_delta'),signed=True),fmt(fm.get('sp_f1_delta'),signed=True),fmt(fs.get('answer_f1',{}).get('p_value')),fmt(fs.get('joint_f1',{}).get('p_value')),'completed_existing','frozen main reader; not rerun'])
# reader output summaries
for d in (OUT/'outputs/reader_outputs').glob('*'):
    if not d.is_dir(): continue
    summ=read_json(d/'reader_run_summary.json',{})
    reader=summ.get('reader_name') or d.name.replace('__','/')
    if reader=='google/flan-t5-large_frozen_main': continue
    m=metrics.get(reader,{})
    sg=sig.get(reader,{})
    status=summ.get('status') or m.get('status','unknown')
    n=m.get('n') or summ.get('num_examples') or 0
    conclusion='completed smoke' if status=='completed' and n<1000 else ('completed full' if status=='completed' else (summ.get('reason') or status))
    rows.append([reader,str(n),fmt(m.get('answer_f1_delta'),signed=True),fmt(m.get('joint_f1_delta'),signed=True),fmt(m.get('support_recall_delta'),signed=True),fmt(m.get('sp_f1_delta'),signed=True),fmt(sg.get('answer_f1_delta',{}).get('p_value')),fmt(sg.get('joint_f1_delta',{}).get('p_value')),status,conclusion])
headers=['reader','n','answer_f1_delta','joint_f1_delta','support_recall@5_delta','sp_f1_delta','answer_f1_p','joint_f1_p','status','conclusion']
(OUT/'outputs/tables/multi_reader_replication_table.md').write_text(md_table(headers, rows))
completed_extra=[r for r in rows if r[0] != 'google/flan-t5-large_frozen_main' and r[8]=='completed']
if completed_extra:
    if any(int(r[1])>=1000 for r in completed_extra): decision='appendix_robustness_possible'
    else: decision='appendix_smoke_only'
else:
    decision='limitation_only'
text=f"""# Final Multi-Reader Decision\n\n## Answers\n\n1. Context materialization succeeded: **{ca.get('status')}**. `num_missing_context={ctx.get('num_missing_context')}`, baseline/selected contexts are available for {ctx.get('num_examples')} examples.\n2. Local reader availability: **{aud.get('decision')}**. Usable readers: `{aud.get('usable_readers', [])}`.\n3. Extra reader replication completed: **{bool(completed_extra)}**.\n4. Joint/support positivity on extra readers: **{'available in table' if completed_extra else 'not evaluated'}**.\n5. answer_f1 remains reader-sensitive / unverified beyond the frozen main reader.\n6. Placement: **{'appendix robustness/smoke' if completed_extra else 'limitation / appendix attempt'}**.\n7. Submission target: **{'NAACL/EMNLP main stretch only if full-reader result is positive; otherwise Findings/COLING' if completed_extra else 'Findings / COLING'}**.\n\n## Runtime Feasibility\n\n- max GPU free MiB: {feas.get('max_gpu_free_mib')}\n- runtime decision: {feas.get('decision')}\n\n## Final Decision\n\n`{decision}`\n\nIf no extra reader completed, do not continue patching experiments now. The current experiments are sufficient for paper writing as a HotpotQA-centered paper with 2Wiki diagnostic limitation and multi-reader as a limitation/replication-ready appendix attempt.\n"""
(OUT/'reports/final_multi_reader_decision.md').write_text(text)
# Update claim boundary and report with final wording.
boundary="""# Multi-Reader Claim Boundary\n\nAllowed if a full additional reader succeeds with positive joint/support deltas:\n\n> The frozen v2.3 selected contexts show consistent joint/support-side improvements across readers, while answer_f1 remains reader-sensitive.\n\nAllowed if only a bounded smoke succeeds:\n\n> A bounded reader replication smoke suggests similar joint/support trends, but full multi-reader validation remains future work.\n\nAllowed if no extra reader succeeds:\n\n> Multi-reader replication was prepared by materializing the frozen final_1000 baseline and selected contexts, but additional reader execution was blocked by local model availability and runtime constraints. We therefore keep multi-reader evaluation as a limitation rather than a strengthened claim.\n\nForbidden:\n\n- v2.3 universally improves all readers\n- answer_f1 significantly improves across readers\n- multi-reader robustness verified\n"""
(OUT/'reports/multi_reader_claim_boundary.md').write_text(boundary)
rep=f"""# Multi-Reader Replication Report\n\nContext materialization: **{ca.get('status')}**. The frozen final_1000 baseline and selected contexts are available for {ctx.get('num_examples')} examples.\n\nAvailable local readers decision: **{aud.get('decision')}**. Runtime decision: **{feas.get('decision')}**.\n\nFinal placement: **{'appendix robustness/smoke' if completed_extra else 'limitation / appendix attempt'}**.\n\nThe HotpotQA v2.3 main result remains unchanged. Multi-reader outputs must not overwrite the frozen main result.\n\nSee `outputs/tables/multi_reader_replication_table.md` for the final table.\n"""
(OUT/'reports/multi_reader_replication_report.md').write_text(rep)
print(json.dumps({'decision':decision,'completed_extra_readers':[r[0] for r in completed_extra]},ensure_ascii=False))
