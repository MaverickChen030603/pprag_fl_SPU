#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAPER="$(cd "$ROOT/../.." && pwd)"
PYTHON="${PYTHON:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
LABELER="$PAPER/v16_action_composition/multi_reader/01_label_oracle_contexts.py"
DATA="$PAPER/v17_fedaction_rag/data/hotpotqa/development.jsonl"
SUPPORT="$PAPER/v16_action_composition/evaluation/checkpoints/hotpotqa_support.joblib"
DIAG="$ROOT/reader_diagnostic"
for INDEX in 0 1 2 3; do
  CONDITION=(frozen centralized fedavg fedprox)
  C="${CONDITION[$INDEX]}"
  CUDA_VISIBLE_DEVICES="$INDEX" "$PYTHON" "$LABELER" --reader flan --model google/flan-t5-large --dataset hotpotqa --split "$DATA" --pool "$DIAG/$C/pool.jsonl" --contexts "$DIAG/$C/contexts.jsonl" --support-predictor "$SUPPORT" --output "$DIAG/$C/flan_outputs.jsonl" --device cuda --batch-size 8 --max-contexts-per-query 1 --resume >"$DIAG/$C/flan.log" 2>&1 &
done
wait
"$PYTHON" - "$DIAG" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[]
for condition in ('frozen','centralized','fedavg','fedprox'):
    path=root/condition/'flan_outputs.jsonl'; values=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    metrics={key:sum(float(x.get(key,0.0)) for x in values)/max(1,len(values)) for key in ('answer_f1','sp_f1','joint_f1')}
    rows.append({'condition':condition,'queries':len(values),**metrics})
(root/'flan_summary.json').write_text(json.dumps({'kind':'exploratory_changed_context_diagnostic','rows':rows},indent=2)+'\n',encoding='utf-8')
print(json.dumps(rows,indent=2))
PY
