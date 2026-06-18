#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/home/iiserver31/anaconda3/envs/supv2/bin/python}"
"$PYTHON_BIN" - <<'PY_FID'
import sentencepiece
print('sentencepiece ok', sentencepiece.__version__)
from transformers import T5Tokenizer
T5Tokenizer.from_pretrained('t5-base')
print('t5 tokenizer ok')
PY_FID
