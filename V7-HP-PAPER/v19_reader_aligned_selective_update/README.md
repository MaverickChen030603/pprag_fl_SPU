# V19: Reader-Aligned Selective Update

This directory contains the new retriever-adaptation research line, separate
from the terminated V17/V18 context-composition line. `stage0_full_upload/`
is intentionally the only executable experiment at initialization. The later
directories are created only after their preceding Go/No-Go conditions pass.

Run the setup from the repository root:

```bash
/home/iiserver31/anaconda3/envs/supv2/bin/python \
  V7-HP-PAPER/v19_reader_aligned_selective_update/protocol/build_v19_manifests.py \
  --v17-manifest V7-HP-PAPER/v17_fedaction_rag/protocol/dataset_split_manifest.json \
  --v17-audit V7-HP-PAPER/v17_fedaction_rag/protocol/no_leak_audit.json \
  --output V7-HP-PAPER/v19_reader_aligned_selective_update/protocol/dataset_manifest.json

/home/iiserver31/anaconda3/envs/supv2/bin/python \
  V7-HP-PAPER/v19_reader_aligned_selective_update/model/build_block_schema.py \
  --schema V7-HP-PAPER/v19_reader_aligned_selective_update/model/block_schema.json \
  --payload-table V7-HP-PAPER/v19_reader_aligned_selective_update/model/block_payload_table.csv
```
