# R2-A.5 Frozen Compression Audit

R2-A.5 reuses R2-A's profiles, Q0 query, Router-Dev first 100 query IDs, and
local dense retrieval. It trains nothing. OracleSubset reads gold clients only
after candidate lists exist; S0--S3 use query/profile scores and frozen client
profiles only. Reader and final test are forbidden.
