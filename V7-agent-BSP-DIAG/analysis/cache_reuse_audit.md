# Cache Reuse Audit

- sensitivity directories: 10
- examples: beam1_len1024_agent_priority, beam1_len1024_gold_oracle_debug, beam1_len1024_retrieval_score, beam1_len512_agent_priority, beam1_len512_gold_oracle_debug, beam1_len512_retrieval_score, beam1_len768_agent_priority, beam1_len768_gold_oracle_debug, beam1_len768_retrieval_score, beam3_len512_retrieval_score

- non-retrieval ordering rows: 60
- zero reader-input diff rate share: 1.000

Finding: most ordering variants produce identical reader input hashes; ordering path is likely ineffective or not connected to reader input.

## sensitivity grid observed
beam_size,max_input_length,passage_ordering,n
1,512,agent_priority,45
1,512,gold_oracle_debug,45
1,512,retrieval_score,45
1,768,agent_priority,45
1,768,gold_oracle_debug,45
1,768,retrieval_score,45
1,1024,agent_priority,45
1,1024,gold_oracle_debug,45
1,1024,retrieval_score,45
3,512,retrieval_score,6
