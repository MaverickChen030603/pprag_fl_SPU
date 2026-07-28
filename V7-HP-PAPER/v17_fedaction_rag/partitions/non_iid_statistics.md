# V17 Non-IID Statistics

All partitions and query origins are label-free. Dirichlet settings are controlled stress tests, not claims about real organizational silos.

## Document Distribution

| Dataset | Partition | M | Documents | Min | Max | Max/mean | CV | Normalized entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | dirichlet_a0p1 | 20 | 369280 | 5707 | 55642 | 3.014 | 0.684 | 0.936 |
| 2wikimultihopqa | dirichlet_a0p3 | 20 | 369280 | 7049 | 39988 | 2.166 | 0.477 | 0.964 |
| 2wikimultihopqa | dirichlet_a1p0 | 20 | 369280 | 12828 | 29910 | 1.620 | 0.248 | 0.991 |
| 2wikimultihopqa | entity_community | 20 | 369280 | 18087 | 25625 | 1.388 | 0.089 | 0.999 |
| 2wikimultihopqa | random_control | 20 | 369280 | 18464 | 18464 | 1.000 | 0.000 | 1.000 |
| 2wikimultihopqa | topic_silo | 20 | 369280 | 5165 | 25976 | 1.407 | 0.257 | 0.987 |
| hotpotqa | dirichlet_a0p1 | 20 | 481959 | 10495 | 47782 | 1.983 | 0.430 | 0.970 |
| hotpotqa | dirichlet_a0p3 | 20 | 481959 | 10670 | 41918 | 1.739 | 0.381 | 0.975 |
| hotpotqa | dirichlet_a1p0 | 20 | 481959 | 16230 | 34252 | 1.421 | 0.199 | 0.993 |
| hotpotqa | entity_community | 20 | 481959 | 24097 | 24098 | 1.000 | 0.000 | 1.000 |
| hotpotqa | random_control | 20 | 481959 | 24097 | 24098 | 1.000 | 0.000 | 1.000 |
| hotpotqa | topic_silo | 20 | 481959 | 14226 | 38216 | 1.586 | 0.253 | 0.990 |
| musique | dirichlet_a0p1 | 20 | 84549 | 953 | 8532 | 2.018 | 0.488 | 0.957 |
| musique | dirichlet_a0p3 | 20 | 84549 | 1134 | 7077 | 1.674 | 0.379 | 0.975 |
| musique | dirichlet_a1p0 | 20 | 84549 | 2189 | 5907 | 1.397 | 0.245 | 0.990 |
| musique | entity_community | 20 | 84549 | 3255 | 6834 | 1.617 | 0.273 | 0.988 |
| musique | random_control | 20 | 84549 | 4227 | 4228 | 1.000 | 0.000 | 1.000 |
| musique | topic_silo | 20 | 84549 | 1370 | 6868 | 1.625 | 0.314 | 0.982 |

## Query-Origin Distribution

| Dataset | Partition | Split | Queries | Min | Max | CV | Normalized entropy |
|---|---|---|---:|---:|---:|---:|---:|
| 2wikimultihopqa | dirichlet_a0p1 | calibration | 1000 | 38 | 72 | 0.171 | 0.995 |
| 2wikimultihopqa | dirichlet_a0p1 | development | 1000 | 31 | 65 | 0.184 | 0.994 |
| 2wikimultihopqa | dirichlet_a0p1 | train | 5000 | 189 | 302 | 0.145 | 0.996 |
| 2wikimultihopqa | dirichlet_a0p3 | calibration | 1000 | 34 | 69 | 0.158 | 0.996 |
| 2wikimultihopqa | dirichlet_a0p3 | development | 1000 | 38 | 73 | 0.165 | 0.996 |
| 2wikimultihopqa | dirichlet_a0p3 | train | 5000 | 207 | 299 | 0.101 | 0.998 |
| 2wikimultihopqa | dirichlet_a1p0 | calibration | 1000 | 37 | 64 | 0.137 | 0.997 |
| 2wikimultihopqa | dirichlet_a1p0 | development | 1000 | 41 | 62 | 0.126 | 0.997 |
| 2wikimultihopqa | dirichlet_a1p0 | train | 5000 | 204 | 281 | 0.085 | 0.999 |
| 2wikimultihopqa | entity_community | calibration | 1000 | 39 | 66 | 0.133 | 0.997 |
| 2wikimultihopqa | entity_community | development | 1000 | 29 | 66 | 0.192 | 0.994 |
| 2wikimultihopqa | entity_community | train | 5000 | 213 | 301 | 0.086 | 0.999 |
| 2wikimultihopqa | random_control | calibration | 1000 | 34 | 59 | 0.124 | 0.997 |
| 2wikimultihopqa | random_control | development | 1000 | 38 | 59 | 0.120 | 0.998 |
| 2wikimultihopqa | random_control | train | 5000 | 232 | 267 | 0.044 | 1.000 |
| 2wikimultihopqa | topic_silo | calibration | 1000 | 17 | 77 | 0.339 | 0.980 |
| 2wikimultihopqa | topic_silo | development | 1000 | 10 | 78 | 0.372 | 0.975 |
| 2wikimultihopqa | topic_silo | train | 5000 | 112 | 385 | 0.309 | 0.983 |
| hotpotqa | dirichlet_a0p1 | calibration | 1000 | 39 | 62 | 0.139 | 0.997 |
| hotpotqa | dirichlet_a0p1 | development | 1000 | 34 | 67 | 0.204 | 0.993 |
| hotpotqa | dirichlet_a0p1 | train | 5000 | 169 | 299 | 0.135 | 0.997 |
| hotpotqa | dirichlet_a0p3 | calibration | 1000 | 35 | 61 | 0.162 | 0.995 |
| hotpotqa | dirichlet_a0p3 | development | 1000 | 40 | 62 | 0.122 | 0.998 |
| hotpotqa | dirichlet_a0p3 | train | 5000 | 210 | 296 | 0.100 | 0.998 |
| hotpotqa | dirichlet_a1p0 | calibration | 1000 | 42 | 65 | 0.115 | 0.998 |
| hotpotqa | dirichlet_a1p0 | development | 1000 | 35 | 64 | 0.156 | 0.996 |
| hotpotqa | dirichlet_a1p0 | train | 5000 | 220 | 304 | 0.081 | 0.999 |
| hotpotqa | entity_community | calibration | 1000 | 40 | 64 | 0.116 | 0.998 |
| hotpotqa | entity_community | development | 1000 | 34 | 70 | 0.179 | 0.995 |
| hotpotqa | entity_community | train | 5000 | 220 | 273 | 0.057 | 0.999 |
| hotpotqa | random_control | calibration | 1000 | 37 | 59 | 0.109 | 0.998 |
| hotpotqa | random_control | development | 1000 | 37 | 60 | 0.135 | 0.997 |
| hotpotqa | random_control | train | 5000 | 230 | 292 | 0.070 | 0.999 |
| hotpotqa | topic_silo | calibration | 1000 | 31 | 71 | 0.171 | 0.995 |
| hotpotqa | topic_silo | development | 1000 | 37 | 67 | 0.188 | 0.994 |
| hotpotqa | topic_silo | train | 5000 | 198 | 290 | 0.129 | 0.997 |
| musique | dirichlet_a0p1 | calibration | 1000 | 30 | 65 | 0.161 | 0.996 |
| musique | dirichlet_a0p1 | development | 1000 | 33 | 60 | 0.125 | 0.997 |
| musique | dirichlet_a0p1 | train | 5000 | 221 | 292 | 0.077 | 0.999 |
| musique | dirichlet_a0p3 | calibration | 1000 | 35 | 72 | 0.184 | 0.995 |
| musique | dirichlet_a0p3 | development | 1000 | 38 | 64 | 0.158 | 0.996 |
| musique | dirichlet_a0p3 | train | 5000 | 217 | 279 | 0.072 | 0.999 |
| musique | dirichlet_a1p0 | calibration | 1000 | 37 | 59 | 0.131 | 0.997 |
| musique | dirichlet_a1p0 | development | 1000 | 36 | 63 | 0.154 | 0.996 |
| musique | dirichlet_a1p0 | train | 5000 | 219 | 284 | 0.074 | 0.999 |
| musique | entity_community | calibration | 1000 | 40 | 64 | 0.164 | 0.996 |
| musique | entity_community | development | 1000 | 36 | 67 | 0.142 | 0.997 |
| musique | entity_community | train | 5000 | 197 | 306 | 0.104 | 0.998 |
| musique | random_control | calibration | 1000 | 38 | 59 | 0.099 | 0.998 |
| musique | random_control | development | 1000 | 32 | 77 | 0.189 | 0.994 |
| musique | random_control | train | 5000 | 225 | 282 | 0.064 | 0.999 |
| musique | topic_silo | calibration | 1000 | 36 | 66 | 0.159 | 0.996 |
| musique | topic_silo | development | 1000 | 34 | 62 | 0.161 | 0.996 |
| musique | topic_silo | train | 5000 | 185 | 299 | 0.113 | 0.998 |
