# RECOMP Fairness Audit

| Audit field | Recorded value |
| --- | --- |
| Official repository | `https://github.com/carriex/recomp` |
| Repository commit | `51d4432` |
| Author checkpoint | `fangyuan/hotpotqa_extractive_compressor` |
| Original RECOMP reader | FLAN-UL2 |
| Adapted reader | Frozen FLAN-T5-Large |
| Input document count | Same baseline Top-5; mean matched count 4.986 |
| Output sentence count | 1 |
| Mean baseline input context tokens | 668.178 |
| Mean RECOMP output context tokens | 47.130 |
| Mean selected-method context tokens | 660.573 |
| RECOMP/baseline token ratio | 7.348% |
| Support treatment | The selected sentence is treated as predicted support |
| Output budget matched | No |

## Fairness questions

1. **Same Top-5 input?** Yes. RECOMP ranks sentences drawn from the exact frozen baseline documents.
2. **Same output token budget?** No. RECOMP emits one sentence averaging 47.13 context tokens; the baseline exposes 668.18 and the selected method 660.57.
3. **Is Top-1 structurally unfavorable for multi-hop evidence?** Yes. Hotpot-style questions commonly require two supporting facts, while one selected sentence can predict at most one support location in this evaluation.
4. **How much of the large gap comes from compression?** It cannot be identified from this run. The 7.35% token ratio is a major confound, so the numerical gap cannot be attributed solely to selection quality.
5. **Can Top-k or token matching be added without changing the frozen comparison?** No. Choosing k or a token budget after observing Top-1 results and rerunning the reader would introduce a new post-hoc condition. The uncompressed Top-5 baseline already provides the non-compressed anchor.

## Paper action

Detailed RECOMP scores are moved to the appendix. The main paper retains one concise supporting row and uses the exact label **official-code reproduction under a standardized reader adaptation**. Approved interpretation: "Under the standardized FLAN reader and the evaluated Top-1 extractive setting, RECOMP is poorly matched to the multi-hop context budget, whereas bounded context actions preserve complementary document evidence." No general superiority claim is allowed.
