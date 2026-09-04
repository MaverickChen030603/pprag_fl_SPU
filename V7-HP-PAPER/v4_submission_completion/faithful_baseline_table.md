# Table 5. Faithful-method external baseline on the 1,000-query development evaluation

| System | Answer F1 | Supporting-fact F1 | Joint F1 | Context protocol |
| --- | ---: | ---: | ---: | --- |
| Frozen Top-5 baseline | 0.6114 | 0.4920 | 0.3241 | Original five documents |
| RECOMP extractive compressor | 0.4437 | 0.3701 | 0.2084 | Official HotpotQA checkpoint, top-1 sentence from Top-5 |
| V4 semantic generator + selector | 0.6247 | 0.4973 | 0.3305 | Bounded five-document context action or fallback |

V4 minus RECOMP: answer F1 +0.1811, supporting-fact F1 +0.1272, joint F1 +0.1221. Classification: `faithful_method_reproduction_with_standardized_reader_adaptation`. We use the official repository at commit `51d4432`, author checkpoint `fangyuan/hotpotqa_extractive_compressor`, and paper settings of five input documents and one selected sentence. The paper's FLAN-UL2 reader is replaced by the frozen V4 FLAN-T5-Large reader to standardize downstream evaluation; this adaptation is stated rather than hidden. Supporting-fact evaluation is an extension that treats the selected sentence as RECOMP's predicted support fact.
