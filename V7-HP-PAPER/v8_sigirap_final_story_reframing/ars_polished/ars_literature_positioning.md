# ARS Literature-Review Positioning

This revision uses only citation keys already present in the manuscript. It does not add unverified references or broaden the empirical claim.

## 1. Multi-hop retrieval and evidence acquisition

HotpotQA and 2WikiMultiHopQA provide answer and supporting-fact supervision [@yang-etal-2018-hotpotqa; @ho-etal-2020-2wiki], while MDR represents multi-step retrieval that changes which evidence enters the pool [@xiong-etal-2021-mdr]. The present paper starts after this stage: retrieval is frozen, and the object is the bounded reader context.

## 2. Reader-aware ranking and set construction

RankRAG, RCPS, and SetR motivate retrieval or passage selection that accounts for downstream generation and set interactions [@yu-etal-2024-rankrag; @xin-etal-2025-rcps; @lee-etal-2025-setr]. Distractor and position studies show why independently relevant passages need not form a reader-compatible context [@shi-etal-2023-distracted; @liu-etal-2024-lost]. Full's specific contribution is to make the finite generated action set explicit and measure both whether useful alternatives exist and whether a frozen policy realizes them.

## 3. Compression and selective prediction

RECOMP represents sentence-level context compression [@xu-etal-2024-recomp]. It controls context budget but exposes a different action space from five-document structural construction. Selective prediction motivates fallback under estimated risk [@geifman-elyaniv-2019-selectivenet]; here fallback preserves the frozen Top-5 context and risk is calibrated empirically rather than certified per query.

## Positioning synthesis

The strongest literature claim is not that earlier work ignores context interactions. It is that candidate generation and selective realization are usually evaluated together, making it difficult to distinguish unavailable repairs from selection misses. The paper contributes a bounded pair-complementary constructor, a leak-controlled selector, and a diagnostic decomposition under a common reader protocol. The matched CrossEncoder result should remain prominent because it clarifies the incremental method claim: independent relevance recovers stronger SP/Joint, while Full occupies a different Answer-oriented point.
