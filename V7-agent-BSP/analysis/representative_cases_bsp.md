# V7-agent-BSP Representative Cases

Generated from official per-query outputs. Selection-trace fields are populated when available.

## dynamic failed but bandit slot succeeded

- query_id: 543
- method: agent_bsp_memory_bandit_no_failure_state, seed: 1
- question: Gabe Turner has collaborated with the actor and film director who won his first Academy Award in which 1995 crime thriller?
- answer_F1: 1.0, support_F1: 0.5, joint_F1: 0.5
- explanation: BSP changes communication structure through slot allocation under fixed top-k; inspect selection trace for block-level rationale when available.

## bandit slot failed but BSP memory bandit succeeded

- query_id: 543
- method: agent_bsp_memory_bandit_no_failure_state, seed: 1
- question: Gabe Turner has collaborated with the actor and film director who won his first Academy Award in which 1995 crime thriller?
- answer_F1: 1.0, support_F1: 0.5, joint_F1: 0.5
- explanation: BSP changes communication structure through slot allocation under fixed top-k; inspect selection trace for block-level rationale when available.

## strict diagnostic signal but official QA flat

- query_id: 543
- method: agent_bsp_memory_bandit_strict, seed: 2
- question: Gabe Turner has collaborated with the actor and film director who won his first Academy Award in which 1995 crime thriller?
- answer_F1: 1.0, support_F1: 0.5, joint_F1: 0.5
- explanation: BSP changes communication structure through slot allocation under fixed top-k; inspect selection trace for block-level rationale when available.

## official QA improved case

- query_id: 543
- method: agent_bsp_memory_bandit_strict, seed: 2
- question: Gabe Turner has collaborated with the actor and film director who won his first Academy Award in which 1995 crime thriller?
- answer_F1: 1.0, support_F1: 0.5, joint_F1: 0.5
- explanation: BSP changes communication structure through slot allocation under fixed top-k; inspect selection trace for block-level rationale when available.

## hard-query early evidence case

- query_id: 543
- method: agent_bsp_memory_bandit_strict, seed: 2
- question: Gabe Turner has collaborated with the actor and film director who won his first Academy Award in which 1995 crime thriller?
- answer_F1: 1.0, support_F1: 0.5, joint_F1: 0.5
- explanation: BSP changes communication structure through slot allocation under fixed top-k; inspect selection trace for block-level rationale when available.

## bridge-heavy failure candidate

- query_id: 543
- method: agent_bsp_memory_bandit_strict, seed: 2
- question: Gabe Turner has collaborated with the actor and film director who won his first Academy Award in which 1995 crime thriller?
- answer_F1: 1.0, support_F1: 0.5, joint_F1: 0.5
- explanation: BSP changes communication structure through slot allocation under fixed top-k; inspect selection trace for block-level rationale when available.

## rare-domain candidate

- query_id: 543
- method: agent_bsp_memory_bandit_strict, seed: 2
- question: Gabe Turner has collaborated with the actor and film director who won his first Academy Award in which 1995 crime thriller?
- answer_F1: 1.0, support_F1: 0.5, joint_F1: 0.5
- explanation: BSP changes communication structure through slot allocation under fixed top-k; inspect selection trace for block-level rationale when available.

## memory conservative failure

- query_id: 682
- method: agent_bsp_memory_bandit_reader, seed: 0
- question: Which baseball player and manager was born in Ellisville, Mississippi, home of 4448 people?
- answer_F1: 0.8, support_F1: 0.5, joint_F1: 0.4
- explanation: BSP changes communication structure through slot allocation under fixed top-k; inspect selection trace for block-level rationale when available.
