from __future__ import annotations

import unittest

from action_atoms.action_atoms import ActionKind, AtomicAction, ContextState, apply_action, apply_trajectory, legal_actions, shortest_repair_trajectory


def state() -> ContextState:
    return ContextState(context=("d0", "d1", "d2", "d3", "d4"), pool=tuple(f"d{i}" for i in range(10)))


class ActionAtomTests(unittest.TestCase):
    def test_every_edit_preserves_budget_and_uniqueness(self) -> None:
        actions = [
            AtomicAction(ActionKind.REPLACE, i=1, doc_id="d7"),
            AtomicAction(ActionKind.SWAP, i=0, j=4),
            AtomicAction(ActionKind.MOVE, i=0, j=3),
            AtomicAction(ActionKind.DROP_ADD, i=2, doc_id="d8"),
        ]
        for action in actions:
            with self.subTest(action=action):
                updated = apply_action(state(), action)
                self.assertEqual(len(updated.context), 5)
                self.assertEqual(len(set(updated.context)), 5)

    def test_second_action_uses_changed_state(self) -> None:
        first = AtomicAction(ActionKind.REPLACE, i=0, doc_id="d5")
        second = AtomicAction(ActionKind.REPLACE, i=1, doc_id="d0")
        trace = apply_trajectory(state(), [first, second])
        self.assertEqual(trace[1].context[0], "d5")
        self.assertEqual(trace[2].context[:2], ("d5", "d0"))

    def test_stop_terminates_trajectory(self) -> None:
        actions = [AtomicAction(ActionKind.STOP), AtomicAction(ActionKind.SWAP, i=0, j=1)]
        trace = apply_trajectory(state(), actions)
        self.assertEqual(len(trace), 2)
        self.assertTrue(trace[-1].stopped)
        with self.assertRaises(ValueError):
            apply_action(trace[-1], AtomicAction(ActionKind.SWAP, i=0, j=1))

    def test_legal_actions_never_add_duplicate(self) -> None:
        current = state()
        for action in legal_actions(current):
            if action.kind not in {ActionKind.STOP, ActionKind.KEEP}:
                updated = apply_action(current, action)
                self.assertEqual(len(set(updated.context)), 5)

    def test_invalid_existing_replacement_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_action(state(), AtomicAction(ActionKind.REPLACE, i=0, doc_id="d1"))

    def test_shortest_repair_witness_is_state_valid(self) -> None:
        target = ("d6", "d1", "d5", "d3", "d4")
        actions = shortest_repair_trajectory(state().context, target, max_depth=3)
        self.assertIsNotNone(actions)
        self.assertEqual(len(actions), 2)
        self.assertEqual(apply_trajectory(state(), actions)[-1].context, target)

    def test_unreachable_four_replacements(self) -> None:
        target = ("d5", "d6", "d7", "d8", "d4")
        self.assertIsNone(shortest_repair_trajectory(state().context, target, max_depth=3))


if __name__ == "__main__":
    unittest.main()
