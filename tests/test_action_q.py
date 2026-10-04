import unittest
from unittest.mock import patch

import torch

import solver_core
from env_puzzle import Puzzle10x16Env


class ActionValueNetworkTests(unittest.TestCase):
    def test_legal_action_encoding_matches_environment_actions(self):
        env = Puzzle10x16Env()
        env.reset(seed=31)
        env.slots = [0, None, 1]

        expected = {solver_core.encode_action(action) for action in env.get_valid_actions()}
        actual = set(solver_core.get_legal_action_encodings(env.board, env.slots))

        self.assertEqual(actual, expected)

    def test_action_value_network_scores_and_backpropagates(self):
        env = Puzzle10x16Env()
        env.reset(seed=37)
        env.slots = [0, None, None]
        actions = [
            solver_core.encode_action(action)
            for action in env.get_valid_actions()[:2]
        ]
        model = solver_core.ActionValueNet().to(solver_core.DEVICE)
        model.train()

        with patch.object(solver_core, "ACTION_Q_NET", model):
            values = solver_core.evaluate_action_values(env.board, env.slots, actions)

        self.assertEqual(len(values), 2)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in values))

    def test_q_solver_plans_each_available_slot_once(self):
        model = solver_core.ActionValueNet().to(solver_core.DEVICE).eval()
        available_pieces = [(0, 0), (1, 0), (2, 0)]

        with patch.object(solver_core, "ACTION_Q_NET", model):
            _, plan = solver_core.solve_remaining_pieces(
                [0] * 16,
                markers=None,
                available_pieces=available_pieces,
            )

        self.assertEqual(len(plan), 3)
        self.assertEqual({item["piece_slot"] for item in plan}, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()