import unittest
from unittest.mock import patch

from env_puzzle import Puzzle10x16Env
from solver_core import FULL_ROW
from train_rl import board_potential, training_reward


class RewardTests(unittest.TestCase):
    def test_line_score_is_retained_when_game_ends(self):
        env = Puzzle10x16Env()
        env.reset(seed=41)
        env.board = [
            FULL_ROW ^ (1 << (9 - (row_idx % 2)))
            for row_idx in range(15)
        ] + [FULL_ROW ^ 1]
        env.slots = [0, 4, 4]

        action = next(
            action
            for action in env.get_valid_actions()
            if action["slot_idx"] == 0 and action["r"] == 15
        )
        _, reward, terminated, _, info = env.step(action)

        self.assertTrue(terminated)
        self.assertEqual(info["step_score"], 300)
        self.assertEqual(info["score"], 300)
        self.assertEqual(info["termination_penalty"], 20.0)
        self.assertEqual(reward, 3.0 - 20.0)

    def test_setup_shaping_is_disabled_for_official_score_baseline(self):
        current_board = [0] * 16
        next_board = [0] * 16
        next_board[15] = 0b1111111100

        with patch("train_rl.SETUP_SHAPING_SCALE", 0.0):
            reward = training_reward(3.0, current_board, next_board, False)

        self.assertEqual(reward, 3.0)

    def test_setup_shaping_uses_potential_delta_and_zero_terminal_potential(self):
        current_board = [0] * 16
        next_board = [0] * 16
        current_board[10] = 0b1111111100
        next_board[15] = 0b1111111110

        self.assertAlmostEqual(board_potential(current_board), (10 / 15) * 0.4)
        with patch("train_rl.SETUP_SHAPING_SCALE", 1.0):
            nonterminal_reward = training_reward(
                0.0, current_board, next_board, False
            )
            terminal_reward = training_reward(
                0.0, current_board, next_board, True
            )

        expected_delta = 0.98 * ((15 / 15) * 0.4) - ((10 / 15) * 0.4)
        self.assertAlmostEqual(nonterminal_reward, expected_delta)
        self.assertAlmostEqual(terminal_reward, -((10 / 15) * 0.4))


if __name__ == "__main__":
    unittest.main()