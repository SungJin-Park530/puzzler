import unittest

from env_puzzle import Puzzle10x16Env
from train_rl import EMPTY_SLOT_TOKEN, observation_to_replay_state


class PuzzleEnvironmentTransitionTests(unittest.TestCase):
    def test_reset_seed_reproduces_initial_hand(self):
        env = Puzzle10x16Env()

        env.reset(seed=123)
        first_hand = list(env.slots)
        env.reset(seed=123)
        second_hand = list(env.slots)

        self.assertEqual(first_hand, second_hand)

    def test_slots_refill_only_after_all_three_pieces_are_used(self):
        env = Puzzle10x16Env()
        env.reset(seed=123)
        env.slots = [0, 0, 0]

        for slot_idx in range(3):
            action = next(
                action
                for action in env.get_valid_actions()
                if action["slot_idx"] == slot_idx
            )
            obs, _, terminated, _, _ = env.step(action)
            self.assertFalse(terminated)

            if slot_idx < 2:
                self.assertIsNone(env.slots[slot_idx])
                self.assertEqual(obs["slots"][slot_idx], -1)
            else:
                self.assertTrue(all(piece is not None for piece in env.slots))
                self.assertTrue(all(piece >= 0 for piece in obs["slots"]))

    def test_replay_state_uses_returned_refill_observation(self):
        env = Puzzle10x16Env()
        env.reset(seed=123)
        env.slots = [0, 0, 0]

        for slot_idx in range(3):
            action = next(
                action
                for action in env.get_valid_actions()
                if action["slot_idx"] == slot_idx
            )
            obs, _, _, _, _ = env.step(action)

        board_state, slot_state = observation_to_replay_state(obs)

        self.assertEqual(board_state, env.board)
        self.assertEqual(slot_state, env.slots)
        self.assertNotIn(EMPTY_SLOT_TOKEN, slot_state)


if __name__ == "__main__":
    unittest.main()