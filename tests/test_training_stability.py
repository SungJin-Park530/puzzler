import unittest
from unittest.mock import patch

import torch

import solver_core
import train_rl
from env_puzzle import Puzzle10x16Env


class TrainingStabilityTests(unittest.TestCase):
    def test_training_and_target_sync_use_separate_step_cadences(self):
        with (
            patch("train_rl.TRAIN_EVERY_STEPS", 4),
            patch("train_rl.TARGET_UPDATE_FREQ", 3),
        ):
            self.assertFalse(train_rl.should_train(0))
            self.assertFalse(train_rl.should_train(3))
            self.assertTrue(train_rl.should_train(4))
            self.assertTrue(train_rl.should_train(8))
            self.assertFalse(train_rl.should_sync_target(0))
            self.assertFalse(train_rl.should_sync_target(2))
            self.assertTrue(train_rl.should_sync_target(3))
            self.assertTrue(train_rl.should_sync_target(6))

    def test_gradient_norm_is_clipped_to_configured_limit(self):
        model = torch.nn.Linear(4, 2)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 100.0)

        unclipped_norm = train_rl.clip_gradient_norm(model)
        clipped_norm = torch.sqrt(
            sum(parameter.grad.square().sum() for parameter in model.parameters())
        ).item()

        self.assertGreater(unclipped_norm, train_rl.MAX_GRAD_NORM)
        self.assertLessEqual(clipped_norm, train_rl.MAX_GRAD_NORM + 1e-5)

    def test_inference_scores_match_in_both_modes_and_restore_mode(self):
        env = Puzzle10x16Env()
        env.reset(seed=53)
        env.slots = [0, None, None]
        action = solver_core.encode_action(env.get_valid_actions()[0])
        model = solver_core.ActionValueNet().to(solver_core.DEVICE)
        boards = [list(env.board)]
        slots = [[0, train_rl.EMPTY_SLOT_TOKEN, train_rl.EMPTY_SLOT_TOKEN]]

        model.train()
        training_mode_score = train_rl.score_action_batch(
            model, boards, slots, [action]
        ).detach()
        self.assertTrue(model.training)

        inference_score = train_rl.score_action_batch_inference(
            model, boards, slots, [action]
        )
        self.assertTrue(model.training)
        torch.testing.assert_close(training_mode_score, inference_score)

        model.eval()
        train_rl.score_action_batch_inference(model, boards, slots, [action])
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()