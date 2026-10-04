# train_rl.py
import random
import time
from collections import deque
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from env_puzzle import Puzzle10x16Env
from solver_core import (
    ACTION_Q_MODEL_PATH,
    ActionEncoding,
    ActionValueNet,
    encode_action,
    get_legal_action_encodings,
)

# ========================================================
# [하이퍼파라미터 및 학습 설정]
# ========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
GAMMA = 0.98
LR = 3e-4
MEMORY_CAPACITY = 200_000
MIN_REPLAY_SIZE = 5_000
TARGET_UPDATE_FREQ = 1_000
TOTAL_EPISODES = 50_000
MAX_Q_EVAL_BATCH = 2_048
TRAIN_EVERY_STEPS = 4
MAX_GRAD_NORM = 10.0
SETUP_SHAPING_SCALE = 0.0

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 120_000       # 충분한 탐색을 위해 스케줄 완화
SEED = 42

EMPTY_SLOT_TOKEN = 19         # 0~18은 조각 ID, 19는 빈 슬롯 토큰

# ========================================================
# [GPU 룩업 테이블 가속]
# ========================================================
LOOKUP_BITS = torch.tensor(
    [[(val >> (9 - c)) & 1 for c in range(10)] for val in range(1024)],
    dtype=torch.float32, device=DEVICE
)

def boards_to_gpu_tensor(boards_list: List[List[int]]) -> torch.Tensor:
    idx_tensor = torch.tensor(boards_list, dtype=torch.long, device=DEVICE)
    return LOOKUP_BITS[idx_tensor].unsqueeze(1)


def observation_to_replay_state(obs: Dict[str, np.ndarray]) -> Tuple[List[int], List[int]]:
    board_rows = [
        sum(int(cell) << (9 - col) for col, cell in enumerate(row))
        for row in obs["board"]
    ]
    slots = [
        int(piece) if piece >= 0 else EMPTY_SLOT_TOKEN
        for piece in obs["slots"]
    ]
    return board_rows, slots


def board_potential(board: List[int]) -> float:
    potential = 0.0
    for row_idx, row_value in enumerate(board):
        filled = row_value.bit_count()
        if filled in (8, 9):
            potential += (row_idx / 15.0) * 0.4
    return potential


def training_reward(
    base_reward: float,
    current_board: List[int],
    next_board: List[int],
    episode_done: bool,
) -> float:
    next_potential = 0.0 if episode_done else board_potential(next_board)
    potential_delta = GAMMA * next_potential - board_potential(current_board)
    return base_reward + SETUP_SHAPING_SCALE * potential_delta


def action_mask_rows(action: ActionEncoding) -> List[int]:
    _, _, orientation, r, _, shift = action
    mask = [0] * 16
    for row_idx, row_bits in enumerate(orientation[2]):
        mask[r + row_idx] = row_bits << shift
    return mask


def score_action_batch(
    model: ActionValueNet,
    boards: List[List[int]],
    slots: List[List[int]],
    actions: List[ActionEncoding],
) -> torch.Tensor:
    values = []
    for start in range(0, len(actions), MAX_Q_EVAL_BATCH):
        end = start + MAX_Q_EVAL_BATCH
        action_chunk = actions[start:end]
        board_tensor = boards_to_gpu_tensor(boards[start:end])
        slots_tensor = torch.tensor(slots[start:end], dtype=torch.long, device=DEVICE)
        placement_tensor = boards_to_gpu_tensor(
            [action_mask_rows(action) for action in action_chunk]
        )
        action_slots = torch.tensor(
            [action[0] for action in action_chunk], dtype=torch.long, device=DEVICE
        )
        values.append(
            model(board_tensor, slots_tensor, placement_tensor, action_slots).squeeze(-1)
        )
    return torch.cat(values)


def score_action_batch_inference(
    model: ActionValueNet,
    boards: List[List[int]],
    slots: List[List[int]],
    actions: List[ActionEncoding],
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            return score_action_batch(model, boards, slots, actions)
    finally:
        model.train(was_training)


def should_train(total_steps: int) -> bool:
    return total_steps > 0 and total_steps % TRAIN_EVERY_STEPS == 0


def should_sync_target(optimizer_steps: int) -> bool:
    return optimizer_steps > 0 and optimizer_steps % TARGET_UPDATE_FREQ == 0


def clip_gradient_norm(model: torch.nn.Module) -> float:
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    return float(norm.item())


# ========================================================
# [학습 루프]
# ========================================================
def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"★ 조각 인지 강화학습 트레이너 구동 - 연산 디바이스: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"-> GPU: {torch.cuda.get_device_name(0)}")

    env = Puzzle10x16Env()
    policy_net = ActionValueNet().to(DEVICE)
    target_net = ActionValueNet().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, weight_decay=1e-4)
    replay_buffer = deque(maxlen=MEMORY_CAPACITY)

    total_steps = 0
    optimizer_steps = 0
    recent_scores = deque(maxlen=50)
    recent_lines = deque(maxlen=50)
    start_time = time.time()

    def optimize_batch():
        if len(replay_buffer) < MIN_REPLAY_SIZE:
            return None

        batch = random.sample(replay_buffer, BATCH_SIZE)
        cur_b, cur_s, cur_actions, next_b, next_s, r_list, d_list = zip(*batch)

        reward_batch = torch.tensor(r_list, dtype=torch.float32, device=DEVICE)
        done_batch = torch.tensor(d_list, dtype=torch.float32, device=DEVICE)

        current_q = score_action_batch(
            policy_net, list(cur_b), list(cur_s), list(cur_actions)
        )

        best_next_actions = [None] * len(batch)
        best_next_values = [-float("inf")] * len(batch)
        candidate_state_indices = []
        candidate_boards = []
        candidate_slots = []
        candidate_actions = []

        def evaluate_next_action_chunk():
            if not candidate_actions:
                return
            candidate_q = score_action_batch_inference(
                policy_net,
                candidate_boards,
                candidate_slots,
                candidate_actions,
            ).tolist()
            for idx, value in enumerate(candidate_q):
                state_idx = candidate_state_indices[idx]
                if value > best_next_values[state_idx]:
                    best_next_values[state_idx] = value
                    best_next_actions[state_idx] = candidate_actions[idx]
            candidate_state_indices.clear()
            candidate_boards.clear()
            candidate_slots.clear()
            candidate_actions.clear()

        for state_idx, (state_board, state_slots, is_done) in enumerate(
            zip(next_b, next_s, d_list)
        ):
            if is_done:
                continue
            for action in get_legal_action_encodings(state_board, state_slots):
                candidate_state_indices.append(state_idx)
                candidate_boards.append(state_board)
                candidate_slots.append(state_slots)
                candidate_actions.append(action)
                if len(candidate_actions) >= MAX_Q_EVAL_BATCH:
                    evaluate_next_action_chunk()
        evaluate_next_action_chunk()

        with torch.no_grad():
            selected_indices = [
                idx for idx, action in enumerate(best_next_actions) if action is not None
            ]
            next_q_values = torch.zeros(len(batch), dtype=torch.float32, device=DEVICE)
            if selected_indices:
                selected_actions = [best_next_actions[idx] for idx in selected_indices]
                selected_boards = [next_b[idx] for idx in selected_indices]
                selected_slots = [next_s[idx] for idx in selected_indices]
                selected_q = score_action_batch(
                    target_net,
                    selected_boards,
                    selected_slots,
                    selected_actions,
                )
                next_q_values[selected_indices] = selected_q
            target_q = reward_batch + (1.0 - done_batch) * GAMMA * next_q_values

        loss = nn.SmoothL1Loss()(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = clip_gradient_norm(policy_net)
        optimizer.step()
        return float(loss.item()), gradient_norm

    for episode in range(1, TOTAL_EPISODES + 1):
        obs, _ = env.reset(seed=SEED if episode == 1 else None)
        done = False
        ep_steps = 0

        while not done:
            total_steps += 1
            ep_steps += 1
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1.0 * total_steps / EPSILON_DECAY)

            cur_slots_encoded = [s if s is not None else EMPTY_SLOT_TOKEN for s in env.slots]
            encoded_actions = [encode_action(action) for action in valid_actions]

            if random.random() < epsilon:
                chosen_idx = random.randint(0, len(valid_actions) - 1)
            else:
                values = score_action_batch_inference(
                    policy_net,
                    [list(env.board)] * len(encoded_actions),
                    [cur_slots_encoded] * len(encoded_actions),
                    encoded_actions,
                )
                chosen_idx = int(torch.argmax(values).item())

            chosen_action = valid_actions[chosen_idx]
            chosen_action_encoding = encoded_actions[chosen_idx]

            # ----------------------------------------------------
            # 환경 전진 및 긍정적 보상 책정
            # ----------------------------------------------------
            prev_b = list(env.board)
            prev_s = list(cur_slots_encoded)

            next_obs, base_reward, terminated, truncated, info = env.step(chosen_action)
            done = terminated or truncated
            next_b_state, next_s_state = observation_to_replay_state(next_obs)

            shaped_reward = training_reward(
                base_reward,
                prev_b,
                next_b_state,
                done,
            )

            # 리플레이 버퍼 저장 (State -> Next State 전이)
            replay_buffer.append((
                prev_b,
                prev_s,
                chosen_action_encoding,
                next_b_state,
                next_s_state,
                shaped_reward,
                done
            ))

            if should_train(total_steps):
                update_result = optimize_batch()
                if update_result is not None:
                    optimizer_steps += 1
                    if should_sync_target(optimizer_steps):
                        target_net.load_state_dict(policy_net.state_dict())

        recent_scores.append(env.total_score)
        recent_lines.append(env.total_lines_cleared)

        if episode % 50 == 0:
            avg_score = np.mean(recent_scores)
            avg_lines = np.mean(recent_lines)
            max_score = max(recent_scores)
            fps = total_steps / (time.time() - start_time)

            print(f"[Ep {episode:05d}] 스텝: {total_steps} | 업데이트: {optimizer_steps} | ε: {epsilon:.3f} | "
                  f"평균점수: {avg_score:6.1f} | 최고점수: {max_score:5d} | "
                  f"평균제거줄: {avg_lines:4.1f}줄 | 속도: {fps:.0f} step/s")

            torch.save(policy_net.state_dict(), ACTION_Q_MODEL_PATH)

    print(f"\n★ 학습 완료: {ACTION_Q_MODEL_PATH}")

if __name__ == "__main__":
    train()