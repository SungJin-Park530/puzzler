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
from solver_core import apply_placement

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

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 120_000       # 충분한 탐색을 위해 스케줄 완화

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


# ========================================================
# [개편된 Board + Hand Value Network]
# ========================================================
class BoardValueNet(nn.Module):
    def __init__(self, num_pieces=20, emb_dim=16):
        super().__init__()
        # 보드 지형 특징 추출
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.board_fc = nn.Sequential(
            nn.Linear(64 * 16 * 10, 128),
            nn.ReLU()
        )
        # 슬롯 잔여 조각 임베딩 (0~18 조각 + 19 빈 슬롯)
        self.piece_embed = nn.Embedding(num_pieces, emb_dim)
        
        # 보드(128) + 3개 슬롯(16 * 3 = 48) = 176 차원
        self.final_fc = nn.Sequential(
            nn.Linear(128 + (3 * emb_dim), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, board_tensor: torch.Tensor, slots_tensor: torch.Tensor) -> torch.Tensor:
        b_feat = self.conv(board_tensor)
        b_feat = b_feat.view(b_feat.size(0), -1)
        b_out = self.board_fc(b_feat)

        p_feat = self.piece_embed(slots_tensor)
        p_feat = p_feat.view(p_feat.size(0), -1)

        combined = torch.cat([b_out, p_feat], dim=1)
        return self.final_fc(combined)


# ========================================================
# [학습 루프]
# ========================================================
def train():
    print(f"★ 조각 인지 강화학습 트레이너 구동 - 연산 디바이스: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"-> GPU: {torch.cuda.get_device_name(0)}")

    env = Puzzle10x16Env()
    policy_net = BoardValueNet().to(DEVICE)
    target_net = BoardValueNet().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, weight_decay=1e-4)
    replay_buffer = deque(maxlen=MEMORY_CAPACITY)

    total_steps = 0
    recent_scores = deque(maxlen=50)
    recent_lines = deque(maxlen=50)
    start_time = time.time()

    def optimize_batch():
        if len(replay_buffer) < MIN_REPLAY_SIZE:
            return

        batch = random.sample(replay_buffer, BATCH_SIZE)
        cur_b, cur_s, next_b, next_s, r_list, d_list = zip(*batch)

        cur_b_tensor = boards_to_gpu_tensor(cur_b)
        cur_s_tensor = torch.tensor(cur_s, dtype=torch.long, device=DEVICE)

        next_b_tensor = boards_to_gpu_tensor(next_b)
        next_s_tensor = torch.tensor(next_s, dtype=torch.long, device=DEVICE)

        reward_batch = torch.tensor(r_list, dtype=torch.float32, device=DEVICE)
        done_batch = torch.tensor(d_list, dtype=torch.float32, device=DEVICE)

        current_q = policy_net(cur_b_tensor, cur_s_tensor).squeeze(-1)
        with torch.no_grad():
            next_q = target_net(next_b_tensor, next_s_tensor).squeeze(-1)
            target_q = reward_batch + (1.0 - done_batch) * GAMMA * next_q

        loss = nn.MSELoss()(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for episode in range(1, TOTAL_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        ep_steps = 0

        while not done:
            total_steps += 1
            ep_steps += 1
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1.0 * total_steps / EPSILON_DECAY)

            # 착수 후 예상되는 보드 및 잔여 슬롯 계산
            candidate_next_boards = []
            candidate_next_slots = []
            cur_slots_encoded = [s if s is not None else EMPTY_SLOT_TOKEN for s in env.slots]

            for act in valid_actions:
                nb, _, _, _ = apply_placement(env.board, None, act["ori"], act["r"], act["shift"])
                candidate_next_boards.append(nb)

                # 행동 후 남은 슬롯 형태 (해당 슬롯은 토큰 19로 변경)
                n_slots = list(cur_slots_encoded)
                n_slots[act["slot_idx"]] = EMPTY_SLOT_TOKEN
                candidate_next_slots.append(n_slots)

            if random.random() < epsilon:
                chosen_idx = random.randint(0, len(valid_actions) - 1)
            else:
                b_tensors = boards_to_gpu_tensor(candidate_next_boards)
                s_tensors = torch.tensor(candidate_next_slots, dtype=torch.long, device=DEVICE)
                with torch.no_grad():
                    values = policy_net(b_tensors, s_tensors).squeeze(-1)
                chosen_idx = torch.argmax(values).item()

            chosen_action = valid_actions[chosen_idx]
            next_b_state = candidate_next_boards[chosen_idx]
            next_s_state = candidate_next_slots[chosen_idx]

            # ----------------------------------------------------
            # 환경 전진 및 긍정적 보상 책정
            # ----------------------------------------------------
            prev_b = list(env.board)
            prev_s = list(cur_slots_encoded)

            _, base_reward, terminated, truncated, info = env.step(chosen_action)
            done = terminated or truncated

            shaped_reward = base_reward

            # 줄 삭제 보너스 강화
            if base_reward >= 3.0:
                shaped_reward *= 1.5
            else:
                # 하단 가중치 빌드업 보너스 (15행일수록 높은 가산점)
                filled_bonus = 0.0
                for r_idx, r_val in enumerate(next_b_state):
                    cnt = bin(r_val).count('1')
                    if cnt in (8, 9):
                        filled_bonus += (r_idx / 15.0) * 0.4
                shaped_reward += filled_bonus

            if terminated:
                shaped_reward = -50.0

            # 리플레이 버퍼 저장 (State -> Next State 전이)
            replay_buffer.append((
                prev_b,
                prev_s,
                next_b_state,
                next_s_state,
                shaped_reward,
                done
            ))

            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

        recent_scores.append(env.total_score)
        recent_lines.append(env.total_lines_cleared)

        # 에피소드 종료 후 몰아서 배치 학습
        for _ in range(ep_steps // 4):
            optimize_batch()

        if episode % 50 == 0:
            avg_score = np.mean(recent_scores)
            avg_lines = np.mean(recent_lines)
            max_score = max(recent_scores)
            fps = total_steps / (time.time() - start_time)

            print(f"[Ep {episode:05d}] 스텝: {total_steps} | ε: {epsilon:.3f} | "
                  f"평균점수: {avg_score:6.1f} | 최고점수: {max_score:5d} | "
                  f"평균제거줄: {avg_lines:4.1f}줄 | 속도: {fps:.0f} step/s")

            torch.save(policy_net.state_dict(), "puzzle_value_net.pth")

    print("\n★ 학습 완료: puzzle_value_net.pth")

if __name__ == "__main__":
    train()