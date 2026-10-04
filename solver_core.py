import os
import itertools
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np

# ========================================================
# [공식 점수 테이블 및 상수 정의]
# ========================================================
FULL_ROW = 0b1111111111  # 1023 (가로 10칸 꽉 참)

LINE_SCORES = {
    0: 0,
    1: 300,
    2: 1200,
    3: 2700,
    4: 4800,
    5: 7500
}

# 10비트 팝카운트 룩업 테이블 (0 ~ 1023)
POPCOUNT_10 = [bin(i).count('1') for i in range(1024)]

# 19종 기본 조각 정의
RAW_PIECE_POOL = [
    [[1]],                                                                     # 0
    [[1, 1, 1]],                                                               # 1
    [[1, 1], [0, 1], [0, 1]],                                                  # 2
    [[1, 0], [1, 1]],                                                          # 3
    [[1, 1], [1, 0], [1, 1]],                                                  # 4
    [[1, 1], [0, 1], [1, 1], [1, 0], [1, 1]],                                  # 5
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]],                                         # 6
    [[1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],                             # 7
    [[0, 1, 0], [1, 0, 1]],                                                    # 8
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]],                                         # 9
    [[1, 1, 1], [0, 1, 0], [1, 0, 1]],                                         # 10
    [[0, 1, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]],                             # 11
    [[1, 1], [0, 1], [1, 1], [0, 1]],                                          # 12
    [[1, 1], [1, 0], [1, 1], [1, 0], [1, 1]],                                  # 13
    [[1, 1, 1, 1], [0, 1, 1, 0], [1, 1, 1, 1]],                               # 14
    [[0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0]],    # 15
    [[1, 0], [1, 1], [1, 0]],                                                  # 16
    [[1, 0], [1, 1], [1, 0], [1, 1], [1, 0]],                                  # 17
    [[1, 1, 1, 1, 1]]                                                          # 18
]

# ========================================================
# [PyTorch 디바이스 및 모델 정의]
# ========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "puzzle_value_net.pth"
ACTION_Q_MODEL_PATH = "puzzle_action_q_net.pth"
EMPTY_SLOT_TOKEN = 19
ActionEncoding = Tuple[int, int, Tuple[int, int, Tuple[int, ...]], int, int, int]


class ActionValueNet(nn.Module):
    def __init__(self, num_pieces=20, emb_dim=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
        )
        self.board_fc = nn.Sequential(nn.Linear(64 * 16 * 10, 128), nn.ReLU())
        self.piece_embed = nn.Embedding(num_pieces, emb_dim)
        self.action_slot_embed = nn.Embedding(3, emb_dim)
        self.final_fc = nn.Sequential(
            nn.Linear(128 + (4 * emb_dim), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        board_tensor: torch.Tensor,
        slots_tensor: torch.Tensor,
        placement_tensor: torch.Tensor,
        action_slots_tensor: torch.Tensor,
    ) -> torch.Tensor:
        state_action = torch.cat([board_tensor, placement_tensor], dim=1)
        board_features = self.board_fc(self.conv(state_action).flatten(start_dim=1))
        piece_features = self.piece_embed(slots_tensor).flatten(start_dim=1)
        action_features = self.action_slot_embed(action_slots_tensor)
        combined = torch.cat([board_features, piece_features, action_features], dim=1)
        return self.final_fc(combined)

class BoardValueNet(nn.Module):
    def __init__(self, num_pieces=20, emb_dim=16):
        super().__init__()
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
        self.piece_embed = nn.Embedding(num_pieces, emb_dim)
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

# 가중치 파일 로드
VALUE_NET: Optional[BoardValueNet] = None
if os.path.exists(MODEL_PATH):
    try:
        model = BoardValueNet().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.eval()
        VALUE_NET = model
        print(f"★ [RL Solver] 조각 인지 가치망({MODEL_PATH}) 로드 완료 - 연산 디바이스: {DEVICE}")
    except Exception as e:
        print(f"[경고] 모델 로드 실패: {e} -> 휴리스틱 폴백 모드로 실행됩니다.")

ACTION_Q_NET: Optional[ActionValueNet] = None
if os.path.exists(ACTION_Q_MODEL_PATH):
    try:
        action_q_model = ActionValueNet().to(DEVICE)
        action_q_model.load_state_dict(
            torch.load(ACTION_Q_MODEL_PATH, map_location=DEVICE, weights_only=True)
        )
        action_q_model.eval()
        ACTION_Q_NET = action_q_model
        print(f"★ [RL Solver] 행동 Q망({ACTION_Q_MODEL_PATH}) 로드 완료 - 연산 디바이스: {DEVICE}")
    except Exception as e:
        print(f"[경고] 행동 Q망 로드 실패: {e} -> 기존 솔버로 실행됩니다.")


def evaluate_boards_batch(boards: List[List[int]], remaining_slots: List[List[int]]) -> List[float]:
    """보드와 남은 손패 조각들을 결합하여 일괄 가치 추론"""
    if not boards:
        return []

    if VALUE_NET is None:
        return [float(-sum(POPCOUNT_10[r] for r in b)) for b in boards]

    arr = np.zeros((len(boards), 1, 16, 10), dtype=np.float32)
    for b_idx, b in enumerate(boards):
        for r in range(16):
            row_val = b[r]
            for c in range(10):
                if (row_val >> (9 - c)) & 1:
                    arr[b_idx, 0, r, c] = 1.0

    slots_arr = np.array(remaining_slots, dtype=np.int64)

    with torch.inference_mode():
        b_tensor = torch.from_numpy(arr).to(DEVICE, non_blocking=True)
        s_tensor = torch.from_numpy(slots_arr).to(DEVICE, non_blocking=True)
        values = VALUE_NET(b_tensor, s_tensor).squeeze(-1).tolist()
        del b_tensor, s_tensor

    if isinstance(values, float):
        return [values]
    return values

def rotate_90(grid: List[List[int]]) -> List[List[int]]:
    return [list(row) for row in zip(*grid[::-1])]

def flip_h(grid: List[List[int]]) -> List[List[int]]:
    return [row[::-1] for row in grid]

def grid_to_bits(grid: List[List[int]]) -> Tuple[int, int, Tuple[int, ...]]:
    h = len(grid)
    w = len(grid[0])
    rows_bits = []
    for r in grid:
        val = 0
        for bit in r:
            val = (val << 1) | bit
        rows_bits.append(val)
    return h, w, tuple(rows_bits)

# 19종 조각의 모든 고유 회전/반전 상태 사전 연산
PROCESSED_PIECES = []
for p_idx, raw in enumerate(RAW_PIECE_POOL):
    unique_orientations = set()
    curr = raw
    for _ in range(4):
        unique_orientations.add(grid_to_bits(curr))
        unique_orientations.add(grid_to_bits(flip_h(curr)))
        curr = rotate_90(curr)
    PROCESSED_PIECES.append({
        "id": p_idx,
        "orientations": list(unique_orientations)
    })


# ========================================================
# [비트마스크 연산 & 보드 시뮬레이션]
# ========================================================
def get_valid_placements(board: List[int], orientation: Tuple[int, int, Tuple[int, ...]]):
    """보드에 조각이 충돌 없이 들어갈 수 있는 모든 (r, c, shift) 탐색"""
    h, w, row_bits = orientation
    valid = []
    max_r = 16 - h + 1
    max_c = 10 - w + 1

    for r in range(max_r):
        for c in range(max_c):
            shift = 10 - w - c
            collision = False
            for i, p_row in enumerate(row_bits):
                if board[r + i] & (p_row << shift):
                    collision = True
                    break
            if not collision:
                valid.append((r, c, shift))
    return valid


def encode_action(action: Dict[str, Any]) -> ActionEncoding:
    return (
        action["slot_idx"], action["piece_idx"], action["ori"],
        action["r"], action["c"], action["shift"],
    )


def get_legal_action_encodings(
    board: List[int], slots: List[Optional[int]]
) -> List[ActionEncoding]:
    actions = []
    for slot_idx, piece_idx in enumerate(slots):
        if piece_idx is None or piece_idx == EMPTY_SLOT_TOKEN:
            continue
        for orientation in PROCESSED_PIECES[piece_idx]["orientations"]:
            for r, c, shift in get_valid_placements(board, orientation):
                actions.append((slot_idx, piece_idx, orientation, r, c, shift))
    return actions


ROW_BITS_LOOKUP = torch.tensor(
    [[(value >> (9 - col)) & 1 for col in range(10)] for value in range(1024)],
    dtype=torch.float32,
    device=DEVICE,
)


def evaluate_action_values(
    board: List[int],
    slots: List[Optional[int]],
    actions: List[ActionEncoding],
) -> List[float]:
    if not actions or ACTION_Q_NET is None:
        return []

    board_rows = torch.tensor([board] * len(actions), dtype=torch.long, device=DEVICE)
    slot_rows = torch.tensor(
        [[EMPTY_SLOT_TOKEN if piece is None else piece for piece in slots]] * len(actions),
        dtype=torch.long,
        device=DEVICE,
    )
    placement_rows = [[0] * 16 for _ in actions]
    for action_idx, (_, _, orientation, r, _, shift) in enumerate(actions):
        for row_idx, row_bits in enumerate(orientation[2]):
            placement_rows[action_idx][r + row_idx] = row_bits << shift
    placement_indices = torch.tensor(placement_rows, dtype=torch.long, device=DEVICE)
    action_slots = torch.tensor([action[0] for action in actions], dtype=torch.long, device=DEVICE)

    with torch.inference_mode():
        values = ACTION_Q_NET(
            ROW_BITS_LOOKUP[board_rows].unsqueeze(1),
            slot_rows,
            ROW_BITS_LOOKUP[placement_indices].unsqueeze(1),
            action_slots,
        ).squeeze(-1)
    return values.cpu().tolist()


def apply_placement(board: List[int], markers: Optional[List[List[int]]],
                    orientation: Tuple[int, int, Tuple[int, ...]], r: int, shift: int):
    """조각을 보드에 비트 연산으로 배치하고 채워진 가로줄을 삭제"""
    h, w, row_bits = orientation
    new_board = list(board)
    new_markers = [list(row) for row in markers] if markers else None

    # 블록 배치
    for i, p_row in enumerate(row_bits):
        new_board[r + i] |= (p_row << shift)

    # 가로줄 삭제 검사
    cleared_lines = 0
    for row_idx in range(16):
        if new_board[row_idx] == FULL_ROW:
            new_board[row_idx] = 0
            cleared_lines += 1
            if new_markers:
                new_markers[row_idx] = [0] * 10

    return new_board, new_markers, cleared_lines, 0


# ========================================================
# [고득점 지향 휴리스틱 평가 함수]
# ========================================================
def evaluate_board_state(board: List[int], accumulated_game_score: int, total_cleared_lines: int) -> float:
    """
    고득점 + 생존력 균형 평가 함수
    - 실제 터진 줄 수 및 획득 점수에 최우선 가중치
    - 보드가 가득 찼을 때 줄 삭제를 통한 빈 공간 창출을 절대적으로 선호
    """
    score = float(accumulated_game_score) * 2.0  # 게임 공식 점수 2배 가중

    # 줄 삭제 누적 우대 (줄을 많이 지울수록 절대적 고득점)
    score += total_cleared_lines * 500.0

    # 빈 행(0인 줄) 개수: 생존 공간 확보
    empty_rows = sum(1 for row in board if row == 0)
    score += empty_rows * 150.0

    # 보드 내 전체 블록 수 감점 (블록이 적고 깨끗할수록 안전)
    total_blocks = sum(POPCOUNT_10[r] for r in board)
    score -= total_blocks * 5.0

    # 다중 줄 폭파 준비도 (8, 9칸 행) - 최대 4줄까지만 인정하여 점수 폭주 방지
    near_full_cnt = 0
    for row in board:
        cnt = POPCOUNT_10[row]
        if cnt == 9:
            near_full_cnt += 1
            score += 80.0
        elif cnt == 8:
            near_full_cnt += 1
            score += 40.0

    if near_full_cnt >= 2:
        score += min(near_full_cnt, 4) * 60.0

    return score


def solve_remaining_pieces_with_q(
    board: List[int], available_pieces: List[Tuple[int, int]]
):
    current_board = list(board)
    current_slots: List[Optional[int]] = [EMPTY_SLOT_TOKEN] * 3
    for slot_idx, piece_idx in available_pieces:
        current_slots[slot_idx] = piece_idx

    plan = []
    earned_score = 0
    first_action_value = -float("inf")
    for step_idx in range(len(available_pieces)):
        actions = get_legal_action_encodings(current_board, current_slots)
        values = evaluate_action_values(current_board, current_slots, actions)
        if not values:
            break

        chosen_idx = int(np.argmax(values))
        slot_idx, piece_idx, orientation, r, c, shift = actions[chosen_idx]
        if step_idx == 0:
            first_action_value = float(values[chosen_idx])
        current_board, _, cleared_lines, _ = apply_placement(
            current_board, None, orientation, r, shift
        )
        earned_score += LINE_SCORES.get(cleared_lines, cleared_lines * 1500)
        current_slots[slot_idx] = EMPTY_SLOT_TOKEN
        plan.append({
            "step": step_idx + 1,
            "piece_slot": slot_idx,
            "piece_id": piece_idx,
            "r": r,
            "c": c,
            "ori": orientation,
            "cleared_lines": cleared_lines,
        })

    if plan:
        total_clears = sum(item["cleared_lines"] for item in plan)
        print(
            f"-> [RL Q 판단] {len(plan)}개 조각 연계 계획 완료 "
            f"(예상 득점: +{earned_score}점, {total_clears}줄 제거, "
            f"첫 행동 Q: {first_action_value:.2f})"
        )
    else:
        print("-> [RL Q 판단] 유효한 배치를 찾지 못했습니다.")
    return first_action_value, plan


def solve_remaining_pieces(board: List[int], markers: Optional[List[List[int]]],
                           available_pieces: List[Tuple[int, int]], beam_width: int = 40):
    if not available_pieces:
        return -float('inf'), None

    if ACTION_Q_NET is not None:
        return solve_remaining_pieces_with_q(board, available_pieces)

    num_pieces = len(available_pieces)
    best_eval = -float('inf')
    best_plan = None
    best_earned_score = 0
    max_steps_found = 0

    order_permutations = list(itertools.permutations(range(num_pieces)))
    MAX_EVAL_CANDIDATES = 120

    # 초기 슬롯 상태 (3개 슬롯 기준 인코딩)
    base_slots = [EMPTY_SLOT_TOKEN] * 3
    for s_idx, p_idx in available_pieces:
        base_slots[s_idx] = p_idx

    for perm_order in order_permutations:
        current_candidates = [(0.0, list(board), list(base_slots), 0, 0, [])]

        for step_idx, order_idx in enumerate(perm_order):
            slot_idx, p_idx = available_pieces[order_idx]
            raw_branches = []

            for cur_eval, b_state, s_state, score_acc, lines_acc, plan_hist in current_candidates:
                orientations = PROCESSED_PIECES[p_idx]["orientations"]
                for ori in orientations:
                    for r, c, shift in get_valid_placements(b_state, ori):
                        new_b, _, l_clr, _ = apply_placement(b_state, None, ori, r, shift)
                        step_game_score = LINE_SCORES.get(l_clr, l_clr * 1500)
                        tot_game_score = score_acc + step_game_score
                        tot_lines = lines_acc + l_clr

                        # 착수 후 슬롯 갱신
                        new_slots = list(s_state)
                        new_slots[slot_idx] = EMPTY_SLOT_TOKEN

                        quick_score = tot_game_score - sum(POPCOUNT_10[row] for row in new_b) * 2

                        step_info = {
                            "step": step_idx + 1,
                            "piece_slot": slot_idx,
                            "piece_id": p_idx,
                            "r": r,
                            "c": c,
                            "ori": ori,
                            "cleared_lines": l_clr
                        }
                        raw_branches.append((quick_score, new_b, new_slots, tot_game_score, tot_lines, plan_hist + [step_info]))

            if not raw_branches:
                if current_candidates:
                    top_partial = max(current_candidates, key=lambda x: x[0])
                    partial_steps = len(top_partial[5])
                    if partial_steps > max_steps_found or (partial_steps == max_steps_found and top_partial[0] > best_eval):
                        max_steps_found = partial_steps
                        best_eval = top_partial[0]
                        best_plan = top_partial[5]
                        best_earned_score = top_partial[3]
                break

            # 1차 필터링
            raw_branches.sort(key=lambda x: x[0], reverse=True)
            trimmed_branches = raw_branches[:MAX_EVAL_CANDIDATES]

            # 2차 정밀 평가: 보드 + 잔여 슬롯 정보를 함께 신경망에 전달
            boards_to_eval = [item[1] for item in trimmed_branches]
            slots_to_eval = [item[2] for item in trimmed_branches]
            neural_values = evaluate_boards_batch(boards_to_eval, slots_to_eval)

            scored_candidates = []
            for (_, new_b, new_slots, tot_score, tot_lines, plan_hist), v_val in zip(trimmed_branches, neural_values):
                total_val = (tot_score / 100.0) + float(v_val)
                scored_candidates.append((total_val, new_b, new_slots, tot_score, tot_lines, plan_hist))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            current_candidates = scored_candidates[:beam_width]

            top_eval, _, _, top_game_score, _, top_plan = current_candidates[0]
            curr_step_len = len(top_plan)
            if curr_step_len > max_steps_found or (curr_step_len == max_steps_found and top_eval > best_eval):
                max_steps_found = curr_step_len
                best_eval = top_eval
                best_plan = top_plan
                best_earned_score = top_game_score

    if best_plan:
        total_clears = sum(p["cleared_lines"] for p in best_plan)
        print(f"-> [RL AI 판단] {len(best_plan)}개 조각 연계 계획 완료 (예상 득점: +{best_earned_score}점, {total_clears}줄 제거, 평가치: {best_eval:.2f})")
    else:
        print("-> [RL AI 판단] 유효한 배치를 찾지 못했습니다.")

    return best_eval, best_plan