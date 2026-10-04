import itertools
from typing import List, Tuple, Dict, Any, Optional

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


# ========================================================
# [남은 조각 대상 빔 서치 탐색 엔진]
# ========================================================
def solve_remaining_pieces(board: List[int], markers: Optional[List[List[int]]],
                           available_pieces: List[Tuple[int, int]], beam_width: int = 50):
    """
    남은 조각에 대해 최적 배치를 탐색.
    3개를 다 놓지 못하는 포화 상태일 경우, 2개 또는 1개만이라도 놓아 줄을 터뜨리는 최적 생존 플랜 반환.
    """
    if not available_pieces:
        return -float('inf'), None

    num_pieces = len(available_pieces)
    best_eval = -float('inf')
    best_plan = None
    best_earned_score = 0
    max_steps_found = 0

    order_permutations = list(itertools.permutations(range(num_pieces)))

    for perm_order in order_permutations:
        current_candidates = [(0.0, list(board), None, 0, 0, [])]

        for step_idx, order_idx in enumerate(perm_order):
            slot_idx, p_idx = available_pieces[order_idx]
            next_candidates = []

            for cur_eval, b_state, m_state, score_acc, lines_acc, plan_hist in current_candidates:
                orientations = PROCESSED_PIECES[p_idx]["orientations"]
                for ori in orientations:
                    for r, c, shift in get_valid_placements(b_state, ori):
                        new_b, _, l_clr, _ = apply_placement(b_state, None, ori, r, shift)

                        step_game_score = LINE_SCORES.get(l_clr, l_clr * 1500)
                        tot_game_score = score_acc + step_game_score
                        tot_lines = lines_acc + l_clr

                        step_eval = evaluate_board_state(new_b, tot_game_score, tot_lines)

                        new_step_info = {
                            "step": step_idx + 1,
                            "piece_slot": slot_idx,
                            "r": r,
                            "c": c,
                            "ori": ori,
                            "cleared_lines": l_clr
                        }
                        next_candidates.append(
                            (step_eval, new_b, None, tot_game_score, tot_lines, plan_hist + [new_step_info])
                        )

            # 만약 이번 단계에서 더 이상 놓을 수 있는 자리가 없다면:
            # 이전 단계까지 만들어둔 플랜이라도 유효 플랜 후보로 기록 (부분 플랜 폴백)
            if not next_candidates:
                if current_candidates:
                    top_partial = max(current_candidates, key=lambda x: x[0])
                    partial_steps = len(top_partial[5])
                    if partial_steps > max_steps_found or (partial_steps == max_steps_found and top_partial[0] > best_eval):
                        max_steps_found = partial_steps
                        best_eval = top_partial[0]
                        best_plan = top_partial[5]
                        best_earned_score = top_partial[3]
                break

            next_candidates.sort(key=lambda x: x[0], reverse=True)
            current_candidates = next_candidates[:beam_width]

            # 정상적으로 이번 단계를 놓았을 때의 최선책 기록
            top_eval, _, _, top_game_score, _, top_plan = current_candidates[0]
            curr_step_len = len(top_plan)
            if curr_step_len > max_steps_found or (curr_step_len == max_steps_found and top_eval > best_eval):
                max_steps_found = curr_step_len
                best_eval = top_eval
                best_plan = top_plan
                best_earned_score = top_game_score

    if best_plan:
        total_clears = sum(p["cleared_lines"] for p in best_plan)
        print(f"-> [AI 판단] {len(best_plan)}개 조각 계획 수립 완료 (예상 득점: +{best_earned_score}점, {total_clears}줄 제거)")
    else:
        print("-> [AI 판단] 현재 보드 상태에서 놓을 수 있는 조각이 없습니다 (게임 오버 위기).")

    return best_eval, best_plan