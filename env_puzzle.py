# env_puzzle.py
from typing import List, Tuple, Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from solver_core import (
    RAW_PIECE_POOL,
    PROCESSED_PIECES,
    FULL_ROW,
    POPCOUNT_10,
    LINE_SCORES,
    get_valid_placements,
    apply_placement,
)

class Puzzle10x16Env(gym.Env):
    """
    10x16 퍼즐 게임 가상 환경
    - State: (16, 10) 바이너리 보드 + 보유 조각 슬롯 3개 정보
    - Step 단위: 3개 조각 중 1개를 골라 유효한 (r, c, ori)에 배치
    """
    metadata = {"render_modes": ["ansi"]}
    REWARD_SCORE_SCALE = 100.0
    TERMINATION_PENALTY = 20.0

    def __init__(self):
        super().__init__()
        # 보드 크기: 16행 10열 (0: 빈칸, 1: 블록)
        self.board: List[int] = [0] * 16
        self.slots: List[Optional[int]] = [None, None, None]  # 3개 슬롯의 조각 ID (0~18)
        self.total_score = 0
        self.total_lines_cleared = 0
        self.steps_taken = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.board = [0] * 16
        self.total_score = 0
        self.total_lines_cleared = 0
        self.steps_taken = 0
        self._refill_slots()
        return self._get_obs(), {}

    def _refill_slots(self):
        """슬롯 3개에 19종 조각 중 3개를 무작위 보급"""
        self.slots = [
            int(self.np_random.integers(0, len(RAW_PIECE_POOL)))
            for _ in range(3)
        ]

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """신경망에 전달할 상태 벡터 (16x10 보드 + 조각 정보)"""
        board_array = np.zeros((16, 10), dtype=np.float32)
        for r in range(16):
            row_val = self.board[r]
            for c in range(10):
                if (row_val >> (9 - c)) & 1:
                    board_array[r, c] = 1.0

        slots_array = np.array([s if s is not None else -1 for s in self.slots], dtype=np.int32)
        return {
            "board": board_array,
            "slots": slots_array
        }

    def get_valid_actions(self) -> List[Dict[str, Any]]:
        """현재 남아있는 슬롯 조각들로 둘 수 있는 모든 합법적인 착수점 목록 생성"""
        actions = []
        for slot_idx, p_idx in enumerate(self.slots):
            if p_idx is None:
                continue

            orientations = PROCESSED_PIECES[p_idx]["orientations"]
            for ori in orientations:
                for r, c, shift in get_valid_placements(self.board, ori):
                    actions.append({
                        "slot_idx": slot_idx,
                        "piece_idx": p_idx,
                        "ori": ori,
                        "r": r,
                        "c": c,
                        "shift": shift
                    })
        return actions

    def step(self, action: Dict[str, Any]):
        """
        선택한 행동을 보드에 반영
        action: {"slot_idx", "ori", "r", "shift", ...}
        """
        slot_idx = action["slot_idx"]
        ori = action["ori"]
        r = action["r"]
        shift = action["shift"]

        # 1. 블록 배치 및 줄 삭제 연산
        new_board, _, cleared_lines, _ = apply_placement(self.board, None, ori, r, shift)
        self.board = new_board
        self.slots[slot_idx] = None  # 해당 슬롯 소모
        self.steps_taken += 1

        # 2. 공식 점수 계산 ([300, 1200, 2700, 4800, 7500])
        step_score = LINE_SCORES.get(cleared_lines, cleared_lines * 1500)
        self.total_score += step_score
        self.total_lines_cleared += cleared_lines

        reward = step_score / self.REWARD_SCORE_SCALE

        # 4. 슬롯 3개를 모두 소진했으면 리필
        if all(s is None for s in self.slots):
            self._refill_slots()

        # 5. 게임 오버(종료) 판정: 남은 슬롯의 조각 중 단 하나도 둘 수 없으면 종료
        valid_next_actions = self.get_valid_actions()
        terminated = len(valid_next_actions) == 0
        truncated = False

        if terminated:
            reward -= self.TERMINATION_PENALTY

        info = {
            "score": self.total_score,
            "step_score": step_score,
            "cleared_lines": cleared_lines,
            "termination_penalty": self.TERMINATION_PENALTY if terminated else 0.0,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        """디버깅용 아스키 아트 출력"""
        print("\n┌" + "──" * 10 + "┐")
        for r in range(16):
            row_str = "".join("■ " if (self.board[r] >> (9 - c)) & 1 else "· " for c in range(10))
            print(f"│{row_str}│ {r:02d}")
        print("└" + "──" * 10 + "┘")
        print(f"슬롯 상태: {self.slots} / 점수: {self.total_score}점 / 누적 줄: {self.total_lines_cleared}줄\n")