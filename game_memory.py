import copy
from config import CONFIG
from typing import List, Tuple, Dict, Any, Optional
from solver_core import apply_placement, POPCOUNT_10, FULL_ROW

def log_print(*args, **kwargs):
    if CONFIG.get("ENABLE_CONSOLE_LOG", True):
        print(*args, **kwargs)

POLL_INTERVAL_SEC = CONFIG.get("POLL_INTERVAL_SEC", 1.0)
BEAM_SEARCH_WIDTH = CONFIG.get("BEAM_SEARCH_WIDTH", 50)
ANCHOR_FILE = "anchor.png"

class GameMemory:
    def __init__(self):
        # 16개 행 비트마스크 (0: 빈 칸, 1: 채워진 블록)
        self.board: List[int] = [0] * 16
        
        # 턴 및 통계 추적
        self.turn_count: int = 0
        self.total_lines_cleared: int = 0
        self.total_pieces_placed: int = 0
        
        # 실행 취소(Undo)를 위한 스택 히스토리
        self.history: List[Dict[str, Any]] = []

    def reset(self):
        """1번 요구사항: 빈 필드로 메모리 초기화"""
        self.board = [0] * 16
        self.turn_count = 0
        self.total_lines_cleared = 0
        self.total_pieces_placed = 0
        self.history.clear()
        log_print("[Memory] 보드 메모리가 빈 상태로 초기화되었습니다.")

    def save_snapshot(self, action_name: str):
        """현재 상태를 히스토리에 기록 (최대 20수 보관)"""
        snapshot = {
            "action": action_name,
            "board": list(self.board),
            "lines": self.total_lines_cleared,
            "pieces": self.total_pieces_placed
        }
        self.history.append(snapshot)
        if len(self.history) > 20:
            self.history.pop(0)

    def undo(self) -> bool:
        """이전 수로 되돌리기"""
        if not self.history:
            log_print("[Memory] 되돌릴 히스토리가 없습니다.")
            return False
        
        last_state = self.history.pop()
        self.board = list(last_state["board"])
        self.total_lines_cleared = last_state["lines"]
        self.total_pieces_placed = last_state["pieces"]
        log_print(f"[Memory Undo] '{last_state['action']}' 직전 상태로 복구 완료.")
        return True

    def place_piece(self, ori: Tuple[int, int, Tuple[int, ...]], r: int, c: int, slot_num: int) -> int:
        """
        추천된 조각을 메모리 보드에 적용하고 줄 삭제 처리
        ori: (h, w, row_bits) 형태의 튜플
        반환값: 이번 배치로 삭제된 가로줄 수
        """
        self.save_snapshot(f"Place Slot {slot_num} at ({r}, {c})")

        h, w, row_bits = ori
        shift = 10 - w - c

        # 보드 복사본 생성
        new_board = list(self.board)
        cleared_lines = 0

        # row_bits(실제 각 행의 비트 패턴)를 순회하며 보드에 배치
        for i, p_row in enumerate(row_bits):
            target_r = r + i
            if 0 <= target_r < 16:
                new_board[target_r] |= (p_row << shift)

        # 가로줄 삭제 검사 (10비트 0b1111111111 == 1023 == FULL_ROW)
        for row_idx in range(16):
            if new_board[row_idx] == FULL_ROW:
                new_board[row_idx] = 0
                cleared_lines += 1

        self.board = new_board
        self.total_pieces_placed += 1
        self.total_lines_cleared += cleared_lines

        log_print(f"[Memory] 슬롯 {slot_num}번 조각 배치 완료 -> 위치: ({r}행, {c}열)")
        if cleared_lines > 0:
            log_print(f"         ★ 가로줄 {cleared_lines}줄 삭제! (누적 삭제: {self.total_lines_cleared}줄)")

        return cleared_lines

    def get_total_occupied_blocks(self) -> int:
        """현재 메모리 보드에 채워져 있는 총 블록(1비트) 개수"""
        return sum(POPCOUNT_10[r] for r in self.board)

    def print_ascii_board(self):
        """디버깅용 콘솔 시각화 (16행 x 10열)"""
        log_print("\n┌" + "──" * 10 + "┐")
        for r_idx, row in enumerate(self.board):
            row_str = ""
            for c in range(10):
                # 최상위 비트(9)부터 0번 비트까지 검사
                if (row >> (9 - c)) & 1:
                    row_str += "■ "
                else:
                    row_str += "· "
            log_print(f"│{row_str}│ {r_idx:02d}")
        log_print("└" + "──" * 10 + "┘")
        log_print(f"현재 블록 수: {self.get_total_occupied_blocks()}개 / 누적 삭제 줄: {self.total_lines_cleared}줄\n")

    def sync_with_vision(self, vision_board: List[int], tolerance: int = 3):
        """
        5번 요구사항: 비전 인식 보드와 메모리 보드 정합성 검증(Sanity Check)
        총 블록 수 차이가 허용 오차 이내면 메모리를 신뢰하고,
        오차가 너무 크면(게임 리셋이나 특수 상황) 사용자에게 경고 및 동기화 옵션 제공
        """
        mem_blocks = self.get_total_occupied_blocks()
        vis_blocks = sum(POPCOUNT_10[r] for r in vision_board)
        diff = abs(mem_blocks - vis_blocks)

        if diff <= tolerance:
            # 정상 범위 내 (마우스 커서나 이펙트 등으로 1~2개 픽셀 차이일 가능성 높음)
            return True
        else:
            log_print(f"[Memory Warning] 비전 블록 수({vis_blocks})와 메모리 블록 수({mem_blocks}) 불일치! (차이: {diff})")
            return False