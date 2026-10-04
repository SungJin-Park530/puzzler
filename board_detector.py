import cv2
import numpy as np
from typing import List, Tuple, Optional
from solver_core import PROCESSED_PIECES, RAW_PIECE_POOL, grid_to_bits, rotate_90, flip_h

# ========================================================
# 1. 앵커 및 좌표 상수 (앞서 계산된 정밀 픽셀 오프셋)
# ========================================================
BOARD_OFFSET_X = -89
BOARD_OFFSET_Y = 108
CELL_SIZE = 26
BOARD_COLS = 10
BOARD_ROWS = 16

SLOT_OFFSET_X = 184
SLOT_WIDTH = 121
SLOT_HEIGHT = 70
SLOT_Y_OFFSETS = [131, 206, 281]

# 16개 행별 배경색 기준 테이블 (세로 그라데이션 완벽 대응)
# Row 0 (상단) ~ Row 15 (하단)
ROW_BASELINE_BGR = np.array([
    [223, 177, 94],  # Row 0
    [221, 178, 93],  # Row 1
    [218, 180, 91],  # Row 2
    [213, 182, 89],  # Row 3
    [207, 185, 88],  # Row 4
    [203, 187, 84],  # Row 5
    [198, 189, 82],  # Row 6
    [196, 190, 81],  # Row 7
    [194, 191, 80],  # Row 8
    [194, 191, 80],  # Row 9
    [194, 191, 80],  # Row 10
    [194, 191, 80],  # Row 11
    [195, 192, 80],  # Row 12
    [198, 195, 82],  # Row 13
    [203, 200, 85],  # Row 14
    [211, 208, 89],  # Row 15
], dtype=np.float32)


def locate_anchor(screen_bgr: np.ndarray, anchor_bgr: np.ndarray) -> Optional[Tuple[int, int]]:
    """화면에서 '한글 모아모아' 앵커 좌상단 (ax, ay) 탐색"""
    res = cv2.matchTemplate(screen_bgr, anchor_bgr, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.85:
        return None
    return max_loc


def parse_board_state_robust(screen_bgr: np.ndarray, ax: int, ay: int) -> List[int]:
    """그라데이션 및 4색 블록을 완벽하게 판별하는 10x16 보드 파서"""
    bx1 = ax + BOARD_OFFSET_X
    by1 = ay + BOARD_OFFSET_Y
    board = [0] * BOARD_ROWS

    for r in range(BOARD_ROWS):
        row_bits = 0
        expected_bg = ROW_BASELINE_BGR[r]

        for c in range(BOARD_COLS):
            cx = int(bx1 + (c + 0.5) * CELL_SIZE)
            cy = int(by1 + (r + 0.5) * CELL_SIZE)

            # 셀 중심 5x5 평균 색상
            patch = screen_bgr[cy - 2 : cy + 3, cx - 2 : cx + 3]
            mean_bgr = np.mean(patch, axis=(0, 1))

            # 해당 행의 기준 배경색과의 거리 측정
            color_diff = np.linalg.norm(mean_bgr - expected_bg)

            # 빈칸(diff < 5), 파랑 블록(diff ~65), 그 외(diff > 170)
            if color_diff > 30.0:
                row_bits |= (1 << (9 - c))

        board[r] = row_bits

    return board


def identify_slot_piece(slot_bgr: np.ndarray) -> Optional[int]:
    """
    좌측 프리뷰 박스 안쪽에서 배경을 뺀 나머지(블록)만 추출하여 19종 조각 매칭.
    """
    # 먼저 활성 슬롯인지 검사
    if not is_slot_active(slot_bgr):
        return None

    # 1. 좌측 프리뷰 박스 안쪽 크롭 (외곽 테두리 3px 배제: y 10~60, x 10~52)
    preview_inner = slot_bgr[10:60, 10:52]
    hsv = cv2.cvtColor(preview_inner, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # 2. 배경(흰색 + 호버/토글 연노랑) 마스크
    # V가 235 이상이면서 S가 110 이하인 픽셀은 모두 배경으로 판정
    is_bg = (v >= 235) & (s <= 110)
    block_mask = (~is_bg).astype(np.uint8)

    contours, _ = cv2.findContours(block_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 면적 15픽셀 이상의 유효 블록 컨투어만 병합
    valid_cnts = [c for c in contours if cv2.contourArea(c) > 15]
    if not valid_cnts:
        return None

    all_points = np.vstack(valid_cnts)
    bx, by, bw, bh = cv2.boundingRect(all_points)

    # 노이즈 크기 필터
    if bw < 8 or bh < 8:
        return None

    # 3. 블록 격자화 (8.25px 단위)
    PREVIEW_CELL_PX = 8.25
    grid_cols = max(1, round(bw / PREVIEW_CELL_PX))
    grid_rows = max(1, round(bh / PREVIEW_CELL_PX))

    piece_grid = [[0] * grid_cols for _ in range(grid_rows)]
    step_x = bw / grid_cols
    step_y = bh / grid_rows

    H_m, W_m = block_mask.shape
    for gr in range(grid_rows):
        for gc in range(grid_cols):
            sample_x = int(bx + (gc + 0.5) * step_x)
            sample_y = int(by + (gr + 0.5) * step_y)
            if 0 <= sample_y < H_m and 0 <= sample_x < W_m:
                if block_mask[sample_y, sample_x]:
                    piece_grid[gr][gc] = 1

    # 4. 사전 정의된 19종 조각과 비교 매칭
    target_bits = grid_to_bits(piece_grid)
    for p_idx, p_data in enumerate(PROCESSED_PIECES):
        for ori in p_data["orientations"]:
            if ori == target_bits:
                return p_idx

    return None

def is_slot_active(slot_bgr: np.ndarray) -> bool:
    """
    슬롯의 좌측 프리뷰 박스 내부 영역만 크롭하여 조각이 존재하는지 판별.
    - 사용 완료: 영역 전체가 짙은 하늘색 단색
    - 조각 존재: 영역 대부분이 흰색(또는 호버 연노랑) 배경으로 구성됨
    """
    # 1. 버튼을 완전히 배제하고 좌측 내부 알맹이만 크롭 (x: 10~52, y: 10~60)
    preview_inner = slot_bgr[10:60, 10:52]
    hsv = cv2.cvtColor(preview_inner, cv2.COLOR_BGR2HSV)
    
    # 2. 밝은 배경 픽셀(흰색 V>240, S<50 또는 연노랑 V>240, S<110) 검출
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    is_card_bg = (v >= 230) & (s <= 110)
    
    # 내부의 40% 이상이 카드 배경(밝은 톤)이면 조각이 들어있는 활성 슬롯
    bg_ratio = np.mean(is_card_bg)
    return bg_ratio > 0.40


def get_game_state(screen_bgr: np.ndarray, anchor_bgr: np.ndarray):
    """스크린샷에서 전체 게임 상태를 추출하는 메인 함수"""
    loc = locate_anchor(screen_bgr, anchor_bgr)
    if not loc:
        return None, None

    ax, ay = loc

    # 1. 보드 상태 추출
    board = parse_board_state_robust(screen_bgr, ax, ay)

    # 2. 우측 3개 슬롯 조각 인식
    piece_indices = []
    for s_dy in SLOT_Y_OFFSETS:
        sx1 = ax + SLOT_OFFSET_X
        sy1 = ay + s_dy
        slot_bgr = screen_bgr[sy1 : sy1 + SLOT_HEIGHT, sx1 : sx1 + SLOT_WIDTH]
        p_idx = identify_slot_piece(slot_bgr)
        piece_indices.append(p_idx)

    return board, piece_indices