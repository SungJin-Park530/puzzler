import cv2
from board_detector import get_game_state

screen = cv2.imread("wide_screenshot.png")
anchor = cv2.imread("anchor.png")

board, pieces = get_game_state(screen, anchor)

if board is not None:
    print("=== [1] 보드 상태 (16행 x 10열) ===")
    for r_idx, r_val in enumerate(board):
        print(f"Row {r_idx:02d}: {r_val:010b}")

    print("\n=== [2] 슬롯 조각 식별 결과 ===")
    for slot_num, p_idx in enumerate(pieces, start=1):
        if p_idx is not None:
            print(f"슬롯 {slot_num}: 조각 {p_idx}번 식별 완료!")
        else:
            print(f"슬롯 {slot_num}: 조각 인식 실패 (매칭 불가 또는 비어있음)")