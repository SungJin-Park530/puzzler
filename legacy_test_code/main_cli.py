import time
import cv2
import numpy as np
import mss
from pynput import mouse
import keyboard

from board_detector import (
    locate_anchor,
    parse_board_state_robust,
    identify_slot_piece,
    SLOT_Y_OFFSETS,
    SLOT_OFFSET_X,
    SLOT_WIDTH,
    SLOT_HEIGHT,
)
from solver_core import solve_3_pieces

# 1. 앵커 이미지 로드 (anchor.png)
ANCHOR_IMG = cv2.imread("anchor.png")
if ANCHOR_IMG is None:
    print("[오류] 'anchor.png'를 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

sct = mss.mss()
is_processing = False  # 연산 중 중복 클릭 방지 락
mouse_controller = mouse.Controller()


def get_current_monitor_screen() -> tuple[np.ndarray, int, int]:
    """
    현재 마우스 커서가 위치한 모니터 화면만 캡처하여
    (이미지_BGR, 모니터_시작_X, 모니터_시작_Y) 튜플로 반환
    """
    cur_x, cur_y = mouse_controller.position

    # 마우스 커서가 속한 모니터 찾기 (sct.monitors[1:]은 개별 모니터 목록)
    target_monitor = None
    for mon in sct.monitors[1:]:
        mx, my, mw, mh = mon["left"], mon["top"], mon["width"], mon["height"]
        if mx <= cur_x < mx + mw and my <= cur_y < my + mh:
            target_monitor = mon
            break

    # 만약 판별 실패 시 기본 1번 모니터 선택
    if target_monitor is None:
        target_monitor = sct.monitors[1]

    screenshot = sct.grab(target_monitor)
    img = np.array(screenshot)
    screen_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return screen_bgr, target_monitor["left"], target_monitor["top"]


def run_solver_turn():
    """마우스 뒤로가기 클릭 시 실행되는 1턴 분석 및 계산"""
    global is_processing
    if is_processing:
        return
    is_processing = True

    try:
        print("\n" + "=" * 50)
        print("[1] 마우스가 있는 모니터 화면 캡처 중...")
        t0 = time.time()
        screen_bgr, mon_x, mon_y = get_current_monitor_screen()
        h, w, _ = screen_bgr.shape
        print(f"      캡처 해상도: {w}x{h} (모니터 시작좌표: {mon_x}, {mon_y})")

        # 디버그용: 실제 파이썬이 본 화면을 바로 저장
        cv2.imwrite("debug_captured_screen.png", screen_bgr)

        # 1. 앵커 찾기
        res = cv2.matchTemplate(screen_bgr, ANCHOR_IMG, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print(f"      기준점 매칭 신뢰도: {max_val * 100:.2f}%")

        if max_val < 0.80:
            print("[실패] 화면에서 '한글 모아모아' 기준점을 찾지 못했습니다!")
            print("      'debug_captured_screen.png' 파일을 열어 게임 창이 잘 찍혔는지 확인해보세요.")
            return

        ax, ay = max_loc
        print(f"      게임 화면 감지 성공! 로컬 기준점 좌표: ({ax}, {ay})")

        # 2. 보드 상태 추출
        board = parse_board_state_robust(screen_bgr, ax, ay)

        # 3. 우측 3개 슬롯 조각 식별
        piece_indices = []
        for slot_idx, s_dy in enumerate(SLOT_Y_OFFSETS, start=1):
            sx1 = ax + SLOT_OFFSET_X
            sy1 = ay + s_dy
            slot_bgr = screen_bgr[sy1 : sy1 + SLOT_HEIGHT, sx1 : sx1 + SLOT_WIDTH]
            p_idx = identify_slot_piece(slot_bgr)
            piece_indices.append(p_idx)

        print(f"[2] 화면 분석 완료 ({time.time() - t0:.3f}초)")
        print(f"    - 감지된 슬롯 조각: {piece_indices}")

        # 유효성 검사
        if any(p is None for p in piece_indices):
            print("[경고] 일부 조각을 인식하지 못했습니다.")
            for idx, p in enumerate(piece_indices, start=1):
                if p is None:
                    print(f"      * 슬롯 {idx}번 조각 미인식")
            return

        # 4. 최적 배치 알고리즘 계산
        print("[3] 3개 조각 최적 배치 계산 중...")
        t1 = time.time()
        best_score, plan = solve_3_pieces(
            board, markers=None, piece_indices=piece_indices, beam_width=40
        )
        calc_time = time.time() - t1

        print(f"    - 계산 완료! 소요 시간: {calc_time:.3f}초 (점수: {best_score:.1f})")
        print("-" * 50)

        if plan:
            for p in plan:
                slot_num = p["piece_slot"] + 1
                r, c = p["r"], p["c"]
                h_cnt, w_cnt, _ = p["ori"]
                print(f"▶ [{p['step']}단계] {slot_num}번 슬롯 조각 사용")
                print(f"   - 위치: 보드 [{r:02d}행, {c:02d}열] (좌상단 기준)")
                print(f"   - 형태: 가로 {w_cnt}칸 x 세로 {h_cnt}칸")
        else:
            print("▶ [배치 불가] 3개 조각을 모두 놓을 수 있는 배치가 없습니다! (능력 사용 필요)")
        print("=" * 50)

    finally:
        is_processing = False


def on_mouse_click(x, y, button, pressed):
    """마우스 클릭 이벤트 리스너"""
    if pressed and button == mouse.Button.x1:
        run_solver_turn()


def main():
    print("=" * 50)
    print(" [블록 게임 AI 어시스턴트 대기 중] ")
    print(" - 게임 화면이 있는 모니터 위에 마우스를 올리고 [뒤로가기(G4)]를 누르세요.")
    print(" - 프로그램 종료: 키보드 [Esc] 키")
    print("=" * 50)

    mouse_listener = mouse.Listener(on_click=on_mouse_click)
    mouse_listener.start()

    keyboard.wait("esc")
    mouse_listener.stop()
    print("\n프로그램을 종료합니다.")


if __name__ == "__main__":
    main()