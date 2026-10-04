import sys
import time
import signal
import cv2
import numpy as np
import mss
from pynput import mouse
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from board_detector import (
    locate_anchor,
    parse_board_state_robust,
    identify_slot_piece,
    BOARD_OFFSET_X,
    BOARD_OFFSET_Y,
    CELL_SIZE,
    SLOT_Y_OFFSETS,
    SLOT_OFFSET_X,
    SLOT_WIDTH,
    SLOT_HEIGHT,
)
from solver_core import solve_remaining_pieces
from overlay_window import TransparentOverlay

class SignalBridge(QObject):
    trigger_turn = pyqtSignal()
    render_plan = pyqtSignal(list)

sct = mss.mss()
mouse_controller = mouse.Controller()
ANCHOR_IMG = cv2.imread("anchor.png")

class AppController:
    def __init__(self, overlay: TransparentOverlay, bridge: SignalBridge):
        self.overlay = overlay
        self.bridge = bridge
        self.is_processing = False

        self.bridge.trigger_turn.connect(self.process_turn)
        self.bridge.render_plan.connect(self.overlay.update_plan)

    def get_current_monitor_screen(self):
        cur_x, cur_y = mouse_controller.position
        target_monitor = None
        for mon in sct.monitors[1:]:
            mx, my, mw, mh = mon["left"], mon["top"], mon["width"], mon["height"]
            if mx <= cur_x < mx + mw and my <= cur_y < my + mh:
                target_monitor = mon
                break
        if target_monitor is None:
            target_monitor = sct.monitors[1]

        screenshot = sct.grab(target_monitor)
        img = np.array(screenshot)
        screen_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return screen_bgr, target_monitor["left"], target_monitor["top"]

    def process_turn(self):
        if self.is_processing:
            return
        self.is_processing = True

        try:
            print("\n[오버레이 분석 시작]")
            screen_bgr, mon_x, mon_y = self.get_current_monitor_screen()

            loc = locate_anchor(screen_bgr, ANCHOR_IMG)
            if not loc:
                print("[실패] 게임 기준점을 찾지 못했습니다.")
                return

            ax, ay = loc
            board = parse_board_state_robust(screen_bgr, ax, ay)

            available_pieces = []
            for slot_idx, s_dy in enumerate(SLOT_Y_OFFSETS):
                sx1 = ax + SLOT_OFFSET_X
                sy1 = ay + s_dy
                slot_bgr = screen_bgr[sy1 : sy1 + SLOT_HEIGHT, sx1 : sx1 + SLOT_WIDTH]
                p_idx = identify_slot_piece(slot_bgr)
                
                if p_idx is not None:
                    available_pieces.append((slot_idx, p_idx))

            print(f"[2] 화면 분석 완료: 남은 조각 {len(available_pieces)}개 감지")
            for s_idx, p_idx in available_pieces:
                print(f"    - 슬롯 {s_idx + 1}번: 조각 {p_idx}번")

            # 남아있는 조각이 하나도 없으면 오버레이를 지우고 종료
            if not available_pieces:
                print("[알림] 모든 조각을 사용했습니다. 다음 턴 조각 지급을 기다립니다.")
                self.bridge.render_plan.emit([])
                return

            # 4. 남은 조각들만 가지고 최적 배치 계산
            _, plan = solve_remaining_pieces(
                board, markers=None, available_pieces=available_pieces, beam_width=40
            )

            if not plan:
                print("[알림] 현재 보드에 남은 조각을 놓을 수 있는 자리가 없습니다!")
                self.bridge.render_plan.emit([])
                return

            board_screen_x = mon_x + ax + BOARD_OFFSET_X
            board_screen_y = mon_y + ay + BOARD_OFFSET_Y

            plan_items = []
            for p in plan:
                h_cnt, w_cnt, row_bits = p["ori"]
                r, c = p["r"], p["c"]
                slot_num = p["piece_slot"] + 1

                base_x = board_screen_x + (c * CELL_SIZE)
                base_y = board_screen_y + (r * CELL_SIZE)

                occupied_cells = []
                for dr, p_row in enumerate(row_bits):
                    for dc in range(w_cnt):
                        if (p_row >> (w_cnt - 1 - dc)) & 1:
                            occupied_cells.append((dr, dc))

                plan_items.append({
                    "step": p["step"],
                    "slot": slot_num,
                    "base_x": base_x,
                    "base_y": base_y,
                    "cell_size": CELL_SIZE,
                    "cells": occupied_cells,
                })

            self.bridge.render_plan.emit(plan_items)
            print(f"[성공] 3개 조각 실제 모양 오버레이 렌더링 완료!")

        finally:
            self.is_processing = False


def main():
    app = QApplication(sys.argv)

    # 1. SIGINT(Ctrl+C) 발생 시 Qt 앱을 즉시 종료하도록 핸들러 설정
    signal.signal(signal.SIGINT, lambda *args: app.quit())

    # 2. 파이썬 인터프리터가 SIGINT 시그널을 주기적으로 체크할 수 있도록 빈 타이머 구동
    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(200)

    total_rect = app.primaryScreen().virtualGeometry()

    overlay = TransparentOverlay()
    overlay.setGeometry(total_rect)
    overlay.show()

    bridge = SignalBridge()
    controller = AppController(overlay, bridge)

    def on_click(x, y, button, pressed):
        if pressed and button == mouse.Button.x1:
            bridge.trigger_turn.emit()

    listener = mouse.Listener(on_click=on_click)
    listener.start()

    print("=" * 50)
    print(" [투명 오버레이 AI 어시스턴트 실행 중] ")
    print(" - 마우스 뒤로가기(G4) 버튼: 추천 배치 화면 렌더링")
    print(" - 터미널에서 [Ctrl + C] 입력 시 즉시 종료됩니다.")
    print("=" * 50)

    try:
        app.exec()
    finally:
        listener.stop()
        print("\n프로그램이 정상 종료되었습니다.")

if __name__ == "__main__":
    main()