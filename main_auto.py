import sys
import os
import ctypes

# Windows DPI Awareness 사전 설정 (PyQt6와의 충돌 방지)
try:
    # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
except Exception:
    pass

import time
import signal
import cv2
import numpy as np
import mss
from pynput import mouse
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from config import CONFIG
from game_memory import GameMemory
from board_detector import (
    locate_anchor,
    parse_board_state_robust,  # 실제 보드 스캔용
    identify_slot_piece,
    is_slot_active,
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
from board_validator import find_actual_placement

# ========================================================
# [설정 및 상수 정의]
# ========================================================
POLL_INTERVAL_SEC = 1.0        # 평상시 슬롯 감시 주기 (초 단위)
ANCHOR_FILE = "anchor.png"     # 앵커 이미지 파일명
BEAM_SEARCH_WIDTH = 40         # 빔 서치 탐색 폭


class SignalBridge(QObject):
    sync_trigger = pyqtSignal()   # 수동 스마트 동기화 트리거
    update_overlay = pyqtSignal(list)


sct = mss.MSS()
mouse_controller = mouse.Controller()
ANCHOR_IMG = cv2.imread(ANCHOR_FILE)
if ANCHOR_IMG is None:
    print(f"[오류] '{ANCHOR_FILE}' 파일을 찾을 수 없습니다. 경로를 확인하세요.")
    sys.exit(1)


class GameTracker:
    def __init__(self, bridge: SignalBridge):
        self.bridge = bridge
        self.memory = GameMemory()
        self.prev_active_slots = [False, False, False]
        self.current_plan = []
        self.anchor_pos = None
        self.mon_offset = (0, 0)
        self.is_running = False

        self.bridge.sync_trigger.connect(self.sync_board_and_solve)

    def get_screen(self):
        cur_x, cur_y = mouse_controller.position
        target_mon = sct.monitors[1]
        for mon in sct.monitors[1:]:
            if (mon["left"] <= cur_x < mon["left"] + mon["width"] and
                mon["top"] <= cur_y < mon["top"] + mon["height"]):
                target_mon = mon
                break

        screenshot = sct.grab(target_mon)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), target_mon["left"], target_mon["top"]

    def sync_board_and_solve(self):
        """
        [마우스 뒤로가기 클릭 시 실행]
        1. 실제 화면의 보드와 슬롯을 강제 캡처
        2. 빈 보드면 메모리 리셋
        3. 빈 보드가 아니면 board_validator로 실제 착수점 검증 및 보정
        4. 남은 조각 기준으로 최적 수 재계산
        """
        print("\n" + "=" * 55)
        print("★ [스마트 동기화] 화면 보드 및 슬롯 강제 스캔 시작...")

        screen_bgr, mon_x, mon_y = self.get_screen()
        self.mon_offset = (mon_x, mon_y)

        loc = locate_anchor(screen_bgr, ANCHOR_IMG)
        if not loc:
            print("[실패] 게임 창 기준점(앵커)을 찾지 못했습니다.")
            return
        self.anchor_pos = loc
        ax, ay = loc

        # 1. 실제 화면 보드 스캔
        scanned_board = parse_board_state_robust(screen_bgr, ax, ay)
        total_blocks = sum(bin(r).count('1') for r in scanned_board)

        # 2. 슬롯 조각 확인 (사라진 조각 파악을 위해 먼저 스캔)
        curr_active_slots = [False, False, False]
        available_pieces = []
        for s_idx, s_dy in enumerate(SLOT_Y_OFFSETS):
            sx1 = ax + SLOT_OFFSET_X
            sy1 = ay + s_dy
            sbgr = screen_bgr[sy1 : sy1 + SLOT_HEIGHT, sx1 : sx1 + SLOT_WIDTH]
            active = is_slot_active(sbgr)
            curr_active_slots[s_idx] = active
            if active:
                p_idx = identify_slot_piece(sbgr)
                if p_idx is not None:
                    available_pieces.append((s_idx, p_idx))

        # ----------------------------------------------------
        # [핵심] 보드 상태 판단 및 board_validator 연계
        # ----------------------------------------------------
        if total_blocks <= 2:
            print("-> [빈 필드 감지] 메모리를 새 게임 상태로 초기화합니다.")
            self.memory.reset()
            self.current_plan = []
        else:
            # 방금 사라진 조각 식별 (이전엔 활성, 지금은 비활성)
            consumed_slots = [i for i in range(3) if self.prev_active_slots[i] and not curr_active_slots[i]]
            validated = False

            if consumed_slots and self.current_plan:
                # 사라진 슬롯 번호에 대응하는 조각의 id 찾기
                target_slot = consumed_slots[0]
                matched_step = next((p for p in self.current_plan if p["piece_slot"] == target_slot), None)

                if matched_step:
                    # 해당 조각의 piece_id 역추적
                    piece_id = matched_step.get("piece_id")
                    if piece_id is not None:
                        status, match_data = find_actual_placement(self.memory.board, scanned_board, piece_id)

                        if status == "MATCH":
                            print(f"-> [검증 성공] 실제 배치 위치: ({match_data['r']}행, {match_data['c']}열) 자동 보정")
                            self.memory.place_piece(match_data["ori"], match_data["r"], match_data["c"], slot_num=target_slot + 1)
                            validated = True
                        elif status == "AMBIGUOUS":
                            print("\n[주의] 1칸 블록 배치와 능력 출현이 동시에 감지되었거나 다중 후보가 존재합니다.")
                            candidates = match_data["candidates"]
                            for c_i, cand in enumerate(candidates):
                                print(f"   {c_i + 1}번 후보: ({cand['r']}행, {cand['c']}열)")
                            choice = input("실제 배치한 번호를 선택하세요 (엔터 시 1번): ").strip()
                            sel_idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(candidates) else 0
                            sel = candidates[sel_idx]
                            self.memory.place_piece(sel["ori"], sel["r"], sel["c"], slot_num=target_slot + 1)
                            validated = True

            # 검증으로 갱신되지 않은 경우(예: 게임 도중 처음 켰거나 복합 예외)는 비전 스캔 보드로 안전 덮어쓰기
            if not validated:
                print(f"-> [직접 동기화] 실제 화면 보드({total_blocks}개 블록)로 메모리를 동기화합니다.")
                self.memory.board = list(scanned_board)

        self.memory.print_ascii_board()
        self.prev_active_slots = curr_active_slots
        self.is_running = True

        # 3. 남은 조각으로 최적 수 재계산
        if available_pieces:
            print(f"[동기화 완료] 유효 조각 {len(available_pieces)}개 배치 계산 시작...")
            _, plan = solve_remaining_pieces(
                self.memory.board,
                markers=None,
                available_pieces=available_pieces,
                beam_width=BEAM_SEARCH_WIDTH,
            )
            self.current_plan = plan if plan else []
            self.render_current_plan()
            print("[계산 완료] 오버레이 가이드 업데이트")
        else:
            print("[알림] 사용할 수 있는 조각이 슬롯에 없습니다.")
            self.current_plan = []
            self.bridge.update_overlay.emit([])

    def poll_screen(self):
        """평상시 1초 주기 감시: 슬롯 소모 추적 및 리필 자동 전진"""
        if not self.is_running:
            return

        screen_bgr, mon_x, mon_y = self.get_screen()
        self.mon_offset = (mon_x, mon_y)

        loc = locate_anchor(screen_bgr, ANCHOR_IMG)
        if not loc:
            return
        self.anchor_pos = loc
        ax, ay = loc

        curr_active_slots = [False, False, False]
        slot_images = []
        for s_idx, s_dy in enumerate(SLOT_Y_OFFSETS):
            sx1 = ax + SLOT_OFFSET_X
            sy1 = ay + s_dy
            sbgr = screen_bgr[sy1 : sy1 + SLOT_HEIGHT, sx1 : sx1 + SLOT_WIDTH]
            slot_images.append(sbgr)
            curr_active_slots[s_idx] = is_slot_active(sbgr)

        prev_count = sum(self.prev_active_slots)
        curr_count = sum(curr_active_slots)

        # 1) 리필 감지 시 이전 턴의 미반영 마지막 조각 자동 커밋
        if curr_count == 3 and prev_count < 3 and self.current_plan:
            for p in self.current_plan:
                self.memory.place_piece(p["ori"], p["r"], p["c"], slot_num=p["piece_slot"] + 1)
            self.current_plan = []
            self.memory.print_ascii_board()

        # 2) 조각 하나 내려놓았을 때 (슬롯 사라짐 감지)
        placed_slots = [
            i for i in range(3)
            if self.prev_active_slots[i] and not curr_active_slots[i]
        ]
        if placed_slots and self.current_plan:
            for p_slot in placed_slots:
                # 1. 기본 매칭: 해당 슬롯 번호에 직접 매정된 플랜 찾기
                matched_step = next(
                    (p for p in self.current_plan if p["piece_slot"] == p_slot), None
                )
                
                # 2. 동일 조각 대응 (순서 엇갈림 보정):
                # 만약 사용자가 '다음 스텝'에 배정된 동일 모양 슬롯을 먼저 썼다면,
                # 가장 앞선 스텝(current_plan[0])과 모양이 같은지 확인하고 앞선 스텝으로 적용!
                if matched_step and len(self.current_plan) > 1:
                    earliest_step = self.current_plan[0]
                    # 모양(ori의 행/열 비트 패턴)이 동일한 경우, 슬롯 번호 무시하고 1순위 스텝으로 처리
                    if matched_step["ori"] == earliest_step["ori"]:
                        matched_step = earliest_step

                if matched_step:
                    self.memory.place_piece(
                        matched_step["ori"],
                        matched_step["r"],
                        matched_step["c"],
                        slot_num=p_slot + 1,
                    )
                    # 처리된 계획은 목록에서 제거
                    self.current_plan.remove(matched_step)

            self.memory.print_ascii_board()
            self.render_current_plan()

        # 3) 새 턴 조각 리필 계산
        if curr_count == 3 and len(self.current_plan) == 0: #
            # 1차 스캔 결과 확인
            available_pieces = []
            for s_idx, active in enumerate(curr_active_slots): #
                if active: #
                    p_idx = identify_slot_piece(slot_images[s_idx]) #
                    if p_idx is not None: #
                        available_pieces.append((s_idx, p_idx)) #

            # ----------------------------------------------------
            # [가드레일] 3개여야 할 새 턴인데 1~2개만 식별된 경우 재시도
            # ----------------------------------------------------
            if len(available_pieces) < 3:
                print(f"[재인식 가드레일] 조각 {len(available_pieces)}개만 식별됨. 재시도 수행...")
                curr_active_slots, available_pieces = self.scan_slots_with_retry(ax, ay, expected_count=3)

            if available_pieces:
                print(f"\n[새 턴 감지] 조각 {len(available_pieces)}개 계산...") #
                _, plan = solve_remaining_pieces( #
                    self.memory.board, #
                    markers=None, #
                    available_pieces=available_pieces, #
                    beam_width=BEAM_SEARCH_WIDTH, #
                )
                self.current_plan = plan if plan else [] #
                self.render_current_plan() #
            else:
                self.bridge.update_overlay.emit([]) #
        elif curr_count == 0: #
            self.bridge.update_overlay.emit([]) #

        self.prev_active_slots = curr_active_slots

    def render_current_plan(self):
        if not self.current_plan or not self.anchor_pos:
            self.bridge.update_overlay.emit([])
            return

        ax, ay = self.anchor_pos
        mon_x, mon_y = self.mon_offset
        board_screen_x = mon_x + ax + BOARD_OFFSET_X
        board_screen_y = mon_y + ay + BOARD_OFFSET_Y

        plan_items = []
        for p in self.current_plan:
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

        self.bridge.update_overlay.emit(plan_items)
    
    def scan_slots_once(self, screen_bgr, ax, ay):
        """슬롯 3개의 활성 여부 및 조각 ID 1회 스캔"""
        active_slots = [False, False, False]
        available_pieces = []
        for s_idx, s_dy in enumerate(SLOT_Y_OFFSETS): #[cite: 1]
            sx1 = ax + SLOT_OFFSET_X #[cite: 1]
            sy1 = ay + s_dy #[cite: 1]
            sbgr = screen_bgr[sy1 : sy1 + SLOT_HEIGHT, sx1 : sx1 + SLOT_WIDTH] #[cite: 1]
            active = is_slot_active(sbgr) #[cite: 1]
            active_slots[s_idx] = active
            if active:
                p_idx = identify_slot_piece(sbgr) #[cite: 1]
                if p_idx is not None:
                    available_pieces.append((s_idx, p_idx))
        return active_slots, available_pieces

    def scan_slots_with_retry(self, ax, ay, expected_count=3):
        """
        새 턴 지급 시 조각이 expected_count(기본 3개) 미만으로 인식되면
        마우스 간섭 또는 애니메이션 지연으로 보고 재시도
        """
        max_retries = CONFIG.get("SLOT_RETRY_COUNT", 3)
        retry_delay = CONFIG.get("SLOT_RETRY_DELAY_SEC", 0.15)

        for attempt in range(max_retries + 1):
            screen_bgr, mon_x, mon_y = self.get_screen() #
            self.mon_offset = (mon_x, mon_y) #
            active_slots, available_pieces = self.scan_slots_once(screen_bgr, ax, ay)

            # 기대한 조각 수(3개)가 온전히 잡혔으면 즉시 반환
            if len(available_pieces) >= expected_count:
                return active_slots, available_pieces

            # 마지막 시도가 아니면 잠깐 대기 후 재촬영
            if attempt < max_retries:
                time.sleep(retry_delay)

        # 재시도 소진 후 감지된 결과라도 반환
        return active_slots, available_pieces


def main():
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, lambda *args: app.quit())

    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(200)

    total_rect = app.primaryScreen().virtualGeometry()
    overlay = TransparentOverlay()
    overlay.setGeometry(total_rect)
    overlay.show()

    bridge = SignalBridge()
    bridge.update_overlay.connect(overlay.update_plan)

    tracker = GameTracker(bridge)

    # 마우스 G4(뒤로가기): 스마트 동기화 트리거
    def on_click(x, y, button, pressed):
        if pressed and button == mouse.Button.x1:
            bridge.sync_trigger.emit()

    listener = mouse.Listener(on_click=on_click)
    listener.start()

    poll_timer = QTimer()
    poll_timer.timeout.connect(tracker.poll_screen)
    poll_timer.start(int(POLL_INTERVAL_SEC * 1000))

    print("=" * 60)
    print(" [스마트 보정 지원 논리 메모리 AI 어시스턴트] ")
    print(f" - 자동 감시: {POLL_INTERVAL_SEC}초 간격 슬롯 추적")
    print(" - 마우스 [뒤로가기(G4)]: 화면 즉시 스캔 & 보드 동기화")
    print("   * 빈 필드에서 누름: 새 게임 시작 (메모리 리셋)")
    print("   * 잘못 놓거나 꼬였을 때 누름: 실제 필드로 메모리 즉시 보정")
    print("=" * 60)

    try:
        app.exec()
    finally:
        listener.stop()
        print("\n프로그램이 정상 종료되었습니다.")


if __name__ == "__main__":
    main()