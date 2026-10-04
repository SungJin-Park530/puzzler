# overlay_window.py
import sys
import ctypes
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QBrush

try:
    from config import CONFIG
except ImportError:
    # config.py가 없을 경우 기본값으로 폴백
    CONFIG = {
        "SHOW_BADGE_NUMBER": True,
        "SHOW_LABEL_TEXT": True,
        "BADGE_RADIUS": 13,
        "LABEL_FONT_SIZE": 9,
        "CELL_PADDING": 1,
        "CELL_CORNER_RADIUS": 4,
        "STEP_COLORS": {
            1: {"border": QColor(46, 204, 113, 240), "fill": QColor(46, 204, 113, 100), "text": QColor(255, 255, 255)},
            2: {"border": QColor(52, 152, 219, 240), "fill": QColor(52, 152, 219, 100), "text": QColor(255, 255, 255)},
            3: {"border": QColor(230, 126, 34, 240), "fill": QColor(230, 126, 34, 100), "text": QColor(255, 255, 255)},
        }
    }

WDA_EXCLUDEFROMCAPTURE = 0x00000011


class TransparentOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowTransparentForInput |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.plan_items = []

    def showEvent(self, event):
        super().showEvent(event)
        try:
            hwnd = int(self.winId())
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        except Exception as e:
            print(f"[오버레이 경고] 캡처 배제 적용 실패: {e}")

    def update_plan(self, plan_items):
        self.plan_items = plan_items
        self.update()

    def clear_plan(self):
        self.plan_items = []
        self.update()

    def paintEvent(self, event):
        if not self.plan_items:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 설정 플래그 읽기
        show_badge = CONFIG.get("SHOW_BADGE_NUMBER", True)
        show_label = CONFIG.get("SHOW_LABEL_TEXT", True)
        pad = CONFIG.get("CELL_PADDING", 1)
        rnd = CONFIG.get("CELL_CORNER_RADIUS", 4)
        badge_r = CONFIG.get("BADGE_RADIUS", 13)
        label_size = CONFIG.get("LABEL_FONT_SIZE", 9)

        for item in self.plan_items:
            step = item["step"]
            cfg_color = CONFIG["STEP_COLORS"].get(step, CONFIG["STEP_COLORS"][1])
            border_pen = QPen(cfg_color["border"], 2, Qt.PenStyle.SolidLine)
            fill_brush = QBrush(cfg_color["fill"])

            base_x = item["base_x"]
            base_y = item["base_y"]
            cs = item["cell_size"]
            cells = item["cells"]

            if not cells:
                continue

            # ----------------------------------------------------
            # 1. 실제 점유 셀 블록 렌더링
            # ----------------------------------------------------
            painter.setBrush(fill_brush)
            painter.setPen(border_pen)

            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')

            for dr, dc in cells:
                cx = base_x + (dc * cs)
                cy = base_y + (dr * cs)
                cell_rect = QRectF(cx + pad, cy + pad, cs - 2 * pad, cs - 2 * pad)
                painter.drawRoundedRect(cell_rect, rnd, rnd)

                min_x = min(min_x, cx)
                min_y = min(min_y, cy)
                max_x = max(max_x, cx + cs)
                max_y = max(max_y, cy + cs)

            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0

            # ----------------------------------------------------
            # 2. 중심 번호 원형 배지 딱지 (True일 때만 표시)
            # ----------------------------------------------------
            if show_badge:
                painter.setBrush(QBrush(cfg_color["border"]))
                painter.setPen(QPen(QColor(255, 255, 255), 1.5))
                painter.drawEllipse(QRectF(center_x - badge_r, center_y - badge_r, badge_r * 2, badge_r * 2))

                painter.setPen(QPen(cfg_color["text"]))
                painter.setFont(QFont("Arial", 11, QFont.Weight.Bold))
                painter.drawText(
                    QRectF(center_x - badge_r, center_y - badge_r, badge_r * 2, badge_r * 2),
                    Qt.AlignmentFlag.AlignCenter,
                    str(step)
                )

            # ----------------------------------------------------
            # 3. 상단 Step / 슬롯 번호 안내 텍스트 (True일 때만 표시)
            # ----------------------------------------------------
            if show_label:
                label_text = f"Step {step} (슬롯 {item['slot']})"
                painter.setFont(QFont("Arial", label_size, QFont.Weight.Bold))
                # 텍스트 가독성을 위한 검은색 그림자
                painter.setPen(QPen(QColor(0, 0, 0, 210)))
                painter.drawText(int(min_x + 2), int(min_y - 4), label_text)
                # 본문 컬러 텍스트
                painter.setPen(QPen(cfg_color["border"]))
                painter.drawText(int(min_x + 1), int(min_y - 5), label_text)

        painter.end()