import cv2
import numpy as np

# 1. 화면 캡처 이미지와 앵커("한글 모아모아") 이미지 로드
screen = cv2.imread("wide_screenshot.png")
anchor = cv2.imread("anchor.png")

if screen is None or anchor is None:
    print("이미지를 찾을 수 없습니다. wide_screenshot.png와 anchor.png 경로를 확인해주세요.")
    exit()

# 2. 템플릿 매칭으로 화면 전체에서 앵커 위치 탐색
res = cv2.matchTemplate(screen, anchor, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

print(f"기준점 매칭 신뢰도: {max_val * 100:.2f}%")
if max_val < 0.85:
    print("경고: 화면에서 기준점(한글 모아모아)을 찾지 못했습니다!")
    exit()

# 앵커 좌상단 좌표 (ax, ay)
ax, ay = max_loc
aw, ah = anchor.shape[1], anchor.shape[0]
print(f"기준점 위치: ({ax}, {ay}), 크기: ({aw}x{ah})")

# ========================================================
# 3. 앵커 기준 정확한 상대 오프셋 (픽셀 분석 결과)
# ========================================================

# [A] 10x16 보드 영역
# 한 칸 크기: 정확히 26px x 26px
BOARD_OFFSET_X = -89   # 앵커 좌상단 기준 보드 좌상단 X (134 - 89 = 45)
BOARD_OFFSET_Y = 108   # 앵커 좌상단 기준 보드 좌상단 Y (15 + 108 = 123)
CELL_SIZE = 26         # 칸당 가로/세로 픽셀
BOARD_COLS = 10        # 가로 10칸 -> 260px
BOARD_ROWS = 16        # 세로 16칸 -> 416px

bx1 = ax + BOARD_OFFSET_X
by1 = ay + BOARD_OFFSET_Y
bx2 = bx1 + (BOARD_COLS * CELL_SIZE)
by2 = by1 + (BOARD_ROWS * CELL_SIZE)

# [B] 우측 보유 조각 카드 슬롯 (3개)
# 각 슬롯 카드 크기: 가로 121px x 세로 70px, 75px 간격 배치
SLOT_OFFSET_X = 184
SLOT_WIDTH = 121
SLOT_HEIGHT = 70
SLOT_Y_OFFSETS = [131, 206, 281]  # 1번, 2번, 3번 슬롯의 Y 오프셋

# ========================================================
# 4. 검증 시각화 (디버그 이미지 생성)
# ========================================================
debug_img = screen.copy()

# 1) 찾은 앵커 표시 (노란색 사각형)
cv2.rectangle(debug_img, (ax, ay), (ax + aw, ay + ah), (0, 255, 255), 2)

# 2) 10x16 보드 외곽선 (초록색)
cv2.rectangle(debug_img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

# 3) 10x16 격자 160칸의 정중앙에 빨간 점 찍기
for r in range(BOARD_ROWS):
    for c in range(BOARD_COLS):
        # 각 셀의 중심 좌표
        cx = int(bx1 + (c + 0.5) * CELL_SIZE)
        cy = int(by1 + (r + 0.5) * CELL_SIZE)
        cv2.circle(debug_img, (cx, cy), 1, (0, 0, 255), -1)

# 4) 우측 3개 슬롯 영역 표시 (파란색) 및 회전/반전 버튼 영역
for i, s_dy in enumerate(SLOT_Y_OFFSETS):
    sx1 = ax + SLOT_OFFSET_X
    sy1 = ay + s_dy
    sx2 = sx1 + SLOT_WIDTH
    sy2 = sy1 + SLOT_HEIGHT
    cv2.rectangle(debug_img, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)
    cv2.putText(debug_img, f"Slot {i+1}", (sx1 + 5, sy1 + 18), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

# 결과 저장
output_filename = "debug_anchor_result.png"
cv2.imwrite(output_filename, debug_img)
print(f"'{output_filename}' 저장 완료! 160칸 격자 중심점과 슬롯 테두리를 확인해보세요.")