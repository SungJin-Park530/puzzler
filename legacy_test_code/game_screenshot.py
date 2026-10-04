import cv2
import numpy as np

# 1. 스크린샷 로드
img = cv2.imread("game_screenshot.png")
if img is None:
    print("이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

h, w, _ = img.shape
print(f"이미지 해상도: {w}x{h}")

# ========================================================
# 2. 주요 UI 영역 대략적 비율 (실제 해상도에 맞춰 자동 스케일)
# 첨부해주신 이미지 비율 기준 근사치입니다.
# ========================================================

# [1] 10x16 보드 필드 영역 (좌표: x1, y1, x2, y2)
# 이미지의 좌측 큰 사각형 영역
board_box = [int(w * 0.03), int(h * 0.03), int(w * 0.66), int(h * 0.98)]

# [2] 우측 3개 조각 카드 슬롯 영역
slot_boxes = [
    [int(w * 0.70), int(h * 0.09), int(w * 0.98), int(h * 0.25)],  # 슬롯 1
    [int(w * 0.70), int(h * 0.26), int(w * 0.98), int(h * 0.42)],  # 슬롯 2
    [int(w * 0.70), int(h * 0.43), int(w * 0.98), int(h * 0.59)],  # 슬롯 3
]

# 디버그용 사각형 그리기
debug_img = img.copy()

# 보드 영역 (초록색)
bx1, by1, bx2, by2 = board_box
cv2.rectangle(debug_img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

# 10x16 격자 중심점 찍어보기
cell_w = (bx2 - bx1) / 10
cell_h = (by2 - by1) / 16

for r in range(16):
    for c in range(10):
        cx = int(bx1 + (c + 0.5) * cell_w)
        cy = int(by1 + (r + 0.5) * cell_h)
        cv2.circle(debug_img, (cx, cy), 2, (0, 0, 255), -1)

# 슬롯 영역 (파란색)
for i, sbox in enumerate(slot_boxes):
    sx1, sy1, sx2, sy2 = sbox
    cv2.rectangle(debug_img, (sx1, sy1), (sx2, sy2), (255, 0, 0), 2)

# 결과 이미지 저장 후 눈으로 확인
cv2.imwrite("debug_crop.png", debug_img)
print("debug_crop.png 저장 완료! 격자 중심점과 슬롯 테두리가 잘 맞는지 확인해보세요.")