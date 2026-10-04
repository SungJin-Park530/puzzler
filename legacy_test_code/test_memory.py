# test_memory.py
from game_memory import GameMemory
from solver_core import PROCESSED_PIECES

mem = GameMemory()
mem.reset()

# 조각 18번(5칸 일자)의 회전 목록 중 너비가 5인 '가로' 형태 선택
piece_5_horizontal = next(ori for ori in PROCESSED_PIECES[18]["orientations"] if ori[1] == 5)

print("1. 좌측 5칸 배치 (15행 0열):")
mem.place_piece(piece_5_horizontal, r=15, c=0, slot_num=1)
mem.print_ascii_board()

print("2. 우측 5칸 배치 (15행 5열 -> 한 줄 가득 차서 즉시 삭제):")
mem.place_piece(piece_5_horizontal, r=15, c=5, slot_num=2)
mem.print_ascii_board()

print("3. Undo(실행 취소) 테스트:")
mem.undo()
mem.print_ascii_board()