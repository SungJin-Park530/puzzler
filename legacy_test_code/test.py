import time
from solver_core import solve_3_pieces

if __name__ == "__main__":
    # 빈 보드 생성 (16행 모두 0)
    test_board = [0] * 16
    
    # 테스트로 밑바닥 2개 줄을 8칸씩 채워둠 (2칸만 채우면 터지는 상태)
    test_board[14] = 0b1111111100  # 15번째 줄
    test_board[15] = 0b0011111111  # 16번째 줄
    
    # 첨부 이미지에 나온 조각들과 유사하게 테스트:
    # 1: 3칸 직선 [[111]]
    # 5: 계단 지그재그 [[11][01][11][10][11]]
    # 3: L자 [[10][11]]
    given_pieces = [1, 5, 3] 
    
    print("계산 시작...")
    start_t = time.time()
    best_score, plan = solve_3_pieces(test_board, markers=None, piece_indices=given_pieces)
    end_t = time.time()
    
    print(f"계산 완료! 소요 시간: {end_t - start_t:.3f}초")
    print(f"예상 휴리스틱 점수: {best_score:.1f}")
    
    if plan:
        for p in plan:
            slot = p["piece_slot"]
            h, w, rows = p["ori"]
            print(f"[{p['step']}단계] 지급 조각 {slot+1}번 사용 -> 위치: ({p['r']}행, {p['c']}열), 크기: ({h}x{w})")
    else:
        print("현재 보드에 3개 조각을 모두 배치할 수 있는 조합이 없습니다! (패배 위험/능력 사용 필요)")