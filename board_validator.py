# board_validator.py
from typing import List, Tuple, Optional, Dict, Any
from solver_core import PROCESSED_PIECES, FULL_ROW, get_valid_placements, apply_placement

def find_actual_placement(
    mem_board: List[int],
    scanned_board: List[int],
    piece_id: int
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    사라진 조각(piece_id)을 mem_board의 어디에 놓아야 scanned_board가 되는지 역추적.
    반환값: (상태코드, 배치정보딕셔너리)
    상태코드:
      - "MATCH": 단 하나의 명확한 착수점 발견 (자동 보정)
      - "AMBIGUOUS": 1칸짜리 조각 + 능력 출현 등으로 후보가 여러 개이거나 설명 불가 (사용자 확인 필요)
      - "MISMATCH": 해당 조각으로 만들 수 없는 비정상 필드 변화
    """
    orientations = PROCESSED_PIECES[piece_id]["orientations"]
    matched_candidates = []

    for ori in orientations:
        for r, c, shift in get_valid_placements(mem_board, ori):
            sim_board, _, _, _ = apply_placement(mem_board, None, ori, r, shift)
            if sim_board == scanned_board:
                matched_candidates.append({
                    "r": r,
                    "c": c,
                    "ori": ori
                })

    if len(matched_candidates) == 1:
        return "MATCH", matched_candidates[0]

    # 1칸짜리 조각인데 후보가 여럿 나오거나, 능력이 동시에 떠서 완전 일치 실패 시
    if len(matched_candidates) > 1:
        return "AMBIGUOUS", {"candidates": matched_candidates}

    return "MISMATCH", None