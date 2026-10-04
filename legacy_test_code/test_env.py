# test_env.py
import time
from env_puzzle import Puzzle10x16Env

env = Puzzle10x16Env()
obs, _ = env.reset()

print("★ 가상 퍼즐 환경 동작 테스트 시작")
start_time = time.time()
episodes = 50
total_steps = 0

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        # 랜덤 액션 선택
        import random
        action = random.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        done = terminated or truncated

elapsed = time.time() - start_time
print(f"-> {episodes}게임 시뮬레이션 완료!")
print(f"-> 총 스텝 수: {total_steps}스텝 (소요 시간: {elapsed:.2f}초, 초당 {total_steps / elapsed:.0f}스텝)")

# 마지막 판 상태 출력
env.render()