# Reinforcement Learning Assignment #1
### SARSA, Q-Learning 실험  
+ **목표** : Q-Learning 및 SARSA 학습에서 사용하는 탐색 정책 ($\epsilon$-greedy)의 차이를 실험을 통해 확인하고 비교 분석한다.
+ **내용**
   - 동일한 환경에서 $\epsilon$-greedy policy를 적용한 Q-Learning, SARSA 알고리즘 각각 구현
   - Q-value 수렴 과정 시각화 (plot 포함)
   - episode reward, 성공률 등을 기준으로 성능 비교

## Requirement

| 패키지 이름        | 버전      | 설명 (선택적으로 기입)       |
|-------------------|-----------|-----------------------------|
| gymnasium         | 0.28.1    | 강화학습 환경 (Gym API)      |
| minigrid          | 3.0.0     | MiniGrid 환경               |
| box2d-py          | 2.3.5     | Box2D 물리 시뮬레이션        |
| pygame            | 2.6.1     | 게임 환경 지원 (렌더링 등)   |
| swig              | 4.3.0     | C/C++ 바인딩 도구            |
| numpy             | 2.0.2     | 수치 계산 라이브러리         |
| pandas            | 2.2.3     | 데이터프레임 처리             |
| matplotlib        | 3.9.4     | 시각화 도구                  |
| imageio-ffmpeg    | 0.6.0     | 영상 인코딩/디코딩           |
| moviepy           | 2.1.2     | 비디오 편집 도구             |
| jupyter           | 1.1.1     | 주피터 노트북                |
| ipykernel         | 6.29.5    | 주피터 커널 실행 환경         |

## SARSA
+ On-policy 방식
+ $\alpha$, $\gamma$를 각각 0.01, 0.1로 고정했을 때 epsilon에 따른 reward의 평균값 그래프
<img width="1037" alt="Image" src="https://github.com/user-attachments/assets/88bb2c64-e840-499e-a01b-4b23c6bf0b5b" /><br><br>

+ $\alpha$, $\epsilon$를 각각 0.01, 0.3로 고정했을 때 discount factor($\gamma$)에 따른 reward의 평균값 그래프
<img width="1044" alt="Image" src="https://github.com/user-attachments/assets/09d8464b-e686-431d-933a-d1c5524da61d" /><br><br>

## Q-Learning
+ Off-policy 방식
+ $\alpha$, $\gamma$를 각각 0.01, 0.1로 고정했을 때 epsilon에 따른 reward의 평균값 그래프
<img width="1033" alt="Image" src="https://github.com/user-attachments/assets/859b73ef-d5c1-4769-a4c0-ad5bb8a1b231" /><br><br>

+ $\alpha$, $\epsilon$를 각각 0.01, 0.3로 고정했을 때 discount factor($\gamma$)에 따른 reward의 평균값 그래프
<img width="1029" alt="Image" src="https://github.com/user-attachments/assets/9f2113b6-77f6-4989-ae03-209e2886a3ac" /><br><br>

+ epsilon=0.1로 학습을 시켰을 때 제자리에서 회전만 하는 문제 발생 <br><br>
![Image](https://github.com/user-attachments/assets/d7941065-4b3e-4130-a4f8-e84c4bd30ee0)

## Comparsion
+ discount factor에 따른 reward 평균값 변화양상 비교 (SARSA, Q-Learning)
<img width="953" alt="Image" src="https://github.com/user-attachments/assets/1109fcc0-73b7-4874-a916-0c281d87cd5c" /><br><br>

+ SARSA Q-value 분포표
<img width="932" alt="Image" src="https://github.com/user-attachments/assets/a9fba647-7c30-4b81-86f9-11b4e0571fda" /><br><br>

+ Q-Learning Q-value 분포표
<img width="937" alt="Image" src="https://github.com/user-attachments/assets/feabfb67-abc3-4532-add1-21af683185e2" /><br><br>
