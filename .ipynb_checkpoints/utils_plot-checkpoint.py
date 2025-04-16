import base64
import glob
import io
import os
import re

import numpy as np
import pandas as pd
import random
from collections import defaultdict

from IPython.display import HTML
from IPython import display
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium import spaces

import minigrid
import minigrid.envs
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

max_env_steps = 50

# 실행 시 최초 한 번 비디오 폴더 정리
def clear_video_folder(folder_path='./video'):
    if os.path.exists(folder_path):
        files = glob.glob(os.path.join(folder_path, '*'))
        for f in files:
            os.remove(f)
    else:
        os.makedirs(folder_path)

def plot_q_value_distribution(agent, title_suffix="Q-value Distribution"):
    q_values = agent.q_values

    # Q값 펼치기
    all_qs = []
    for q_list in q_values.values():
        all_qs.extend(q_list)
    all_qs = np.array(all_qs)

    # 알고리즘 이름 감지
    algorithm_name = agent.__class__.__name__.lower()
    if algorithm_name == "qlearning":
        algo_title = "Q-Learning"
    elif algorithm_name == "sarsa":
        algo_title = "SARSA"
    else:
        algo_title = algorithm_name.capitalize()

    # epsilon과 gamma 값 읽기 (없으면 생략)
    epsilon_str = f"ε = {getattr(agent, 'epsilon', '?'):.2f}" if hasattr(agent, 'epsilon') else ""
    gamma_str = f"γ = {getattr(agent, 'gamma', '?'):.2f}" if hasattr(agent, 'gamma') else ""

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.hist(all_qs, bins=30, color="skyblue", edgecolor="black")
    plt.xlabel("Q-value")
    plt.ylabel("Frequency")
    plt.title(f"{algo_title} - {title_suffix}")
    plt.grid(True)

    # 좌측 상단 텍스트로 epsilon, gamma 표시
    text_str = f"{epsilon_str}\n{gamma_str}"
    plt.text(0.02, 0.95, text_str, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

    plt.show()

# rewards_log: 보상 로그를 저장하는 함수
def save_rewards_log(rewards, variable_name, agent, folder="./logs"):
    os.makedirs(folder, exist_ok=True)

    if variable_name not in ["epsilon", "gamma"]:
        raise ValueError("variable_name must be 'epsilon' or 'gamma'")

    # 알고리즘 이름 추출 (예: QLearning, SARSA)
    algorithm = agent.__class__.__name__.lower()  # 'qlearning' or 'sarsa'

    # epsilon 또는 gamma 값 가져오기
    variable_value = getattr(agent, variable_name)

    # 값 문자열 포맷 (0.10 → 0_10)
    var_str = f"{variable_value:.2f}".replace(".", "_")
    filename = f"{algorithm}_rewards_{variable_name}_{var_str}.csv"
    path = os.path.join(folder, filename)

    pd.Series(rewards).to_csv(path, index=False)
    print(f"Saved to: {path}")
    return path

def plot_avgreward(variable_name, algorithm, folder="./logs"):
    if variable_name not in ["epsilon", "gamma"]:
        raise ValueError("variable_name must be 'epsilon' or 'gamma'")

    # 입력된 algorithm 문자열을 파일명용 형식으로 변환
    algorithm_key = algorithm.lower().replace("-", "").replace(" ", "")  # 예: "Q-Learning" → "qlearning"

    # 그래프 제목에 표시할 포맷
    def format_algorithm_title(name):
        if name == "qlearning":
            return "Q-Learning"
        elif name == "sarsa":
            return "SARSA"
        else:
            return name.capitalize()

    # 해당 알고리즘 + 조작변인에 해당하는 파일만 수집
    pattern = os.path.join(folder, f"{algorithm_key}_rewards_{variable_name}_*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"No CSV files found for {algorithm} with variable '{variable_name}'")
        return

    algo_title = format_algorithm_title(algorithm_key)

    plt.figure(figsize=(12, 6))
    cmap = plt.get_cmap("viridis")
    n = len(csv_files)

    for idx, file in enumerate(csv_files):
        base = os.path.basename(file)

        # 정규식으로 변수 값 추출 (예: 0_90 → 0.90)
        match = re.search(f"{variable_name}_(\\d+_\\d+)", base)
        if not match:
            print(f"Skipped file (invalid name format): {base}")
            continue

        var_value_str = match.group(1)
        var_value = float(var_value_str.replace("_", "."))

        rewards = pd.read_csv(file).iloc[:, 0]
        avg_rewards = rewards.cumsum() / np.arange(1, len(rewards) + 1)

        color = cmap(idx / max(n - 1, 1))

        plt.plot(
            avg_rewards,
            label=f"{variable_name}={var_value:.2f}",
            color=color,
            linestyle="-"
        )

    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"{algo_title} - Avg Reward by {variable_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_algorithms(variable_name, variable_value, folder="./logs"):
    if variable_name not in ["epsilon", "gamma"]:
        raise ValueError("variable_name must be 'epsilon' or 'gamma'")

    # 값 문자열 포맷 (0.10 → 0_10)
    var_str = f"{variable_value:.2f}".replace(".", "_")

    algorithms = ["qlearning", "sarsa"]
    color_map = {
        "qlearning": "blue",
        "sarsa": "red"
    }

    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        filename = f"{algo}_rewards_{variable_name}_{var_str}.csv"
        path = os.path.join(folder, filename)

        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue

        rewards = pd.read_csv(path).iloc[:, 0]
        avg_rewards = rewards.cumsum() / np.arange(1, len(rewards) + 1)

        algo_label = "Q-Learning" if algo == "qlearning" else "SARSA"

        plt.plot(
            avg_rewards,
            label=f"{algo_label}",
            color=color_map[algo],
            linestyle="-"
        )

    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"Algorithm Comparison at {variable_name} = {variable_value:.2f}")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def policy_heatmap(q_values, grid_size=6, goal_pos=(5, 5), title="Policy Map"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, grid_size + 1, 1))
    ax.set_yticks(np.arange(0, grid_size + 1, 1))
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_title(title)

    # 방향 벡터 (turn left, turn right, move forward)
    action_dirs = {
        0: (-1, 0),  # ←
        1: (1, 0),   # →
        2: (0, -1),  # ↑
    }

    # 바탕 칠하기 (벽은 회색, goal은 연두색)
    for y in range(grid_size):
        for x in range(grid_size):
            # 벽: 가장자리
            if x == 0 or x == grid_size - 1 or y == 0 or y == grid_size - 1:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color="lightgray"))
            # goal: 연두색
            elif (x, y) == goal_pos:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color="lightgreen"))

    # Q-table에서 최적 행동 방향 그리기 (내부 5x5)
    for state, q in q_values.items():
        best_action = int(np.argmax(q))
        dx, dy = action_dirs.get(best_action, (0, 0))

        x = state % grid_size
        y = state // grid_size

        # 내부 영역에만 화살표 그림
        if 1 <= x <= grid_size - 2 and 1 <= y <= grid_size - 2:
            ax.arrow(
                x + 0.5, y + 0.5,
                dx * 0.3, dy * 0.3,
                head_width=0.2,
                head_length=0.2,
                fc='black',
                ec='black'
            )

    plt.show()