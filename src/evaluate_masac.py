# ✅ evaluate_masac.py：评估 + 可视化 + 分析工具

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from src.acn_multiagent import ACNPettingZooEnv
from src.models import calculate_transmission_metrics, calculate_total_metrics
import pandas as pd
import os

def evaluate_masac_model(device_params, task_params, model_path="results/sb3_masac_model"):
    aec_env = ACNPettingZooEnv()
    aec_env.device_params = device_params
    aec_env.task_params = task_params

    obs = aec_env.reset()
    model = SAC.load(model_path)
    action, _ = model.predict(obs)

    I = task_params["I"]
    actions = np.split(action, I)
    wi = np.array([a[0] for a in actions])
    bi = np.array([a[1] for a in actions])

    # clip 并归一化
    epsilon = 0.01
    wi = np.clip(wi, epsilon, 1.0)
    bi = np.clip(bi, epsilon, 1.0)

    w_total = task_params["w_over_v"] / (task_params["w_over_v"] + 1)
    wi_sum = np.sum(wi)
    bi_sum = np.sum(bi)
    wi = wi * w_total / wi_sum
    bi = bi * task_params["B"] / bi_sum

    trans_metrics = calculate_transmission_metrics(wi, bi, device_params, task_params)
    T, E = calculate_total_metrics(trans_metrics)

    # ✅ 保存动作记录
    actions_df = pd.DataFrame({
        "Agent": [f"Agent-{i+1}" for i in range(I)],
        "wi (task ratio)": wi,
        "bi (bandwidth)": bi / 1e6,
        "pi (power, mW)": trans_metrics["pi"] * 1e3
    })
    actions_path = os.path.join("results", "masac_actions.csv")
    actions_df.to_csv(actions_path, index=False)
    print(f"✅ 动作记录已保存至: {actions_path}")

    # ✅ 可视化动作柱状图
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    agent_labels = [f"A{i+1}" for i in range(I)]

    ax[0].bar(agent_labels, wi, color='skyblue')
    ax[0].set_title("任务分配 wi")
    ax[0].set_ylim(0, 1)

    ax[1].bar(agent_labels, bi / 1e6, color='orange')
    ax[1].set_title("带宽分配 bi (MHz)")

    ax[2].bar(agent_labels, trans_metrics["pi"] * 1e3, color='green')
    ax[2].set_title("功率分配 pi (mW)")

    for a in ax:
        a.set_xlabel("智能体")
        a.set_ylabel("值")
        a.grid(True)

    plt.tight_layout()
    plt.savefig("results/masac_actions_bar.png", dpi=300)
    plt.close()
    print("✅ 动作分配柱状图已保存至: results/masac_actions_bar.png")

    return {
        "wi": wi,
        "bi": bi,
        "pi": trans_metrics["pi"],
        "T": T,
        "E": E
    }

if __name__ == "__main__":
    from src.parameters import DEVICE_PARAMS, TASK_PARAMS
    result = evaluate_masac_model(DEVICE_PARAMS, TASK_PARAMS)
    print("MASAC优化结果:")
    print(f"任务分配: {result['wi']}")
    print(f"带宽分配: {result['bi']}")
    print(f"功率分配: {result['pi']}")
    print(f"总时延: {result['T']:.6f}s")
    print(f"总能耗: {result['E']:.10f}J")
