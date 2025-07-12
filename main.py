# main.py

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到系统路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src import parameters, models
from src.train_masac_sb3 import train_masac_model
from src.evaluate_masac import evaluate_masac_model
from pettingzoo.utils.conversions import aec_to_parallel
from src.acn_multiagent import ACNPettingZooEnv

def run_baseline_simulation(device_params, task_params):
    """
    运行均匀分配（baseline）仿真
    """
    I = task_params["I"]
    w_total = task_params["w_over_v"] / (task_params["w_over_v"] + 1)
    wi_baseline = np.full(I, w_total / I)
    bi_baseline = np.full(I, task_params["B"] / I)

    trans_metrics = models.calculate_transmission_metrics(
        wi_baseline, bi_baseline, device_params, task_params
    )
    T_baseline, E_baseline = models.calculate_total_metrics(trans_metrics)

    return {
        "wi": wi_baseline,
        "bi": bi_baseline,
        "pi": trans_metrics["pi"],
        "T": T_baseline,
        "E": E_baseline
    }

def run_masac_simulation(device_params, task_params):
    """
    运行MASAC优化仿真
    """
    # 创建环境
    aec_env = ACNPettingZooEnv()
    aec_env.device_params = device_params
    aec_env.task_params = task_params
    
    # 训练模型（如果不存在）
    model_path = "results/sb3_masac_model"
    if not os.path.exists(model_path + ".zip"):
        print("训练MASAC模型...")
        model = train_masac_model(aec_env)
        model.save(model_path)
    
    # 评估模型
    return evaluate_masac_model(device_params, task_params, model_path)

def save_results(df, filename):
    """保存结果到CSV文件"""
    filepath = os.path.join(parameters.RESULTS_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"结果已保存至：{filepath}")

def plot_wv_simulation(results_df):
    """绘制w/v比例与总时延、总能耗的关系图"""
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["w/v"], results_df["总时延 (s)"], "o-", color="blue", label="总时延")
    plt.xlabel("公共/特定任务比例 (w/v)")
    plt.ylabel("总时延 (s)")
    plt.title("w/v比例对总时延的影响")
    plt.legend()
    plt.grid(True)
    delay_filepath = os.path.join(parameters.RESULTS_DIR, "wv_simulation_delay.png")
    plt.savefig(delay_filepath, dpi=300, bbox_inches="tight")
    print(f"总时延仿真图已保存至：{delay_filepath}")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(results_df["w/v"], results_df["总能耗 (J)"] * 1e9, "s-", color="orange", label="总能耗")
    plt.xlabel("公共/特定任务比例 (w/v)")
    plt.ylabel("总能耗 (nJ)")
    plt.title("w/v比例对总能耗的影响")
    plt.legend()
    plt.grid(True)
    energy_filepath = os.path.join(parameters.RESULTS_DIR, "wv_simulation_energy.png")
    plt.savefig(energy_filepath, dpi=300, bbox_inches="tight")
    print(f"总能耗仿真图已保存至：{energy_filepath}")
    plt.close()

def run_wv_simulation(device_params, task_params, wv_values):
    """
    运行不同w/v比例的仿真
    """
    results = []
    for wv in wv_values:
        task_params["w_over_v"] = wv
        masac_result = run_masac_simulation(device_params, task_params)
        results.append({
            "w/v": wv,
            "总时延 (s)": masac_result["T"],
            "总能耗 (J)": masac_result["E"]
        })
    return pd.DataFrame(results)

def create_results_table(result, title):
    """创建结果表格"""
    I = len(result["wi"])
    df = pd.DataFrame({
        "指标": ["公共任务比例", "带宽分配 (MHz)", "下行功率 (mW)", "总时延 (s)", "总能耗 (J)"],
        "设备1": [
            f"{result['wi'][0]:.4f}",
            f"{result['bi'][0]/1e6:.2f}",
            f"{result['pi'][0]*1e3:.2f}",
            f"{result['T']:.6f}",
            f"{result['E']:.10f}"
        ]
    })
    
    for i in range(1, I):
        df[f"设备{i+1}"] = [
            f"{result['wi'][i]:.4f}",
            f"{result['bi'][i]/1e6:.2f}",
            f"{result['pi'][i]*1e3:.2f}",
            "-",
            "-"
        ]
    return df

if __name__ == "__main__":
    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 加载参数
    device_params = parameters.DEVICE_PARAMS
    task_params = parameters.TASK_PARAMS.copy()
    
    print("===== 运行Baseline（均匀分配）仿真 =====")
    baseline_result = run_baseline_simulation(device_params, task_params)
    baseline_df = create_results_table(baseline_result, "Baseline")
    save_results(baseline_df, "baseline_results.csv")
    print(baseline_df)
    
    print("\n===== 运行MASAC优化仿真 =====")
    masac_result = run_masac_simulation(device_params, task_params)
    masac_df = create_results_table(masac_result, "MASAC优化")
    save_results(masac_df, "masac_results.csv")
    print(masac_df)
    
    # 比较结果
    print("\n===== 结果比较 =====")
    improvement_df = pd.DataFrame({
        "指标": ["总时延 (s)", "总能耗 (J)"],
        "Baseline": [f"{baseline_result['T']:.6f}", f"{baseline_result['E']:.10f}"],
        "MASAC优化": [f"{masac_result['T']:.6f}", f"{masac_result['E']:.10f}"],
        "改进": [
            f"{(baseline_result['T'] - masac_result['T'])/baseline_result['T']*100:.2f}%",
            f"{(baseline_result['E'] - masac_result['E'])/baseline_result['E']*100:.2f}%"
        ]
    })
    save_results(improvement_df, "comparison_results.csv")
    print(improvement_df)
    
    print("\n===== 运行w/v比例仿真 =====")
    wv_test_values = np.arange(0.5, 5.5, 0.5).tolist()
    wv_results_df = run_wv_simulation(device_params, task_params, wv_test_values)
    save_results(wv_results_df, "wv_simulation_results.csv")
    plot_wv_simulation(wv_results_df)
    
    print("\n===== 所有仿真完成 =====")