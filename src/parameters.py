# src/parameters.py

import math
"""全局参数管理：加载设备参数，定义常量"""
import os
import pandas as pd

# ---------------------- 通用路径配置 ----------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录 （自动获取）
DATA_DIR = os.path.join(PROJECT_ROOT, "data")  # 数据目录
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")  # 结果目录  （存放仿真输出的表格、图表）

# 创建结果目录（若不存在）
os.makedirs(RESULTS_DIR, exist_ok=True)  # 如果results文件夹不存在，自动创建


# ---------------------- 固定参数 ----------------------------
TASK_PARAMS = {
    "I": 3,                  # AI-Agent设备数量
    "C": 1e6,                # 总计算量 (FLOPs)
    "D": 1e7,                # 输入数据量 (bits)
    "O": 2e6,                # 输出数据量 (bits)
    "B": 10e6,               # 总带宽 (Hz)
    "Pdown": 100e-3,         # 下行总功率 (W)
    "N0": 1e-18/10e6,        # 单位带宽噪声功率 (W/Hz)
    "w_over_v": 2,           # 公共/特定任务比例
    "alpha_path": 4,         # 路径损耗指数为4
    "a": 2 * math.log(2),    # 可靠性参数a（2ln2）
    "b": 50 * math.log(2),   # 可靠性参数b（50ln2）
}

# 修正后的设备参数（基于文档位置）
device_positions = [(6, 8), (-20, 0), (0, 30)]  # 文档给定位置
distances = [math.hypot(x, y) for x, y in device_positions]  # 计算距离（米）

# 设备参数
DEVICE_PARAMS = pd.DataFrame({
    "device_id": [1, 2, 3],
    "distance": distances,     # 正确距离
    "fi": [0.5e7, 1e7, 2e7],   # 算力 (FLOPs/s)
    "kappa": [1e-27] * 3,      # 能耗系数 (J·s²)
    "Piup": [50e-3] * 3        # 上行功率 (W)
})