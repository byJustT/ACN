# 多智能体资源分配优化项目

[![Python 3.10](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable_Baselines3-2.3.0-red)](https://stable-baselines3.readthedocs.io/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-1.24.3-green)](https://pettingzoo.farama.org/)

> 基于MASAC强化学习的AI设备协同网络资源分配方案

## 项目背景

本项目探索多智能体强化学习在边缘计算资源分配中的应用。针对3个AI设备的协同网络，设计MASAC（Multi-Agent Soft Actor-Critic）算法优化任务量分配(wi)和带宽分配(bi)，目标是最小化系统总时延和能耗。

## 当前问题

**优化结果异常**：  
训练后的MASAC模型分配方案效果显著差于均匀分配方案：
- 🟢 均匀分配：时延120秒，能耗5焦耳
- 🔴 MASAC分配：时延12万秒，能耗173焦耳

## 🔍 核心疑问与问题诊断

### 1. 环境状态转移逻辑
- **可疑代码**：`src/train_masac_sb3.py`中的`DirectEnv.step()`方法
- **具体问题**：
  ```python
  def step(self, action):
      action = np.clip(action + np.random.normal(0, 0.02, size=action.shape), 0.01, 1.0)
      actions = np.split(action, self.num_agents)
      self.aec_env.reset()  # ⚠️ 每次step都调用reset()是否合理？
      for i, agent in enumerate(self.aec_env.agents):
          self.aec_env.step(actions[i])
  ```
- **疑问**：每次step都调用reset()是否会导致环境状态被错误重置？

### 2. 奖励函数设计
- **可疑代码**：`src/acn_multiagent.py`中的`_calculate_rewards()`方法
- **具体问题**：
  ```python
  def _calculate_rewards(self):
      # 计算总时延T和总能耗E
      ...
      # 惩罚项：不均衡性（方差）+ 总和偏差
      penalty = (
          0.1 * np.var(wi) +
          0.1 * np.var(bi) +
          0.05 * (np.abs(np.sum(wi) - self.w_total) + np.abs(np.sum(bi) - self.B))
      )
      reward = - (0.5 * T + 0.5 * E + penalty)
      reward = np.log1p(-reward)  # ⚠️ 对数转换是否合理？
  ```
- **疑问**：
  1. 奖励函数使用负值（-T-E）是否会导致模型学习到错误方向？
  2. 对数转换`np.log1p(-reward)`在reward为负值时可能产生NaN或异常值

### 3. 多智能体协同机制
- **可疑代码**：`src/acn_multiagent.py`中的`step()`和`_calculate_rewards()`方法
- **具体问题**：
  ```python
  def step(self, action):
      # 所有智能体执行完动作后才计算奖励
      ...
      
  def _calculate_rewards(self):
      # 所有智能体获得相同的全局奖励
      for agent in self.possible_agents:
          self.rewards[agent] = reward  # ⚠️ 相同奖励是否合理？
  ```
- **疑问**：
  1. 所有智能体获得相同奖励是否削弱了多智能体特性？
  2. 是否存在"搭便车"问题—部分智能体贡献小但获得相同奖励？

### 4. 动作归一化处理
- **可疑代码**：`src/evaluate_masac.py`中的动作后处理
- **具体问题**：
  ```python
  # 动作归一化处理
  wi = wi * w_total / wi_sum  # ⚠️ 是否与训练时一致？
  bi = bi * task_params["B"] / bi_sum
  ```
- **疑问**：这种后处理是否与训练时的动作空间不一致？

## 运行环境

### 依赖安装
```bash
git clone https://github.com/yourusername/acn-resource-allocation.git
cd acn-resource-allocation
pip install -r requirements.txt
```

### 主要依赖
```
gymnasium==0.29.1
stable-baselines3==2.3.0
pettingzoo==1.24.3
pandas==2.2.1
matplotlib==3.8.3
numpy==1.26.4
```

## 项目结构

```
acn-resource-allocation/
├── src/                     # 核心源代码
│   ├── acn_multiagent.py    # 多智能体环境实现（重点检查）
│   ├── train_masac_sb3.py   # MASAC训练脚本（重点检查）
│   ├── evaluate_masac.py    # 模型评估与可视化
│   ├── parameters.py        # 全局参数配置
│   └── models.py            # 时延/能耗计算模型
├── data/                    # 设备参数数据
├── results/                 # 输出结果目录
│   ├── masac_actions.csv    # 动作分配记录
│   └── masac_actions_bar.png # 资源分配可视化
├── main.py                  # 主运行入口
├── requirements.txt         # 依赖列表
└── README.md                # 项目说明
```

## 运行流程

### 完整执行
```bash
python main.py
```

### 执行流程
1. 运行Baseline（均匀分配）仿真
2. 训练并评估MASAC模型
3. 对比两种方案的性能差异
4. 进行w/v比例敏感性分析
5. 生成结果图表和CSV文件



