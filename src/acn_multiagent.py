# src/acn_multiagent.py

from pettingzoo.utils.env import AECEnv
from gymnasium import spaces
import numpy as np
from src.parameters import DEVICE_PARAMS, TASK_PARAMS
from src.models import calculate_transmission_metrics, calculate_total_metrics
import math

class ACNPettingZooEnv(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "acn_masac_v0",
        "is_parallelizable": True 
    }

    def __init__(self):
        super().__init__()
        self.render_mode = None
        self._num_agents = TASK_PARAMS["I"]
        self.agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.possible_agents = self.agents[:]

        self.device_params = DEVICE_PARAMS
        self.task_params = TASK_PARAMS
        self.B = TASK_PARAMS["B"]
        self.w_total = TASK_PARAMS["w_over_v"] / (TASK_PARAMS["w_over_v"] + 1)
        self.v_total = 1 / (TASK_PARAMS["w_over_v"] + 1)
        self.vi = self.v_total / self._num_agents
        self.max_wi = 1 - self.vi
        self.state_size = self._num_agents * 2 + 1
        
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=np.inf, shape=(self.state_size,), dtype=np.float32)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Box(low=np.array([0.0, 0.0]), high=np.array([self.max_wi, self.B]), dtype=np.float32)
            for agent in self.agents
        }

        self.state = None
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.reset()

    def observe(self, agent):
        return self.state.copy() if self.state is not None else np.zeros(self.state_size)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        d = self.device_params["distance"].values.astype(np.float32)
        fi = self.device_params["fi"].values.astype(np.float32)
        wv = np.array([self.task_params["w_over_v"]], dtype=np.float32)
        self.state = np.concatenate([d, fi, wv])
        self.actions = {}
        self._agent_selector = iter(self.agents)
        self.agent_selection = next(self._agent_selector)
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        return self.state

    def step(self, action):
        if self.agent_selection in self.agents:
            self.actions[self.agent_selection] = action
        try:
            self.agent_selection = next(self._agent_selector)
        except StopIteration:
            self._calculate_rewards()

    def _calculate_rewards(self):
        I = self._num_agents
        wi = np.array([self.actions.get(f"agent_{i}", [0, 0])[0] for i in range(I)])
        bi = np.array([self.actions.get(f"agent_{i}", [0, 0])[1] for i in range(I)])
        
        wi_sum = np.sum(wi)
        bi_sum = np.sum(bi)
        if wi_sum > 0:
            wi = wi * self.w_total / wi_sum
        if bi_sum > 0:
            bi = bi * self.B / bi_sum

        trans_metrics = calculate_transmission_metrics(wi, bi, self.device_params, self.task_params)
        T, E = calculate_total_metrics(trans_metrics)

        # 惩罚项：不均衡性（方差）+ 总和偏差（偏离目标值）
        penalty = (
            0.1 * np.var(wi) +
            0.1 * np.var(bi) +
            0.05 * (np.abs(np.sum(wi) - self.w_total) + np.abs(np.sum(bi) - self.B))
        )

        reward = - (0.5 * T + 0.5 * E + penalty)
        reward = np.log1p(-reward)  # log 奖励转换，避免巨大梯度

        for agent in self.possible_agents:
            self.rewards[agent] = reward
            self.terminations[agent] = True
            self._cumulative_rewards[agent] += reward

    def last(self, observe=True):
        agent = self.agent_selection
        observation = self.observe(agent) if observe else None
        return (
            observation,
            self._cumulative_rewards.get(agent, 0),
            self.terminations.get(agent, False),
            self.truncations.get(agent, False),
            self.infos.get(agent, {}),
        )
