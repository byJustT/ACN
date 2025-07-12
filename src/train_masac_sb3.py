# src/train_masac_sb3.py

import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces, Env
from src.acn_multiagent import ACNPettingZooEnv

class DirectEnv(Env):
    def __init__(self, aec_env):
        super().__init__()
        self.aec_env = aec_env
        self.num_agents = aec_env._num_agents
        self.state_size = aec_env.state_size
        self.action_space = spaces.Box(low=0.01, high=1.0, shape=(self.num_agents * 2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.state_size,), dtype=np.float32)
        self.epsilon = 0.01

    def reset(self, seed=None, options=None):
        obs = self.aec_env.reset()
        return obs, {}

    def step(self, action):
        action = np.clip(action + np.random.normal(0, 0.02, size=action.shape), 0.01, 1.0)
        actions = np.split(action, self.num_agents)
        self.aec_env.reset()
        for i, agent in enumerate(self.aec_env.agents):
            self.aec_env.step(actions[i])
        reward = self.aec_env.rewards.get(self.aec_env.agents[0], 0)
        done = all(self.aec_env.terminations.values())
        state = self.aec_env.state
        return state, reward, done, False, {}

def train_masac_model(aec_env, total_timesteps=50000):
    env = DummyVecEnv([lambda: DirectEnv(aec_env)])
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/",
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=256,
        learning_starts=1000,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[128, 128])
    )
    model.learn(total_timesteps=total_timesteps)
    return model

def main():
    aec_env = ACNPettingZooEnv()
    model = train_masac_model(aec_env, total_timesteps=100000)
    os.makedirs("results", exist_ok=True)
    model.save("results/sb3_masac_model")
    print("✅ 模型已保存至：results/sb3_masac_model.zip")

if __name__ == "__main__":
    main()
