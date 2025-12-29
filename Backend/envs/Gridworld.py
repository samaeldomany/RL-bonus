import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridWorld(gym.Env):
    def __init__(self, size=5, step_reward=-1, goal_reward=10):
        super().__init__()
        self.size = size
        self.observation_space = spaces.Discrete(size * size)
        self.action_space = spaces.Discrete(4)
        self.goal_state = size * size - 1
        self.state = 0
        self.step_reward = step_reward
        self.goal_reward = goal_reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

    def step(self, action):
        row, col = divmod(self.state, self.size)
        if action == 0: row = max(0, row - 1)    
        elif action == 1: col = min(self.size - 1, col + 1) 
        elif action == 2: row = min(self.size - 1, row + 1) 
        elif action == 3: col = max(0, col - 1)  
        
        self.state = row * self.size + col
        terminated = self.state == self.goal_state
        reward = self.goal_reward if terminated else self.step_reward
        return self.state, reward, terminated, False, {}