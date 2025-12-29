import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Gym4ReaL(gym.Env):

    def __init__(self, size=7):
        super().__init__()
        self.size = size
        self.observation_space = spaces.Discrete(size * size)
        self.action_space = spaces.Discrete(4)
        self.state = 0
        self.goal = size * size - 1
        self.obstacles = [10, 11, 12, 25, 30] 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

    def step(self, action):
        row, col = divmod(self.state, self.size)
        
        if np.random.rand() < 0.05:
            action = np.random.randint(4)

        if action == 0: row = max(0, row - 1)    
        elif action == 1: col = min(self.size - 1, col + 1) 
        elif action == 2: row = min(self.size - 1, row + 1) 
        elif action == 3: col = max(0, col - 1) 

        new_state = row * self.size + col
        
        if new_state in self.obstacles:
            reward = -5
        else:
            self.state = new_state
            reward = 10 if self.state == self.goal else -1

        terminated = self.state == self.goal
        return self.state, reward, terminated, False, {}

    def render(self):
        img = np.zeros((350, 350, 3), dtype=np.uint8) + 255
        cell_size = 50
        
        for obs in self.obstacles:
            r, c = divmod(obs, self.size)
            img[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = [100, 100, 100]
            
        r, c = divmod(self.goal, self.size)
        img[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = [255, 215, 0]
        
        r, c = divmod(self.state, self.size)
        img[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = [255, 0, 0]
        
        return img