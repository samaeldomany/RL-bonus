import gymnasium as gym

class FrozenLake:
    def __init__(self, size=4, is_slippery=True):
        self.env = gym.make("FrozenLake-v1", 
                             map_name="4x4" if size == 4 else "8x8",
                             is_slippery=is_slippery)
        self.actions = [0, 1, 2, 3]  
        self.start_pos = (0, 0)

    def reset(self):
        return self.env.reset()

    def get_transitions(self, state, action):
        next_state, reward, done, info = self.env.step(action)
        return [(next_state, reward, done)]