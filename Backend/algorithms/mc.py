import numpy as np
from collections import defaultdict

class MCAgent:
    def __init__(self, n_states, n_actions, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.q_table = np.zeros((n_states, n_actions))
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, episode_data):
        G = 0
        visited_state_actions = set()
        
        for t in reversed(range(len(episode_data))):
            state, action, reward = episode_data[t]
            G = self.gamma * G + reward
            
            state_action = (state, action)
            if state_action not in visited_state_actions:
                self.returns_sum[state_action] += G
                self.returns_count[state_action] += 1
                self.q_table[state][action] = (
                    self.returns_sum[state_action] / self.returns_count[state_action]
                )
                visited_state_actions.add(state_action)

    def get_value_function(self):
        return np.max(self.q_table, axis=1)