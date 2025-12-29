import numpy as np

class TDAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.9, epsilon=0.1, n_step=1):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.n_step = n_step
        self.memory = [] 

    def get_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.n_actions)
        return np.argmax(self.q_table[state])

    def update_q_learning(self, s, a, r, s_next):
        target = r + self.gamma * np.max(self.q_table[s_next])
        self.q_table[s, a] += self.lr * (target - self.q_table[s, a])

    def update_sarsa(self, s, a, r, s_next, a_next):
        target = r + self.gamma * self.q_table[s_next, a_next]
        self.q_table[s, a] += self.lr * (target - self.q_table[s, a])

    def update_n_step(self, n_step_data):
        G = 0
        for i, (s, a, r) in enumerate(n_step_data):
            G += (self.gamma**i) * r

        last_s = n_step_data[-1][0]
        G += (self.gamma**len(n_step_data)) * np.max(self.q_table[last_s])
        
        s0, a0, _ = n_step_data[0]
        self.q_table[s0, a0] += self.lr * (G - self.q_table[s0, a0])