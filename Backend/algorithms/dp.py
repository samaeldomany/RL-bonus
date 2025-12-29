import numpy as np

def value_iteration(env, gamma=0.99, theta=1e-6):
    n_s = env.observation_space.n
    n_a = env.action_space.n
    V = np.zeros(n_s)
    
    while True:
        delta = 0
        for s in range(n_s):
            v = V[s]
            q_values = [-1 + gamma * V[min(n_s-1, s+1)] for a in range(n_a)]
            V[s] = max(q_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta: break
    return V

def policy_iteration(env, gamma=0.99, theta=1e-6):
    n_s = env.observation_space.n
    n_a = env.action_space.n
    V = np.zeros(n_s)
    policy = np.zeros(n_s, dtype=int) 

    def policy_evaluation(V, policy):
        while True:
            delta = 0
            for s in range(n_s):
                v = V[s]
                a = policy[s]
                V[s] = -1 + gamma * V[min(n_s-1, s+1)]
                delta = max(delta, abs(v - V[s]))
            if delta < theta: break
        return V

    while True:
        V = policy_evaluation(V, policy)
        policy_stable = True
        for s in range(n_s):
            old_action = policy[s]
            q_values = [-1 + gamma * V[min(n_s-1, s+1)] for a in range(n_a)]
            policy[s] = np.argmax(q_values)
            if old_action != policy[s]:
                policy_stable = False
        if policy_stable: break
    
    return V, policy