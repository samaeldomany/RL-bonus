import streamlit as st
import numpy as np
import time
import gymnasium as gym
import plotly.graph_objects as go
from Backend.envs.Gridworld import GridWorld
from Backend.algorithms.td import TDAgent
from Backend.algorithms.mc import MCAgent
from Backend.algorithms.dp import value_iteration, policy_iteration

st.set_page_config(page_title="RL Master Tool", layout="wide")

st.sidebar.title("🛠️ Global Configuration")
env_name = st.sidebar.selectbox("Environment", ["GridWorld", "FrozenLake-v1", "CartPole-v1", "MountainCar-v0"])
algo_name = st.sidebar.selectbox("Algorithm", ["Q-Learning", "SARSA", "n-step TD", "Monte Carlo", "Value Iteration", "Policy Iteration"])

n_step = st.sidebar.slider("n-step", 1, 10, 3, 
    help="Used in n-step TD. The agent looks 'n' steps ahead before updating its value estimate. Higher values bridge the gap between TD and Monte Carlo.")

gamma = st.sidebar.slider("Discount (γ)", 0.1, 1.0, 0.95, 
    help="Determines the importance of future rewards. 0 = short-sighted, 1 = far-sighted.")

alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 0.5, 0.1, 
    help="How much new information overrides old information.")

epsilon = st.sidebar.slider("Epsilon (ε)", 0.0, 1.0, 0.1, 
    help="The probability of choosing a random action (Exploration) vs. the best action (Exploitation).")

episodes = st.sidebar.number_input("Episodes", 10, 100, 50, 
    help="Total number of training cycles. More episodes usually lead to better convergence but take more time.")

def render_live_view(env, state, env_name):
    if "GridWorld" in env_name or "FrozenLake" in env_name:
        size = 5 if "GridWorld" in env_name else 4
        grid = np.full((size, size), "⬜")
        r, c = divmod(state, size)
        grid[size-1, size-1] = "🚩"
        grid[r, c] = "🤖"
        grid_html = "".join([f"<div>{''.join(row)}</div>" for row in grid])
        return f"<div style='font-size:30px; line-height:1;'>{grid_html}</div>"
    else:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=state[0] if isinstance(state, np.ndarray) else state))
        return fig

st.title(f"🚀 Training {algo_name} on {env_name}")
col1, col2 = st.columns([1, 1])
metrics_plot = col1.empty()
live_vis = col2.empty()

if st.button("Start Global Pipeline"):
    if env_name == "GridWorld": env = GridWorld(size=5)
    else: env = gym.make(env_name, render_mode="rgb_array")
    
    n_s = env.observation_space.n if hasattr(env.observation_space, 'n') else 100 
    n_a = env.action_space.n
    
    if algo_name == "Monte Carlo": agent = MCAgent(n_s, n_a, gamma, epsilon)
    elif algo_name == "Value Iteration":
        V = value_iteration(env, gamma)
    elif algo_name == "Policy Iteration":
        V, policy = policy_iteration(env, gamma) 
        st.plotly_chart(go.Figure(data=go.Heatmap(z=V.reshape(-1, int(np.sqrt(n_s))))))
        st.stop()
    else: agent = TDAgent(n_s, n_a, alpha, gamma, epsilon, n_step)

    rewards_history = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_r = 0
        step_memory = [] 
        
        while not done:
            obs = state if isinstance(state, (int, np.int64)) else 0 
            
            action = agent.get_action(obs)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            next_obs = next_state if isinstance(next_state, (int, np.int64)) else 0

            if algo_name == "Q-Learning":
                agent.update_q_learning(obs, action, reward, next_obs)
            elif algo_name == "SARSA":
                next_act = agent.get_action(next_obs)
                agent.update_sarsa(obs, action, reward, next_obs, next_act)
            elif algo_name == "n-step TD":
                step_memory.append((obs, action, reward))
                if len(step_memory) >= n_step:
                    agent.update_n_step(step_memory)
                    step_memory.pop(0)
            elif algo_name == "Monte Carlo":
                step_memory.append((obs, action, reward))
            
            state = next_state
            total_r += reward

            if ep > episodes - 5:
                res = render_live_view(env, state, env_name)
                if isinstance(res, str): live_vis.markdown(res, unsafe_allow_html=True)
                else: live_vis.plotly_chart(res, use_container_width=True)
                time.sleep(0.1)

        if algo_name == "Monte Carlo": agent.update(step_memory)
        rewards_history.append(total_r)
        if ep % 5 == 0: metrics_plot.line_chart(rewards_history)

    st.success("Target Objective Achieved.")