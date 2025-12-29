import gymnasium as gym

def get_gym_env(env_name):
    if "Breakout" in env_name:
        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    else:
        env = gym.make(env_name, render_mode="rgb_array")
    return env

def get_env_metadata(env_name):
    env = get_gym_env(env_name)
    metadata = {
        "action_space": str(env.action_space),
        "observation_space": str(env.observation_space)
    }
    env.close()
    return metadata