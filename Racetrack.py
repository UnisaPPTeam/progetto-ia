import gymnasium as gym
from gymnasium import Space
import numpy as np

def pick_action(observation: np.ndarray, action_list: Space):
    return action_list.sample()

# Calculate the TD
def get_temporal_difference(reward, gamma, new_q_values, old_q_value):
    temporal_difference = reward + (gamma * np.max(new_q_values)) - old_q_value
    return temporal_difference

def main():
    env = gym.make('racetrack-v0', render_mode='rgb_array')
    # Set action space to discrete
    env.unwrapped.configure({
        "observation": {
            "type": "OccupancyGrid",
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
            "grid_step": [5, 5],
            "absolute": False,
        },
        "action": {
            "type": "DiscreteAction"
        }
    })
    observation, info = env.reset()
    action_size = env.action_space.n
    
    # define Qlearning hyperparams
    discount_factor = 0.9
    learning_rate = 0.7
    for _ in range(1000):
        # We need to preserve the old observation to be sure we can access the old q value
        old_observation = observation
        action_list = env.action_space
        action = pick_action(observation, action_list) # agent policy that uses the observation and info  
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    
    
if __name__ == '__main__':
    main()