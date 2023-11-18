import gymnasium as gym
from gymnasium import Space
import numpy as np

def pick_action(observation: np.ndarray, action_list: Space):
    return action_list.sample()

# Calculate the TD
def get_temporal_difference(reward, gamma, new_q_values, old_q_value):
    temporal_difference = reward + (gamma * np.max(new_q_values)) - old_q_value

def main():
    env = gym.make('racetrack-v0', render_mode='rgb_array')
    observation, info = env.reset()
    
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