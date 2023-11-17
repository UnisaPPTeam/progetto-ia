import gymnasium as gym
from gymnasium import Space
import numpy

def pick_action(observation: numpy.ndarray, action_list: Space):
    return action_list.sample()


def main():
    env = gym.make('racetrack-v0', render_mode='rgb_array')
    observation, info = env.reset()

    for _ in range(1000):
        action_list = env.action_space
        action = pick_action(observation, action_list) # agent policy that uses the observation and info  
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    
    
if __name__ == '__main__':
    main()