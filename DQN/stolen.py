from logging import config
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
import racetrack_env
import highway_env


gym.register(
    id='custom-highway-v1',
    entry_point='racetrack_env:RacetrackEnvBello',
)


_config = {"action": {
    "type": "DiscreteAction",
    "longitudinal": True,
    

},
    "lane_centering_cost": 1.3,
    "action_reward": 3.0,
    "duration": 150,
    "off_road_penalty": -20,
    'collision_reward': -30,
    "other_vehicles": 9,
    "policy_frequency": 9,
}

TRAIN = True

if __name__ == '__main__':
    n_cpu = 8
    env = make_vec_env("custom-highway-v1", n_envs=n_cpu, env_kwargs=dict(config=_config), vec_env_cls=SubprocVecEnv)
    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=128,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1)
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=150000)
        model.save("racetrack_DQN/model")
        del model

    # Run the algorithm
    model = DQN.load("racetrack_DQN/model", env=env)

    env = gym.make("custom-highway-v1", render_mode='rgb_array', config=_config)
    env = RecordVideo(env, video_folder="racetrack_DQN/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)

    for video in range(5):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()
