import itertools
from logging import config
import random
from torch import nn 
import torch 
import gymnasium as gym
from collections import deque
import numpy as np
from DQN import DQN

BUFFER_SIZE = 50000
GAMMA = 0.99
BATCH_SIZE = 32
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END= 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000
SCORE_BREAKPOINT = 375

_config = { "action": {
                            "type": "DiscreteAction",
                            "lateral": True,
                            "actions_per_axis": 5,
                                    },
                        }


def train(env, replay_buffer):
    device = 'cpu'
    print(device)
    episode_reward = 0.0
    rew_buffer = deque([0,0], maxlen=100)
    online_net = DQN(env)
    target_net = DQN(env)
    target_net.load_state_dict(online_net.state_dict())
    optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)
    obs, _ = env.reset()
    for step in itertools.count():
        # follow an epsilon-greedy policy, given an epsilon the changes of picking the optimal move (exploitation) will be 1-epsilon, the chances of taking a random move (exploration) will be epsilon 
        epsilon = np.interp(step,[0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        rnd_sample = random.random()
        if rnd_sample <= epsilon:
            action = env.action_space.sample()
        else:
            action = online_net.act(obs)
        # Execute the action and populate the replay buffer
        new_obs, rew, terminated, truncated, info = env.step(action)
        transition = (obs, action, rew, terminated, truncated, new_obs)
        replay_buffer.append(transition)
        obs = new_obs
        episode_reward += rew
        if terminated or truncated:
            obs, _ = env.reset()
            rew_buffer.append(episode_reward)
            episode_reward = 0.0
        
        if np.mean(rew_buffer) >= SCORE_BREAKPOINT:
            break

        # Start gradient step
        transitions = random.sample(replay_buffer, BATCH_SIZE)

        obses = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        terminateds = np.asarray([t[3] for t in transitions])
        truncateds = np.asarray([t[4] for t in transitions])
        new_obses = np.asarray([t[5] for t in transitions])

        obses_t = torch.as_tensor(obses, dtype=torch.float32).reshape(32,288)
        actions_t = torch.as_tensor(actions,dtype=torch.int64).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
        terminateds_t = torch.as_tensor(terminateds, dtype=torch.float32).unsqueeze(-1)
        truncateds_t = torch.as_tensor(truncateds, dtype=torch.float32).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32).reshape(32,288)
        # Compute targets 
        target_q_values = target_net(new_obses_t)
        max_target_q_value = target_q_values.max(dim = 1, keepdim = True)[0]
        
        targets = rews_t + GAMMA * (1 - terminateds_t - truncateds_t) * max_target_q_value
        # Compute Loss
        q_values = online_net(obses_t)
        action_q_values = torch.gather(input=q_values,dim=1,index=actions_t)
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)
        # GD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update target network
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())
            
        if step % 1000 == 0:
            print()
            print('Step', step)
            print('Avg Rew ', np.mean(rew_buffer))
    
    return online_net


def showcaseAgent(env, online_net):
    while True:
        obs, info = env.reset()
        total_reward = 0
        action = online_net.act(obs)
        done = truncated = False
        while not (done or truncated):
            env.render()
            action= online_net.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        print(total_reward)
    

def initReplayBuffer(env):
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    obs, _ = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        # Make a random action
        action = env.action_space.sample()
        new_obs, rew, terminated, truncated, info = env.step(action)
        # Populate the replay buffer with this transition
        transition = (obs, action, rew, terminated, truncated, new_obs)
        replay_buffer.append(transition)
        obs = new_obs
        if terminated or truncated:
            obs, _ = env.reset()
    return replay_buffer
    
def main():
    env = gym.make('racetrack-v0', render_mode="rgb_array", config = _config)
    replay_buffer = initReplayBuffer(env)
    print("Replay buffer init done")
    trained_net = train(env, replay_buffer)
    env = gym.make('racetrack-v0', render_mode="human", config = _config)
    showcaseAgent(env, online_net = trained_net)
        

if __name__ == '__main__':
    main()

        

            

