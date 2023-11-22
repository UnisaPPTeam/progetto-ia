from torch import nn 
import torch
import numpy as np

device = 'cpu'

class DQN(nn.Module):
    def __init__(self, env):
        super().__init__()
        in_features = int(np.prod(env.observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Tanh(),
            nn.Linear(128, env.action_space.n)
        )
    
    def forward(self,x):
        return self.net(x)
    
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).reshape(1,288)
        q_values = self(obs_t)
        max_q_index = torch.argmax(q_values, dim = 1)[0]
        action = max_q_index.detach().item()
        return action