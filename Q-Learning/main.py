import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_Q_Table(action_count):
    qtable = np.zeros((2, FLOAT_QUANTIZATION, FLOAT_QUANTIZATION, FLOAT_QUANTIZATION, FLOAT_QUANTIZATION,
                       ANGLE_QUANTIZATION, FLOAT_QUANTIZATION,
                       FLOAT_QUANTIZATION, FLOAT_QUANTIZATION, ANGLE_QUANTIZATION, action_count))
    return qtable


def epsilon_greedy_policy(env, Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(env.action_space.n)
    else:
        return np.argmax(Q[state, :])


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# presence 2
# x 4
# y 4
# vx 4
# vy 4
# heading 8
# long_off 4
# lat_off 4
# ang:off 8

ANGLE_QUANTIZATION = 8
FLOAT_QUANTIZATION = 5


def approssima_angolo(angolo):
    # Calcola l'angolo più vicino che è un multiplo di π/4
    angolo_approssimato = (round(angolo / (np.pi / (ANGLE_QUANTIZATION // 2)))) % ANGLE_QUANTIZATION
    return angolo_approssimato


def quantize_float(value):
    # Calculate the step size
    step_size = 2.0 / FLOAT_QUANTIZATION

    # Quantize the value to the nearest step
    quantized_value = round(value / step_size)

    return quantized_value + (FLOAT_QUANTIZATION // 2)


# RENDO OGNI ELEMENTO DELL'array un intero
# tutti i float da -1 a 1 diventano numeri da 0 a 4
# tutti gli angoli diventano numeri da 0 a 7

def process_observation(observation: list[float]):
    new_observation = [0] * len(observation)
    new_observation[0] = int(observation[0])
    new_observation[1] = quantize_float(observation[1])
    new_observation[2] = quantize_float(observation[2])
    new_observation[3] = quantize_float(observation[3])
    new_observation[4] = quantize_float(observation[4])
    new_observation[5] = approssima_angolo(observation[5])
    new_observation[6] = quantize_float(observation[6])
    new_observation[7] = quantize_float(observation[7])
    new_observation[8] = approssima_angolo(observation[8])
    return new_observation


def main():
    env = gym.make("racetrack-v0", render_mode="rgb_array",
                   config={"show_trajectories": True,
                           "action": {
                               "type": "DiscreteAction",  # DiscreteMetaAction  ContinuousAction
                               "longitudinal": True,
                               "lateral": True,
                               "actions_per_axis": 11
                           }, "observation": {
                           "type": "Kinematics",
                           "vehicles_count": 2,
                           #                0        1    2    3     4      5             6        7            8
                           "features": ["presence", "x", "y", "vx", "vy", "heading", "long_off", "lat_off", "ang_off"],
                           "features_range": {
                               "x": [-100, 100],
                               "y": [-100, 100],
                               "vx": [-100, 100],
                               "vy": [-100, 100],
                               "long_off": [-100, 100],
                               "lat_off": [-100, 100],
                           },
                           "absolute": False,
                       },
                           }
                   )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset = True
    for _ in range(10000):
        if reset:
            _observation, _info = env.reset()
            env.render()
            reset = False
        action = env.action_space.sample()
        print(action)
        _observation, _reward, terminated, truncated, _info = env.step(action)
        for o in _observation.tolist():
            print(o, process_observation(o))
        if terminated or truncated:
            reset = True
    env.close()


if __name__ == '__main__':
    main()
