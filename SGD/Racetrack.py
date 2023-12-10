import itertools
import gymnasium as gym
import numpy as np
from estimator import Estimator

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
    "other_vehicles": 1,
    "policy_frequency": 9,
}


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def make_estimator(action_count, initial_state):
    model = Estimator(action_count, initial_state)
    return model


# Calculate the TD
def get_temporal_difference(reward, gamma, new_q_values, old_q_value):
    temporal_difference = reward + (gamma * np.max(new_q_values)) - old_q_value
    return temporal_difference


def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    for i_episode in range(num_episodes):
        total_reward = 0
        # The policy we're following
        policy = make_epsilon_greedy_policy(
            estimator, epsilon, env.action_space.n)

        # Reset the environment and pick the first action
        state, _ = env.reset()

        # One step in the environment
        for t in itertools.count():
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            # Take a step
            next_state, reward, terminated, truncated, _ = env.step(action)
            # TD Update
            q_values_next = estimator.predict(next_state)

            # Q-Value TD Target to approximate
            td_target = reward + discount_factor * np.max(q_values_next)
            total_reward += reward
            # Update the function approximator using our target
            estimator.update(state, action, td_target)

            if terminated or truncated:
                break
            env.render()
            state = next_state
        print(f"total reward: {total_reward}")
        print("Episode {} Done".format(i_episode + 1))


if __name__ == '__main__':
    # Init environment
    env = gym.make('custom-highway-v1', render_mode="rgb_array", config=_config)
    # Reset (start) the environment 
    initial_state, _ = env.reset()
    estimator = make_estimator(env.action_space.n, initial_state)
    # q_learning(env, estimator, num_episodes=10000, discount_factor=0.8, epsilon=0.15)
    # After the training we can perform optimal operations and record them
    # The policy we're following
    policy = make_epsilon_greedy_policy(estimator, 0, env.action_space.n)
    while True:
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            actions = estimator.predict(obs)
            obs, reward, done, truncated, info = env.step(np.argmax(actions))
            env.render()
