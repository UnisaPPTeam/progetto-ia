import gymnasium


def main():
    env = gymnasium.make("custom-highway-v0", render_mode="human")
    env.configure({"other_vehicles": 2, "scaling": 4})
    reset = True
    for _ in range(10000):
        if reset:
            _observation, _info = env.reset()
            env.render()
            reset = False
        action = env.action_space.sample()
        _observation, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            reset = True
    env.close()


if __name__ == '__main__':
    main()
