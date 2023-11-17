import gymnasium

import src.envs.highway_env_dup

gymnasium.register(
    id='custom-highway-v0',
    entry_point='src.envs.highway_env_dup:CustomHighwayEnv',
)
