from gymnasium.envs.registration import register

register(
    id="FrankaPick-v0",
    entry_point="gym_franka_sim.envs:FrankaPickEnv",
    max_episode_steps=100,
)

register(
    id="FrankaPush-v0",
    entry_point="gym_franka_sim.envs:FrankaPushEnv",
    max_episode_steps=100,
)

register(
    id="FrankaHIL-v0",
    entry_point="gym_franka_sim.envs:FrankaHILEnv",
    max_episode_steps=100,
)
