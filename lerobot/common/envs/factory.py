#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import einops
import gymnasium as gym
import importlib
import numpy as np
import torch
from collections import deque
from mani_skill.utils import common
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from omegaconf import DictConfig


def make_env(cfg: DictConfig, n_envs: int | None = None) -> gym.vector.VectorEnv | None:
    """Makes a gym vector environment according to the evaluation config.

    n_envs can be used to override eval.batch_size in the configuration. Must be at least 1.
    """
    if n_envs is not None and n_envs < 1:
        raise ValueError("`n_envs must be at least 1")

    if cfg.env.name == "real_world":
        return

    if "maniskill" in cfg.env.name:
        env = make_maniskill_env(cfg, n_envs if n_envs is not None else cfg.eval.batch_size)
        return env

    package_name = f"gym_{cfg.env.name}"

    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError as e:
        print(
            f"{package_name} is not installed. Please install it with `pip install 'lerobot[{cfg.env.name}]'`"
        )
        raise e

    gym_handle = f"{package_name}/{cfg.env.task}"
    gym_kwgs = dict(cfg.env.get("gym", {}))

    if cfg.env.get("episode_length"):
        gym_kwgs["max_episode_steps"] = cfg.env.episode_length

    # batched version of the env that returns an observation of shape (b, c)
    env_cls = gym.vector.AsyncVectorEnv if cfg.eval.use_async_envs else gym.vector.SyncVectorEnv
    env = env_cls(
        [
            lambda: gym.make(gym_handle, disable_env_checker=True, **gym_kwgs)
            for _ in range(n_envs if n_envs is not None else cfg.eval.batch_size)
        ]
    )

    return env


def make_maniskill_env(cfg: DictConfig, n_envs: int | None = None) -> gym.vector.VectorEnv | None:
    """Make ManiSkill3 gym environment"""
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    env = gym.make(
        cfg.env.task,
        obs_mode=cfg.env.obs,
        control_mode=cfg.env.control_mode,
        render_mode=cfg.env.render_mode,
        sensor_configs=dict(width=cfg.env.image_size, height=cfg.env.image_size),
        num_envs=n_envs,
    )
    # cfg.env_cfg.control_mode = cfg.eval_env_cfg.control_mode = env.control_mode
    env = ManiSkillVectorEnv(env, ignore_terminations=True)
    # state should have the size of 25
    # env = ConvertToLeRobotEnv(env, n_envs)
    # env = PixelWrapper(cfg, env, n_envs)
    env._max_episode_steps = env.max_episode_steps = 50  # gym_utils.find_max_episode_steps_value(env)
    env.unwrapped.metadata["render_fps"] = 20

    return env


def preprocess_maniskill_observation(observations: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    # TODO: You have to merge all tensors from agent key and extra key
    # You don't keep sensor param key in the observation
    # And you keep sensor data rgb
    q_pos = observations["agent"]["qpos"]
    q_vel = observations["agent"]["qvel"]
    tcp_pos = observations["extra"]["tcp_pose"]
    img = observations["sensor_data"]["base_camera"]["rgb"]

    _, h, w, c = img.shape
    assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

    # sanity check that images are uint8
    assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

    # convert to channel first of type float32 in range [0,1]
    img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
    img = img.type(torch.float32)
    img /= 255

    state = torch.cat([q_pos, q_vel, tcp_pos], dim=-1)

    return_observations["observation.image"] = img
    return_observations["observation.state"] = state
    return return_observations


class ManiSkillObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, device: torch.device = "cuda"):
        super().__init__(env)
        self.device = device

    def observation(self, observation):
        observation = preprocess_maniskill_observation(observation)
        observation = {k: v.to(self.device) for k, v in observation.items()}
        return observation


class ManiSkillCompat(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = reward.item()
        terminated = terminated.item()
        truncated = truncated.item()
        return obs, reward, terminated, truncated, info


class ManiSkillActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Tuple(spaces=(env.action_space, gym.spaces.Discrete(2)))

    def action(self, action):
        action, telop = action
        return action


class ManiSkillMultiplyActionWrapper(gym.Wrapper):
    def __init__(self, env, multiply_factor: float = 1):
        super().__init__(env)
        self.multiply_factor = multiply_factor
        action_space_agent: gym.spaces.Box = env.action_space[0]
        action_space_agent.low = action_space_agent.low * multiply_factor
        action_space_agent.high = action_space_agent.high * multiply_factor
        self.action_space = gym.spaces.Tuple(spaces=(action_space_agent, gym.spaces.Discrete(2)))

    def step(self, action):
        if isinstance(action, tuple):
            action, telop = action
        else:
            telop = 0
        action = action / self.multiply_factor
        obs, reward, terminated, truncated, info = self.env.step((action, telop))
        return obs, reward, terminated, truncated, info


def make_maniskill(
    cfg: DictConfig,
    n_envs: int | None = None,
) -> gym.Env:
    """
    Factory function to create a ManiSkill environment with standard wrappers.

    Args:
        task: Name of the ManiSkill task
        obs_mode: Observation mode (rgb, rgbd, etc)
        control_mode: Control mode for the robot
        render_mode: Rendering mode
        sensor_configs: Camera sensor configurations
        n_envs: Number of parallel environments

    Returns:
        A wrapped ManiSkill environment
    """

    env = gym.make(
        cfg.env.task,
        obs_mode=cfg.env.obs,
        control_mode=cfg.env.control_mode,
        render_mode=cfg.env.render_mode,
        sensor_configs={"width": cfg.env.image_size, "height": cfg.env.image_size},
        num_envs=n_envs,
    )
    env = ManiSkillObservationWrapper(env, device=cfg.env.device)
    env = ManiSkillVectorEnv(env, ignore_terminations=True)
    env._max_episode_steps = env.max_episode_steps = 50  # gym_utils.find_max_episode_steps_value(env)
    env.unwrapped.metadata["render_fps"] = 20

    return env