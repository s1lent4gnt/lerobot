#!/usr/bin/env python

"""
Example of integrating gym_franka_sim with LeRobot.
"""

import gymnasium as gym
import numpy as np
import torch
import gym_franka_sim  # This import is necessary to register the environments

from lerobot.common.envs.factory import make_env_config, make_env
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.parser import parse_args

def main():
    # Create a configuration for the Franka simulation environment
    env_config = make_env_config("franka_sim", task="FrankaPick-v0")
    
    # Create a configuration for training
    cfg = TrainPipelineConfig()
    cfg.env = env_config
    cfg.policy.type = "sac"
    cfg.policy.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create the environment using LeRobot's factory
    env = make_env(cfg.env)
    
    # Create a policy
    policy = SACPolicy(
        state_dim=8,  # 7 joints + gripper
        action_dim=8,  # 7 joints + gripper
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.1,
        alpha_lr=1e-4,
        alpha_betas=(0.9, 0.999),
        actor_lr=1e-4,
        actor_betas=(0.9, 0.999),
        actor_update_frequency=1,
        critic_lr=1e-4,
        critic_betas=(0.9, 0.999),
        critic_tau=0.005,
        critic_target_update_frequency=2,
        batch_size=1024,
        learnable_temperature=True,
        device=cfg.policy.device
    )
    
    # Reset the environment
    obs, _ = env.reset()
    
    # Run for a few steps
    for i in range(100):
        # Get action from policy
        with torch.no_grad():
            action = policy.select_action(obs)
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print some information
        print(f"Step {i}, Reward: {reward}")
        
        if terminated or truncated:
            print("Episode finished")
            obs, _ = env.reset()
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
