#!/usr/bin/env python

"""
Example of integrating the FrankaHIL environment with LeRobot's HIL-SERL framework.
This demonstrates how to use the make_robot_env function with LeRobot's training pipeline.
"""

import gymnasium as gym
import numpy as np
import torch
import gym_franka_sim  # This import is necessary to register the environments

from gym_franka_sim.envs.franka_hil_env import make_robot_env
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.parser import parse_args
from lerobot.common.envs.configs import HILSerlRobotEnvConfig, EnvWrapperConfig


def main():
    # Create a configuration for the HIL-SERL environment
    wrapper_config = EnvWrapperConfig(
        display_cameras=True,
        use_relative_joint_positions=True,
        add_joint_velocity_to_observation=True,
        control_time_s=20.0,
        use_gripper=True,
        gripper_quantization_threshold=0.8,
        gripper_penalty=-0.1,
        gripper_penalty_in_reward=True,
    )
    
    # Create the environment config
    env_config = HILSerlRobotEnvConfig(
        wrapper=wrapper_config,
        fps=30,
        name="franka_hil",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Create a configuration for training
    cfg = TrainPipelineConfig()
    cfg.env = env_config
    cfg.policy.type = "sac"
    cfg.policy.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create the environment using the make_robot_env function
    env_kwargs = {
        "render_mode": "human",
        "image_obs": True,
        "reward_type": "sparse",
        "use_delta_action_space": True,
        "delta": 0.05,
    }
    
    env = make_robot_env(
        env_kwargs=env_kwargs,
        env_name="FrankaHIL-v0",
        use_camera=True,
        use_gripper=True,
        add_joint_velocity_to_observation=wrapper_config.add_joint_velocity_to_observation,
        fps=env_config.fps,
        control_time_s=wrapper_config.control_time_s,
        gripper_quantization_threshold=wrapper_config.gripper_quantization_threshold,
        gripper_penalty=wrapper_config.gripper_penalty,
        gripper_penalty_in_reward=wrapper_config.gripper_penalty_in_reward,
    )
    
    # Create a policy
    policy = SACPolicy(
        state_dim=16,  # 7 joints + gripper + 8 joint velocities
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
    obs, info = env.reset()
    
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
            obs, info = env.reset()
    
    # Close the environment
    env.close()


if __name__ == "__main__":
    main()
