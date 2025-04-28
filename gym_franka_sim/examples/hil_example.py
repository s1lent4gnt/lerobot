#!/usr/bin/env python

"""
Example of using the FrankaHIL environment with the make_robot_env function.
This demonstrates how to create a Franka gym environment that follows the HIL-SERL structure.
"""

import gymnasium as gym
import numpy as np
import torch
import time
import gym_franka_sim  # This import is necessary to register the environments

from gym_franka_sim.envs.franka_hil_env import make_robot_env


def main():
    # Create the environment using the make_robot_env function
    env_kwargs = {
        "render_mode": "human",
        "image_obs": True,
        "reward_type": "sparse",
        "use_delta_action_space": True,
        "delta": 0.05,
    }
    
    # Additional wrapper parameters
    wrapper_kwargs = {
        "add_joint_velocity_to_observation": True,
        "fps": 30,
        "control_time_s": 20.0,
        "use_gripper": True,
        "gripper_quantization_threshold": 0.8,
        "gripper_penalty": -0.1,
        "gripper_penalty_in_reward": True,
    }
    
    # Create the environment
    env = make_robot_env(
        env_kwargs=env_kwargs,
        env_name="FrankaHIL-v0",
        use_camera=True,
        use_gripper=True,
        **wrapper_kwargs
    )
    
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    # Reset the environment
    obs, info = env.reset()
    
    # Run for a few steps
    for i in range(100):
        # Sample a random action
        action = (env.action_space[0].sample(), False)  # (joint_action, intervention_flag)
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print some information
        print(f"Step {i}, Reward: {reward}")
        
        # Sleep to slow down the visualization
        time.sleep(0.1)
        
        if terminated or truncated:
            print("Episode finished")
            obs, info = env.reset()
    
    # Close the environment
    env.close()


if __name__ == "__main__":
    main()
