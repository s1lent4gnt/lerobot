#!/usr/bin/env python

"""
Example of using the FrankaPush environment from gym_franka_sim.
"""

import gymnasium as gym
import numpy as np
import time
import gym_franka_sim  # This import is necessary to register the environments

def main():
    # Create the FrankaPush environment
    env = gym.make('FrankaPush-v0', render_mode="human")
    
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
