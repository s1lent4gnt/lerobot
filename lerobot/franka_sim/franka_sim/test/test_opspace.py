import numpy as np
from time import sleep

import argparse
import time
import mujoco
import mujoco.viewer
import numpy as np

from franka_sim import envs
import gymnasium as gym

# import joystick wrapper
from franka_env.envs.wrappers import JoystickIntervention
from franka_env.spacemouse.spacemouse_expert import ControllerType

from franka_sim.utils.viewer_utils import DualMujocoViewer

def move_to_cube(env, max_steps=100):
    """
    Moves the end-effector to the cube's position.

    Args:
        env: The Mujoco environment instance.
        max_steps: Maximum number of steps to attempt reaching the cube.

    Returns:
        success: True if the end-effector reaches the cube, False otherwise.
    """
    # Define success threshold
    position_tolerance = 0.05

    for step in range(max_steps):
        # Get observations
        tcp_pos = env._data.sensor("2f85/pinch_pos").data  # End-effector position
        block_pos = env._data.sensor("block_pos").data  # Cube position

        # Compute the distance to the cube
        dist_to_cube = np.linalg.norm(tcp_pos - block_pos)

        # Check if the end-effector is close enough to the cube
        if dist_to_cube <= position_tolerance:
            print(f"Reached cube at step {step+1}. Distance: {dist_to_cube:.4f}")
            return True

        # Compute action to move toward the cube
        direction = block_pos - tcp_pos
        action = np.zeros(env.action_space.shape)
        action[:3] = direction / np.linalg.norm(direction)  # Normalize direction vector

        # Take a step in the environment
        env.step(action)
        env.render()

        # Delay for visualization
        sleep(0.05)

    print(f"Failed to reach the cube within {max_steps} steps.")
    return False


def sample_new_cube_position(env):
    """
    Samples a new random position for the cube and resets the environment.

    Args:
        env: The Mujoco environment instance.
    """
    obs, _ = env.reset()
    block_pos = obs["state"]["block_pos"]
    print(f"New cube position sampled: {block_pos}")


if __name__ == "__main__":
    env = gym.make("PandaPickCubeVision-v0", render_mode="human", image_obs=True)
    try:
        for _ in range(10):  # Repeat the process 10 times
            print("Moving to cube...")
            success = move_to_cube(env)

            if success:
                print("Sampling a new position for the cube...")
                sample_new_cube_position(env)
            else:
                print("Retrying...")
            
            sleep(1)  # Pause between iterations for clarity
    finally:
        env.close()