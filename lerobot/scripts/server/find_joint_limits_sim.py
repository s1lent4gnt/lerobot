import argparse
import time

import cv2
import numpy as np

from lerobot.common.utils.utils import init_hydra_config
from lerobot.scripts.server.gym_manipulator import get_classifier, make_robot_env
from lerobot.franka_sim.franka_sim.utils.viewer_utils import DualMujocoViewer


def find_joint_bounds(
    env,
    control_time_s=30,
):
    start_episode_t = time.perf_counter()
    pos_list = []

    # Create the dual viewer
    dual_viewer = DualMujocoViewer(env.model, env.data)

    env.reset()

    with dual_viewer as viewer:
        while viewer.is_running():
            env.step(np.array([0.0, 0.0, 0.0]))
            viewer.sync()

            # Wait for 5 seconds to stabilize the robot initial position
            if time.perf_counter() - start_episode_t < 5:
                continue

            pos_list.append(env.data.qpos[env.panda_dof_ids].copy())

            if time.perf_counter() - start_episode_t > control_time_s:
                max = np.max(np.stack(pos_list), 0)
                min = np.min(np.stack(pos_list), 0)
                print(f"Max angle position per joint {max}")
                print(f"Min angle position per joint {min}")
                break


def find_ee_bounds(
    env,
    control_time_s=30,
):
    start_episode_t = time.perf_counter()
    ee_list = []

    # Create the dual viewer
    dual_viewer = DualMujocoViewer(env.model, env.data)

    env.reset()

    with dual_viewer as viewer:
        while viewer.is_running():
            env.step(np.array([0.0, 0.0, 0.0]))
            viewer.sync()

            # Wait for 5 seconds to stabilize the robot initial position
            if time.perf_counter() - start_episode_t < 5:
                continue

            ee_list.append(env.data.site_xpos[env.model.site("pinch").id].copy())

            if time.perf_counter() - start_episode_t > control_time_s:
                max = np.max(np.stack(ee_list), 0)
                min = np.min(np.stack(ee_list), 0)
                print(f"Max ee position {max}")
                print(f"Min ee position {min}")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-env-path",
        type=str,
        default="lerobot/configs/robot/koch.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    parser.add_argument(
        "--robot-env-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="joint",
        choices=["joint", "ee"],
        help="Mode to run the script in. Can be 'joint' or 'ee'.",
    )
    parser.add_argument(
        "--control-time-s",
        type=int,
        default=30,
        help="Time step to use for control.",
    )
    args = parser.parse_args()
    robot_env_cfg = init_hydra_config(args.robot_env_path, args.robot_env_overrides)

    env = make_robot_env(
        robot=None,
        reward_classifier=None,
        cfg=robot_env_cfg,
    )

    if args.mode == "joint":
        find_joint_bounds(env, args.control_time_s)
    elif args.mode == "ee":
        find_ee_bounds(env, args.control_time_s)
