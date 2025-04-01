import argparse
import sys

import logging
import time
from threading import Lock
from typing import Annotated, Any, Dict, Tuple
import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.robot_devices.control_utils import (
    busy_wait,
    is_headless,
    reset_follower_position,
)
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config, log_say

from lerobot.scripts.server.kinematics import RobotKinematics

logging.basicConfig(level=logging.INFO)

from lerobot.franka_sim.franka_sim.envs.panda_push_gym_env import PandaPushCubeGymEnv, _PANDA_HOME
from lerobot.franka_sim.franka_sim.envs.panda_pick_gym_env import PandaPickCubeGymEnv, _PANDA_HOME
from lerobot.franka_sim.franka_sim.controllers import opspace
import mujoco
from lerobot.franka_sim.franka_sim.utils.viewer_utils import DualMujocoViewer


class HILSerlRobotEnv(gym.Env):
    """
    Gym-compatible environment for evaluating robotic control policies with integrated human intervention.

    This environment wraps a robot interface to provide a consistent API for policy evaluation. It supports both relative (delta)
    and absolute joint position commands and automatically configures its observation and action spaces based on the robot's
    sensors and configuration.

    The environment can switch between executing actions from a policy or using teleoperated actions (human intervention) during
    each step. When teleoperation is used, the override action is captured and returned in the `info` dict along with a flag
    `is_intervention`.
    """

    def __init__(
        self,
        robot,
        use_delta_action_space: bool = True,
        delta: float | None = None,
        display_cameras: bool = False,
    ):
        """
        Initialize the HILSerlRobotEnv environment.

        The environment is set up with a robot interface, which is used to capture observations and send joint commands. The setup
        supports both relative (delta) adjustments and absolute joint positions for controlling the robot.

        Args:
            robot: The robot interface object used to connect and interact with the physical robot.
            use_delta_action_space (bool): If True, uses a delta (relative) action space for joint control. Otherwise, absolute
                joint positions are used.
            delta (float or None): A scaling factor for the relative adjustments applied to joint positions. Should be a value between
                0 and 1 when using a delta action space.
            display_cameras (bool): If True, the robot's camera feeds will be displayed during execution.
        """
        super().__init__()

        self.robot = robot
        self.display_cameras = display_cameras

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        self.initial_follower_position = robot.follower_arms["main"].read(
            "Present_Position"
        )

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self.delta = delta
        self.use_delta_action_space = use_delta_action_space
        self.current_joint_positions = self.robot.follower_arms["main"].read(
            "Present_Position"
        )

        # Retrieve the size of the joint position interval bound.
        self.relative_bounds_size = (
            (
                self.robot.config.joint_position_relative_bounds["max"]
                - self.robot.config.joint_position_relative_bounds["min"]
            )
            if self.robot.config.joint_position_relative_bounds is not None
            else None
        )

        self.robot.config.max_relative_target = (
            self.relative_bounds_size.float()
            if self.relative_bounds_size is not None
            else None
        )

        # Dynamically configure the observation and action spaces.
        self._setup_spaces()

    def _setup_spaces(self):
        """
        Dynamically configure the observation and action spaces based on the robot's capabilities.

        Observation Space:
            - For keys with "image": A Box space with pixel values ranging from 0 to 255.
            - For non-image keys: A nested Dict space is created under 'observation.state' with a suitable range.

        Action Space:
            - The action space is defined as a Tuple where:
                • The first element is a Box space representing joint position commands. It is defined as relative (delta)
                  or absolute, based on the configuration.
                • ThE SECONd element is a Discrete space (with 2 values) serving as a flag for intervention (teleoperation).
        """
        example_obs = self.robot.capture_observation()

        # Define observation spaces for images and other states.
        image_keys = [key for key in example_obs if "image" in key]
        observation_spaces = {
            key: gym.spaces.Box(
                low=0, high=255, shape=example_obs[key].shape, dtype=np.uint8
            )
            for key in image_keys
        }
        observation_spaces["observation.state"] = gym.spaces.Box(
            low=0,
            high=10,
            shape=example_obs["observation.state"].shape,
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = len(self.robot.follower_arms["main"].read("Present_Position"))
        if self.use_delta_action_space:
            bounds = (
                self.relative_bounds_size
                if self.relative_bounds_size is not None
                else np.ones(action_dim) * 1000
            )
            action_space_robot = gym.spaces.Box(
                low=-bounds,
                high=bounds,
                shape=(action_dim,),
                dtype=np.float32,
            )
        else:
            bounds_min = (
                self.robot.config.joint_position_relative_bounds["min"].cpu().numpy()
                if self.robot.config.joint_position_relative_bounds is not None
                else np.ones(action_dim) * -1000
            )
            bounds_max = (
                self.robot.config.joint_position_relative_bounds["max"].cpu().numpy()
                if self.robot.config.joint_position_relative_bounds is not None
                else np.ones(action_dim) * 1000
            )
            action_space_robot = gym.spaces.Box(
                low=bounds_min,
                high=bounds_max,
                shape=(action_dim,),
                dtype=np.float32,
            )

        self.action_space = gym.spaces.Tuple(
            (
                action_space_robot,
                gym.spaces.Discrete(2),
            ),
        )

    def reset(
        self, seed=None, options=None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to its initial state.
        This method resets the step counter and clears any episodic data.

        Args:
            seed (Optional[int]): A seed for random number generation to ensure reproducibility.
            options (Optional[dict]): Additional options to influence the reset behavior.

        Returns:
            A tuple containing:
                - observation (dict): The initial sensor observation.
                - info (dict): A dictionary with supplementary information, including the key "initial_position".
        """
        super().reset(seed=seed, options=options)

        # Capture the initial observation.
        observation = self.robot.capture_observation()

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None

        return observation, {}

    def step(
        self, action: Tuple[np.ndarray, bool]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute a single step within the environment using the specified action.

        The provided action is a tuple comprised of:
            • A policy action (joint position commands) that may be either in absolute values or as a delta.
            • A boolean flag indicating whether teleoperation (human intervention) should be used for this step.

        Behavior:
            - When the intervention flag is False, the environment processes and sends the policy action to the robot.
            - When True, a teleoperation step is executed. If using a delta action space, an absolute teleop action is converted
              to relative change based on the current joint positions.

        Args:
            action (tuple): A tuple with two elements:
                - policy_action (np.ndarray or torch.Tensor): The commanded joint positions.
                - intervention_bool (bool): True if the human operator intervenes by providing a teleoperation input.

        Returns:
            tuple: A tuple containing:
                - observation (dict): The new sensor observation after taking the step.
                - reward (float): The step reward (default is 0.0 within this wrapper).
                - terminated (bool): True if the episode has reached a terminal state.
                - truncated (bool): True if the episode was truncated (e.g., time constraints).
                - info (dict): Additional debugging information including:
                    ◦ "action_intervention": The teleop action if intervention was used.
                    ◦ "is_intervention": Flag indicating whether teleoperation was employed.
        """
        policy_action, intervention_bool = action
        teleop_action = None
        self.current_joint_positions = self.robot.follower_arms["main"].read(
            "Present_Position"
        )
        if isinstance(policy_action, torch.Tensor):
            policy_action = policy_action.cpu().numpy()
            policy_action = np.clip(
                policy_action, self.action_space[0].low, self.action_space[0].high
            )

        if not intervention_bool:
            if self.use_delta_action_space:
                target_joint_positions = (
                    self.current_joint_positions + self.delta * policy_action
                )
            else:
                target_joint_positions = policy_action
            self.robot.send_action(torch.from_numpy(target_joint_positions))
            observation = self.robot.capture_observation()
        else:
            observation, teleop_action = self.robot.teleop_step(record_data=True)
            teleop_action = teleop_action[
                "action"
            ]  # Convert tensor to appropriate format

            # When applying the delta action space, convert teleop absolute values to relative differences.
            if self.use_delta_action_space:
                teleop_action = (
                    teleop_action - self.current_joint_positions
                ) / self.delta
                if self.relative_bounds_size is not None and (
                    torch.any(teleop_action < -self.relative_bounds_size)
                    and torch.any(teleop_action > self.relative_bounds_size)
                ):
                    logging.debug(
                        f"Relative teleop delta exceeded bounds {self.relative_bounds_size}, teleop_action {teleop_action}\n"
                        f"lower bounds condition {teleop_action < -self.relative_bounds_size}\n"
                        f"upper bounds condition {teleop_action > self.relative_bounds_size}"
                    )

                    teleop_action = torch.clamp(
                        teleop_action,
                        -self.relative_bounds_size,
                        self.relative_bounds_size,
                    )
            # NOTE: To mimic the shape of a neural network output, we add a batch dimension to the teleop action.
            if teleop_action.dim() == 1:
                teleop_action = teleop_action.unsqueeze(0)

        # self.render()

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return (
            observation,
            reward,
            terminated,
            truncated,
            {
                "action_intervention": teleop_action,
                "is_intervention": teleop_action is not None,
            },
        )

    def render(self):
        """
        Render the current state of the environment by displaying the robot's camera feeds.
        """
        import cv2

        observation = self.robot.capture_observation()
        image_keys = [key for key in observation if "image" in key]

        for key in image_keys:
            cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def close(self):
        """
        Close the environment and clean up resources by disconnecting the robot.

        If the robot is currently connected, this method properly terminates the connection to ensure that all
        associated resources are released.
        """
        if self.robot.is_connected:
            self.robot.disconnect()


class AddJointVelocityToObservation(gym.ObservationWrapper):
    def __init__(self, env, joint_velocity_limits=100.0, fps=30):
        super().__init__(env)

        # Extend observation space to include joint velocities
        old_low = self.observation_space["observation.state"].low
        old_high = self.observation_space["observation.state"].high
        old_shape = self.observation_space["observation.state"].shape

        # Assuming the last element is the gripper
        self.arm_joint_count = old_shape[0] - 1  # Number of arm joints (excluding gripper)

        self.last_joint_positions = np.zeros(self.arm_joint_count)

        new_low = np.concatenate(
            [old_low, np.ones(self.arm_joint_count) * -joint_velocity_limits]
        )
        new_high = np.concatenate(
            [old_high, np.ones(self.arm_joint_count) * joint_velocity_limits]
        )

        # New shape: original observation plus velocities for arm joints only
        new_shape = (old_shape[0] + self.arm_joint_count,)

        self.observation_space["observation.state"] = gym.spaces.Box(
            low=new_low,
            high=new_high,
            shape=new_shape,
            dtype=np.float32,
        )

        self.dt = 1.0 / fps

    def observation(self, observation):
        # Extract arm joint positions (exclude gripper)
        arm_joint_positions = observation["observation.state"][:-1]
        gripper_position = observation["observation.state"][-1].unsqueeze(0)
        
        joint_velocities = (
            arm_joint_positions - self.last_joint_positions
        ) / self.dt
        self.last_joint_positions = arm_joint_positions.clone()
        observation["observation.state"] = torch.cat(
            [arm_joint_positions, gripper_position, joint_velocities], dim=-1
        )
        return observation


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, nb_repeat: int = 1):
        super().__init__(env)
        self.nb_repeat = nb_repeat

    def step(self, action):
        for _ in range(self.nb_repeat):
            obs, reward, done, truncated, info = self.env.step(action)
            if done or truncated:
                break
        return obs, reward, done, truncated, info


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_classifier, device: torch.device = "cuda"):
        """
        Wrapper to add reward prediction to the environment, it use a trained classifer.

        Args:
            env: The environment to wrap
            reward_classifier: The reward classifier model
            device: The device to run the model on
        """
        self.env = env

        # NOTE: We got 15% speedup by compiling the model
        self.reward_classifier = torch.compile(reward_classifier)

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        images = [
            observation[key].to(self.device, non_blocking=self.device.type == "cuda")
            for key in observation
            if "image" in key
        ]
        start_time = time.perf_counter()
        with torch.inference_mode():
            reward = (
                self.reward_classifier.predict_reward(images, threshold=0.8)
                if self.reward_classifier is not None
                else 0.0
            )
        info["Reward classifer frequency"] = 1 / (time.perf_counter() - start_time)

        if reward == 1.0:
            terminated = True
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)


class JointMaskingActionSpace(gym.Wrapper):
    def __init__(self, env, mask):
        """
        Wrapper to mask out dimensions of the action space.

        Args:
            env: The environment to wrap
            mask: Binary mask array where 0 indicates dimensions to remove
        """
        super().__init__(env)

        # Validate mask matches action space

        # Keep only dimensions where mask is 1
        self.active_dims = np.where(mask)[0]

        if isinstance(env.action_space, gym.spaces.Box):
            if len(mask) != env.action_space.shape[0]:
                raise ValueError("Mask length must match action space dimensions")
            low = env.action_space.low[self.active_dims]
            high = env.action_space.high[self.active_dims]
            self.action_space = gym.spaces.Box(
                low=low, high=high, dtype=env.action_space.dtype
            )

        if isinstance(env.action_space, gym.spaces.Tuple):
            if len(mask) != env.action_space[0].shape[0]:
                raise ValueError("Mask length must match action space 0 dimensions")

            low = env.action_space[0].low[self.active_dims]
            high = env.action_space[0].high[self.active_dims]
            action_space_masked = gym.spaces.Box(
                low=low, high=high, dtype=env.action_space[0].dtype
            )
            self.action_space = gym.spaces.Tuple(
                (action_space_masked, env.action_space[1])
            )
            # Create new action space with masked dimensions

    def action(self, action):
        """
        Convert masked action back to full action space.

        Args:
            action: Action in masked space. For Tuple spaces, the first element is masked.

        Returns:
            Action in original space with masked dims set to 0.
        """

        # Determine whether we are handling a Tuple space or a Box.
        if isinstance(self.env.action_space, gym.spaces.Tuple):
            # Extract the masked component from the tuple.
            masked_action = action[0] if isinstance(action, tuple) else action
            # Create a full action for the Box element.
            full_box_action = np.zeros(
                self.env.action_space[0].shape, dtype=self.env.action_space[0].dtype
            )
            full_box_action[self.active_dims] = masked_action
            # Return a tuple with the reconstructed Box action and the unchanged remainder.
            return (full_box_action, action[1])
        else:
            # For Box action spaces.
            masked_action = action if not isinstance(action, tuple) else action[0]
            full_action = np.zeros(
                self.env.action_space.shape, dtype=self.env.action_space.dtype
            )
            full_action[self.active_dims] = masked_action
            return full_action

    def step(self, action):
        action = self.action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "action_intervention" in info and info["action_intervention"] is not None:
            if info["action_intervention"].dim() == 1:
                info["action_intervention"] = info["action_intervention"][
                    self.active_dims
                ]
            else:
                info["action_intervention"] = info["action_intervention"][
                    :, self.active_dims
                ]
        return obs, reward, terminated, truncated, info


class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, control_time_s, fps):
        self.env = env
        self.control_time_s = control_time_s
        self.fps = fps

        self.last_timestamp = 0.0
        self.episode_time_in_s = 0.0

        self.max_episode_steps = int(self.control_time_s * self.fps)

        self.current_step = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        time_since_last_step = time.perf_counter() - self.last_timestamp
        self.episode_time_in_s += time_since_last_step
        self.last_timestamp = time.perf_counter()
        self.current_step += 1
        # check if last timestep took more time than the expected fps
        if 1.0 / time_since_last_step < self.fps:
            logging.debug(f"Current timestep exceeded expected fps {self.fps}")

        if self.current_step >= self.max_episode_steps:
            terminated = True
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.episode_time_in_s = 0.0
        self.last_timestamp = time.perf_counter()
        self.current_step = 0
        return self.env.reset(seed=seed, options=options)


class ImageCropResizeWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        crop_params_dict: Dict[str, Annotated[Tuple[int], 4]],
        resize_size=None,
    ):
        super().__init__(env)
        self.env = env
        self.crop_params_dict = crop_params_dict
        print(f"obs_keys , {self.env.observation_space}")
        print(f"crop params dict {crop_params_dict.keys()}")
        for key_crop in crop_params_dict:
            if key_crop not in self.env.observation_space.keys():  # noqa: SIM118
                raise ValueError(f"Key {key_crop} not in observation space")
        for key in crop_params_dict:
            top, left, height, width = crop_params_dict[key]
            new_shape = (top + height, left + width)
            self.observation_space[key] = gym.spaces.Box(
                low=0, high=255, shape=new_shape
            )

        self.resize_size = resize_size
        if self.resize_size is None:
            self.resize_size = (128, 128)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for k in self.crop_params_dict:
            device = obs[k].device
            if obs[k].dim() >= 3:
                # Reshape to combine height and width dimensions for easier calculation
                batch_size = obs[k].size(0)
                channels = obs[k].size(1)
                flattened_spatial_dims = obs[k].view(batch_size, channels, -1)

                # Calculate standard deviation across spatial dimensions (H, W)
                std_per_channel = torch.std(flattened_spatial_dims, dim=2)

                # If any channel has std=0, all pixels in that channel have the same value
                if (std_per_channel <= 0.02).any():
                    logging.warning(
                        f"Potential hardware issue detected: All pixels have the same value in observation {k}"
                    )
            # Check for NaNs before processing
            if torch.isnan(obs[k]).any():
                logging.error(
                    f"NaN values detected in observation {k} before crop and resize"
                )

            if device == torch.device("mps:0"):
                obs[k] = obs[k].cpu()

            obs[k] = F.crop(obs[k], *self.crop_params_dict[k])
            obs[k] = F.resize(obs[k], self.resize_size)

            # Check for NaNs after processing
            if torch.isnan(obs[k]).any():
                logging.error(
                    f"NaN values detected in observation {k} after crop and resize"
                )

            obs[k] = obs[k].to(device)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        for k in self.crop_params_dict:
            device = obs[k].device
            if device == torch.device("mps:0"):
                obs[k] = obs[k].cpu()
            obs[k] = F.crop(obs[k], *self.crop_params_dict[k])
            obs[k] = F.resize(obs[k], self.resize_size)
            obs[k] = obs[k].to(device)
        return obs, info


class ConvertToLeRobotObservation(gym.ObservationWrapper):
    def __init__(self, env, device):
        super().__init__(env)

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

    def observation(self, observation):
        observation = preprocess_observation(observation)

        observation = {
            key: observation[key].to(
                self.device, non_blocking=self.device.type == "cuda"
            )
            for key in observation
        }
        observation = {
            k: torch.tensor(v, device=self.device) for k, v in observation.items()
        }
        return observation


class KeyboardInterfaceWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.listener = None
        self.events = {
            "exit_early": False,
            "pause_policy": False,
            "reset_env": False,
            "human_intervention_step": False,
            "episode_success": False,
        }
        self.event_lock = Lock()  # Thread-safe access to events
        self._init_keyboard_listener()

    def _init_keyboard_listener(self):
        """Initialize keyboard listener if not in headless mode"""

        if is_headless():
            logging.warning(
                "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
            )
            return
        try:
            from pynput import keyboard

            def on_press(key):
                with self.event_lock:
                    try:
                        if key == keyboard.Key.right or key == keyboard.Key.esc:
                            print("Right arrow key pressed. Exiting loop...")
                            self.events["exit_early"] = True
                            return
                        if hasattr(key, "char") and key.char == "s":
                            print("Key 's' pressed. Episode success triggered.")
                            self.events["episode_success"] = True
                            return
                        if key == keyboard.Key.space and not self.events["exit_early"]:
                            if not self.events["pause_policy"]:
                                print(
                                    "Space key pressed. Human intervention required.\n"
                                    "Place the leader in similar pose to the follower and press space again."
                                )
                                self.events["pause_policy"] = True
                                log_say(
                                    "Human intervention stage. Get ready to take over.",
                                    play_sounds=True,
                                )
                                return
                            if (
                                self.events["pause_policy"]
                                and not self.events["human_intervention_step"]
                            ):
                                self.events["human_intervention_step"] = True
                                print("Space key pressed. Human intervention starting.")
                                log_say(
                                    "Starting human intervention.", play_sounds=True
                                )
                                return
                            if (
                                self.events["pause_policy"]
                                and self.events["human_intervention_step"]
                            ):
                                self.events["pause_policy"] = False
                                self.events["human_intervention_step"] = False
                                print("Space key pressed for a third time.")
                                log_say(
                                    "Continuing with policy actions.", play_sounds=True
                                )
                                return
                    except Exception as e:
                        print(f"Error handling key press: {e}")

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
        except ImportError:
            logging.warning(
                "Could not import pynput. Keyboard interface will not be available."
            )
            self.listener = None

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        is_intervention = False
        terminated_by_keyboard = False

        # Extract policy_action if needed
        if isinstance(self.env.action_space, gym.spaces.Tuple):
            policy_action = action[0]

        # Check the event flags without holding the lock for too long.
        with self.event_lock:
            if self.events["exit_early"]:
                terminated_by_keyboard = True
            pause_policy = self.events["pause_policy"]

        if pause_policy:
            # Now, wait for human_intervention_step without holding the lock
            while True:
                with self.event_lock:
                    if self.events["human_intervention_step"]:
                        is_intervention = True
                        break
                time.sleep(0.1)  # Check more frequently if desired

        # Execute the step in the underlying environment
        obs, reward, terminated, truncated, info = self.env.step(
            (policy_action, is_intervention)
        )

        # Override reward and termination if episode success event triggered
        with self.event_lock:
            if self.events["episode_success"]:
                reward = 1
                terminated_by_keyboard = True

        return obs, reward, terminated or terminated_by_keyboard, truncated, info

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """
        Reset the environment and clear any pending events
        """
        with self.event_lock:
            self.events = {k: False for k in self.events}
        return self.env.reset(**kwargs)

    def close(self):
        """
        Properly clean up the keyboard listener when the environment is closed
        """
        if self.listener is not None:
            self.listener.stop()
        super().close()


class ResetWrapper(gym.Wrapper):
    def __init__(
        self,
        env: HILSerlRobotEnv,
        reset_pose: np.ndarray | None = None,
        reset_time_s: float = 5,
    ):
        super().__init__(env)
        self.reset_time_s = reset_time_s
        self.reset_pose = reset_pose

    def reset(self, *, seed=None, options=None):
        if self.reset_pose is not None:
            start_time = time.perf_counter()
            log_say("Reset the environment.", play_sounds=True)
            mujoco.mj_resetData(self.env.model, self.env.data)

            # Reset arm to home position.
            self.env.data.qpos[self.env.panda_dof_ids] = np.asarray(self.reset_pose)
            # Gripper
            self.env.data.ctrl[self.env.gripper_ctrl_id] = 255
            mujoco.mj_forward(self.env.model, self.env.data)

            # Reset mocap body to home position.
            tcp_pos = self.env.data.sensor("2f85/pinch_pos").data
            self.env.data.mocap_pos[0] = tcp_pos

            # z pos
            self.z_init = self.data.sensor("block_pos").data[2]
            self.z_success = self.z_init + 0.2

            # Sample a new block position.
            # block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
            block_xy = np.array([0.5, 0.0])
            self.env.data.jnt("block").qpos[:3] = (*block_xy, 0.02)
            mujoco.mj_forward(self.env.model, self.env.data)

            # Sample a new target position
            # target_region_xy = np.random.uniform(*_SAMPLING_BOUNDS)
            target_region_xy = np.array([0.5, 0.10])
            self.env.model.geom("target_region").pos = (*target_region_xy, 0.005)
            mujoco.mj_forward(self.env.model, self.env.data)

            busy_wait(self.reset_time_s - (time.perf_counter() - start_time))
            log_say("Reset the environment done.", play_sounds=True)
        else:
            # log_say(
            #     f"Manually reset the environment for {self.reset_time_s} seconds.",
            #     play_sounds=True,
            # )
            # start_time = time.perf_counter()
            # while time.perf_counter() - start_time < self.reset_time_s:
            #     self.robot.teleop_step()

            # log_say("Manual reseting of the environment done.", play_sounds=True)
            # TODO (lilkm): add manual reset
            pass
        return super().reset(seed=seed, options=options)


class BatchCompitableWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(
        self, observation: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        for key in observation:
            if "image" in key and observation[key].dim() == 3:
                observation[key] = observation[key].unsqueeze(0)
            if "state" in key and observation[key].dim() == 1:
                observation[key] = observation[key].unsqueeze(0)
            if "velocity" in key and observation[key].dim() == 1:
                observation[key] = observation[key].unsqueeze(0)
        return observation


class EEActionWrapper(gym.ActionWrapper):
    def __init__(self, env, ee_action_space_params=None):
        super().__init__(env)
        self.ee_action_space_params = ee_action_space_params

        action_space_bounds = np.array(
            [
                ee_action_space_params.x_step_size,
                ee_action_space_params.y_step_size,
                ee_action_space_params.z_step_size,
            ]
        )
        # ee_action_space = gym.spaces.Box(
        #     low=-action_space_bounds,
        #     high=action_space_bounds,
        #     shape=(3,),
        #     dtype=np.float32,
        # )

        # # Gripper action space: -1 (open), 0 (no change), 1 (close)
        # gripper_action_space = gym.spaces.Box(
        #     low=-1.0,
        #     high=1.0,
        #     shape=(1,),
        #     dtype=np.float32,
        # )

        # Combined action space including gripper
        full_action_space = gym.spaces.Box(
            low=np.concatenate([-action_space_bounds, np.array([-1.0])]),
            high=np.concatenate([action_space_bounds, np.array([1.0])]),
            shape=(4,),
            dtype=np.float32,
        )

        # Set the action space based on environment configuration
        if isinstance(self.action_space, gym.spaces.Tuple):
            self.action_space = gym.spaces.Tuple(
                (full_action_space, self.action_space[1])
            )
        else:
            self.action_space = full_action_space

        self.bounds = ee_action_space_params.bounds

    def action(self, action):
        is_intervention = False
        if isinstance(action, tuple):
            action, _ = action

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        # Extract position and gripper actions
        position_action = action[:3]
        gripper_action = action[3] if len(action) > 3 else 0.0

        # print(f"gripper action = {gripper_action}")

        # Set the mocap position.
        current_ee_pos = self.env.data.mocap_pos[0].copy()
        npos = np.clip(current_ee_pos + position_action * 0.1, self.bounds["min"], self.bounds["max"]) # TODO (lilkm): 0.1 is the step size, should be a parameter
        self.env.data.mocap_pos[0] = npos

        # Get current gripper state (normalized to 0-1)
        current_gripper = self.env.data.ctrl[self.env.gripper_ctrl_id] / 255.0
        
        # Calculate gripper delta based on action and scale
        gripper_delta = gripper_action * 1.0 # TODO (lilkm): 0.1 is the step size, should be a parameter

        # if gripper_action > 0:  # Close gripper
        #     # Use random values between 0.9-1.0 for closing, like the reference
        #     new_gripper = 1.0
        # elif gripper_action < 0:  # Open gripper
        #     # Use random values between 0-0.1 for opening
        #     new_gripper = -1.0
        # else:
        #     new_gripper = current_gripper
        
        # Compute new gripper position and clip to valid range
        new_gripper = np.clip(current_gripper + gripper_delta, 0.0, 1.0)
        
        # Apply to environment (scaling back to 0-255 range)
        # self.env.data.ctrl[self.env.gripper_ctrl_id] = new_gripper * 255.0

        for _ in range(49):
            target_joint_pos = opspace(
                model=self.env.model,
                data=self.env.data,
                site_id=self.env.model.site("pinch").id,
                dof_ids=self.env.panda_dof_ids,
                pos=self.env.data.mocap_pos[0],
                ori=self.env.data.mocap_quat[0],
                joint=_PANDA_HOME,
                gravity_comp=True,
            )
            self.env.data.ctrl[self.env.panda_ctrl_ids] = target_joint_pos
            mujoco.mj_step(self.env.model, self.env.data)

        target_joint_pos = opspace(
            model=self.env.model,
            data=self.env.data,
            site_id=self.env.model.site("pinch").id,
            dof_ids=self.env.panda_dof_ids,
            pos=self.env.data.mocap_pos[0],
            ori=self.env.data.mocap_quat[0],
            joint=_PANDA_HOME,
            gravity_comp=True,
        )

        output = np.concatenate([target_joint_pos, [new_gripper * 255.0]])

        return output, is_intervention


class EEObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, ee_pose_limits):
        super().__init__(env)

        # Extend observation space to include end effector pose
        prev_space = self.observation_space["observation.state"]

        self.observation_space["observation.state"] = gym.spaces.Box(
            low=np.concatenate([prev_space.low, ee_pose_limits["min"]]),
            high=np.concatenate([prev_space.high, ee_pose_limits["max"]]),
            shape=(prev_space.shape[0] + 3,),
            dtype=np.float32,
        )

    def observation(self, observation):
        current_ee_pos = self.env.data.sensor("2f85/pinch_pos").data
        observation["observation.state"] = torch.cat(
            [
                observation["observation.state"],
                torch.from_numpy(current_ee_pos),
            ],
            dim=-1,
        )
        return observation


class GamepadControlWrapper(gym.Wrapper):
    """
    Wrapper that allows controlling a gym environment with a gamepad.

    This wrapper intercepts the step method and allows human input via gamepad
    to override the agent's actions when desired.
    """

    def __init__(
        self,
        env,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        auto_reset=False,
        input_threshold=0.001,
    ):
        """
        Initialize the gamepad controller wrapper.

        Args:
            env: The environment to wrap
            x_step_size: Base movement step size for X axis in meters
            y_step_size: Base movement step size for Y axis in meters
            z_step_size: Base movement step size for Z axis in meters
            vendor_id: USB vendor ID of the gamepad (default: Logitech)
            product_id: USB product ID of the gamepad (default: RumblePad 2)
            auto_reset: Whether to auto reset the environment when episode ends
            input_threshold: Minimum movement delta to consider as active input
        """
        super().__init__(env)
        from lerobot.scripts.server.end_effector_control_utils import (
            GamepadControllerHID,
            GamepadController,
        )

        # use HidApi for macos
        if sys.platform == "darwin":
            self.controller = GamepadControllerHID(
                x_step_size=x_step_size,
                y_step_size=y_step_size,
                z_step_size=z_step_size,
            )
        else:
            self.controller = GamepadController(
                x_step_size=x_step_size,
                y_step_size=y_step_size,
                z_step_size=z_step_size,
            )
        self.auto_reset = auto_reset
        self.input_threshold = input_threshold
        self.controller.start()

        logging.info("Gamepad control wrapper initialized")
        print("Gamepad controls:")
        print("  Left analog stick: Move in X-Y plane")
        print("  Right analog stick: Move in Z axis (up/down)")
        print("  L3 button (press left stick): Open gripper")
        print("  R3 button (press right stick): Close gripper")
        print("  X/Square button: End episode (FAILURE)")
        print("  Y/Triangle button: End episode (SUCCESS)")
        print("  B/Circle button: Exit program")

    def get_gamepad_action(self):
        """
        Get the current action from the gamepad if any input is active.

        Returns:
            Tuple of (is_active, action, terminate_episode, success)
        """
        # Update the controller to get fresh inputs
        self.controller.update()

        # Get movement deltas from the controller
        delta_x, delta_y, delta_z = self.controller.get_deltas()

        # Get gripper action
        gripper_action = self.controller.get_gripper_action()

        if gripper_action == -1: # close gripper
            gripper_action = float(np.random.uniform(-1, -0.9))
        elif gripper_action == 1: # open gripper
            gripper_action = float(np.random.uniform(0.9, 1))
        else:
            gripper_action = 0.0

        intervention_is_active = self.controller.should_intervene()

        # Create action from gamepad input
        gamepad_action = np.array([delta_x, delta_y, delta_z, gripper_action], dtype=np.float32)

        # Check episode ending buttons
        # We'll rely on controller.get_episode_end_status() which returns "success", "failure", or None
        episode_end_status = self.controller.get_episode_end_status()
        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        rerecord_episode = episode_end_status == "rerecord_episode"

        return (
            intervention_is_active,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        )

    def step(self, action):
        """
        Step the environment, using gamepad input to override actions when active.

        Args:
            action: Original action from agent

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get gamepad state and action
        (
            is_intervention,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        ) = self.get_gamepad_action()

        # Update episode ending state if requested
        if terminate_episode:
            logging.info(
                f"Episode manually ended: {'SUCCESS' if success else 'FAILURE'}"
            )

        # Only override the action if gamepad is active
        if is_intervention:
            # Format according to the expected action type
            if isinstance(self.action_space, gym.spaces.Tuple):
                # For environments that use (action, is_intervention) tuples
                final_action = (torch.from_numpy(gamepad_action), False)
            else:
                final_action = torch.from_numpy(gamepad_action)
        else:
            # Use the original action
            final_action = action

        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(final_action)

        # Add episode ending if requested via gamepad
        terminated = terminated or truncated or terminate_episode

        if success:
            reward = 1.0
            logging.info("Episode ended successfully with reward 1.0")

        info["is_intervention"] = is_intervention
        action_intervention = (
            final_action[0] if isinstance(final_action, Tuple) else final_action
        )
        if isinstance(action_intervention, np.ndarray):
            action_intervention = torch.from_numpy(action_intervention)
        info["action_intervention"] = action_intervention
        info["rerecord_episode"] = rerecord_episode

        # If episode ended, reset the state
        if terminated or truncated:
            # Add success/failure information to info dict
            info["next.success"] = success

            # Auto reset if configured
            if self.auto_reset:
                obs, reset_info = self.reset()
                info.update(reset_info)

        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up resources when environment closes."""
        # Stop the controller
        if hasattr(self, "controller"):
            self.controller.stop()

        # Call the parent close method
        return self.env.close()
    
class SimRewardWrapper(gym.Wrapper):
    """
    This wrapper is used to help label transitions via the sim env.
    """

    def __init__(self, env, reward_type):
        super().__init__(env)
        self.reward_type = reward_type

    def compute_reward(self) -> float:
        if self.reward_type == "dense":
            block_pos = self.data.sensor("block_pos").data
            tcp_pos = self.data.sensor("2f85/pinch_pos").data
            dist = np.linalg.norm(block_pos - tcp_pos)
            r_close = np.exp(-20 * dist)
            r_lift = (block_pos[2] - self.z_init) / (self._z_success - self.z_init)
            r_lift = np.clip(r_lift, 0.0, 1.0)
            rew = 0.3 * r_close + 0.7 * r_lift
            return rew
        else:
            block_pos = self.data.sensor("block_pos").data
            lift = block_pos[2] - self.z_init
            return float(lift > 0.2)

    def is_success(self) -> bool:
        block_pos = self.data.sensor("block_pos").data
        tcp_pos = self.data.sensor("2f85/pinch_pos").data
        dist = np.linalg.norm(block_pos - tcp_pos)
        lift = block_pos[2] - self.z_init
        return dist < 0.05 and lift > 0.2
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        reward = self.compute_reward()

        # move reward to device
        reward = torch.tensor(reward, device=self.device)
        if reward == 1.0:
            terminated = True
        return observation, reward, terminated, truncated, info

class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["observation.state"][0, 7] # TODO (lilkm) : gripper joint in the first index
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        print(f"*** Gripper action = {action[-1]} *** last gripper pos = {self.last_gripper_pos}")

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (action[-1] > 0.5 and self.last_gripper_pos < 0.9):
            print("Penalty!!!")
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["observation.state"][0, 7]
        return observation, reward, terminated, truncated, info


def make_robot_env(
    robot,
    reward_classifier,
    cfg,
    n_envs: int = 1,
) -> gym.vector.VectorEnv:
    """
    Factory function to create a vectorized robot environment.

    Args:
        robot: Robot instance to control
        reward_classifier: Classifier model for computing rewards
        cfg: Configuration object containing environment parameters
        n_envs: Number of environments to create in parallel. Defaults to 1.

    Returns:
        A vectorized gym environment with all the necessary wrappers applied.
    """
    if "maniskill" in cfg.env.name:
        from lerobot.scripts.server.maniskill_manipulator import make_maniskill

        logging.warning("WE SHOULD REMOVE THE MANISKILL BEFORE THE MERGE INTO MAIN")
        env = make_maniskill(
            cfg=cfg,
            n_envs=1,
        )
        return env
    # Create base environment
    # env = PandaPushCubeGymEnv(
    #     cfg=cfg,
    #     delta=cfg.env.wrapper.delta_action,
    #     use_delta_action_space=cfg.env.wrapper.use_relative_joint_positions,
    #     render_mode="human",
    #     image_obs=True,
    #     reward_type="sparse",
    #     time_limit=60.0,
    #     control_dt=0.1
    # )

    env = PandaPickCubeGymEnv(
        cfg=cfg,
        delta=cfg.env.wrapper.delta_action,
        use_delta_action_space=cfg.env.wrapper.use_relative_joint_positions,
        render_mode="human",
        image_obs=True,
        reward_type="sparse",
        time_limit=60.0,
        control_dt=0.1
    )

    # Add observation and image processing
    if cfg.env.wrapper.add_joint_velocity_to_observation:
        env = AddJointVelocityToObservation(env=env, fps=cfg.fps)
    if cfg.env.wrapper.add_ee_pose_to_observation:
        env = EEObservationWrapper(
            env=env, ee_pose_limits=cfg.env.wrapper.ee_action_space_params.bounds
        )

    env = ConvertToLeRobotObservation(env=env, device=cfg.env.device)

    if cfg.env.wrapper.crop_params_dict is not None:
        env = ImageCropResizeWrapper(
            env=env,
            crop_params_dict=cfg.env.wrapper.crop_params_dict,
            resize_size=cfg.env.wrapper.resize_size,
        )

    # Add reward computation and control wrappers
    if reward_classifier is not None:
        env = RewardWrapper(env=env, reward_classifier=reward_classifier, device="cuda")
    else:
        env = SimRewardWrapper(env=env, reward_type="sparse")
    env = TimeLimitWrapper(
        env=env, control_time_s=cfg.env.wrapper.control_time_s, fps=cfg.fps
    )
    if cfg.env.wrapper.ee_action_space_params is not None:
        env = EEActionWrapper(
            env=env, ee_action_space_params=cfg.env.wrapper.ee_action_space_params
        )
    if cfg.env.wrapper.ee_action_space_params.use_gamepad:
        env = GamepadControlWrapper(
            env=env,
            x_step_size=cfg.env.wrapper.ee_action_space_params.x_step_size,
            y_step_size=cfg.env.wrapper.ee_action_space_params.y_step_size,
            z_step_size=cfg.env.wrapper.ee_action_space_params.z_step_size,
        )
    else:
        env = KeyboardInterfaceWrapper(env=env)

    env = ResetWrapper(
        env=env,
        reset_pose=cfg.env.wrapper.fixed_reset_joint_positions,
        reset_time_s=cfg.env.wrapper.reset_time_s,
    )
    if (
        cfg.env.wrapper.ee_action_space_params is None
        and cfg.env.wrapper.joint_masking_action_space is not None
    ):
        env = JointMaskingActionSpace(
            env=env, mask=cfg.env.wrapper.joint_masking_action_space
        )
    env = BatchCompitableWrapper(env=env)

    env = GripperPenaltyWrapper(env=env)

    return env



def get_classifier(pretrained_path, config_path, device="cuda"):
    if pretrained_path is None or config_path is None:
        return None

    from lerobot.common.policies.factory import _policy_cfg_from_hydra_cfg
    from lerobot.common.policies.hilserl.classifier.configuration_classifier import (
        ClassifierConfig,
    )
    from lerobot.common.policies.hilserl.classifier.modeling_classifier import (
        Classifier,
    )

    cfg = init_hydra_config(config_path)

    classifier_config = _policy_cfg_from_hydra_cfg(ClassifierConfig, cfg)
    classifier_config.num_cameras = len(
        cfg.training.image_keys
    )  # TODO automate these paths
    model = Classifier(classifier_config)
    model.load_state_dict(Classifier.from_pretrained(pretrained_path).state_dict())
    model = model.to(device)
    return model


def record_dataset(
    env,
    repo_id,
    root=None,
    num_episodes=1,
    control_time_s=20,
    fps=30,
    push_to_hub=True,
    task_description="",
    policy=None,
):
    """
    Record a dataset of robot interactions using either a policy or teleop.

    Args:
        env: The environment to record from
        repo_id: Repository ID for dataset storage
        root: Local root directory for dataset (optional)
        num_episodes: Number of episodes to record
        control_time_s: Maximum episode length in seconds
        fps: Frames per second for recording
        push_to_hub: Whether to push dataset to Hugging Face Hub
        task_description: Description of the task being recorded
        policy: Optional policy to generate actions (if None, uses teleop)
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # Setup initial action (zero action if using teleop)
    dummy_action = env.action_space.sample()
    dummy_action = (torch.from_numpy(dummy_action[0] * 0.0), False)
    action = dummy_action

    # Configure dataset features based on environment spaces
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": env.observation_space["observation.state"].shape,
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": env.action_space[0].shape,
            "names": None,
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }

    # Add image features
    for key in env.observation_space:
        if "image" in key:
            features[key] = {
                "dtype": "video",
                "shape": env.observation_space[key].shape,
                "names": None,
            }

    # Create dataset
    dataset = LeRobotDataset.create(
        repo_id,
        fps,
        root=root,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        features=features,
    )
    
    # Create the dual viewer
    # dual_viewer = DualMujocoViewer(env.model, env.data)

    # with dual_viewer as viewer:
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        episode_index = 0
        while viewer.is_running():
            obs, _ = env.reset()
            start_episode_t = time.perf_counter()
            log_say(f"Recording episode {episode_index}", play_sounds=True)

            # Run episode steps
            while time.perf_counter() - start_episode_t < control_time_s:
                start_loop_t = time.perf_counter()
                # Get action from policy if available
                if policy is not None:
                    action = policy.select_action(obs)
                # Step environment
                obs, reward, terminated, truncated, info = env.step(dummy_action)
                viewer.sync()
                if info.get("rerecord_episode", False):
                    break
                # For teleop, get action from intervention
                if policy is None:
                    action = {
                        "action": info["action_intervention"].cpu().squeeze(0).float()
                    }
                # Process observation for dataset
                obs = {k: v.cpu().squeeze(0).float() for k, v in obs.items()}

                # Add frame to dataset
                frame = {**obs, **action}
                frame["next.reward"] = reward.cpu().squeeze(0).float()
                frame["next.done"] = terminated or truncated
                dataset.add_frame(frame)
                # Maintain consistent timing
                if fps:
                    dt_s = time.perf_counter() - start_loop_t
                    busy_wait(1 / fps - dt_s)

                if terminated or truncated:
                    break

            # Handle episode recording
            if info.get("rerecord_episode", False):
                dataset.clear_episode_buffer()
                logging.info(f"Re-recording episode {episode_index}")
                continue

            if episode_index >= num_episodes:
                break

            dataset.save_episode(task_description)
            episode_index += 1
    dataset.consolidate(run_compute_stats=True)

    if push_to_hub:
        dataset.push_to_hub(repo_id)


def replay_episode(env, repo_id, root=None, episode=0):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    local_files_only = root is not None
    dataset = LeRobotDataset(
        repo_id, root=root, episodes=[episode], local_files_only=local_files_only
    )
    env.reset()

    actions = dataset.hf_dataset.select_columns("action")

    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action = actions[idx]["action"][:4]
        env.step((action, False))
        # env.step((action / env.unwrapped.delta, False))

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / 10 - dt_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=30, help="control frequency")
    parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    parser.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    parser.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
        ),
    )
    parser.add_argument(
        "--display-cameras",
        help=("Whether to display the camera feed while the rollout is happening"),
    )
    parser.add_argument(
        "--reward-classifier-pretrained-path",
        type=str,
        default=None,
        help="Path to the pretrained classifier weights.",
    )
    parser.add_argument(
        "--reward-classifier-config-file",
        type=str,
        default=None,
        help="Path to a yaml config file that is necessary to build the reward classifier model.",
    )
    parser.add_argument(
        "--env-path", type=str, default=None, help="Path to the env yaml file"
    )
    parser.add_argument(
        "--env-overrides",
        type=str,
        default=None,
        help="Overrides for the env yaml file",
    )
    parser.add_argument(
        "--control-time-s",
        type=float,
        default=20,
        help="Maximum episode length in seconds",
    )
    parser.add_argument(
        "--reset-follower-pos",
        type=int,
        default=1,
        help="Reset follower between episodes",
    )
    parser.add_argument(
        "--replay-repo-id",
        type=str,
        default=None,
        help="Repo ID of the episode to replay",
    )
    parser.add_argument(
        "--dataset-root", type=str, default=None, help="Root of the dataset to replay"
    )
    parser.add_argument(
        "--replay-episode", type=int, default=0, help="Episode to replay"
    )
    parser.add_argument(
        "--record-repo-id",
        type=str,
        default=None,
        help="Repo ID of the dataset to record",
    )
    parser.add_argument(
        "--record-num-episodes",
        type=int,
        default=1,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--record-episode-task",
        type=str,
        default="",
        help="Single line description of the task to record",
    )

    args = parser.parse_args()

    robot_cfg = init_hydra_config(args.robot_path, args.robot_overrides)
    robot = make_robot(robot_cfg)

    reward_classifier = get_classifier(
        args.reward_classifier_pretrained_path, args.reward_classifier_config_file
    )
    user_relative_joint_positions = True

    cfg = init_hydra_config(args.env_path, args.env_overrides)
    # env = make_robot_env(
    #     robot=None,
    #     reward_classifier=reward_classifier,
    #     cfg=cfg,  # .wrapper,
    # )

    env = make_robot_env(
        robot=None,
        reward_classifier=None,
        cfg=cfg,  # .wrapper,
    )

    if args.record_repo_id is not None:
        policy = None
        if args.pretrained_policy_name_or_path is not None:
            from lerobot.common.policies.sac.modeling_sac import SACPolicy

            policy = SACPolicy.from_pretrained(args.pretrained_policy_name_or_path)
            policy.to(cfg.device)
            policy.eval()

        record_dataset(
            env,
            args.record_repo_id,
            root=args.dataset_root,
            num_episodes=args.record_num_episodes,
            fps=args.fps,
            task_description=args.record_episode_task,
            policy=policy,
        )
        exit()

    if args.replay_repo_id is not None:
        replay_episode(
            env,
            args.replay_repo_id,
            root=args.dataset_root,
            episode=args.replay_episode,
        )
        exit()

    env.reset()

    # Retrieve the robot's action space for joint commands.
    action_space_robot = env.action_space.spaces[0]

    # Initialize the smoothed action as a random sample.
    smoothed_action = action_space_robot.sample()

    # Smoothing coefficient (alpha) defines how much of the new random sample to mix in.
    # A value close to 0 makes the trajectory very smooth (slow to change), while a value close to 1 is less smooth.
    alpha = 1.0

    num_episode = 0
    sucesses = []
    while num_episode < 20:
        start_loop_s = time.perf_counter()
        # Sample a new random action from the robot's action space.
        new_random_action = action_space_robot.sample()
        # Update the smoothed action using an exponential moving average.
        smoothed_action = alpha * new_random_action + (1 - alpha) * smoothed_action

        # Execute the step: wrap the NumPy action in a torch tensor.
        obs, reward, terminated, truncated, info = env.step(
            (torch.from_numpy(smoothed_action), False)
        )
        if terminated or truncated:
            sucesses.append(reward)
            env.reset()
            num_episode += 1

        dt_s = time.perf_counter() - start_loop_s
        busy_wait(1 / args.fps - dt_s)

    logging.info(f"Success after 20 steps {sucesses}")
    logging.info(f"success rate {sum(sucesses)/ len(sucesses)}")