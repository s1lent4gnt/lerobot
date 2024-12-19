"""Contains logic to instantiate a robot, read information from its motors and cameras,
and send orders to its motors.
"""
# TODO(rcadene, aliberts): reorganize the codebase into one file per robot, with the associated
# calibration procedure, to make it easy for people to add their own robot.

import json
import logging
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import mujoco
import mink
import multiprocessing
import inputs
from enum import Enum

from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.motors.utils import MotorsBus
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError


def ensure_safe_goal_position(
    goal_pos: torch.Tensor, present_pos: torch.Tensor, max_relative_target: float | list[float]
):
    # Cap relative action target magnitude for safety.
    diff = goal_pos - present_pos
    max_relative_target = torch.tensor(max_relative_target)
    safe_diff = torch.minimum(diff, max_relative_target)
    safe_diff = torch.maximum(safe_diff, -max_relative_target)
    safe_goal_pos = present_pos + safe_diff

    if not torch.allclose(goal_pos, safe_goal_pos):
        logging.warning(
            "Relative goal position magnitude had to be clamped to be safe.\n"
            f"  requested relative goal position target: {diff}\n"
            f"    clamped relative goal position target: {safe_diff}"
        )

    return safe_goal_pos


@dataclass
class ManipulatorRobotConfig:
    """
    Example of usage:
    ```python
    ManipulatorRobotConfig()
    ```
    """

    # Define all components of the robot
    robot_type: str = "koch"
    leader_arms: dict[str, MotorsBus] = field(default_factory=lambda: {})
    follower_arms: dict[str, MotorsBus] = field(default_factory=lambda: {})
    cameras: dict[str, Camera] = field(default_factory=lambda: {})

    # Optionally limit the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length
    # as the number of motors in your follower arms (assumes all follower arms have the same number of
    # motors).
    max_relative_target: list[float] | float | None = None

    # Optionally set the leader arm in torque mode with the gripper motor set to this angle. This makes it
    # possible to squeeze the gripper and have it spring back to an open position on its own. If None, the
    # gripper is not put in torque mode.
    gripper_open_degree: float | None = None

    def __setattr__(self, prop: str, val):
        if prop == "max_relative_target" and val is not None and isinstance(val, Sequence):
            for name in self.follower_arms:
                if len(self.follower_arms[name].motors) != len(val):
                    raise ValueError(
                        f"len(max_relative_target)={len(val)} but the follower arm with name {name} has "
                        f"{len(self.follower_arms[name].motors)} motors. Please make sure that the "
                        f"`max_relative_target` list has as many parameters as there are motors per arm. "
                        "Note: This feature does not yet work with robots where different follower arms have "
                        "different numbers of motors."
                    )
        super().__setattr__(prop, val)

    def __post_init__(self):
        if self.robot_type not in ["koch", "koch_bimanual", "aloha", "so100", "moss"]:
            raise ValueError(f"Provided robot type ({self.robot_type}) is not supported.")


class ManipulatorRobot:
    # TODO(rcadene): Implement force feedback
    """This class allows to control any manipulator robot of various number of motors.

    Non exaustive list of robots:
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow expansion, developed
    by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    - [Aloha](https://www.trossenrobotics.com/aloha-kits) developed by Trossen Robotics

    Example of highest frequency teleoperation without camera:
    ```python
    # Defines how to communicate with the motors of the leader and follower arms
    leader_arms = {
        "main": DynamixelMotorsBus(
            port="/dev/tty.usbmodem575E0031751",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl330-m077"),
                "shoulder_lift": (2, "xl330-m077"),
                "elbow_flex": (3, "xl330-m077"),
                "wrist_flex": (4, "xl330-m077"),
                "wrist_roll": (5, "xl330-m077"),
                "gripper": (6, "xl330-m077"),
            },
        ),
    }
    follower_arms = {
        "main": DynamixelMotorsBus(
            port="/dev/tty.usbmodem575E0032081",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl430-w250"),
                "shoulder_lift": (2, "xl430-w250"),
                "elbow_flex": (3, "xl330-m288"),
                "wrist_flex": (4, "xl330-m288"),
                "wrist_roll": (5, "xl330-m288"),
                "gripper": (6, "xl330-m288"),
            },
        ),
    }
    robot = ManipulatorRobot(
        robot_type="koch",
        calibration_dir=".cache/calibration/koch",
        leader_arms=leader_arms,
        follower_arms=follower_arms,
    )

    # Connect motors buses and cameras if any (Required)
    robot.connect()

    while True:
        robot.teleop_step()
    ```

    Example of highest frequency data collection without camera:
    ```python
    # Assumes leader and follower arms have been instantiated already (see first example)
    robot = ManipulatorRobot(
        robot_type="koch",
        calibration_dir=".cache/calibration/koch",
        leader_arms=leader_arms,
        follower_arms=follower_arms,
    )
    robot.connect()
    while True:
        observation, action = robot.teleop_step(record_data=True)
    ```

    Example of highest frequency data collection with cameras:
    ```python
    # Defines how to communicate with 2 cameras connected to the computer.
    # Here, the webcam of the laptop and the phone (connected in USB to the laptop)
    # can be reached respectively using the camera indices 0 and 1. These indices can be
    # arbitrary. See the documentation of `OpenCVCamera` to find your own camera indices.
    cameras = {
        "laptop": OpenCVCamera(camera_index=0, fps=30, width=640, height=480),
        "phone": OpenCVCamera(camera_index=1, fps=30, width=640, height=480),
    }

    # Assumes leader and follower arms have been instantiated already (see first example)
    robot = ManipulatorRobot(
        robot_type="koch",
        calibration_dir=".cache/calibration/koch",
        leader_arms=leader_arms,
        follower_arms=follower_arms,
        cameras=cameras,
    )
    robot.connect()
    while True:
        observation, action = robot.teleop_step(record_data=True)
    ```

    Example of controlling the robot with a policy (without running multiple policies in parallel to ensure highest frequency):
    ```python
    # Assumes leader and follower arms + cameras have been instantiated already (see previous example)
    robot = ManipulatorRobot(
        robot_type="koch",
        calibration_dir=".cache/calibration/koch",
        leader_arms=leader_arms,
        follower_arms=follower_arms,
        cameras=cameras,
    )
    robot.connect()
    while True:
        # Uses the follower arms and cameras to capture an observation
        observation = robot.capture_observation()

        # Assumes a policy has been instantiated
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Orders the robot to move
        robot.send_action(action)
    ```

    Example of disconnecting which is not mandatory since we disconnect when the object is deleted:
    ```python
    robot.disconnect()
    ```
    """

    def __init__(
        self,
        config: ManipulatorRobotConfig | None = None,
        calibration_dir: Path = ".cache/calibration/koch",
        **kwargs,
    ):
        if config is None:
            config = ManipulatorRobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)
        self.calibration_dir = Path(calibration_dir)

        self.robot_type = self.config.robot_type
        self.leader_arms = self.config.leader_arms
        self.follower_arms = self.config.follower_arms
        self.cameras = self.config.cameras
        self.is_connected = False
        self.logs = {}

    def get_motor_names(self, arm: dict[str, MotorsBus]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arm.items() for motor in bus.motors]

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = self.get_motor_names(self.leader_arms)
        state_names = self.get_motor_names(self.leader_arms)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available_arms = []
        for name in self.follower_arms:
            arm_id = get_arm_id(name, "follower")
            available_arms.append(arm_id)
        for name in self.leader_arms:
            arm_id = get_arm_id(name, "leader")
            available_arms.append(arm_id)
        return available_arms

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras:
            raise ValueError(
                "ManipulatorRobot doesn't have any device to connect. See example of usage in docstring of the class."
            )

        # Connect the arms
        for name in self.follower_arms:
            print(f"Connecting {name} follower arm.")
            self.follower_arms[name].connect()
        for name in self.leader_arms:
            print(f"Connecting {name} leader arm.")
            self.leader_arms[name].connect()

        if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
        elif self.robot_type in ["so100", "moss"]:
            from lerobot.common.robot_devices.motors.feetech import TorqueMode

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        for name in self.follower_arms:
            self.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
        for name in self.leader_arms:
            self.leader_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)

        self.activate_calibration()

        # Set robot preset (e.g. torque in leader gripper for Koch v1.1)
        if self.robot_type in ["koch", "koch_bimanual"]:
            self.set_koch_robot_preset()
        elif self.robot_type == "aloha":
            self.set_aloha_robot_preset()
        elif self.robot_type in ["so100", "moss"]:
            self.set_so100_robot_preset()

        # Enable torque on all motors of the follower arms
        for name in self.follower_arms:
            print(f"Activating torque on {name} follower arm.")
            self.follower_arms[name].write("Torque_Enable", 1)

        if self.config.gripper_open_degree is not None:
            if self.robot_type not in ["koch", "koch_bimanual"]:
                raise NotImplementedError(
                    f"{self.robot_type} does not support position AND current control in the handle, which is require to set the gripper open."
                )
            # Set the leader arm in torque mode with the gripper motor set to an angle. This makes it possible
            # to squeeze the gripper and have it spring back to an open position on its own.
            for name in self.leader_arms:
                self.leader_arms[name].write("Torque_Enable", 1, "gripper")
                self.leader_arms[name].write("Goal_Position", self.config.gripper_open_degree, "gripper")

        # Check both arms can be read
        for name in self.follower_arms:
            self.follower_arms[name].read("Present_Position")
        for name in self.leader_arms:
            self.leader_arms[name].read("Present_Position")

        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True

    def activate_calibration(self):
        """After calibration all motors function in human interpretable ranges.
        Rotations are expressed in degrees in nominal range of [-180, 180],
        and linear motions (like gripper of Aloha) in nominal range of [0, 100].
        """

        def load_or_run_calibration_(name, arm, arm_type):
            arm_id = get_arm_id(name, arm_type)
            arm_calib_path = self.calibration_dir / f"{arm_id}.json"

            if arm_calib_path.exists():
                with open(arm_calib_path) as f:
                    calibration = json.load(f)
            else:
                # TODO(rcadene): display a warning in __init__ if calibration file not available
                print(f"Missing calibration file '{arm_calib_path}'")

                if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
                    from lerobot.common.robot_devices.robots.dynamixel_calibration import run_arm_calibration

                    calibration = run_arm_calibration(arm, self.robot_type, name, arm_type)

                elif self.robot_type in ["so100", "moss"]:
                    from lerobot.common.robot_devices.robots.feetech_calibration import (
                        run_arm_manual_calibration,
                    )

                    calibration = run_arm_manual_calibration(arm, self.robot_type, name, arm_type)

                print(f"Calibration is done! Saving calibration file '{arm_calib_path}'")
                arm_calib_path.parent.mkdir(parents=True, exist_ok=True)
                with open(arm_calib_path, "w") as f:
                    json.dump(calibration, f)

            return calibration

        for name, arm in self.follower_arms.items():
            calibration = load_or_run_calibration_(name, arm, "follower")
            arm.set_calibration(calibration)
        for name, arm in self.leader_arms.items():
            calibration = load_or_run_calibration_(name, arm, "leader")
            arm.set_calibration(calibration)

    def set_koch_robot_preset(self):
        def set_operating_mode_(arm):
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode

            if (arm.read("Torque_Enable") != TorqueMode.DISABLED.value).any():
                raise ValueError("To run set robot preset, the torque must be disabled on all motors.")

            # Use 'extended position mode' for all motors except gripper, because in joint mode the servos can't
            # rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling the arm,
            # you could end up with a servo with a position 0 or 4095 at a crucial point See [
            # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
            all_motors_except_gripper = [name for name in arm.motor_names if name != "gripper"]
            if len(all_motors_except_gripper) > 0:
                # 4 corresponds to Extended Position on Koch motors
                arm.write("Operating_Mode", 4, all_motors_except_gripper)

            # Use 'position control current based' for gripper to be limited by the limit of the current.
            # For the follower gripper, it means it can grasp an object without forcing too much even tho,
            # it's goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
            # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
            # to make it move, and it will move back to its original target position when we release the force.
            # 5 corresponds to Current Controlled Position on Koch gripper motors "xl330-m077, xl330-m288"
            arm.write("Operating_Mode", 5, "gripper")

        for name in self.follower_arms:
            set_operating_mode_(self.follower_arms[name])

            # Set better PID values to close the gap between recorded states and actions
            # TODO(rcadene): Implement an automatic procedure to set optimial PID values for each motor
            self.follower_arms[name].write("Position_P_Gain", 1500, "elbow_flex")
            self.follower_arms[name].write("Position_I_Gain", 0, "elbow_flex")
            self.follower_arms[name].write("Position_D_Gain", 600, "elbow_flex")

        if self.config.gripper_open_degree is not None:
            for name in self.leader_arms:
                set_operating_mode_(self.leader_arms[name])

                # Enable torque on the gripper of the leader arms, and move it to 45 degrees,
                # so that we can use it as a trigger to close the gripper of the follower arms.
                self.leader_arms[name].write("Torque_Enable", 1, "gripper")
                self.leader_arms[name].write("Goal_Position", self.config.gripper_open_degree, "gripper")

    def set_aloha_robot_preset(self):
        def set_shadow_(arm):
            # Set secondary/shadow ID for shoulder and elbow. These joints have two motors.
            # As a result, if only one of them is required to move to a certain position,
            # the other will follow. This is to avoid breaking the motors.
            if "shoulder_shadow" in arm.motor_names:
                shoulder_idx = arm.read("ID", "shoulder")
                arm.write("Secondary_ID", shoulder_idx, "shoulder_shadow")

            if "elbow_shadow" in arm.motor_names:
                elbow_idx = arm.read("ID", "elbow")
                arm.write("Secondary_ID", elbow_idx, "elbow_shadow")

        for name in self.follower_arms:
            set_shadow_(self.follower_arms[name])

        for name in self.leader_arms:
            set_shadow_(self.leader_arms[name])

        for name in self.follower_arms:
            # Set a velocity limit of 131 as advised by Trossen Robotics
            self.follower_arms[name].write("Velocity_Limit", 131)

            # Use 'extended position mode' for all motors except gripper, because in joint mode the servos can't
            # rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling the arm,
            # you could end up with a servo with a position 0 or 4095 at a crucial point See [
            # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
            all_motors_except_gripper = [
                name for name in self.follower_arms[name].motor_names if name != "gripper"
            ]
            if len(all_motors_except_gripper) > 0:
                # 4 corresponds to Extended Position on Aloha motors
                self.follower_arms[name].write("Operating_Mode", 4, all_motors_except_gripper)

            # Use 'position control current based' for follower gripper to be limited by the limit of the current.
            # It can grasp an object without forcing too much even tho,
            # it's goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
            # 5 corresponds to Current Controlled Position on Aloha gripper follower "xm430-w350"
            self.follower_arms[name].write("Operating_Mode", 5, "gripper")

            # Note: We can't enable torque on the leader gripper since "xc430-w150" doesn't have
            # a Current Controlled Position mode.

        if self.config.gripper_open_degree is not None:
            warnings.warn(
                f"`gripper_open_degree` is set to {self.config.gripper_open_degree}, but None is expected for Aloha instead",
                stacklevel=1,
            )

    def set_so100_robot_preset(self):
        for name in self.follower_arms:
            # Mode=0 for Position Control
            self.follower_arms[name].write("Mode", 0)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.follower_arms[name].write("P_Coefficient", 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.follower_arms[name].write("I_Coefficient", 0)
            self.follower_arms[name].write("D_Coefficient", 32)
            # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
            # which is mandatory for Maximum_Acceleration to take effect after rebooting.
            self.follower_arms[name].write("Lock", 0)
            # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
            # the motors. Note: this configuration is not in the official STS3215 Memory Table
            self.follower_arms[name].write("Maximum_Acceleration", 254)
            self.follower_arms[name].write("Acceleration", 254)

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Prepare to assign the position of the leader to the follower
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

        # Send goal position to the follower
        follower_goal_pos = {}
        for name in self.follower_arms:
            before_fwrite_t = time.perf_counter()
            goal_pos = leader_pos[name]

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Used when record_data=True
            follower_goal_pos[name] = goal_pos

            goal_pos = goal_pos.numpy().astype(np.int32)
            self.follower_arms[name].write("Goal_Position", goal_pos)
            self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t

        # Early exit when recording data is not requested
        if not record_data:
            return

        # TODO(rcadene): Add velocity and other info
        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Create action by concatenating follower goal position
        action = []
        for name in self.follower_arms:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action = torch.cat(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionnaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionnaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Command the follower arms to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action: tensor containing the concatenated goal positions for the follower arms.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        from_idx = 0
        to_idx = 0
        action_sent = []
        for name in self.follower_arms:
            # Get goal position of each follower arm by splitting the action vector
            to_idx += len(self.follower_arms[name].motor_names)
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Save tensor to concat and return
            action_sent.append(goal_pos)

            # Send goal position to each follower
            goal_pos = goal_pos.numpy().astype(np.int32)
            self.follower_arms[name].write("Goal_Position", goal_pos)

        return torch.cat(action_sent)

    def print_logs(self):
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        for name in self.follower_arms:
            self.follower_arms[name].disconnect()

        for name in self.leader_arms:
            self.leader_arms[name].disconnect()

        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

class ControllerType(Enum):
    PS5 = "ps5"
    XBOX = "xbox"

@dataclass
class ControllerConfig:
    resolution: dict
    scale: dict

class JoystickExpert:
    """
    This class provides an interface to the Joystick/Gamepad.
    It continuously reads the joystick state and provides
    a "get_action" method to get the latest action and button state.
    """

    CONTROLLER_CONFIGS = {
        ControllerType.PS5: ControllerConfig(
            # PS5 controller joystick values have 8 bit resolution [0, 255]
            resolution={
                'ABS_X': 2**8,
                'ABS_Y': 2**8,
                'ABS_RX': 2**8,
                'ABS_RY': 2**8,
                'ABS_Z': 2**8,
                'ABS_RZ': 2**8,
                'ABS_HAT0X': 1.0,
            },
            scale={
                'ABS_X': 0.4,
                'ABS_Y': 0.4,
                'ABS_RX': 0.5,
                'ABS_RY': 0.5,
                'ABS_Z': 0.8,
                'ABS_RZ': 1.2,
                'ABS_HAT0X': 0.5,
            }
        ),
        ControllerType.XBOX: ControllerConfig(
            # XBOX controller joystick values have 16 bit resolution [0, 65535]
            resolution={
                'ABS_X': 2**16,
                'ABS_Y': 2**16,
                'ABS_RX': 2**16,
                'ABS_RY': 2**16,
                'ABS_Z': 2**8,
                'ABS_RZ': 2**8,
                'ABS_HAT0X': 1.0,
            },
            scale={
                'ABS_X': 0.05,
                'ABS_Y': -0.05,
                'ABS_RX': -0.03,
                'ABS_RY': -0.03,
                'ABS_Z': 0.05,
                'ABS_RZ': 0.05,
                'ABS_HAT0X': 0.03,
            }
        ),
    }

    def __init__(self, controller_type=ControllerType.XBOX):
        self.controller_type = controller_type
        self.controller_config = self.CONTROLLER_CONFIGS[controller_type]

        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 3
        self.latest_data["buttons"] = [0.0]

        # Start a process to continuously read Joystick state
        self.process = multiprocessing.Process(target=self._read_joystick)
        self.process.daemon = True
        self.process.start()


    def _read_joystick(self):        
        action = [0.0] * 3
        buttons = [0.0]
        
        while True:
            try:
                # Get fresh events
                events = inputs.get_gamepad()
          
                # Process events
                for event in events:
                    if event.code in self.controller_config.resolution:
                        # Calculate relative changes based on the axis
                        # Normalize the joystick input values to range [-1, 1] expected by the environment
                        resolution = self.controller_config.resolution[event.code]
                        if self.controller_type == ControllerType.PS5:
                            normalized_value = (event.state - (resolution / 2)) / (resolution / 2)
                        else:
                            normalized_value = event.state / (resolution / 2)
                        scaled_value = normalized_value * self.controller_config.scale[event.code]

                        if event.code == 'ABS_X':
                            action[0] = scaled_value
                        elif event.code == 'ABS_Y':
                            action[1] = scaled_value
                        elif event.code == 'ABS_RY':
                            action[2] = scaled_value

                        # Handle button events
                        elif event.code == 'ABS_RZ':
                            buttons[0] = scaled_value
                        elif event.code == 'ABS_Z':
                            # Flip sign so this will go in the down direction
                            buttons[0] = -scaled_value

                # Update the shared state
                self.latest_data["action"] = action
                self.latest_data["buttons"] = buttons
                
            except inputs.UnpluggedError:
                print("No controller found. Retrying...")
                time.sleep(1)

    def get_action(self):
        """Returns the latest action and button state from the Joystick."""
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons

class SimManipulatorRobot(ManipulatorRobot):
    def __init__(
        self,
        config: ManipulatorRobotConfig | None = None,
        calibration_dir: Path = ".cache/calibration/koch",
        controller_type=ControllerType.XBOX,
        **kwargs,
    ):
        super().__init__(config, calibration_dir, **kwargs)
        self.expert = JoystickExpert(controller_type=controller_type)
        self._action_scale = np.array([0.05, 1])
        self.bounds = np.asarray([[-0.1, 0.015, 0.0], [0.1, 0.2, 0.2]])
        self.cube_low = np.array([-0.2 / 2, 0.015, .0055])
        self.cube_high = np.array([0.2 / 2, 0.2 / 2, .0055])
        self.target_low = np.array([-0.2 / 2, 0.015 / 2, .0055])
        self.target_high = np.array([0.2 / 2, 0.2 / 2, 0.1])
        self.num_dof = 6

    def initialize_episode(self, model, data):
        """Initialize the robot state at the start of each episode"""
        self.configuration = mink.Configuration(model)

        self.end_effector_task = mink.FrameTask(
            frame_name="end_effector_site",
            frame_type="site",
            position_cost=200.0,
            orientation_cost=0.0,
            lm_damping=0.01,
        )

        self.tasks = [self.end_effector_task]

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        self.configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        
        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        # Update cube and target position
        cube_pos = np.random.uniform(self.cube_low, self.cube_high)
        cube_rot = np.array([1.0, 0.0, 0.0, 0.0])
        data.qpos[self.num_dof : self.num_dof + 7] = np.concatenate([cube_pos, cube_rot])

        # Sample the target position
        target_pos = np.random.uniform(self.target_low, self.target_high).astype(np.float32)

        # Update visualization
        model.geom("target_region").pos = target_pos[:]

        # Step the simulation
        mujoco.mj_forward(model, data)
    
    def mink_solve_ik(
        self,
        model,
        data,
        solver: str = "quadprog",
        pos_threshold: float = 1e-4,
        ori_threshold: float = 1e-4,
        max_iters: int = 20,
    ) -> np.ndarray:
        """Solve IK using mink"""
        # Update end-effector task
        t_wt = mink.SE3.from_mocap_name(model, data, "target")
        self.end_effector_task.set_target(t_wt)

        # Compute velocity and integrate into the next configuration.
        for _ in range(max_iters):
            vel = mink.solve_ik(self.configuration, self.tasks, model.opt.timestep, solver, 1e-3)
            self.configuration.integrate_inplace(vel, model.opt.timestep)
            err = self.end_effector_task.compute_error(self.configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
            if pos_achieved and ori_achieved:
                break
        
        return self.configuration.q[:4]

    def diffik(
        self,
        model,
        data,
        site_name,
        joint_names,
        body_names,
        damping: float = 0.1,
        integration_dt: float = 1.0,
        gravity_compensation: bool = True,
        max_angvel: float = 0.785,
    ):
        site_id = model.site(site_name).id

        body_ids = [model.body(name).id for name in body_names]
        if gravity_compensation:
            model.body_gravcomp[body_ids] = 1.0

        dof_ids = np.array([model.joint(name).id for name in joint_names])

        # Mocap body we will control with our mouse.
        mocap_id = model.body("target").mocapid[0]

        # Pre-allocate numpy arrays.
        jac = np.zeros((3, model.nv))
        diag = damping * np.eye(3)
        twist = np.zeros(3)
        eye = np.eye(model.nv)
        Kpos: float = 0.95

        # Nullspace P gain.
        Kn = np.asarray([0.0, 0.0, 0.0, 0.0])

        # Initial joint configuration saved as a keyframe in the XML file.
        key_name = "home"
        q0 = model.key(key_name).qpos
            
        dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
        twist = Kpos * dx / integration_dt

        # Get the Jacobian with respect to the end-effector site.
        mujoco.mj_jacSite(model, data, jac, None, site_id)

        # Solve system of equations: J @ dq = error.
        dq = jac[:, dof_ids].T @ np.linalg.solve(jac[:, dof_ids] @ jac[:, dof_ids].T + diag, twist)

        # Nullspace control biasing joint velocities towards the home configuration.
        dq += (eye[dof_ids,dof_ids] - np.linalg.pinv(jac[:, dof_ids], rcond=1e-4) @ jac[:, dof_ids]) @ (Kn * (q0 - data.qpos)[dof_ids])

        # Scale down joint velocities if they exceed maximum.
        if max_angvel > 0:
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > max_angvel:
                dq *= max_angvel / dq_abs_max

        # Integrate joint velocities to obtain joint positions.
        q = data.qpos[dof_ids].copy()
        q += dq * integration_dt

        # Set the control signal.
        np.clip(q[dof_ids], *model.jnt_range[dof_ids].T, out=q[dof_ids])
        
        return q[dof_ids]
        
    def teleop_step(self, record_data=False):
        """Override teleop_step to use joystick control instead of real leader arm"""
        print("Starting teleop_step...")  # Debug print
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Get joystick action
        try:
            action, buttons = self.expert.get_action()
            # print(f"Got joystick action: {action}, buttons: {buttons}")  # Debug print
        except Exception as e:
            print(f"Error getting joystick action: {e}")  # Debug print
            raise
       
        # Update follower arm positions through inverse kinematics
        follower_pos = {}
        for name in self.follower_arms:
            before_lread_t = time.perf_counter()
            
            deadzone = 0.001
            if np.linalg.norm(action) < deadzone:
                action = np.zeros_like(action)

            x, y, z = action
            grasp = buttons[0]

            # Set the mocap position.
            pos = self.follower_arms[name].data.mocap_pos[0].copy()
            dpos = np.asarray([x, y, z]) * self._action_scale[0]
            npos = np.clip(pos + dpos, *self.bounds)
            self.follower_arms[name].data.mocap_pos[0] = npos

            # Set gripper grasp.
            g = self.follower_arms[name].data.ctrl[5]
            dg = grasp * self._action_scale[1]
            ng = np.clip(g + dg, -2.3, 0.032)
            self.follower_arms[name].data.ctrl[5] = ng

            tau = self.mink_solve_ik(
                self.follower_arms[name].model,
                self.follower_arms[name].data,
            )

            for _ in range(20):
                # Set the target position
                self.follower_arms[name].data.ctrl[:4] = tau

                # Step the simulation forward
                mujoco.mj_step(self.follower_arms[name].model, self.follower_arms[name].data)

            # Combine all joint positions
            follower_pos[name] = self.follower_arms[name].data.qpos[:6].copy()
            
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

        if not record_data:
            return

        # Record data if requested (same as parent class)
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        action = []
        for name in self.follower_arms:
            if name in follower_pos:
                action.append(follower_pos[name])
        action = torch.cat(action)

        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].read(self.follower_arms["main"].model, self.follower_arms["main"].data)
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict
