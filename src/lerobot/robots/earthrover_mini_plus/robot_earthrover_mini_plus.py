#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""EarthRover Mini Plus robot implementation using low-level TCP SDK."""

import logging
from functools import cached_property
from typing import Any

from earth_rover_mini_sdk import EarthRoverMini_API
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_earthrover_mini_plus import EarthRoverMiniPlusConfig

logger = logging.getLogger(__name__)


class EarthRoverMiniPlus(Robot):
    """
    EarthRover Mini Plus mobile robot with low-level TCP SDK control.

    This robot uses the earth_rover_mini_sdk package for direct TCP communication
    with the robot (192.168.1.84:8888) and RTSP camera streams for vision.

    The robot provides:
    - Differential drive control (linear and angular velocity)
    - RTSP camera feeds (front and rear)
    - Telemetry data (motor RPMs, IMU, battery, etc.)

    Example:
        ```python
        from lerobot.robots.earthrover_mini_plus import EarthRoverMiniPlus, EarthRoverMiniPlusConfig

        # Create robot with default config
        robot = EarthRoverMiniPlus(EarthRoverMiniPlusConfig())

        # Connect to robot and cameras
        robot.connect()

        # Send movement command
        action = {"speed": 50, "heading": 0}
        robot.send_action(action)

        # Get observation
        obs = robot.get_observation()
        print(f"Front camera shape: {obs['front'].shape}")
        print(f"Motor RPMs: {obs['motor_Fl']}")

        # Disconnect
        robot.disconnect()
        ```

    Note:
        Requires earth_rover_mini_sdk: pip install earth-rover-mini-sdk
    """

    config_class = EarthRoverMiniPlusConfig
    name = "earthrover_mini_plus"

    def __init__(self, config: EarthRoverMiniPlusConfig):
        """
        Initialize EarthRover Mini Plus robot.

        Args:
            config: Robot configuration including camera settings
        """
        super().__init__(config)
        self.config = config

        # TCP SDK connection (will be initialized in connect())
        self.earth_rover: EarthRoverMini_API | None = None

        # RTSP cameras from configuration
        self.cameras = make_cameras_from_configs(config.cameras)

        # Connection state
        self._is_connected = False

        logger.info(f"Initialized {self.name} with {len(self.cameras)} cameras")

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to robot via TCP SDK and initialize cameras.

        Args:
            calibrate: Whether to run calibration (not required for this robot)

        Raises:
            DeviceAlreadyConnectedError: If robot is already connected
            DeviceNotConnectedError: If connection to robot or cameras fails
        """
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected")

        # Connect to robot via TCP SDK
        logger.info(f"Connecting to {self.name} at {self.config.robot_ip}:{self.config.robot_port}")
        self.earth_rover = EarthRoverMini_API(
            ip=self.config.robot_ip,
            port=self.config.robot_port
        )
        self.earth_rover.connect()

        # Connect to RTSP cameras
        for cam_name, cam in self.cameras.items():
            logger.info(f"Connecting to camera '{cam_name}' at {cam.config.index_or_path}")
            cam.connect()
            if not cam.is_connected:
                raise DeviceNotConnectedError(
                    f"Failed to connect to camera '{cam_name}' at {cam.config.index_or_path}"
                )
            logger.info(f"✓ Camera '{cam_name}' connected")

        # Run configuration
        self.configure()

        # Update connection state
        self._is_connected = True
        logger.info(f"✓ {self.name} connected successfully")

        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        """Calibration not required for EarthRover Mini Plus."""
        logger.info("Calibration not required for this robot")

    @property
    def is_calibrated(self) -> bool:
        """Robot doesn't require calibration."""
        return True

    def configure(self) -> None:
        """Configure robot settings (currently no configuration needed)."""
        pass

    @cached_property
    def observation_features(self) -> dict:
        """
        Define observation space for the robot.

        Returns:
            Dictionary mapping feature names to their types/shapes:
            - Cameras: (height, width, 3) tuples
            - Motor RPMs: float values
            - IMU data: float values (accelerometer, gyro, magnetometer)
            - Speed and heading: float values
        """
        features = {}

        # Camera features
        for cam_name, cam in self.cameras.items():
            features[cam_name] = (cam.height, cam.width, 3)

        # Motor RPM features
        features.update({
            "motor_Fl": float,
            "motor_Fr": float,
            "motor_Br": float,
            "motor_Bl": float,
        })

        # IMU features
        features.update({
            "accel_x": float,
            "accel_y": float,
            "accel_z": float,
            "gyro_x": float,
            "gyro_y": float,
            "gyro_z": float,
            "mag_x": float,
            "mag_y": float,
            "mag_z": float,
        })

        # Speed and heading features
        features.update({
            "speed": float,
            "heading": float,
        })

        return features

    @cached_property
    def action_features(self) -> dict:
        """
        Define action space for the robot.

        Returns:
            Dictionary with speed and heading control:
            - speed: Linear velocity (float)
            - heading: Angular velocity (float)
        """
        return {
            "speed": float,
            "heading": float,
        }

    def get_observation(self) -> dict[str, Any]:
        """
        Get current robot observation including cameras and telemetry.

        Returns:
            Dictionary containing:
            - Camera frames (numpy arrays)
            - Motor RPMs
            - IMU data (accelerometer, gyro, magnetometer)
            - Speed and heading

        Raises:
            DeviceNotConnectedError: If robot is not connected
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected")

        observation = {}

        # Get telemetry from SDK (RPMs, IMU, etc.)
        telemetry = self.earth_rover.get_telemetry()
        
        # Transform SDK telemetry format to match observation_features
        # SDK returns arrays, we need individual named fields
        if 'rpm' in telemetry and len(telemetry['rpm']) == 4:
            observation['motor_Fl'] = float(telemetry['rpm'][0])
            observation['motor_Fr'] = float(telemetry['rpm'][1])
            observation['motor_Br'] = float(telemetry['rpm'][2])
            observation['motor_Bl'] = float(telemetry['rpm'][3])
        
        # IMU accelerometer (use m/s^2 if available, otherwise g)
        acc_data = telemetry.get('acc_ms2', telemetry.get('acc_g', [0, 0, 0]))
        if len(acc_data) == 3:
            observation['accel_x'] = float(acc_data[0])
            observation['accel_y'] = float(acc_data[1])
            observation['accel_z'] = float(acc_data[2])
        
        # IMU gyroscope
        gyro_data = telemetry.get('gyro_dps', [0, 0, 0])
        if len(gyro_data) == 3:
            observation['gyro_x'] = float(gyro_data[0])
            observation['gyro_y'] = float(gyro_data[1])
            observation['gyro_z'] = float(gyro_data[2])
        
        # IMU magnetometer
        mag_data = telemetry.get('mag_uT', [0, 0, 0])
        if len(mag_data) == 3:
            observation['mag_x'] = float(mag_data[0])
            observation['mag_y'] = float(mag_data[1])
            observation['mag_z'] = float(mag_data[2])
        
        # Speed and heading (these might not be in telemetry, use 0 as default)
        observation['speed'] = float(telemetry.get('speed', 0.0))
        observation['heading'] = float(telemetry.get('heading_deg', 0.0))

        # Get camera frames
        for cam_name, cam in self.cameras.items():
            observation[cam_name] = cam.async_read()

        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send control command to robot.

        Args:
            action: Dictionary with movement commands:
                - speed: Linear velocity (int, typically 0-100)
                - heading: Angular velocity (int, typically -100 to 100)
                Alternative keys also supported:
                - linear_velocity, linear.vel: Alias for speed
                - angular_velocity, angular.vel: Alias for heading

        Returns:
            The action that was sent (in standard format)

        Raises:
            DeviceNotConnectedError: If robot is not connected
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected")

        # Extract action values (support multiple key formats)
        speed = action.get("speed", action.get("linear_velocity", action.get("linear.vel", 0)))
        heading = action.get("heading", action.get("angular_velocity", action.get("angular.vel", 0)))

        # Send command to robot via SDK
        self.earth_rover.move_continuous_loop(speed=int(speed), angular=int(heading))

        logger.debug(f"Sent action: speed={speed}, heading={heading}")

        # Return in standard format (must match action_features)
        return {
            "speed": float(speed),
            "heading": float(heading),
        }

    def disconnect(self) -> None:
        """
        Disconnect from robot and cameras.

        Raises:
            DeviceNotConnectedError: If robot is not connected
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self.name} is not connected")

        # Disconnect cameras
        for cam_name, cam in self.cameras.items():
            logger.info(f"Disconnecting camera '{cam_name}'")
            cam.disconnect()

        # Disconnect robot
        if self.earth_rover is not None:
            logger.info(f"Disconnecting from {self.name}")
            self.earth_rover.disconnect()
            self.earth_rover = None

        self._is_connected = False
        logger.info(f"✓ {self.name} disconnected")
