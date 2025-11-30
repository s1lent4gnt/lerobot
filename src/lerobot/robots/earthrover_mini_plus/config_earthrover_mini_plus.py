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
"""Configuration for EarthRover Mini Plus robot."""

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.earthrover_mini_camera.configuration_earthrover_mini import (
    ColorMode,
    EarthRoverMiniCameraConfig,
)

from ..config import RobotConfig


def earthrover_mini_plus_cameras() -> dict[str, CameraConfig]:
    """
    Default camera configuration for EarthRover Mini Plus.

    Configures two RTSP camera streams:
    - front: Front main camera (1920x1080 @ 30fps)
    - rear: Rear main camera (1920x1080 @ 30fps)

    Returns:
        Dictionary mapping camera names to their configurations
    """
    return {
        "front": EarthRoverMiniCameraConfig(
            index_or_path=EarthRoverMiniCameraConfig.FRONT_CAM_MAIN,
            color_mode=ColorMode.RGB,
        ),
        "rear": EarthRoverMiniCameraConfig(
            index_or_path=EarthRoverMiniCameraConfig.REAR_CAM_MAIN,
            color_mode=ColorMode.RGB,
        ),
    }


@RobotConfig.register_subclass("earthrover_mini_plus")
@dataclass
class EarthRoverMiniPlusConfig(RobotConfig):
    """
    Configuration for EarthRover Mini Plus mobile robot.

    This robot uses TCP communication for control and RTSP for camera streams.
    The default configuration connects to the robot at 1192.168.1.84:8888.

    Attributes:
        robot_ip: IP address of the robot (default: "1192.168.1.84")
        robot_port: TCP port for robot communication (default: 8888)
        cameras: Dictionary of camera configurations

    Example:
        ```python
        from lerobot.robots.earthrover_mini_plus import EarthRoverMiniPlusConfig

        # Use default configuration
        config = EarthRoverMiniPlusConfig()

        # Or customize
        config = EarthRoverMiniPlusConfig(
            robot_ip="192.168.1.84",
            robot_port=8888
        )
        ```
    """

    robot_ip: str = "192.168.1.84"
    robot_port: int = 8888
    cameras: dict[str, CameraConfig] = field(default_factory=earthrover_mini_plus_cameras)
