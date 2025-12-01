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


def earthrover_mini_plus_cameras(robot_ip: str = "172.18.130.174") -> dict[str, CameraConfig]:
    """
    Camera configuration for EarthRover Mini Plus.

    Configures two RTSP camera streams using the robot's IP address:
    - front: Front main camera (1920x1080 @ 30fps)
    - rear: Rear main camera (1920x1080 @ 30fps)

    Args:
        robot_ip: IP address of the robot (default: "172.18.130.174")

    Returns:
        Dictionary mapping camera names to their configurations

    Example:
        >>> cameras = earthrover_mini_plus_cameras("192.168.0.100")
        >>> cameras["front"].index_or_path
        'rtsp://192.168.0.100/live/0'
    """
    return {
        "front": EarthRoverMiniCameraConfig(
            index_or_path=EarthRoverMiniCameraConfig.get_camera_url(robot_ip, "front", "main"),
            color_mode=ColorMode.RGB,
        ),
        "rear": EarthRoverMiniCameraConfig(
            index_or_path=EarthRoverMiniCameraConfig.get_camera_url(robot_ip, "rear", "main"),
            color_mode=ColorMode.RGB,
        ),
    }


@RobotConfig.register_subclass("earthrover_mini_plus")
@dataclass
class EarthRoverMiniPlusConfig(RobotConfig):
    """
    Configuration for EarthRover Mini Plus mobile robot.

    This robot uses TCP communication for control and RTSP for camera streams.
    The camera URLs are automatically configured based on the robot_ip.

    Attributes:
        robot_ip: IP address of the robot (default: "172.18.130.174")
        robot_port: TCP port for robot communication (default: 8888)
        cameras: Dictionary of camera configurations (auto-generated from robot_ip)

    Example:
        ```python
        from lerobot.robots.earthrover_mini_plus import EarthRoverMiniPlusConfig

        # Use default configuration (172.18.130.174)
        config = EarthRoverMiniPlusConfig()

        # Customize robot IP - cameras will automatically use this IP
        config = EarthRoverMiniPlusConfig(robot_ip="192.168.0.100")
        # Cameras will be at rtsp://192.168.0.100/live/0 and rtsp://192.168.0.100/live/2
        ```
    """

    robot_ip: str = "172.18.130.174"
    robot_port: int = 8888
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})

    def __post_init__(self):
        """Initialize cameras with the configured robot_ip."""
        # Only generate cameras if not explicitly provided
        if not self.cameras:
            self.cameras = earthrover_mini_plus_cameras(self.robot_ip)
