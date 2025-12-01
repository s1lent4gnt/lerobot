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

import warnings
from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("earthrover_mini_camera")
@dataclass
class EarthRoverMiniCameraConfig(CameraConfig):
    """Configuration class for EarthRoverMini camera devices.

    This class provides configuration options for cameras accessed through OpenCV,
    supporting both physical camera devices and video files. It includes settings
    for resolution, frame rate, color mode, and image rotation.

    Example configuration with default IP:
    ```python
    EarthRoverMiniCameraConfig(
        index_or_path=EarthRoverMiniCameraConfig.FRONT_CAM_MAIN,  # front main stream
        color_mode=ColorMode.RGB,
    )
    ```

    Example configuration with custom robot IP:
    ```python
    robot_ip = "192.168.0.100"
    EarthRoverMiniCameraConfig(
        index_or_path=EarthRoverMiniCameraConfig.get_camera_url(robot_ip, "front", "main"),
        color_mode=ColorMode.RGB,
    )
    ```

    Attributes:
        index_or_path: Either an integer representing the camera device index,
                      or a Path object pointing to a video file.
        fps: Requested frames per second for the color stream.
        width: Requested frame width in pixels for the color stream.
        height: Requested frame height in pixels for the color stream.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)

    Note:
        - Only 3-channel color output (RGB/BGR) is currently supported.
    """

    # Default camera URLs (using default robot IP)
    FRONT_CAM_MAIN: str = "rtsp://192.168.1.84/live/0"
    FRONT_CAM_SUB: str = "rtsp://192.168.1.84/live/1"
    REAR_CAM_MAIN: str = "rtsp://192.168.1.84/live/2"
    REAR_CAM_SUB: str = "rtsp://192.168.1.84/live/3"

    @staticmethod
    def get_camera_url(robot_ip: str, position: str, stream: str) -> str:
        """
        Build camera RTSP URL for a specific robot IP.

        Args:
            robot_ip: IP address of the robot (e.g., "192.168.1.84")
            position: Camera position - "front" or "rear"
            stream: Stream quality - "main" (1920x1080) or "sub" (720x576)

        Returns:
            RTSP URL string

        Example:
            >>> EarthRoverMiniCameraConfig.get_camera_url("192.168.0.100", "front", "main")
            'rtsp://192.168.0.100/live/0'
        """
        stream_ids = {
            ("front", "main"): 0,
            ("front", "sub"): 1,
            ("rear", "main"): 2,
            ("rear", "sub"): 3,
        }

        key = (position.lower(), stream.lower())
        if key not in stream_ids:
            raise ValueError(
                f"Invalid position/stream combination: {position}/{stream}. "
                f"Valid combinations: front/main, front/sub, rear/main, rear/sub"
            )

        stream_id = stream_ids[key]
        return f"rtsp://{robot_ip}/live/{stream_id}"

    # Does not change actual fps, width, and height, just for camera info/logging purposes
    DEFAULT_FPS: float = 30.0

    MAIN_WIDTH: int = 1920
    MAIN_HEIGHT: int = 1080

    SUB_WIDTH: int = 720
    SUB_HEIGHT: int = 576

    index_or_path: str = FRONT_CAM_MAIN
    fps: float = DEFAULT_FPS
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self):
        main_cams = {self.FRONT_CAM_MAIN, self.REAR_CAM_MAIN}
        sub_cams = {self.FRONT_CAM_SUB, self.REAR_CAM_SUB}
        if self.index_or_path not in main_cams and self.index_or_path not in sub_cams:
            raise ValueError(f"index_or_path must be one of four allowed camera URLS: {main_cams} {sub_cams}")

        if self.fps is not None and self.fps != self.DEFAULT_FPS:
            warnings.warn("FPS cannot be modified for this camera — using default (30).", stacklevel=2)
            self.fps = self.DEFAULT_FPS

        if self.index_or_path in main_cams:
            if (self.width is not None and self.width != self.MAIN_WIDTH) or (
                self.height is not None and self.height != self.MAIN_HEIGHT
            ):
                warnings.warn(
                    "Resolution cannot be modified for this main camera — using default (1920x1080).",
                    stacklevel=2,
                )
            self.width = self.MAIN_WIDTH
            self.height = self.MAIN_HEIGHT

        if self.index_or_path in sub_cams:
            if (self.width is not None and self.width != self.SUB_WIDTH) or (
                self.height is not None and self.height != self.SUB_HEIGHT
            ):
                warnings.warn(
                    "Resolution cannot be modified for this sub camera — using default (720x576).", stacklevel=2
                )
            self.width = self.SUB_WIDTH
            self.height = self.SUB_HEIGHT

        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` is expected to be in {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, but {self.rotation} is provided."
            )
