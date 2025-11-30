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

"""Test script for EarthRover Mini Plus cameras with warmup."""

import time
import cv2

from lerobot.cameras.earthrover_mini_camera import EarthRoverMiniCamera, EarthRoverMiniCameraConfig
from lerobot.cameras.earthrover_mini_camera.configuration_earthrover_mini import ColorMode

print("=== Testing EarthRover Mini Plus Cameras ===\n")

# Test front camera
print("1. Testing FRONT camera...")
front_config = EarthRoverMiniCameraConfig(
    index_or_path=EarthRoverMiniCameraConfig.FRONT_CAM_MAIN,
    color_mode=ColorMode.RGB,
)
front_cam = EarthRoverMiniCamera(front_config)

print("   Connecting to front camera...")
front_cam.connect()
print("   ✓ Front camera connected")

# WARMUP: Read and discard first 10 frames
print("   Warming up (discarding first 10 frames)...")
for i in range(10):
    _ = front_cam.read()
    time.sleep(0.1)
print("   ✓ Warmup complete")

print("   Capturing front camera frames...")
for i in range(5):
    frame = front_cam.read()
    filename = f"test_front_{i:03d}.jpg"
    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"   ✓ Saved {filename} - shape: {frame.shape}")

front_cam.disconnect()
print("   ✓ Front camera disconnected\n")

# Test rear camera
print("2. Testing REAR camera...")
rear_config = EarthRoverMiniCameraConfig(
    index_or_path=EarthRoverMiniCameraConfig.REAR_CAM_MAIN,
    color_mode=ColorMode.RGB,
)
rear_cam = EarthRoverMiniCamera(rear_config)

print("   Connecting to rear camera...")
rear_cam.connect()
print("   ✓ Rear camera connected")

# WARMUP: Read and discard first 10 frames
print("   Warming up (discarding first 10 frames)...")
for i in range(10):
    _ = rear_cam.read()
    time.sleep(0.1)
print("   ✓ Warmup complete")

print("   Capturing rear camera frames...")
for i in range(5):
    frame = rear_cam.read()
    filename = f"test_rear_{i:03d}.jpg"
    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"   ✓ Saved {filename} - shape: {frame.shape}")

rear_cam.disconnect()
print("   ✓ Rear camera disconnected\n")

print("=" * 50)
print("✅ Test complete!")
print("=" * 50)
print("Front camera frames: test_front_000.jpg to test_front_004.jpg")
print("Rear camera frames:  test_rear_000.jpg to test_rear_004.jpg")