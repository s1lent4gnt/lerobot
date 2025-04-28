from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import gymnasium as gym
import mujoco
import numpy as np
import torch
from gymnasium import spaces

from lerobot.franka_sim.franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from lerobot.franka_sim.franka_sim.controllers import opspace

_HERE = Path(__file__).parent
_XML_PATH = Path("/home/khalil/Documents/Khalil/Projects/Embodied-AI/lerobot/franka_sim/franka_sim/envs/xmls/arena.xml")
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.5, 0.15]])


class FrankaPushEnv(MujocoGymEnv):
    """
    Gymnasium environment for Franka robot push task.
    
    This environment simulates a Franka robot pushing a cube to a target location.
    The observation space includes joint positions and camera images.
    The action space is a tuple of joint positions and an intervention flag.
    """
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        use_delta_action_space: bool = True,
        delta: float = 0.05,
        action_scale: np.ndarray = np.asarray([0.05, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 20.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = True,
        reward_type: str = "sparse",
    ):
        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.control_dt)),
        }

        self._action_scale = action_scale
        self.reward_type = reward_type
        self.render_mode = render_mode
        camera_name_1 = "front"
        camera_name_2 = "handcam_rgb"
        camera_id_1 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_1)
        camera_id_2 = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_2)
        self.camera_id = (camera_id_1, camera_id_2)
        self.image_obs = image_obs
        
        # Caching.
        self.panda_dof_ids = np.asarray(
            [self._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self.panda_ctrl_ids = np.asarray(
            [self._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )

        self.gripper_ctrl_id = self._model.actuator("fingers_actuator").id
        self.pinch_site_id = self._model.site("pinch").id
        
        # Target region parameters
        self.target_tolerance = 0.035  # Distance threshold for success
        self.ee_min_dist = 0.08  # Minimum distance between end-effector and block for success

        self.initial_follower_position = self.data.qpos[self.panda_dof_ids].astype(np.float32)

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self.delta = delta
        self.use_delta_action_space = use_delta_action_space
        self.current_joint_positions = self.data.qpos[self.panda_dof_ids].astype(np.float32)

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Dict(
                    {
                        "state": spaces.Box(
                            -np.inf, np.inf, shape=(7,), dtype=np.float32  # 7 joints
                        ),
                        "images": spaces.Dict(
                            {
                                "front": spaces.Box(
                                    low=0,
                                    high=255,
                                    shape=(render_spec.height, render_spec.width, 3),
                                    dtype=np.uint8,
                                ),
                                "wrist": spaces.Box(
                                    low=0,
                                    high=255,
                                    shape=(render_spec.height, render_spec.width, 3),
                                    dtype=np.uint8,
                                ),
                            }
                        ),
                    }
                )
            }
        )

        # Define action space
        action_dim = 7  # 7 joints (no gripper for push task)
        action_space_robot = gym.spaces.Box(
            low=-np.ones(action_dim),
            high=np.ones(action_dim),
            shape=(action_dim,),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Tuple(
            (
                action_space_robot,
                gym.spaces.Discrete(2),  # Intervention flag
            ),
        )

        self._viewer = mujoco.Renderer(
            self.model,
            height=render_spec.height,
            width=render_spec.width
        )
        self._viewer.render()

    def reset(
        self, seed=None, options=None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            self._random = np.random.RandomState(seed)
            
        mujoco.mj_resetData(self._model, self._data)

        # Reset arm to home position.
        self._data.qpos[self.panda_dof_ids] = _PANDA_HOME
        # Gripper - keep it open for pushing
        self._data.ctrl[self.gripper_ctrl_id] = 0  # Open gripper
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        self._data.mocap_pos[0] = tcp_pos

        # Sample a new block position.
        block_xy = np.array([0.38, -0.02])
        self._data.jnt("usb_joint").qpos[:3] = (*block_xy, 0.02)
        mujoco.mj_forward(self._model, self._data)

        # Sample a new target position
        target_xy = np.array([0.5, 0.10])
        self._model.geom("target_region").pos = (*target_xy, 0.005)
        mujoco.mj_forward(self._model, self._data)

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None

        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: Tuple[np.ndarray, bool]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Tuple of (joint_action, intervention_flag)
                joint_action: Joint position targets (7 joints)
                intervention_flag: Whether this is a human intervention
                
        Returns:
            observation: Dict of observations
            reward: Reward value
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional info
        """
        policy_action, intervention_bool = action
        teleop_action = None
        self.current_joint_positions = self.data.qpos[self.panda_dof_ids].astype(np.float32)
        
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

            # Keep gripper open for pushing
            self._data.ctrl[self.gripper_ctrl_id] = 0
            self._data.ctrl[self.panda_ctrl_ids] = target_joint_positions
            mujoco.mj_step(self._model, self._data)

            observation = self._compute_observation()
        else:
            # Teleoperation would be handled here
            # For now, we'll just use the policy action
            observation = self._compute_observation()

        self.current_step += 1

        # Compute reward
        reward = self._compute_reward()
        
        # Check if episode is done
        terminated = self._is_success()
        truncated = self.time_limit_exceeded()

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

    def _get_target_position(self) -> np.ndarray:
        """Get the target position from the target region geom."""
        target_pos = self._model.geom("target_region").pos
        return np.array(target_pos[:2])

    def _compute_reward(self) -> float:
        """Compute the reward based on the current state."""
        if self.reward_type == "dense":
            block_pos = self._data.sensor("usb_pos").data
            block_xy = block_pos[:2]
            target_xy = self._get_target_position()
            dist = np.linalg.norm(block_xy - target_xy)
            return np.exp(-10 * dist)
        else:
            return float(self._is_success())

    def _is_block_in_target(self) -> bool:
        """Check if the block is in the target region."""
        block_pos = self._data.sensor("usb_pos").data
        block_xy = block_pos[:2]
        target_xy = self._get_target_position()
        dist = np.linalg.norm(block_xy - target_xy)
        return dist < self.target_tolerance

    def _is_ee_far_from_block(self) -> bool:
        """Check if the end-effector is far enough from the block."""
        block_pos = self._data.sensor("usb_pos").data
        ee_pos = self._data.sensor("pinch_pos").data
        dist = np.linalg.norm(ee_pos - block_pos)
        return dist > self.ee_min_dist

    def _is_success(self) -> bool:
        """Check if the task is successful (block in target and ee far from block)."""
        return self._is_block_in_target() and self._is_ee_far_from_block()

    def render(self):
        """Render the environment from multiple camera views."""
        rendered_frames = []
        for cam_id in self.camera_id:
            self._viewer.update_scene(self.data, camera=cam_id)
            rendered_frames.append(
                self._viewer.render()
            )
        return rendered_frames

    def _compute_observation(self) -> dict:
        """Compute the observation dictionary."""
        obs = {"observation": {}}
        
        # Joint positions
        qpos = self.data.qpos[self.panda_dof_ids].astype(np.float32)
        obs["observation"]["state"] = qpos

        # Camera images
        if self.image_obs:
            front, wrist = self.render()
            obs["observation"]["images"] = {
                "front": front,
                "wrist": wrist
            }

        return obs
