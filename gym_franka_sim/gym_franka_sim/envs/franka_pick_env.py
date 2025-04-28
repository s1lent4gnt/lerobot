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


class FrankaPickEnv(MujocoGymEnv):
    """
    Gymnasium environment for Franka robot pick task.
    
    This environment simulates a Franka robot picking up a cube and lifting it.
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
        self.block_z = self._model.site("usb_center").size[2]

        self.z_init = self._data.sensor("usb_pos").data[2]
        self.z_success = self.z_init + 0.2

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
                            -np.inf, np.inf, shape=(8,), dtype=np.float32  # 7 joints + gripper state
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
        action_dim = 8  # 7 joints + gripper
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
        # Gripper
        self._data.ctrl[self.gripper_ctrl_id] = 0  # Open gripper
        mujoco.mj_forward(self._model, self._data)

        # Reset mocap body to home position.
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        self._data.mocap_pos[0] = tcp_pos

        # Sample a new block position.
        block_xy = np.array([0.38, -0.02])
        self._data.jnt("usb_joint").qpos[:3] = (*block_xy, 0.02)
        mujoco.mj_forward(self._model, self._data)

        # Cache the initial block height.
        self.z_init = self._data.sensor("usb_pos").data[2]
        self.z_success = self.z_init + 0.2

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
                joint_action: Joint position targets (7 joints + gripper)
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

        joint_action = policy_action[:7]
        gripper_action = policy_action[7]

        if not intervention_bool:
            if self.use_delta_action_space:
                target_joint_positions = (
                    self.current_joint_positions + self.delta * joint_action
                )
            else:
                target_joint_positions = joint_action

            # Apply gripper action
            gripper_command = 0
            if gripper_action > 0.5:  # Open gripper
                gripper_command = 0
            else:  # Close gripper
                gripper_command = 255
                
            self._data.ctrl[self.gripper_ctrl_id] = gripper_command
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

    def _compute_reward(self) -> float:
        """Compute the reward based on the current state."""
        if self.reward_type == "dense":
            block_pos = self._data.sensor("usb_pos").data
            tcp_pos = self._data.sensor("pinch_pos").data
            dist = np.linalg.norm(block_pos - tcp_pos)
            r_close = np.exp(-20 * dist)
            r_lift = (block_pos[2] - self.z_init) / (self.z_success - self.z_init)
            r_lift = np.clip(r_lift, 0.0, 1.0)
            return 0.3 * r_close + 0.7 * r_lift
        else:
            return float(self._is_success())

    def _is_success(self) -> bool:
        """Check if the task is successful (block lifted)."""
        block_pos = self._data.sensor("usb_pos").data
        lift = block_pos[2] - self.z_init
        return lift > 0.1

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
        
        # Joint positions and gripper state
        qpos = self.data.qpos[self.panda_dof_ids].astype(np.float32)
        gripper_pose = np.array([self._data.ctrl[self.gripper_ctrl_id] / 255.0], dtype=np.float32)  # Normalize to [0,1]
        state = np.concatenate([qpos, gripper_pose])
        obs["observation"]["state"] = state

        # Camera images
        if self.image_obs:
            front, wrist = self.render()
            obs["observation"]["images"] = {
                "front": front,
                "wrist": wrist
            }

        return obs
