from pathlib import Path
from typing import Any, Dict, Literal, Tuple, Optional

import gymnasium as gym
import mujoco
import numpy as np
import torch
from gymnasium import spaces

from lerobot.franka_sim.franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from lerobot.scripts.server.gym_manipulator import HILSerlRobotEnv

_HERE = Path(__file__).parent
_XML_PATH = Path("/home/khalil/Documents/Khalil/Projects/Embodied-AI/lerobot/franka_sim/franka_sim/envs/xmls/arena.xml")
_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))


class FrankaRobot:
    """
    A robot interface class that mimics the structure expected by HILSerlRobotEnv.
    This class wraps a MujocoGymEnv to provide a consistent interface for the HIL-SERL framework.
    """
    def __init__(
        self,
        env,
        config=None,
    ):
        self.env = env
        self.config = config
        self.is_connected = True
        self.follower_arms = {"main": FrankaArm(env)}
        
    def connect(self):
        self.is_connected = True
        
    def disconnect(self):
        self.is_connected = False
        
    def capture_observation(self):
        """Capture observation from the environment."""
        obs = {}
        
        # Joint positions and gripper state
        qpos = self.env.data.qpos[self.env.panda_dof_ids].astype(np.float32)
        gripper_pose = np.array([self.env._data.ctrl[self.env.gripper_ctrl_id] / 255.0], dtype=np.float32)  # Normalize to [0,1]
        state = np.concatenate([qpos, gripper_pose])
        obs["observation.state"] = torch.from_numpy(state)
        
        # Camera images
        front, wrist = self.env.render()
        obs["observation.images.front"] = torch.from_numpy(front)
        obs["observation.images.wrist"] = torch.from_numpy(wrist)
        
        return obs
    
    def send_action(self, action):
        """Send action to the robot."""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
            
        joint_action = action[:7]
        gripper_action = action[7]
        
        # Apply gripper action
        gripper_command = 0
        if gripper_action > 0.5:  # Open gripper
            gripper_command = 0
        else:  # Close gripper
            gripper_command = 255
            
        self.env._data.ctrl[self.env.gripper_ctrl_id] = gripper_command
        self.env._data.ctrl[self.env.panda_ctrl_ids] = joint_action
        mujoco.mj_step(self.env._model, self.env._data)
    
    def teleop_step(self, record_data=False):
        """Simulate a teleoperation step."""
        # In a real implementation, this would get input from a human operator
        # For simulation, we'll just use a random action
        action = np.random.uniform(-1, 1, 8)
        self.send_action(torch.from_numpy(action))
        obs = self.capture_observation()
        return obs, {"action": torch.from_numpy(action)}


class FrankaArm:
    """A simple wrapper for a Franka arm in the simulation."""
    def __init__(self, env):
        self.env = env
        
    def read(self, property_name):
        """Read a property from the arm."""
        if property_name == "Present_Position":
            return self.env.data.qpos[self.env.panda_dof_ids].astype(np.float32)
        return None


class FrankaHILEnv(gym.Env):
    """
    Gymnasium environment for Franka robot using the HIL-SERL framework.
    
    This environment follows the structure of HILSerlRobotEnv from gym_manipulator.py
    but is specifically designed for the Franka robot in simulation.
    """
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        use_delta_action_space: bool = True,
        delta: float = 0.05,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 20.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = True,
        reward_type: str = "sparse",
        display_cameras: bool = False,
        **kwargs
    ):
        # Create the underlying MujocoGymEnv
        self.mujoco_env = MujocoGymEnv(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )
        
        # Set up the environment
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / control_dt)),
        }
        
        self.render_mode = render_mode
        camera_name_1 = "front"
        camera_name_2 = "handcam_rgb"
        camera_id_1 = mujoco.mj_name2id(self.mujoco_env._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_1)
        camera_id_2 = mujoco.mj_name2id(self.mujoco_env._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name_2)
        self.camera_id = (camera_id_1, camera_id_2)
        self.image_obs = image_obs
        
        # Caching
        self.panda_dof_ids = np.asarray(
            [self.mujoco_env._model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self.panda_ctrl_ids = np.asarray(
            [self.mujoco_env._model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self.gripper_ctrl_id = self.mujoco_env._model.actuator("fingers_actuator").id
        
        # Create a robot config
        class RobotConfig:
            def __init__(self):
                self.joint_position_relative_bounds = None
                self.max_relative_target = None
                
        robot_config = RobotConfig()
        
        # Create a robot interface
        self.robot = FrankaRobot(self.mujoco_env, config=robot_config)
        
        # Create the HIL-SERL environment
        self.hil_env = HILSerlRobotEnv(
            robot=self.robot,
            use_delta_action_space=use_delta_action_space,
            display_cameras=display_cameras,
        )
        
        # Set up spaces
        self.observation_space = self.hil_env.observation_space
        self.action_space = self.hil_env.action_space
        
        # Set up renderer
        self._viewer = mujoco.Renderer(
            self.mujoco_env.model,
            height=render_spec.height,
            width=render_spec.width
        )
        self._viewer.render()
        
        # Additional properties
        self.reward_type = reward_type
        self.delta = delta
        self.use_delta_action_space = use_delta_action_space

    def reset(
        self, seed=None, options=None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            self.mujoco_env._random = np.random.RandomState(seed)
            
        mujoco.mj_resetData(self.mujoco_env._model, self.mujoco_env._data)

        # Reset arm to home position
        self.mujoco_env._data.qpos[self.panda_dof_ids] = _PANDA_HOME
        # Gripper
        self.mujoco_env._data.ctrl[self.gripper_ctrl_id] = 0  # Open gripper
        mujoco.mj_forward(self.mujoco_env._model, self.mujoco_env._data)

        # Reset mocap body to home position
        tcp_pos = self.mujoco_env._data.sensor("2f85/pinch_pos").data
        self.mujoco_env._data.mocap_pos[0] = tcp_pos

        # Sample a new block position
        block_xy = np.array([0.38, -0.02])
        self.mujoco_env._data.jnt("usb_joint").qpos[:3] = (*block_xy, 0.02)
        mujoco.mj_forward(self.mujoco_env._model, self.mujoco_env._data)
        
        # Reset the HIL-SERL environment
        return self.hil_env.reset(seed=seed, options=options)

    def step(
        self, action: Tuple[np.ndarray, bool]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Step the HIL-SERL environment
        obs, reward, terminated, truncated, info = self.hil_env.step(action)
        
        # Compute reward based on task success
        if self.reward_type == "sparse":
            # Check if the task is successful (block lifted)
            block_pos = self.mujoco_env._data.sensor("usb_pos").data
            z_init = self.mujoco_env._data.sensor("usb_pos").data[2]
            lift = block_pos[2] - z_init
            success = lift > 0.1
            reward = float(success)
            terminated = success or terminated
        
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        rendered_frames = []
        for cam_id in self.camera_id:
            self._viewer.update_scene(self.mujoco_env.data, camera=cam_id)
            rendered_frames.append(
                self._viewer.render()
            )
        return rendered_frames

    def close(self):
        """Close the environment."""
        self.hil_env.close()
        self.mujoco_env.close()


def make_robot_env(
    robot_type=None,
    robot_kwargs=None,
    env_kwargs=None,
    env_name="FrankaHIL-v0",
    env_type="gym",
    use_camera=True,
    camera_kwargs=None,
    use_gripper=True,
    gripper_kwargs=None,
    **kwargs
):
    """
    Factory function to create a Franka robot environment following the HIL-SERL structure.
    
    Args:
        robot_type: Not used for simulation, included for compatibility
        robot_kwargs: Not used for simulation, included for compatibility
        env_kwargs: Dictionary of environment parameters
        env_name: Name of the environment to create
        env_type: Type of environment (always "gym" for this implementation)
        use_camera: Whether to use camera observations
        camera_kwargs: Camera parameters
        use_gripper: Whether to use the gripper
        gripper_kwargs: Gripper parameters
        **kwargs: Additional parameters
        
    Returns:
        A gym environment instance
    """
    if env_kwargs is None:
        env_kwargs = {}
    
    # Create the environment
    env = FrankaHILEnv(
        image_obs=use_camera,
        use_delta_action_space=True,
        **env_kwargs
    )
    
    # Apply wrappers as needed
    if kwargs.get("add_joint_velocity_to_observation", False):
        from lerobot.scripts.server.gym_manipulator import AddJointVelocityToObservation
        env = AddJointVelocityToObservation(env=env, fps=kwargs.get("fps", 30))
    
    if kwargs.get("add_ee_pose_to_observation", False):
        from lerobot.scripts.server.gym_manipulator import EEObservationWrapper
        env = EEObservationWrapper(env=env, ee_pose_limits=kwargs.get("ee_pose_limits", {"min": [-1, -1, -1], "max": [1, 1, 1]}))
    
    if kwargs.get("crop_params_dict") is not None:
        from lerobot.scripts.server.gym_manipulator import ImageCropResizeWrapper
        env = ImageCropResizeWrapper(
            env=env,
            crop_params_dict=kwargs.get("crop_params_dict"),
            resize_size=kwargs.get("resize_size", (128, 128))
        )
    
    if kwargs.get("control_time_s") is not None:
        from lerobot.scripts.server.gym_manipulator import TimeLimitWrapper
        env = TimeLimitWrapper(
            env=env,
            control_time_s=kwargs.get("control_time_s", 20.0),
            fps=kwargs.get("fps", 30)
        )
    
    if use_gripper:
        from lerobot.scripts.server.gym_manipulator import GripperActionWrapper
        env = GripperActionWrapper(
            env=env,
            quantization_threshold=kwargs.get("gripper_quantization_threshold", 0.8)
        )
        
        if kwargs.get("gripper_penalty") is not None:
            from lerobot.scripts.server.gym_manipulator import GripperPenaltyWrapper
            env = GripperPenaltyWrapper(
                env=env,
                penalty=kwargs.get("gripper_penalty", -0.1),
                gripper_penalty_in_reward=kwargs.get("gripper_penalty_in_reward", False)
            )
    
    if kwargs.get("ee_action_space_params") is not None:
        from lerobot.scripts.server.gym_manipulator import EEActionWrapper
        env = EEActionWrapper(
            env=env,
            ee_action_space_params=kwargs.get("ee_action_space_params"),
            use_gripper=use_gripper
        )
    
    if kwargs.get("joint_masking_action_space") is not None:
        from lerobot.scripts.server.gym_manipulator import JointMaskingActionSpace
        env = JointMaskingActionSpace(
            env=env,
            mask=kwargs.get("joint_masking_action_space")
        )
    
    # Add keyboard interface
    from lerobot.scripts.server.gym_manipulator import KeyboardInterfaceWrapper
    env = KeyboardInterfaceWrapper(env=env)
    
    # Add batch compatibility
    from lerobot.scripts.server.gym_manipulator import BatchCompitableWrapper
    env = BatchCompitableWrapper(env=env)
    
    return env
