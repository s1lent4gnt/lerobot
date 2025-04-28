# gym-franka-sim

A Gymnasium environment for the Franka robot in simulation, designed to work with the LeRobot framework.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gym-franka-sim.git

# Install the package
cd gym-franka-sim
pip install -e .
```

## Environments

### FrankaPick-v0

This environment simulates a Franka robot picking up a cube and lifting it. The task is considered successful when the cube is lifted above a certain height.

#### Observation Space

The observation space is a dictionary with the following structure:
- `observation.state`: Joint positions (7) and gripper state (1)
- `observation.images.front`: RGB image from the front camera
- `observation.images.wrist`: RGB image from the wrist camera

#### Action Space

The action space is a tuple of:
- Joint position targets (7 joints + gripper)
- Intervention flag (boolean)

### FrankaPush-v0

This environment simulates a Franka robot pushing a cube to a target location. The task is considered successful when the cube is within a certain distance of the target and the end-effector is far enough from the cube.

#### Observation Space

The observation space is a dictionary with the following structure:
- `observation.state`: Joint positions (7)
- `observation.images.front`: RGB image from the front camera
- `observation.images.wrist`: RGB image from the wrist camera

#### Action Space

The action space is a tuple of:
- Joint position targets (7 joints)
- Intervention flag (boolean)

### FrankaHIL-v0

This environment follows the HIL-SERL gym_manipulator.py structure with the make_robot_env function. It provides a Franka robot environment that is compatible with the Human-In-the-Loop Self-Reinforcement Learning (HIL-SERL) framework.

#### Observation Space

The observation space is a dictionary with the following structure:
- `observation.state`: Joint positions (7) and gripper state (1)
- `observation.images.front`: RGB image from the front camera
- `observation.images.wrist`: RGB image from the wrist camera

Additional observations can be added through wrappers:
- Joint velocities
- End-effector pose

#### Action Space

The action space is a tuple of:
- Joint position targets (7 joints + gripper)
- Intervention flag (boolean)

#### Using make_robot_env

The environment provides a `make_robot_env` function that follows the same structure as the one in HIL-SERL's gym_manipulator.py:

```python
from gym_franka_sim.envs.franka_hil_env import make_robot_env

# Create the environment
env = make_robot_env(
    env_kwargs={
        "render_mode": "human",
        "image_obs": True,
        "reward_type": "sparse",
        "use_delta_action_space": True,
    },
    use_camera=True,
    use_gripper=True,
    add_joint_velocity_to_observation=True,
    control_time_s=20.0,
    gripper_quantization_threshold=0.8,
)
```

## Usage with LeRobot

To use these environments with the LeRobot framework, you need to update the LeRobot configuration to recognize the new environments.

### Example

```python
import gymnasium as gym
import gym_franka_sim

# Create the environment
env = gym.make('FrankaPick-v0')

# Reset the environment
obs, info = env.reset()

# Take a step
action = (env.action_space[0].sample(), False)  # Random action, no intervention
obs, reward, terminated, truncated, info = env.step(action)
```

### HIL-SERL Integration Example

```python
import gym_franka_sim
from gym_franka_sim.envs.franka_hil_env import make_robot_env
from lerobot.common.envs.configs import HILSerlRobotEnvConfig, EnvWrapperConfig

# Create a configuration for the HIL-SERL environment
wrapper_config = EnvWrapperConfig(
    display_cameras=True,
    use_relative_joint_positions=True,
    add_joint_velocity_to_observation=True,
    control_time_s=20.0,
    use_gripper=True,
)

# Create the environment config
env_config = HILSerlRobotEnvConfig(
    wrapper=wrapper_config,
    fps=30,
    name="franka_hil",
)

# Create the environment
env = make_robot_env(
    env_kwargs={"render_mode": "human"},
    use_camera=True,
    use_gripper=True,
    add_joint_velocity_to_observation=wrapper_config.add_joint_velocity_to_observation,
    fps=env_config.fps,
    control_time_s=wrapper_config.control_time_s,
)
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
