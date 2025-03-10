import gymnasium as gym
import numpy as np
from lerobot.franka_sim.franka_sim.gamepad.joystick_expert import JoystickExpert, ControllerType

class JoystickIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None, controller_type=ControllerType.XBOX):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.action_indices = action_indices

        self.expert = JoystickExpert(controller_type=controller_type)
        self.left, self.right = False, False

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: joystick action if nonezero; else, policy action
        """
        deadzone = 0.03

        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)
        intervened = False

        if np.linalg.norm(expert_a) > deadzone:
            intervened = True

        if self.gripper_enabled:
            if self.left: # close gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right: # open gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)
            # expert_a[:6] += np.random.uniform(-0.5, 0.5, size=6)

        if self.action_indices is not None:
            filtred_expert_a = np.zeros_like(expert_a)
            filtred_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtred_expert_a

        if intervened:
            return expert_a, True
        
        return action, False
    
    def step(self, action):
        
        new_action, replaced = self.action(action)
        
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info
    
    def close(self):
        self.expert.close()
        super().close()