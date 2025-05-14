import os
import time
import logging
import torch
import mujoco.viewer

from torch import nn
from safetensors.torch import load_file as safe_load_file

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.policies.factory import make_policy
from lerobot.scripts.server.mujoco_manipulator import make_robot_env


@parser.wrap()
def actor_eval(cfg: TrainPipelineConfig):
    cfg.validate()
    set_seed(cfg.seed)

    # Logging setup
    log_dir = os.path.join(cfg.output_dir or ".", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_eval_{cfg.job_name}.log")
    init_logging(log_file=log_file)
    logging.info(f"[EVAL] Logging initialized to {log_file}")

    # Set device
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Environment
    logging.info("[EVAL] Creating environment")
    env = make_robot_env(cfg.env)

    # Policy
    logging.info("[EVAL] Creating policy")
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.eval()
    policy.to(device)

    # Load policy weights (support .pt or .safetensors)
    pretrained_dir = os.path.join("outputs/train/2025-05-14/09-06-34_default", "checkpoints", "last", "pretrained_model")
    pt_path = os.path.join(pretrained_dir, "policy_model.pt")
    safetensors_path = os.path.join(pretrained_dir, "model.safetensors")

    if os.path.exists(pt_path):
        logging.info(f"[EVAL] Loading weights from .pt file: {pt_path}")
        state_dict = torch.load(pt_path, map_location=device)
    elif os.path.exists(safetensors_path):
        logging.info(f"[EVAL] Loading weights from .safetensors file: {safetensors_path}")
        state_dict = safe_load_file(safetensors_path, device=str(device))
    else:
        raise FileNotFoundError(f"No supported checkpoint found in {pretrained_dir}")

    policy.actor.load_state_dict(state_dict)
    logging.info("[EVAL] Weights loaded into policy.")

    # Evaluate
    num_episodes = 5
    with mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False) as viewer:
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            logging.info(f"[EVAL] Starting episode {ep + 1}")
            while not done:
                viewer.sync()
                with torch.no_grad():
                    action = policy.select_action(batch=obs)
                action_np = action.squeeze(dim=0).cpu().numpy()
                obs, reward, done, _, _ = env.step(action_np)
                total_reward += reward
            logging.info(f"[EVAL] Episode {ep + 1} reward: {total_reward:.2f}")


if __name__ == "__main__":
    actor_eval()
