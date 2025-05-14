import os
import time
import logging
from pathlib import Path

import torch
import mujoco.viewer
import safetensors.torch
from torch import nn

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.scripts.server.mujoco_manipulator import make_robot_env


def utils_load_training_state(checkpoint_path: Path, optimizers, device=None):
    """Mimics the logic from learner_server to load model & optimizer."""
    pretrained_dir = checkpoint_path / "pretrained_model"
    state_dict = {}

    safetensor_path = pretrained_dir / "model.safetensors"
    pt_path = pretrained_dir / "policy_model.pt"

    if safetensor_path.exists():
        state_dict = safetensors.torch.load_file(str(safetensor_path), device=str(device or "cpu"))
    elif pt_path.exists():
        state_dict = torch.load(str(pt_path), map_location=device or "cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {pretrained_dir}")

    # resume_step = 0
    # resume_interaction_step = 0
    # training_state_path = checkpoint_path / "training_state" / "training_state.pt"
    # if training_state_path.exists():
    #     training_state = torch.load(str(training_state_path), map_location="cpu")
    #     resume_step = training_state.get("step", 0)
    #     resume_interaction_step = training_state.get("interaction_step", 0)
    #     if optimizers is not None and "optimizers" in training_state:
    #         for key, opt in optimizers.items():
    #             opt.load_state_dict(training_state["optimizers"].get(key, {}))

    # return resume_step, resume_interaction_step, state_dict
    return state_dict


@parser.wrap()
def actor_cli(cfg: TrainPipelineConfig):
    cfg.validate()
    set_seed(cfg.seed)

    # Setup logging
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"inference_{cfg.job_name}.log")
    init_logging(log_file=log_file)
    logging.info(f"[INFER] Logging initialized at {log_file}")

    # Setup device
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Create environment
    env = make_robot_env(cfg=cfg.env)
    logging.info("[INFER] Environment created")

    # Create policy
    policy: SACPolicy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.eval()
    policy.to(device)
    logging.info("[INFER] Policy created")

    # Load actor weights using official logic
    checkpoint_dir = os.path.join("outputs/train/2025-05-14/15-44-22_default", "checkpoints", "last")
    full_state_dict = utils_load_training_state(Path(checkpoint_dir), optimizers=None, device=device)

    actor_state_dict = {k.replace("actor.", ""): v for k, v in full_state_dict.items() if k.startswith("actor.")}
    missing, unexpected = policy.actor.load_state_dict(actor_state_dict, strict=False)
    logging.info(f"[INFER] Loaded actor weights (missing={len(missing)}, unexpected={len(unexpected)})")

    # Run evaluation
    obs, _ = env.reset()
    total_episodes = 5
    ep = 0

    with mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False) as viewer:
        # for ep in range(total_episodes):
        while True:
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            logging.info(f"[INFER] Starting episode {ep + 1}")
            while not done:
                viewer.sync()
                with torch.no_grad():
                    action = policy.select_action(batch=obs)
                action_np = action.squeeze(dim=0).cpu().numpy()
                obs, reward, done, _, _ = env.step(action_np)
                total_reward += reward
            logging.info(f"[INFER] Episode {ep + 1} reward: {total_reward:.2f}")
            ep += 1


if __name__ == "__main__":
    actor_cli()
