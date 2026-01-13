# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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
"""
Learner server runner for distributed HILSerl robot policy training.

This script implements the learner component of the distributed HILSerl architecture.
It initializes the policy network, maintains replay buffers, and updates
the policy based on transitions received from the actor server.

Examples of usage:

- Start a learner server for training:
```bash
python -m lerobot.rl.learner --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**NOTE**: Start the learner server before launching the actor server. The learner opens a gRPC server
to communicate with actors.

**NOTE**: Training progress can be monitored through Weights & Biases if wandb.enable is set to true
in your configuration.

**WORKFLOW**:
1. Create training configuration with proper policy, dataset, and environment settings
2. Start this learner server with the configuration
3. Start an actor server with the same configuration
4. Monitor training progress through wandb dashboard

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pformat

import grpc
import torch
from termcolor import colored
from torch import nn
from torch.multiprocessing import Queue
from torch.optim.optimizer import Optimizer

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.buffer import (
    ReplayBuffer,
    ReplayBufferNSteps,
    concatenate_batch_transitions,
    concatenate_batch_transitions_nstep,
)
from lerobot.policies.acfql.modeling_acfql import ACFQLPolicy
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.robots import so100_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport import services_pb2_grpc
from lerobot.transport.utils import (
    MAX_MESSAGE_SIZE,
    bytes_to_python_object,
    bytes_to_transitions,
    state_to_bytes,
)
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state as utils_load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.transition import move_state_dict_to_device, move_transition_to_device
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)

from .learner_service import MAX_WORKERS, SHUTDOWN_TIMEOUT, LearnerService


@parser.wrap()
def train_cli(cfg: TrainRLServerPipelineConfig):
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")

    # Use the job_name from the config
    train(
        cfg,
        job_name=cfg.job_name,
    )

    logging.info("[LEARNER] train_cli finished")


def train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None):
    """
    Main training function that initializes and runs the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): The training configuration
        job_name (str | None, optional): Job name for logging. Defaults to None.
    """

    cfg.validate()

    if job_name is None:
        job_name = cfg.job_name

    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")

    display_pid = False
    if not use_threads(cfg):
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"learner_{job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Learner logging initialized, writing to {log_file}")
    logging.info(pformat(cfg.to_dict()))

    # Setup WandB logging if enabled
    if cfg.wandb.enable and cfg.wandb.project:
        from lerobot.rl.wandb_utils import WandBLogger

        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # Handle resume logic
    cfg = handle_resume_logic(cfg)

    set_seed(seed=cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    start_learner_threads(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
    )


def start_learner_threads(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
) -> None:
    """
    Start the learner threads for training.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        wandb_logger (WandBLogger | None): Logger for metrics
        shutdown_event: Event to signal shutdown
    """
    # Create multiprocessing queues
    transition_queue = Queue()
    interaction_message_queue = Queue()
    parameters_queue = Queue()

    concurrency_entity = None

    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from torch.multiprocessing import Process

        concurrency_entity = Process

    communication_process = concurrency_entity(
        target=start_learner,
        args=(
            parameters_queue,
            transition_queue,
            interaction_message_queue,
            shutdown_event,
            cfg,
        ),
        daemon=True,
    )
    communication_process.start()

    add_actor_information_and_train(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        parameters_queue=parameters_queue,
    )
    logging.info("[LEARNER] Training process stopped")

    logging.info("[LEARNER] Closing queues")
    transition_queue.close()
    interaction_message_queue.close()
    parameters_queue.close()

    communication_process.join()
    logging.info("[LEARNER] Communication process joined")

    logging.info("[LEARNER] join queues")
    transition_queue.cancel_join_thread()
    interaction_message_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[LEARNER] queues closed")


# Core algorithm functions


def run_acfql_offline_phase(
    cfg: TrainRLServerPipelineConfig,
    policy: nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    offline_replay_buffer: ReplayBufferNSteps,
    preprocessor,
    wandb_logger: WandBLogger | None,
    shutdown_event,
    start_optimization_step: int = 0,
) -> int:
    """
    Run ACFQL offline pretraining phase.

    Trains the policy on demonstration data before starting online training.
    Uses n-step returns for Q-learning.

    Args:
        cfg: Training configuration
        policy: ACFQLPolicy model
        optimizers: Dictionary of optimizers (actor_bc_flow, actor_onestep_flow, critic)
        offline_replay_buffer: Replay buffer with demonstration data
        preprocessor: Preprocessor pipeline for observations/actions
        wandb_logger: Logger for tracking progress
        shutdown_event: Event to signal shutdown
        start_optimization_step: Starting step (for resuming)

    Returns:
        int: Final optimization step after offline training
    """
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    utd_ratio = cfg.policy.utd_ratio
    log_freq = cfg.log_freq
    policy_update_freq = cfg.policy.policy_update_freq
    async_prefetch = cfg.policy.async_prefetch
    batch_size = cfg.batch_size
    offline_steps = cfg.policy.offline_steps

    if offline_steps <= 0 or offline_replay_buffer is None:
        logging.info("[LEARNER] No offline steps configured, skipping offline phase")
        return start_optimization_step

    logging.info(f"[LEARNER] Starting ACFQL offline pretraining for {offline_steps} steps")

    offline_iterator = offline_replay_buffer.get_iterator_nstep(
        batch_size=batch_size,
        n_steps=cfg.policy.chunk_size,
        gamma=cfg.policy.discount,
        async_prefetch=async_prefetch,
        queue_size=2,
    )

    optimization_step = start_optimization_step
    offline_step = min(offline_steps, optimization_step) if start_optimization_step > 0 else 0

    for _ in range(offline_step, offline_steps):
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[LEARNER] Shutdown signal received during offline training. Exiting...")
            return optimization_step

        time_for_one_optimization_step = time.time()

        # UTD ratio - 1 critic updates
        for _ in range(utd_ratio - 1):
            batch = next(offline_iterator)
            forward_batch = _prepare_acfql_batch(batch, preprocessor, policy)

            # Critic optimization
            critic_output = policy.forward(forward_batch, model="critic")
            loss_critic = critic_output["loss_critic"]
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
            )
            optimizers["critic"].step()
            policy.update_target_networks()

        # Last UTD update with logging
        batch = next(offline_iterator)
        forward_batch = _prepare_acfql_batch(batch, preprocessor, policy)

        critic_output = policy.forward(forward_batch, model="critic")
        loss_critic = critic_output["loss_critic"]
        optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
        ).item()
        optimizers["critic"].step()

        training_infos = {
            f"critic/{k}": v.item() if isinstance(v, torch.Tensor) else v
            for k, v in critic_output["info"].items()
        }
        training_infos["critic/grad_norm"] = critic_grad_norm

        # Actor optimization (at specified frequency)
        if optimization_step % policy_update_freq == 0:
            for _ in range(policy_update_freq):
                # Actor BC flow optimization
                actor_bc_flow_output = policy.forward(forward_batch, model="actor_bc_flow")
                loss_actor_bc_flow = actor_bc_flow_output["loss_actor_bc_flow"]
                optimizers["actor_bc_flow"].zero_grad()
                loss_actor_bc_flow.backward()
                actor_bc_flow_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.actor_bc_flow.parameters(), max_norm=clip_grad_norm_value
                ).item()
                optimizers["actor_bc_flow"].step()

                training_infos["actor_bc/grad_norm"] = actor_bc_flow_grad_norm
                training_infos.update(
                    {
                        f"actor_bc/{k}": v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in actor_bc_flow_output["info"].items()
                    }
                )

                # Actor onestep flow optimization
                actor_onestep_flow_output = policy.forward(forward_batch, model="actor_onestep_flow")
                loss_actor_onestep_flow = actor_onestep_flow_output["loss_actor_onestep_flow"]
                optimizers["actor_onestep_flow"].zero_grad()
                loss_actor_onestep_flow.backward()
                actor_onestep_flow_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.actor_onestep_flow.parameters(), max_norm=clip_grad_norm_value
                ).item()
                optimizers["actor_onestep_flow"].step()

                training_infos["actor_one/grad_norm"] = actor_onestep_flow_grad_norm
                training_infos.update(
                    {
                        f"actor_one/{k}": v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in actor_onestep_flow_output["info"].items()
                    }
                )

        # Logging
        if optimization_step % log_freq == 0:
            training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
            training_infos["Optimization step"] = optimization_step
            training_infos["phase"] = "offline"

            time_for_one_optimization_step = time.time() - time_for_one_optimization_step
            frequency = 1 / (time_for_one_optimization_step + 1e-9)
            training_infos["Optimization frequency loop [Hz]"] = frequency

            if wandb_logger:
                wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")

            logging.info(
                f"[LEARNER] Offline step {offline_step}/{offline_steps}, optimization step {optimization_step}"
            )

        optimization_step += 1
        offline_step += 1

    logging.info(f"[LEARNER] Completed ACFQL offline pretraining after {offline_steps} steps")
    return optimization_step


def run_acfql_online_phase(
    cfg: TrainRLServerPipelineConfig,
    policy: nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    replay_buffer: ReplayBufferNSteps,
    offline_replay_buffer: ReplayBufferNSteps | None,
    preprocessor,
    wandb_logger: WandBLogger | None,
    shutdown_event,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    parameters_queue: Queue,
    start_optimization_step: int = 0,
    interaction_step_shift: int = 0,
) -> None:
    """
    Run ACFQL online fine-tuning phase.

    Collects transitions from actors and fine-tunes the policy.
    Uses n-step returns and mixes online/offline data.

    Args:
        cfg: Training configuration
        policy: ACFQLPolicy model
        optimizers: Dictionary of optimizers
        replay_buffer: Online replay buffer
        offline_replay_buffer: Optional offline replay buffer for mixing
        preprocessor: Preprocessor pipeline
        wandb_logger: Logger
        shutdown_event: Event to signal shutdown
        transition_queue: Queue for receiving transitions
        interaction_message_queue: Queue for interaction messages
        parameters_queue: Queue for sending policy parameters
        start_optimization_step: Starting step
        interaction_step_shift: Shift for interaction step
    """
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    online_step_before_learning = cfg.policy.online_step_before_learning
    utd_ratio = cfg.policy.utd_ratio
    fps = cfg.env.fps
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    policy_update_freq = cfg.policy.policy_update_freq
    policy_parameters_push_frequency = cfg.policy.actor_learner_config.policy_parameters_push_frequency
    saving_checkpoint = cfg.save_checkpoint
    online_steps = cfg.policy.online_steps
    async_prefetch = cfg.policy.async_prefetch
    batch_size = cfg.batch_size

    if online_steps == 0:
        logging.info("[LEARNER] No online steps specified, training complete.")
        return

    logging.info("[LEARNER] Starting ACFQL online fine-tuning phase")

    dataset_repo_id = cfg.dataset.repo_id if cfg.dataset else None
    if dataset_repo_id is not None:
        batch_size = batch_size // 2  # Mix online/offline

    # Push policy to actors to start collecting transitions
    push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)
    last_time_policy_pushed = time.time()

    optimization_step = start_optimization_step
    interaction_message = None
    online_iterator = None
    offline_iterator = None

    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[LEARNER] Shutdown signal received. Exiting...")
            break

        # Process transitions from actors
        process_transitions(
            transition_queue=transition_queue,
            replay_buffer=replay_buffer,
            offline_replay_buffer=offline_replay_buffer,
            device=device,
            dataset_repo_id=dataset_repo_id,
            shutdown_event=shutdown_event,
        )

        # Process interaction messages
        interaction_message = process_interaction_messages(
            interaction_message_queue=interaction_message_queue,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
            shutdown_event=shutdown_event,
        )

        # Wait for enough samples
        if len(replay_buffer) < online_step_before_learning:
            continue

        # Initialize iterators
        if online_iterator is None:
            online_iterator = replay_buffer.get_iterator_nstep(
                batch_size=batch_size,
                n_steps=cfg.policy.chunk_size,
                gamma=cfg.policy.discount,
                async_prefetch=async_prefetch,
                queue_size=2,
            )

        if offline_replay_buffer is not None and offline_iterator is None:
            offline_iterator = offline_replay_buffer.get_iterator_nstep(
                batch_size=batch_size,
                n_steps=cfg.policy.chunk_size,
                gamma=cfg.policy.discount,
                async_prefetch=async_prefetch,
                queue_size=2,
            )

        time_for_one_optimization_step = time.time()

        # UTD ratio - 1 critic updates
        for _ in range(utd_ratio - 1):
            batch = next(online_iterator)
            if dataset_repo_id is not None:
                batch_offline = next(offline_iterator)
                batch = concatenate_batch_transitions_nstep(
                    left_batch_transitions=batch, right_batch_transition=batch_offline
                )

            forward_batch = _prepare_acfql_batch(batch, preprocessor, policy)

            critic_output = policy.forward(forward_batch, model="critic")
            loss_critic = critic_output["loss_critic"]
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
            )
            optimizers["critic"].step()
            policy.update_target_networks()

        # Last UTD update with logging
        batch = next(online_iterator)
        if dataset_repo_id is not None:
            batch_offline = next(offline_iterator)
            batch = concatenate_batch_transitions_nstep(
                left_batch_transitions=batch, right_batch_transition=batch_offline
            )

        forward_batch = _prepare_acfql_batch(batch, preprocessor, policy)

        critic_output = policy.forward(forward_batch, model="critic")
        loss_critic = critic_output["loss_critic"]
        optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
        ).item()
        optimizers["critic"].step()

        training_infos = {
            f"critic/{k}": v.item() if isinstance(v, torch.Tensor) else v
            for k, v in critic_output["info"].items()
        }
        training_infos["critic/grad_norm"] = critic_grad_norm

        # Actor optimization
        if optimization_step % policy_update_freq == 0:
            for _ in range(policy_update_freq):
                # Actor BC flow
                actor_bc_flow_output = policy.forward(forward_batch, model="actor_bc_flow")
                loss_actor_bc_flow = actor_bc_flow_output["loss_actor_bc_flow"]
                optimizers["actor_bc_flow"].zero_grad()
                loss_actor_bc_flow.backward()
                actor_bc_flow_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.actor_bc_flow.parameters(), max_norm=clip_grad_norm_value
                ).item()
                optimizers["actor_bc_flow"].step()

                training_infos["actor_bc/grad_norm"] = actor_bc_flow_grad_norm
                training_infos.update(
                    {
                        f"actor_bc/{k}": v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in actor_bc_flow_output["info"].items()
                    }
                )

                # Actor onestep flow
                actor_onestep_flow_output = policy.forward(forward_batch, model="actor_onestep_flow")
                loss_actor_onestep_flow = actor_onestep_flow_output["loss_actor_onestep_flow"]
                optimizers["actor_onestep_flow"].zero_grad()
                loss_actor_onestep_flow.backward()
                actor_onestep_flow_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.actor_onestep_flow.parameters(), max_norm=clip_grad_norm_value
                ).item()
                optimizers["actor_onestep_flow"].step()

                training_infos["actor_one/grad_norm"] = actor_onestep_flow_grad_norm
                training_infos.update(
                    {
                        f"actor_one/{k}": v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in actor_onestep_flow_output["info"].items()
                    }
                )

        # Push policy to actors
        if time.time() - last_time_policy_pushed > policy_parameters_push_frequency:
            push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)
            last_time_policy_pushed = time.time()

        policy.update_target_networks()

        # Logging
        if optimization_step % log_freq == 0:
            training_infos["replay_buffer_size"] = len(replay_buffer)
            if offline_replay_buffer is not None:
                training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
            training_infos["Optimization step"] = optimization_step
            training_infos["phase"] = "online"

            if wandb_logger:
                wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")

        time_for_one_optimization_step = time.time() - time_for_one_optimization_step
        frequency = 1 / (time_for_one_optimization_step + 1e-9)
        logging.info(f"[LEARNER] Optimization frequency loop [Hz]: {frequency}")

        if wandb_logger:
            wandb_logger.log_dict(
                {"Optimization frequency loop [Hz]": frequency, "Optimization step": optimization_step},
                mode="train",
                custom_step_key="Optimization step",
            )

        optimization_step += 1
        if optimization_step % log_freq == 0:
            logging.info(f"[LEARNER] Number of optimization step: {optimization_step}")

        # Checkpointing
        if saving_checkpoint and (optimization_step % save_freq == 0 or optimization_step == online_steps):
            save_training_checkpoint(
                cfg=cfg,
                optimization_step=optimization_step,
                online_steps=online_steps,
                interaction_message=interaction_message,
                policy=policy,
                optimizers=optimizers,
                replay_buffer=replay_buffer,
                offline_replay_buffer=offline_replay_buffer,
                dataset_repo_id=dataset_repo_id,
                fps=fps,
            )


def _prepare_acfql_batch(batch: dict, preprocessor, policy) -> dict:
    """Prepare batch for ACFQL forward pass with preprocessing."""
    actions = batch[ACTION]
    observations = batch["state"]
    next_observations = batch["next_state"]

    # Preprocess observations and actions
    obs_action_batch = {**observations, "action": actions}
    obs_action_batch = preprocessor(obs_action_batch)
    observations = {k: v for k, v in obs_action_batch.items() if k.startswith("observation.")}
    actions = obs_action_batch["action"]

    # Preprocess next observations
    next_batch = {**next_observations}
    next_batch = preprocessor(next_batch)
    next_observations = {k: v for k, v in next_batch.items() if k.startswith("observation.")}

    check_nan_in_transition(
        observations=observations,
        actions=actions.reshape(actions.shape[0], -1),
        next_state=next_observations,
    )

    # Get observation features (for frozen encoders)
    observation_features, next_observation_features = None, None
    if policy.config.vision_encoder_name is not None and policy.config.freeze_vision_encoder:
        with torch.no_grad():
            observation_features = policy.actor_onestep_flow.encoder.get_cached_image_features(observations)
            next_observation_features = policy.actor_onestep_flow.encoder.get_cached_image_features(
                next_observations
            )

    return {
        "state": observations,
        ACTION: actions,
        "reward": batch["reward"],
        "terminal": batch.get("terminals"),
        "mask": batch.get("masks"),
        "valid": batch.get("valid"),
        "next_state": next_observations,
        "observation_feature": observation_features,
        "next_observation_feature": next_observation_features,
        "complementary_info": batch.get("complementary_info"),
    }


def run_sac_training_phase(
    cfg: TrainRLServerPipelineConfig,
    policy: SACPolicy,
    optimizers: dict[str, torch.optim.Optimizer],
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer | None,
    preprocessor,
    wandb_logger: WandBLogger | None,
    shutdown_event,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    parameters_queue: Queue,
    start_optimization_step: int = 0,
    interaction_step_shift: int = 0,
) -> None:
    """
    Run SAC online training phase.

    Collects transitions from actors and trains the SAC policy.
    Uses 1-step TD learning with optional offline data mixing.

    Args:
        cfg: Training configuration
        policy: SACPolicy model
        optimizers: Dictionary of optimizers (actor, critic, temperature, discrete_critic)
        replay_buffer: Online replay buffer
        offline_replay_buffer: Optional offline replay buffer for mixing
        wandb_logger: Logger
        shutdown_event: Event to signal shutdown
        transition_queue: Queue for receiving transitions
        interaction_message_queue: Queue for interaction messages
        parameters_queue: Queue for sending policy parameters
        start_optimization_step: Starting step
        interaction_step_shift: Shift for interaction step
    """
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    online_step_before_learning = cfg.policy.online_step_before_learning
    utd_ratio = cfg.policy.utd_ratio
    fps = cfg.env.fps
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    policy_update_freq = cfg.policy.policy_update_freq
    policy_parameters_push_frequency = cfg.policy.actor_learner_config.policy_parameters_push_frequency
    saving_checkpoint = cfg.save_checkpoint
    online_steps = cfg.policy.online_steps
    async_prefetch = cfg.policy.async_prefetch
    batch_size = cfg.batch_size

    logging.info("[LEARNER] Starting SAC online training phase")

    dataset_repo_id = cfg.dataset.repo_id if cfg.dataset else None
    if dataset_repo_id is not None:
        batch_size = batch_size // 2  # Mix online/offline

    # Push policy to actors to start collecting transitions
    push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)
    last_time_policy_pushed = time.time()

    optimization_step = start_optimization_step
    interaction_message = None
    online_iterator = None
    offline_iterator = None

    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[LEARNER] Shutdown signal received. Exiting...")
            break

        # Process transitions from actors
        process_transitions(
            transition_queue=transition_queue,
            replay_buffer=replay_buffer,
            offline_replay_buffer=offline_replay_buffer,
            device=device,
            dataset_repo_id=dataset_repo_id,
            shutdown_event=shutdown_event,
        )

        # Process interaction messages
        interaction_message = process_interaction_messages(
            interaction_message_queue=interaction_message_queue,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
            shutdown_event=shutdown_event,
        )

        # Wait for enough samples
        if len(replay_buffer) < online_step_before_learning:
            continue

        # Initialize iterators
        if online_iterator is None:
            online_iterator = replay_buffer.get_iterator(
                batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            )

        if offline_replay_buffer is not None and offline_iterator is None:
            offline_iterator = offline_replay_buffer.get_iterator(
                batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            )

        time_for_one_optimization_step = time.time()

        # UTD ratio - 1 critic updates
        for _ in range(utd_ratio - 1):
            batch = next(online_iterator)
            if dataset_repo_id is not None:
                batch_offline = next(offline_iterator)
                batch = concatenate_batch_transitions(
                    left_batch_transitions=batch, right_batch_transition=batch_offline
                )

            forward_batch = _prepare_sac_batch(batch, preprocessor, policy)

            # Critic optimization
            critic_output = policy.forward(forward_batch, model="critic")
            loss_critic = critic_output["loss_critic"]
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
            )
            optimizers["critic"].step()

            # Discrete critic optimization (if available)
            if policy.config.num_discrete_actions is not None:
                discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
                loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
                optimizers["discrete_critic"].zero_grad()
                loss_discrete_critic.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
                )
                optimizers["discrete_critic"].step()

            policy.update_target_networks()

        # Last UTD update with logging
        batch = next(online_iterator)
        if dataset_repo_id is not None:
            batch_offline = next(offline_iterator)
            batch = concatenate_batch_transitions(
                left_batch_transitions=batch, right_batch_transition=batch_offline
            )

        forward_batch = _prepare_sac_batch(batch, preprocessor,policy)

        critic_output = policy.forward(forward_batch, model="critic")
        loss_critic = critic_output["loss_critic"]
        optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
        ).item()
        optimizers["critic"].step()

        training_infos = {
            "loss_critic": loss_critic.item(),
            "critic_grad_norm": critic_grad_norm,
        }

        # Discrete critic optimization (if available)
        if policy.config.num_discrete_actions is not None:
            discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
            loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
            optimizers["discrete_critic"].zero_grad()
            loss_discrete_critic.backward()
            discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
            ).item()
            optimizers["discrete_critic"].step()

            training_infos["loss_discrete_critic"] = loss_discrete_critic.item()
            training_infos["discrete_critic_grad_norm"] = discrete_critic_grad_norm

        # Actor and temperature optimization (at specified frequency)
        if optimization_step % policy_update_freq == 0:
            for _ in range(policy_update_freq):
                # Actor optimization
                actor_output = policy.forward(forward_batch, model="actor")
                loss_actor = actor_output["loss_actor"]
                optimizers["actor"].zero_grad()
                loss_actor.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.actor.parameters(), max_norm=clip_grad_norm_value
                ).item()
                optimizers["actor"].step()

                training_infos["loss_actor"] = loss_actor.item()
                training_infos["actor_grad_norm"] = actor_grad_norm

                # Temperature optimization
                temperature_output = policy.forward(forward_batch, model="temperature")
                loss_temperature = temperature_output["loss_temperature"]
                optimizers["temperature"].zero_grad()
                loss_temperature.backward()
                temp_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=[policy.log_alpha], max_norm=clip_grad_norm_value
                ).item()
                optimizers["temperature"].step()

                training_infos["loss_temperature"] = loss_temperature.item()
                training_infos["temperature_grad_norm"] = temp_grad_norm
                training_infos["temperature"] = policy.temperature

                policy.update_temperature()

        # Push policy to actors
        if time.time() - last_time_policy_pushed > policy_parameters_push_frequency:
            push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)
            last_time_policy_pushed = time.time()

        policy.update_target_networks()

        # Logging
        if optimization_step % log_freq == 0:
            training_infos["replay_buffer_size"] = len(replay_buffer)
            if offline_replay_buffer is not None:
                training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
            training_infos["Optimization step"] = optimization_step
            training_infos["phase"] = "online"

            if wandb_logger:
                wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")

        time_for_one_optimization_step = time.time() - time_for_one_optimization_step
        frequency = 1 / (time_for_one_optimization_step + 1e-9)
        logging.info(f"[LEARNER] Optimization frequency loop [Hz]: {frequency}")

        if wandb_logger:
            wandb_logger.log_dict(
                {"Optimization frequency loop [Hz]": frequency, "Optimization step": optimization_step},
                mode="train",
                custom_step_key="Optimization step",
            )

        optimization_step += 1
        if optimization_step % log_freq == 0:
            logging.info(f"[LEARNER] Number of optimization step: {optimization_step}")

        # Checkpointing
        if saving_checkpoint and (optimization_step % save_freq == 0 or optimization_step == online_steps):
            save_training_checkpoint(
                cfg=cfg,
                optimization_step=optimization_step,
                online_steps=online_steps,
                interaction_message=interaction_message,
                policy=policy,
                optimizers=optimizers,
                replay_buffer=replay_buffer,
                offline_replay_buffer=offline_replay_buffer,
                dataset_repo_id=dataset_repo_id,
                fps=fps,
            )


def _prepare_sac_batch(batch: dict, preprocessor, policy: SACPolicy) -> dict:
    """Prepare batch for SAC forward pass."""
    actions = batch[ACTION]
    observations = batch["state"]
    next_observations = batch["next_state"]

    # Preprocess observations and actions
    obs_action_batch = {**observations, "action": actions}
    obs_action_batch = preprocessor(obs_action_batch)
    observations = {k: v for k, v in obs_action_batch.items() if k.startswith("observation.")}
    actions = obs_action_batch["action"]

    # Preprocess next observations
    next_batch = {**next_observations}
    next_batch = preprocessor(next_batch)
    next_observations = {k: v for k, v in next_batch.items() if k.startswith("observation.")}

    check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

    observation_features, next_observation_features = get_observation_features(
        policy=policy, observations=observations, next_observations=next_observations
    )

    return {
        ACTION: actions,
        "reward": batch["reward"],
        "state": observations,
        "next_state": next_observations,
        "done": batch["done"],
        "observation_feature": observation_features,
        "next_observation_feature": next_observation_features,
        "complementary_info": batch.get("complementary_info"),
    }


def add_actor_information_and_train(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    parameters_queue: Queue,
):
    """
    Handles data transfer from the actor to the learner, manages training updates,
    and logs training progress in an online reinforcement learning setup.

    This function continuously:
    - Transfers transitions from the actor to the replay buffer.
    - Logs received interaction messages.
    - Ensures training begins only when the replay buffer has a sufficient number of transitions.
    - Samples batches from the replay buffer and performs multiple critic updates.
    - Periodically updates the actor, critic, and temperature optimizers.
    - Logs training statistics, including loss values and optimization frequency.

    NOTE: This function doesn't have a single responsibility, it should be split into multiple functions
    in the future. The reason why we did that is the  GIL in Python. It's super slow the performance
    is divided by 200. So we need to have a single thread that does all the work.

    Args:
        cfg (TrainRLServerPipelineConfig): Configuration object containing hyperparameters.
        wandb_logger (WandBLogger | None): Logger for tracking training progress.
        shutdown_event (Event): Event to signal shutdown.
        transition_queue (Queue): Queue for receiving transitions from the actor.
        interaction_message_queue (Queue): Queue for receiving interaction messages from the actor.
        parameters_queue (Queue): Queue for sending policy parameters to the actor.
    """
    # Extract all configuration variables at the beginning, it improve the speed performance
    # of 7%
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)

    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_train_process_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Initialized logging for actor information and training process")

    logging.info("Initializing policy")

    policy: nn.Module = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )

    assert isinstance(policy, nn.Module)

    policy.train()

    # Detect policy type for algorithm-specific training
    is_acfql = isinstance(policy, ACFQLPolicy)

    # ACFQL-specific: Set up preprocessor/postprocessor
    preprocessor, postprocessor = None, None
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = cfg.policy.dataset_stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": cfg.policy.dataset_stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": cfg.policy.dataset_stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg=cfg, policy=policy)

    # If we are resuming, we need to load the training state
    resume_optimization_step, resume_interaction_step = load_training_state(cfg=cfg, optimizers=optimizers)

    log_training_info(cfg=cfg, policy=policy)

    replay_buffer = initialize_replay_buffer(cfg, device, storage_device)
    offline_replay_buffer = None

    if cfg.dataset is not None:
        offline_replay_buffer = initialize_offline_replay_buffer(
            cfg=cfg,
            device=device,
            storage_device=storage_device,
        )

    logging.info("Starting learner thread")
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0
    interaction_step_shift = resume_interaction_step if resume_interaction_step is not None else 0

    # Dispatch to algorithm-specific training
    if is_acfql:
        logging.info("[LEARNER] Detected ACFQL policy, using two-phase training")

        # Phase 1: Offline pretraining (if configured)
        optimization_step = run_acfql_offline_phase(
            cfg=cfg,
            policy=policy,
            optimizers=optimizers,
            offline_replay_buffer=offline_replay_buffer,
            preprocessor=preprocessor,
            wandb_logger=wandb_logger,
            shutdown_event=shutdown_event,
            start_optimization_step=optimization_step,
        )

        # Check if shutdown was requested during offline phase
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[LEARNER] Shutdown signal received after offline phase. Exiting...")
            return

        # Phase 2: Online fine-tuning
        run_acfql_online_phase(
            cfg=cfg,
            policy=policy,
            optimizers=optimizers,
            replay_buffer=replay_buffer,
            offline_replay_buffer=offline_replay_buffer,
            preprocessor=preprocessor,
            wandb_logger=wandb_logger,
            shutdown_event=shutdown_event,
            transition_queue=transition_queue,
            interaction_message_queue=interaction_message_queue,
            parameters_queue=parameters_queue,
            start_optimization_step=optimization_step,
            interaction_step_shift=interaction_step_shift,
        )
    else:
        logging.info("[LEARNER] Detected SAC policy, using online training")

        # SAC: Single-phase online training
        run_sac_training_phase(
            cfg=cfg,
            policy=policy,
            optimizers=optimizers,
            replay_buffer=replay_buffer,
            offline_replay_buffer=offline_replay_buffer,
            wandb_logger=wandb_logger,
            shutdown_event=shutdown_event,
            transition_queue=transition_queue,
            interaction_message_queue=interaction_message_queue,
            parameters_queue=parameters_queue,
            start_optimization_step=optimization_step,
            interaction_step_shift=interaction_step_shift,
        )


def start_learner(
    parameters_queue: Queue,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    shutdown_event: any,  # Event,
    cfg: TrainRLServerPipelineConfig,
):
    """
    Start the learner server for training.
    It will receive transitions and interaction messages from the actor server,
    and send policy parameters to the actor server.

    Args:
        parameters_queue: Queue for sending policy parameters to the actor
        transition_queue: Queue for receiving transitions from the actor
        interaction_message_queue: Queue for receiving interaction messages from the actor
        shutdown_event: Event to signal shutdown
        cfg: Training configuration
    """
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_process_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Learner server process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        # Return back for MP
        # TODO: Check if its useful
        _ = ProcessSignalHandler(False, display_pid=True)

    service = LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=cfg.policy.actor_learner_config.policy_parameters_push_frequency,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        queue_get_timeout=cfg.policy.actor_learner_config.queue_get_timeout,
    )

    server = grpc.server(
        ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ],
    )

    services_pb2_grpc.add_LearnerServiceServicer_to_server(
        service,
        server,
    )

    host = cfg.policy.actor_learner_config.learner_host
    port = cfg.policy.actor_learner_config.learner_port

    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info("[LEARNER] gRPC server started")

    shutdown_event.wait()
    logging.info("[LEARNER] Stopping gRPC server...")
    server.stop(SHUTDOWN_TIMEOUT)
    logging.info("[LEARNER] gRPC server stopped")


def save_training_checkpoint(
    cfg: TrainRLServerPipelineConfig,
    optimization_step: int,
    online_steps: int,
    interaction_message: dict | None,
    policy: nn.Module,
    optimizers: dict[str, Optimizer],
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer | None = None,
    dataset_repo_id: str | None = None,
    fps: int = 30,
) -> None:
    """
    Save training checkpoint and associated data.

    This function performs the following steps:
    1. Creates a checkpoint directory with the current optimization step
    2. Saves the policy model, configuration, and optimizer states
    3. Saves the current interaction step for resuming training
    4. Updates the "last" checkpoint symlink to point to this checkpoint
    5. Saves the replay buffer as a dataset for later use
    6. If an offline replay buffer exists, saves it as a separate dataset

    Args:
        cfg: Training configuration
        optimization_step: Current optimization step
        online_steps: Total number of online steps
        interaction_message: Dictionary containing interaction information
        policy: Policy model to save
        optimizers: Dictionary of optimizers
        replay_buffer: Replay buffer to save as dataset
        offline_replay_buffer: Optional offline replay buffer to save
        dataset_repo_id: Repository ID for dataset
        fps: Frames per second for dataset
    """
    logging.info(f"Checkpoint policy after step {optimization_step}")
    _num_digits = max(6, len(str(online_steps)))
    interaction_step = interaction_message["Interaction step"] if interaction_message is not None else 0

    # Create checkpoint directory
    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, online_steps, optimization_step)

    # Save checkpoint
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=optimization_step,
        cfg=cfg,
        policy=policy,
        optimizer=optimizers,
        scheduler=None,
    )

    # Save interaction step manually
    training_state_dir = os.path.join(checkpoint_dir, TRAINING_STATE_DIR)
    os.makedirs(training_state_dir, exist_ok=True)
    training_state = {"step": optimization_step, "interaction_step": interaction_step}
    torch.save(training_state, os.path.join(training_state_dir, "training_state.pt"))

    # Update the "last" symlink
    update_last_checkpoint(checkpoint_dir)

    # TODO : temporary save replay buffer here, remove later when on the robot
    # We want to control this with the keyboard inputs
    dataset_dir = os.path.join(cfg.output_dir, "dataset")
    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)

    # Save dataset
    # NOTE: Handle the case where the dataset repo id is not specified in the config
    # eg. RL training without demonstrations data
    repo_id_buffer_save = cfg.env.task if dataset_repo_id is None else dataset_repo_id
    replay_buffer.to_lerobot_dataset(repo_id=repo_id_buffer_save, fps=fps, root=dataset_dir)

    if offline_replay_buffer is not None:
        dataset_offline_dir = os.path.join(cfg.output_dir, "dataset_offline")
        if os.path.exists(dataset_offline_dir) and os.path.isdir(dataset_offline_dir):
            shutil.rmtree(dataset_offline_dir)

        offline_replay_buffer.to_lerobot_dataset(
            cfg.dataset.repo_id,
            fps=fps,
            root=dataset_offline_dir,
        )

    logging.info("Resume training")


def make_optimizers_and_scheduler(cfg: TrainRLServerPipelineConfig, policy: nn.Module):
    """
    Creates and returns optimizers for the actor, critic, and temperature components of a reinforcement learning policy.

    This function sets up Adam optimizers for:
    - The **actor network**, ensuring that only relevant parameters are optimized.
    - The **critic ensemble**, which evaluates the value function.
    - The **temperature parameter**, which controls the entropy in soft actor-critic (SAC)-like methods.

    It also initializes a learning rate scheduler, though currently, it is set to `None`.

    NOTE:
    - If the encoder is shared, its parameters are excluded from the actor's optimization process.
    - The policy's log temperature (`log_alpha`) is wrapped in a list to ensure proper optimization as a standalone tensor.

    Args:
        cfg: Configuration object containing hyperparameters.
        policy (nn.Module): The policy model containing the actor, critic, and temperature components.

    Returns:
        Tuple[Dict[str, torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
        A tuple containing:
        - `optimizers`: A dictionary mapping component names ("actor", "critic", "temperature") to their respective Adam optimizers.
        - `lr_scheduler`: Currently set to `None` but can be extended to support learning rate scheduling.

    """
    # ACFQL-specific optimizers
    if isinstance(policy, ACFQLPolicy):
        # Get optimizer parameters from the policy
        optimizer_params = policy.get_optim_params()

        optimizers = {
            "actor_bc_flow": torch.optim.Adam(
                params=optimizer_params["actor_bc_flow"],
                lr=cfg.policy.actor_lr,
            ),
            "actor_onestep_flow": torch.optim.Adam(
                params=optimizer_params["actor_onestep_flow"],
                lr=cfg.policy.actor_lr,
            ),
            "critic": torch.optim.Adam(
                params=optimizer_params["critic"],
                lr=cfg.policy.critic_lr,
            ),
        }

        lr_scheduler = None

        logging.info("[LEARNER] Created ACFQL optimizers (actor_bc_flow, actor_onestep_flow, critic)")
        return optimizers, lr_scheduler

    # SAC-specific optimizers (default)
    optimizer_actor = torch.optim.Adam(
        params=[
            p
            for n, p in policy.actor.named_parameters()
            if not policy.config.shared_encoder or not n.startswith("encoder")
        ],
        lr=cfg.policy.actor_lr,
    )
    optimizer_critic = torch.optim.Adam(params=policy.critic_ensemble.parameters(), lr=cfg.policy.critic_lr)

    if cfg.policy.num_discrete_actions is not None:
        optimizer_discrete_critic = torch.optim.Adam(
            params=policy.discrete_critic.parameters(), lr=cfg.policy.critic_lr
        )
    optimizer_temperature = torch.optim.Adam(params=[policy.log_alpha], lr=cfg.policy.critic_lr)
    lr_scheduler = None
    optimizers = {
        "actor": optimizer_actor,
        "critic": optimizer_critic,
        "temperature": optimizer_temperature,
    }
    if cfg.policy.num_discrete_actions is not None:
        optimizers["discrete_critic"] = optimizer_discrete_critic
    return optimizers, lr_scheduler


# Training setup functions


def handle_resume_logic(cfg: TrainRLServerPipelineConfig) -> TrainRLServerPipelineConfig:
    """
    Handle the resume logic for training.

    If resume is True:
    - Verifies that a checkpoint exists
    - Loads the checkpoint configuration
    - Logs resumption details
    - Returns the checkpoint configuration

    If resume is False:
    - Checks if an output directory exists (to prevent accidental overwriting)
    - Returns the original configuration

    Args:
        cfg (TrainRLServerPipelineConfig): The training configuration

    Returns:
        TrainRLServerPipelineConfig: The updated configuration

    Raises:
        RuntimeError: If resume is True but no checkpoint found, or if resume is False but directory exists
    """
    out_dir = cfg.output_dir

    # Case 1: Not resuming, but need to check if directory exists to prevent overwrites
    if not cfg.resume:
        checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
        if os.path.exists(checkpoint_dir):
            raise RuntimeError(
                f"Output directory {checkpoint_dir} already exists. Use `resume=true` to resume training."
            )
        return cfg

    # Case 2: Resuming training
    checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"No model checkpoint found in {checkpoint_dir} for resume=True")

    # Log that we found a valid checkpoint and are resuming
    logging.info(
        colored(
            "Valid checkpoint found: resume=True detected, resuming previous run",
            color="yellow",
            attrs=["bold"],
        )
    )

    # Load config using Draccus
    checkpoint_cfg_path = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR, "train_config.json")
    checkpoint_cfg = TrainRLServerPipelineConfig.from_pretrained(checkpoint_cfg_path)

    # Ensure resume flag is set in returned config
    checkpoint_cfg.resume = True

    # This is needed to populate pretrained_path and checkpoint_path
    checkpoint_cfg.validate()

    return checkpoint_cfg


def load_training_state(
    cfg: TrainRLServerPipelineConfig,
    optimizers: Optimizer | dict[str, Optimizer],
):
    """
    Loads the training state (optimizers, step count, etc.) from a checkpoint.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        optimizers (Optimizer | dict): Optimizers to load state into

    Returns:
        tuple: (optimization_step, interaction_step) or (None, None) if not resuming
    """
    if not cfg.resume:
        return None, None

    # Construct path to the last checkpoint directory
    checkpoint_dir = os.path.join(cfg.output_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)

    logging.info(f"Loading training state from {checkpoint_dir}")

    try:
        # Use the utility function from train_utils which loads the optimizer state
        step, optimizers, _ = utils_load_training_state(Path(checkpoint_dir), optimizers, None)

        # Load interaction step separately from training_state.pt
        training_state_path = os.path.join(checkpoint_dir, TRAINING_STATE_DIR, "training_state.pt")
        interaction_step = 0
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, weights_only=False)  # nosec B614: Safe usage of torch.load
            interaction_step = training_state.get("interaction_step", 0)

        logging.info(f"Resuming from step {step}, interaction step {interaction_step}")
        return step, interaction_step

    except Exception as e:
        logging.error(f"Failed to load training state: {e}")
        return None, None


def log_training_info(cfg: TrainRLServerPipelineConfig, policy: nn.Module) -> None:
    """
    Log information about the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        policy (nn.Module): Policy model
    """
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.policy.online_steps=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


def initialize_replay_buffer(
    cfg: TrainRLServerPipelineConfig, device: str, storage_device: str
) -> ReplayBuffer:
    """
    Initialize a replay buffer, either empty or from a dataset if resuming.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized replay buffer
    """
    dataset_path = os.path.join(cfg.output_dir, "dataset")

    if cfg.resume and os.path.exists(dataset_path):
        logging.info("Resume training load the online dataset")

        # NOTE: In RL is possible to not have a dataset.
        repo_id = None
        if cfg.dataset is not None:
            repo_id = cfg.dataset.repo_id
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=dataset_path,
        )
    elif not cfg.resume and cfg.online_dataset is not None:
        logging.info(f"Load the online dataset from the repo {cfg.online_dataset.repo_id}")
        dataset = LeRobotDataset(
            repo_id=cfg.online_dataset.repo_id,
            # root=cfg.online_dataset.path,
        )
    else:
        logging.info("Make an empty online replay buffer")
        return ReplayBuffer(
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
        )

    return ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=dataset,
        capacity=cfg.policy.online_buffer_capacity,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        optimize_memory=True,
    )


def initialize_offline_replay_buffer(
    cfg: TrainRLServerPipelineConfig,
    device: str,
    storage_device: str,
) -> ReplayBuffer:
    """
    Initialize an offline replay buffer from a dataset.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized offline replay buffer
    """
    dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")
    if not cfg.resume or not os.path.exists(dataset_offline_path):
        logging.info("make_dataset offline buffer")
        offline_dataset = make_dataset(cfg)
    else:
        logging.info("load offline dataset")
        offline_dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=dataset_offline_path,
        )

    logging.info("Convert to a offline replay buffer")
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
    )
    return offline_replay_buffer


# Utilities/Helpers functions


def get_observation_features(
    policy: nn.Module, observations: torch.Tensor, next_observations: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Get observation features from the policy encoder. It act as cache for the observation features.
    when the encoder is frozen, the observation features are not updated.
    We can save compute by caching the observation features.

    Args:
        policy: The policy model
        observations: The current observations
        next_observations: The next observations

    Returns:
        tuple: observation_features, next_observation_features
    """

    if policy.config.vision_encoder_name is None or not policy.config.freeze_vision_encoder:
        return None, None

    if isinstance(policy, SACPolicy):
        with torch.no_grad():
            observation_features = policy.actor.encoder.get_cached_image_features(observations)
            next_observation_features = policy.actor.encoder.get_cached_image_features(next_observations)

    elif isinstance(policy, ACFQLPolicy):
        with torch.no_grad():
            observation_features = policy.actor_onestep_flow.encoder.get_cached_image_features(observations)
            next_observation_features = policy.actor_onestep_flow.encoder.get_cached_image_features(
                next_observations
            )


    return observation_features, next_observation_features


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.learner == "threads"


def check_nan_in_transition(
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_state: torch.Tensor,
    raise_error: bool = False,
) -> bool:
    """
    Check for NaN values in transition data.

    Args:
        observations: Dictionary of observation tensors
        actions: Action tensor
        next_state: Dictionary of next state tensors
        raise_error: If True, raises ValueError when NaN is detected

    Returns:
        bool: True if NaN values were detected, False otherwise
    """
    nan_detected = False

    # Check observations
    for key, tensor in observations.items():
        if torch.isnan(tensor).any():
            logging.error(f"observations[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in observations[{key}]")

    # Check next state
    for key, tensor in next_state.items():
        if torch.isnan(tensor).any():
            logging.error(f"next_state[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in next_state[{key}]")

    # Check actions
    if torch.isnan(actions).any():
        logging.error("actions contains NaN values")
        nan_detected = True
        if raise_error:
            raise ValueError("NaN detected in actions")

    return nan_detected


def push_actor_policy_to_queue(parameters_queue: Queue, policy: nn.Module):
    logging.debug("[LEARNER] Pushing actor policy to the queue")

    if isinstance(policy, SACPolicy):
        # SAC: Push actor state dict
        # Create a dictionary to hold all the state dicts
        state_dicts = {"policy": move_state_dict_to_device(policy.actor.state_dict(), device="cpu")}

        # Add discrete critic if it exists
        if hasattr(policy, "discrete_critic") and policy.discrete_critic is not None:
            state_dicts["discrete_critic"] = move_state_dict_to_device(
                policy.discrete_critic.state_dict(), device="cpu"
            )
            logging.debug("[LEARNER] Including discrete critic in state dict push")

    elif isinstance(policy, ACFQLPolicy):
        # ACFQL: Push actor_onestep_flow state dict
        state_dicts = {
            "policy": move_state_dict_to_device(policy.actor_onestep_flow.state_dict(), device="cpu")
        }
        logging.debug("[LEARNER] Pushing ACFQL actor_onestep_flow state dict")

    state_bytes = state_to_bytes(state_dicts)
    parameters_queue.put(state_bytes)


def process_interaction_message(
    message, interaction_step_shift: int, wandb_logger: WandBLogger | None = None
):
    """Process a single interaction message with consistent handling."""
    message = bytes_to_python_object(message)
    # Shift interaction step for consistency with checkpointed state
    message["Interaction step"] += interaction_step_shift

    # Log if logger available
    if wandb_logger:
        wandb_logger.log_dict(d=message, mode="train", custom_step_key="Interaction step")

    return message


def process_transitions(
    transition_queue: Queue,
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer,
    device: str,
    dataset_repo_id: str | None,
    shutdown_event: any,
):
    """Process all available transitions from the queue.

    Args:
        transition_queue: Queue for receiving transitions from the actor
        replay_buffer: Replay buffer to add transitions to
        offline_replay_buffer: Offline replay buffer to add transitions to
        device: Device to move transitions to
        dataset_repo_id: Repository ID for dataset
        shutdown_event: Event to signal shutdown
    """
    while not transition_queue.empty() and not shutdown_event.is_set():
        transition_list = transition_queue.get()
        transition_list = bytes_to_transitions(buffer=transition_list)

        for transition in transition_list:
            transition = move_transition_to_device(transition=transition, device=device)

            # Skip transitions with NaN values
            if check_nan_in_transition(
                observations=transition["state"],
                actions=transition[ACTION],
                next_state=transition["next_state"],
            ):
                logging.warning("[LEARNER] NaN detected in transition, skipping")
                continue

            replay_buffer.add(**transition)

            # Add to offline buffer if it's an intervention
            # TODO(jpizarrom): single intervention should not be added to offline buffer when using action chunks, but a chunk where there are intervention make sense
            # TODO(jpizarrom): Review if the enum or the str value is available in the complementary info
            # if dataset_repo_id is not None and transition.get("complementary_info", {}).get(
            #     TeleopEvents.IS_INTERVENTION
            # ):
            #     offline_replay_buffer.add(**transition)


def process_interaction_messages(
    interaction_message_queue: Queue,
    interaction_step_shift: int,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,
) -> dict | None:
    """Process all available interaction messages from the queue.

    Args:
        interaction_message_queue: Queue for receiving interaction messages
        interaction_step_shift: Amount to shift interaction step by
        wandb_logger: Logger for tracking progress
        shutdown_event: Event to signal shutdown

    Returns:
        dict | None: The last interaction message processed, or None if none were processed
    """
    last_message = None
    while not interaction_message_queue.empty() and not shutdown_event.is_set():
        message = interaction_message_queue.get()
        last_message = process_interaction_message(
            message=message,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
        )

    return last_message


if __name__ == "__main__":
    train_cli()
    logging.info("[LEARNER] main finished")
