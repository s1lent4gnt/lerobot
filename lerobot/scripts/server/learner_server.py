#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
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
import logging
import shutil
import time
from pprint import pformat
from concurrent.futures import ThreadPoolExecutor

# from torch.multiprocessing import Event, Queue, Process
# from threading import Event, Thread
# from torch.multiprocessing import Queue, Event
from torch.multiprocessing import Queue

from lerobot.scripts.server.utils import setup_process_handlers

import grpc

# Import generated stubs
import hilserl_pb2_grpc  # type: ignore
import hydra
import torch
from deepdiff import DeepDiff
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from torch import nn
from torch.optim.optimizer import Optimizer

from lerobot.common.datasets.factory import make_dataset

# TODO: Remove the import of maniskill
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.logger import Logger, log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.common.utils.utils import (
    format_big_number,
    get_global_random_state,
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    set_global_random_state,
    set_global_seed,
)

from lerobot.scripts.server.buffer import (
    ReplayBuffer,
    concatenate_batch_transitions,
    move_transition_to_device,
    move_state_dict_to_device,
    bytes_to_transitions,
    state_to_bytes,
    bytes_to_python_object,
)

from lerobot.scripts.server import learner_service


def handle_resume_logic(cfg: DictConfig, out_dir: str) -> DictConfig:
    if not cfg.resume:
        if Logger.get_last_checkpoint_dir(out_dir).exists():
            raise RuntimeError(
                f"Output directory {Logger.get_last_checkpoint_dir(out_dir)} already exists. "
                "Use `resume=true` to resume training."
            )
        return cfg

    # if resume == True
    checkpoint_dir = Logger.get_last_checkpoint_dir(out_dir)
    if not checkpoint_dir.exists():
        raise RuntimeError(
            f"No model checkpoint found in {checkpoint_dir} for resume=True"
        )

    checkpoint_cfg_path = str(
        Logger.get_last_pretrained_model_dir(out_dir) / "config.yaml"
    )
    logging.info(
        colored(
            "Resume=True detected, resuming previous run",
            color="yellow",
            attrs=["bold"],
        )
    )

    checkpoint_cfg = init_hydra_config(checkpoint_cfg_path)
    diff = DeepDiff(OmegaConf.to_container(checkpoint_cfg), OmegaConf.to_container(cfg))

    if "values_changed" in diff and "root['resume']" in diff["values_changed"]:
        del diff["values_changed"]["root['resume']"]

    if len(diff) > 0:
        logging.warning(
            f"Differences between the checkpoint config and the provided config detected: \n{pformat(diff)}\n"
            "Checkpoint configuration takes precedence."
        )

    checkpoint_cfg.resume = True
    return checkpoint_cfg


def load_training_state(
    cfg: DictConfig,
    logger: Logger,
    optimizers: Optimizer | dict,
):
    if not cfg.resume:
        return None, None

    training_state = torch.load(
        logger.last_checkpoint_dir / logger.training_state_file_name, weights_only=False
    )

    if isinstance(training_state["optimizer"], dict):
        assert set(training_state["optimizer"].keys()) == set(optimizers.keys())
        for k, v in training_state["optimizer"].items():
            optimizers[k].load_state_dict(v)
    else:
        optimizers.load_state_dict(training_state["optimizer"])

    set_global_random_state({k: training_state[k] for k in get_global_random_state()})
    return training_state["step"], training_state["interaction_step"]


def log_training_info(cfg: DictConfig, out_dir: str, policy: nn.Module) -> None:
    num_learnable_params = sum(
        p.numel() for p in policy.parameters() if p.requires_grad
    )
    num_total_params = sum(p.numel() for p in policy.parameters())

    log_output_dir(out_dir)
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.training.online_steps=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


def initialize_replay_buffer(
    cfg: DictConfig, logger: Logger, device: str, storage_device: str
) -> ReplayBuffer:
    if not cfg.resume:
        return ReplayBuffer(
            capacity=cfg.training.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_shapes.keys(),
            storage_device=storage_device,
            optimize_memory=True,
        )

    logging.info("Resume training load the online dataset")
    dataset = LeRobotDataset(
        repo_id=cfg.dataset_repo_id,
        local_files_only=True,
        root=logger.log_dir / "dataset",
    )
    return ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=dataset,
        capacity=cfg.training.online_buffer_capacity,
        device=device,
        state_keys=cfg.policy.input_shapes.keys(),
        optimize_memory=True,
    )


def initialize_offline_replay_buffer(
    cfg: DictConfig,
    logger: Logger,
    device: str,
    storage_device: str,
    active_action_dims: list[int] | None = None,
) -> ReplayBuffer:
    if not cfg.resume:
        logging.info("make_dataset offline buffer")
        offline_dataset = make_dataset(cfg)
    if cfg.resume:
        logging.info("load offline dataset")
        offline_dataset = LeRobotDataset(
            repo_id=cfg.dataset_repo_id,
            local_files_only=True,
            root=logger.log_dir / "dataset_offline",
        )

    logging.info("Convert to a offline replay buffer")
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_shapes.keys(),
        action_mask=active_action_dims,
        action_delta=cfg.env.wrapper.delta_action,
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.training.offline_buffer_capacity,
    )
    return offline_replay_buffer


def get_observation_features(
    policy: SACPolicy, observations: torch.Tensor, next_observations: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if (
        policy.config.vision_encoder_name is None
        or not policy.config.freeze_vision_encoder
    ):
        return None, None

    with torch.no_grad():
        observation_features = (
            policy.actor.encoder(observations)
            if policy.actor.encoder is not None
            else None
        )
        next_observation_features = (
            policy.actor.encoder(next_observations)
            if policy.actor.encoder is not None
            else None
        )

    return observation_features, next_observation_features


def use_threads(cfg: DictConfig) -> bool:
    return cfg.actor_learner_config.concurrency.learner == "threads"


def start_learner_threads(
    cfg: DictConfig,
    logger: Logger,
    out_dir: str,
    shutdown_event: any,  # Event,
) -> None:
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
        target=start_learner_server,
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
        cfg,
        logger,
        out_dir,
        shutdown_event,
        transition_queue,
        interaction_message_queue,
        parameters_queue,
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


def start_learner_server(
    parameters_queue: Queue,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    shutdown_event: any,  # Event,
    cfg: DictConfig,
):
    if not use_threads(cfg):
        # We need init logging for MP separataly
        init_logging()

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        # Return back for MP
        setup_process_handlers(False)

    service = learner_service.LearnerService(
        shutdown_event,
        parameters_queue,
        cfg.actor_learner_config.policy_parameters_push_frequency,
        transition_queue,
        interaction_message_queue,
    )

    server = grpc.server(
        ThreadPoolExecutor(max_workers=learner_service.MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", learner_service.MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", learner_service.MAX_MESSAGE_SIZE),
        ],
    )

    hilserl_pb2_grpc.add_LearnerServiceServicer_to_server(
        service,
        server,
    )

    host = cfg.actor_learner_config.learner_host
    port = cfg.actor_learner_config.learner_port

    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info("[LEARNER] gRPC server started")

    shutdown_event.wait()
    logging.info("[LEARNER] Stopping gRPC server...")
    server.stop(learner_service.STUTDOWN_TIMEOUT)
    logging.info("[LEARNER] gRPC server stopped")


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
    state_dict = move_state_dict_to_device(policy.actor.state_dict(), device="cpu")
    state_bytes = state_to_bytes(state_dict)
    parameters_queue.put(state_bytes)


def add_actor_information_and_train(
    cfg,
    logger: Logger,
    out_dir: str,
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

    **NOTE:**
    - This function performs multiple responsibilities (data transfer, training, and logging).
      It should ideally be split into smaller functions in the future.
    - Due to Python's **Global Interpreter Lock (GIL)**, running separate threads for different tasks
      significantly reduces performance. Instead, this function executes all operations in a single thread.

    Args:
        cfg: Configuration object containing hyperparameters.
        device (str): The computing device (`"cpu"` or `"cuda"`).
        logger (Logger): Logger instance for tracking training progress.
        out_dir (str): The output directory for storing training checkpoints and logs.
        shutdown_event (Event): Event to signal shutdown.
        transition_queue (Queue): Queue for receiving transitions from the actor.
        interaction_message_queue (Queue): Queue for receiving interaction messages from the actor.
        parameters_queue (Queue): Queue for sending policy parameters to the actor.
    """

    device = get_safe_torch_device(cfg.device, log=True)
    storage_device = get_safe_torch_device(cfg_device=cfg.training.storage_device)

    logging.info("Initializing policy")
    ### Instantiate the policy in both the actor and learner processes
    ### To avoid sending a SACPolicy object through the port, we create a policy intance
    ### on both sides, the learner sends the updated parameters every n steps to update the actor's parameters
    # TODO: At some point we should just need make sac policy

    policy: SACPolicy = make_policy(
        hydra_cfg=cfg,
        # dataset_stats=offline_dataset.meta.stats if not cfg.resume else None,
        # Hack: But if we do online traning, we do not need dataset_stats
        dataset_stats=None,
        pretrained_policy_name_or_path=str(logger.last_pretrained_model_dir)
        if cfg.resume
        else None,
    )

    # Update the policy config with the grad_clip_norm value from training config if it exists
    clip_grad_norm_value = cfg.training.grad_clip_norm

    # compile policy
    policy = torch.compile(policy)
    assert isinstance(policy, nn.Module)

    push_actor_policy_to_queue(parameters_queue, policy)

    last_time_policy_pushed = time.time()

    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg, policy)
    resume_optimization_step, resume_interaction_step = load_training_state(
        cfg, logger, optimizers
    )

    log_training_info(cfg, out_dir, policy)

    replay_buffer = initialize_replay_buffer(cfg, logger, device, storage_device)
    batch_size = cfg.training.batch_size
    offline_replay_buffer = None

    if cfg.dataset_repo_id is not None:
        active_action_dims = None
        if cfg.env.wrapper.joint_masking_action_space is not None:
            active_action_dims = [
                i
                for i, mask in enumerate(cfg.env.wrapper.joint_masking_action_space)
                if mask
            ]
        offline_replay_buffer = initialize_offline_replay_buffer(
            cfg=cfg,
            logger=logger,
            device=device,
            storage_device=storage_device,
            active_action_dims=active_action_dims,
        )
        batch_size: int = batch_size // 2  # We will sample from both replay buffer

    # NOTE: This function doesn't have a single responsibility, it should be split into multiple functions
    # in the future. The reason why we did that is the  GIL in Python. It's super slow the performance
    # are divided by 200. So we need to have a single thread that does all the work.
    time.time()
    logging.info("Starting learner thread")
    interaction_message, transition = None, None
    optimization_step = (
        resume_optimization_step if resume_optimization_step is not None else 0
    )
    interaction_step_shift = (
        resume_interaction_step if resume_interaction_step is not None else 0
    )

    # Extract variables from cfg
    online_step_before_learning = cfg.training.online_step_before_learning
    utd_ratio = cfg.policy.utd_ratio
    dataset_repo_id = cfg.dataset_repo_id
    fps = cfg.fps
    log_freq = cfg.training.log_freq
    save_freq = cfg.training.save_freq
    device = cfg.device
    storage_device = cfg.training.storage_device
    policy_update_freq = cfg.training.policy_update_freq
    policy_parameters_push_frequency = (
        cfg.actor_learner_config.policy_parameters_push_frequency
    )
    save_checkpoint = cfg.training.save_checkpoint
    online_steps = cfg.training.online_steps

    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[LEARNER] Shutdown signal received. Exiting...")
            break

        logging.debug("[LEARNER] Waiting for transitions")
        while not transition_queue.empty() and not shutdown_event.is_set():
            transition_list = transition_queue.get()
            transition_list = bytes_to_transitions(transition_list)

            for transition in transition_list:
                transition = move_transition_to_device(transition, device=device)
                if check_nan_in_transition(
                    transition["state"], transition["action"], transition["next_state"]
                ):
                    logging.warning("NaN detected in transition, skipping")
                    continue
                replay_buffer.add(**transition)

                if cfg.dataset_repo_id is not None and transition.get(
                    "complementary_info", {}
                ).get("is_intervention"):
                    offline_replay_buffer.add(**transition)

        logging.debug("[LEARNER] Received transitions")
        logging.debug("[LEARNER] Waiting for interactions")
        while not interaction_message_queue.empty() and not shutdown_event.is_set():
            interaction_message = interaction_message_queue.get()
            interaction_message = bytes_to_python_object(interaction_message)
            # If cfg.resume, shift the interaction step with the last checkpointed step in order to not break the logging
            interaction_message["Interaction step"] += interaction_step_shift
            logger.log_dict(
                interaction_message, mode="train", custom_step_key="Interaction step"
            )

        logging.debug("[LEARNER] Received interactions")

        if len(replay_buffer) < online_step_before_learning:
            continue

        logging.debug("[LEARNER] Starting optimization loop")
        time_for_one_optimization_step = time.time()
        for _ in range(utd_ratio - 1):
            batch = replay_buffer.sample(batch_size)

            if dataset_repo_id is not None:
                batch_offline = offline_replay_buffer.sample(batch_size)
                batch = concatenate_batch_transitions(batch, batch_offline)

            actions = batch["action"]
            rewards = batch["reward"]
            observations = batch["state"]
            next_observations = batch["next_state"]
            done = batch["done"]
            check_nan_in_transition(
                observations=observations, actions=actions, next_state=next_observations
            )

            observation_features, next_observation_features = get_observation_features(
                policy, observations, next_observations
            )
            loss_critic = policy.compute_loss_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
            )
            optimizers["critic"].zero_grad()
            loss_critic.backward()

            # clip gradients
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.critic_ensemble.parameters(), clip_grad_norm_value
            )

            optimizers["critic"].step()

            policy.update_target_networks()
            
            # Add gripper critic optimization
            loss_grasp_critic = policy.compute_loss_grasp_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
                complementary_info=batch.get("complementary_info")  # Pass complementary_info
            )
            optimizers["grasp_critic"].zero_grad()
            loss_grasp_critic.backward()
            grasp_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.grasp_critic.parameters(), clip_grad_norm_value
            ).item()
            optimizers["grasp_critic"].step()

            # update grasp critic target network
            policy.update_grasp_target_networks()

        batch = replay_buffer.sample(batch_size)

        if dataset_repo_id is not None:
            batch_offline = offline_replay_buffer.sample(batch_size)
            batch = concatenate_batch_transitions(
                left_batch_transitions=batch, right_batch_transition=batch_offline
            )

        actions = batch["action"]
        rewards = batch["reward"]
        observations = batch["state"]
        next_observations = batch["next_state"]
        done = batch["done"]

        check_nan_in_transition(
            observations=observations, actions=actions, next_state=next_observations
        )

        observation_features, next_observation_features = get_observation_features(
            policy, observations, next_observations
        )
        loss_critic = policy.compute_loss_critic(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            done=done,
            observation_features=observation_features,
            next_observation_features=next_observation_features,
        )
        optimizers["critic"].zero_grad()
        loss_critic.backward()

        # clip gradients
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.critic_ensemble.parameters(), clip_grad_norm_value
        ).item()

        optimizers["critic"].step()

        training_infos = {}
        training_infos["loss_critic"] = loss_critic.item()
        training_infos["critic_grad_norm"] = critic_grad_norm

        # Add gripper critic optimization
        loss_grasp_critic = policy.compute_loss_grasp_critic(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            done=done,
            observation_features=observation_features,
            next_observation_features=next_observation_features,
            complementary_info=batch.get("complementary_info")  # Pass complementary_info
        )
        optimizers["grasp_critic"].zero_grad()
        loss_grasp_critic.backward()
        grasp_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.grasp_critic.parameters(), clip_grad_norm_value
        ).item()
        optimizers["grasp_critic"].step()

        # Add training info for the grasp critic
        training_infos["loss_grasp_critic"] = loss_grasp_critic.item()
        training_infos["grasp_critic_grad_norm"] = grasp_critic_grad_norm

        # if optimization_step % policy_update_freq == 0:
            # for _ in range(policy_update_freq):
        loss_actor = policy.compute_loss_actor(
            observations=observations,
            observation_features=observation_features,
        )

        optimizers["actor"].zero_grad()
        loss_actor.backward()

        # clip gradients
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.actor.parameters_to_optimize, clip_grad_norm_value
        ).item()

        optimizers["actor"].step()

        training_infos["loss_actor"] = loss_actor.item()
        training_infos["actor_grad_norm"] = actor_grad_norm

        # Temperature optimization
        loss_temperature = policy.compute_loss_temperature(
            observations=observations,
            observation_features=observation_features,
        )
        optimizers["temperature"].zero_grad()
        loss_temperature.backward()

        #  clip gradients
        temp_grad_norm = torch.nn.utils.clip_grad_norm_(
            [policy.log_alpha], clip_grad_norm_value
        ).item()

        optimizers["temperature"].step()

        training_infos["loss_temperature"] = loss_temperature.item()
        training_infos["temperature_grad_norm"] = temp_grad_norm
        training_infos["temperature"] = policy.temperature

        policy.update_target_networks()

        # update grasp critic target network
        policy.update_grasp_target_networks()

        if time.time() - last_time_policy_pushed > policy_parameters_push_frequency:
            push_actor_policy_to_queue(parameters_queue, policy)
            last_time_policy_pushed = time.time()

        # policy.update_target_networks()

        if optimization_step % log_freq == 0:
            training_infos["replay_buffer_size"] = len(replay_buffer)
            if offline_replay_buffer is not None:
                training_infos["offline_replay_buffer_size"] = len(
                    offline_replay_buffer
                )
            training_infos["Optimization step"] = optimization_step
            logger.log_dict(
                d=training_infos, mode="train", custom_step_key="Optimization step"
            )
            # logging.info(f"Training infos: {training_infos}")

        time_for_one_optimization_step = time.time() - time_for_one_optimization_step
        frequency_for_one_optimization_step = 1 / (
            time_for_one_optimization_step + 1e-9
        )

        logging.info(
            f"[LEARNER] Optimization frequency loop [Hz]: {frequency_for_one_optimization_step}"
        )

        logger.log_dict(
            {
                "Optimization frequency loop [Hz]": frequency_for_one_optimization_step,
                "Optimization step": optimization_step,
            },
            mode="train",
            custom_step_key="Optimization step",
        )

        optimization_step += 1
        if optimization_step % log_freq == 0:
            logging.info(f"[LEARNER] Number of optimization step: {optimization_step}")

        # if save_checkpoint and (
        #     optimization_step % save_freq == 0 or optimization_step == online_steps
        # ):
        #     logging.info(f"Checkpoint policy after step {optimization_step}")
        #     _num_digits = max(6, len(str(online_steps)))
        #     step_identifier = f"{optimization_step:0{_num_digits}d}"
        #     interaction_step = (
        #         interaction_message["Interaction step"]
        #         if interaction_message is not None
        #         else 0
        #     )
        #     logger.save_checkpoint(
        #         optimization_step,
        #         policy,
        #         optimizers,
        #         scheduler=None,
        #         identifier=step_identifier,
        #         interaction_step=interaction_step,
        #     )

        #     # TODO : temporarly save replay buffer here, remove later when on the robot
        #     # We want to control this with the keyboard inputs
        #     dataset_dir = logger.log_dir / "dataset"
        #     if dataset_dir.exists() and dataset_dir.is_dir():
        #         shutil.rmtree(
        #             dataset_dir,
        #         )
        #     replay_buffer.to_lerobot_dataset(
        #         dataset_repo_id, fps=fps, root=logger.log_dir / "dataset"
        #     )
        #     if offline_replay_buffer is not None:
        #         dataset_dir = logger.log_dir / "dataset_offline"

        #         if dataset_dir.exists() and dataset_dir.is_dir():
        #             shutil.rmtree(
        #                 dataset_dir,
        #             )

        #         offline_replay_buffer.to_lerobot_dataset(
        #             cfg.dataset_repo_id,
        #             fps=cfg.fps,
        #             root=logger.log_dir / "dataset_offline",
        #         )

            logging.info("Resume training")


def make_optimizers_and_scheduler(cfg, policy: nn.Module):
    """
    Creates and returns optimizers for the actor, critic, and temperature components of a reinforcement learning policy.

    This function sets up Adam optimizers for:
    - The **actor network**, ensuring that only relevant parameters are optimized.
    - The **critic ensemble**, which evaluates the value function.
    - The **temperature parameter**, which controls the entropy in soft actor-critic (SAC)-like methods.

    It also initializes a learning rate scheduler, though currently, it is set to `None`.

    **NOTE:**
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
    optimizer_actor = torch.optim.Adam(
        # NOTE: Handle the case of shared encoder where the encoder weights are not optimized with the gradient of the actor
        params=policy.actor.parameters_to_optimize,
        lr=policy.config.actor_lr,
    )
    optimizer_critic = torch.optim.Adam(
        params=policy.critic_ensemble.parameters(), lr=policy.config.critic_lr
    )
    optimizer_temperature = torch.optim.Adam(
        params=[policy.log_alpha], lr=policy.config.critic_lr
    )

    # Add grasp critic optimizer
    optimizer_grasp_critic = torch.optim.Adam(
        params=policy.grasp_critic.parameters(), 
        lr=policy.config.critic_lr
    )

    lr_scheduler = None
    optimizers = {
        "actor": optimizer_actor,
        "critic": optimizer_critic,
        "temperature": optimizer_temperature,
        "grasp_critic": optimizer_grasp_critic,
    }
    return optimizers, lr_scheduler


def train(cfg: DictConfig, out_dir: str | None = None, job_name: str | None = None):
    if out_dir is None:
        raise NotImplementedError()
    if job_name is None:
        raise NotImplementedError()

    init_logging()
    logging.info(pformat(OmegaConf.to_container(cfg)))

    logger = Logger(cfg, out_dir, wandb_job_name=job_name)
    cfg = handle_resume_logic(cfg, out_dir)

    set_global_seed(cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    shutdown_event = setup_process_handlers(use_threads(cfg))

    start_learner_threads(
        cfg,
        logger,
        out_dir,
        shutdown_event,
    )


@hydra.main(version_base="1.2", config_name="default", config_path="../../configs")
def train_cli(cfg: dict):
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")

    train(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )

    logging.info("[LEARNER] train_cli finished")


if __name__ == "__main__":
    train_cli()

    logging.info("[LEARNER] main finished")
