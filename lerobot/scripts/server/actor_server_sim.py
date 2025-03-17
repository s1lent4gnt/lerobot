#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from statistics import mean, quantiles
from functools import lru_cache
from lerobot.scripts.server.utils import setup_process_handlers

# from lerobot.scripts.eval import eval_policy

import grpc
import hydra
import torch
from omegaconf import DictConfig
from torch import nn
import time

# TODO: Remove the import of maniskill
# from lerobot.common.envs.factory import make_maniskill_env
# from lerobot.common.envs.utils import preprocess_maniskill_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.utils.utils import (
    TimerManager,
    get_safe_torch_device,
    set_global_seed,
)
from lerobot.scripts.server import hilserl_pb2, hilserl_pb2_grpc
from lerobot.scripts.server.buffer import (
    Transition,
    move_state_dict_to_device,
    move_transition_to_device,
    python_object_to_bytes,
    transitions_to_bytes,
    bytes_to_state_dict,
)
from lerobot.scripts.server.network_utils import (
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
)
from lerobot.scripts.server.gym_manipulator import get_classifier, make_robot_env
from lerobot.scripts.server import learner_service
from lerobot.common.robot_devices.utils import busy_wait

from torch.multiprocessing import Queue, Event
from queue import Empty

from lerobot.common.utils.utils import init_logging

from lerobot.scripts.server.utils import get_last_item_from_queue

from lerobot.franka_sim.franka_sim.utils.viewer_utils import DualMujocoViewer

ACTOR_SHUTDOWN_TIMEOUT = 30


def receive_policy(
    cfg: DictConfig,
    parameters_queue: Queue,
    shutdown_event: any,  # Event,
    learner_client: hilserl_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
):
    logging.info("[ACTOR] Start receiving parameters from the Learner")

    if not use_threads(cfg):
        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        setup_process_handlers(False)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.actor_learner_config.learner_host,
            port=cfg.actor_learner_config.learner_port,
        )

    try:
        iterator = learner_client.StreamParameters(hilserl_pb2.Empty())
        receive_bytes_in_chunks(
            iterator,
            parameters_queue,
            shutdown_event,
            log_prefix="[ACTOR] parameters",
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Received policy loop stopped")


def transitions_stream(
    shutdown_event: Event, transitions_queue: Queue
) -> hilserl_pb2.Empty:
    while not shutdown_event.is_set():
        try:
            message = transitions_queue.get(block=True, timeout=5)
        except Empty:
            logging.debug("[ACTOR] Transition queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message, hilserl_pb2.Transition, log_prefix="[ACTOR] Send transitions"
        )

    return hilserl_pb2.Empty()


def interactions_stream(
    shutdown_event: any,  # Event,
    interactions_queue: Queue,
) -> hilserl_pb2.Empty:
    while not shutdown_event.is_set():
        try:
            message = interactions_queue.get(block=True, timeout=5)
        except Empty:
            logging.debug("[ACTOR] Interaction queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message,
            hilserl_pb2.InteractionMessage,
            log_prefix="[ACTOR] Send interactions",
        )

    return hilserl_pb2.Empty()


def send_transitions(
    cfg: DictConfig,
    transitions_queue: Queue,
    shutdown_event: any,  # Event,
    learner_client: hilserl_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> hilserl_pb2.Empty:
    """
    Sends transitions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - **Transition Data:**
        - A batch of transitions (observation, action, reward, next observation) is collected.
        - Transitions are moved to the CPU and serialized using PyTorch.
        - The serialized data is wrapped in a `hilserl_pb2.Transition` message and sent to the learner.
    """

    if not use_threads(cfg):
        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        setup_process_handlers(False)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.actor_learner_config.learner_host,
            port=cfg.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendTransitions(
            transitions_stream(shutdown_event, transitions_queue)
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming transitions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Transitions process stopped")


def send_interactions(
    cfg: DictConfig,
    interactions_queue: Queue,
    shutdown_event: any,  # Event,
    learner_client: hilserl_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> hilserl_pb2.Empty:
    """
    Sends interactions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - **Interaction Messages:**
        - Contains useful statistics about episodic rewards and policy timings.
        - The message is serialized using `pickle` and sent to the learner.
    """

    if not use_threads(cfg):
        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        setup_process_handlers(False)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.actor_learner_config.learner_host,
            port=cfg.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendInteractions(
            interactions_stream(shutdown_event, interactions_queue)
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming interactions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Interactions process stopped")


@lru_cache(maxsize=1)
def learner_service_client(
    host="127.0.0.1", port=50051
) -> tuple[hilserl_pb2_grpc.LearnerServiceStub, grpc.Channel]:
    import json

    """
    Returns a client for the learner service.

    GRPC uses HTTP/2, which is a binary protocol and multiplexes requests over a single connection.
    So we need to create only one client and reuse it.
    """

    service_config = {
        "methodConfig": [
            {
                "name": [{}],  # Applies to ALL methods in ALL services
                "retryPolicy": {
                    "maxAttempts": 5,  # Max retries (total attempts = 5)
                    "initialBackoff": "0.1s",  # First retry after 0.1s
                    "maxBackoff": "2s",  # Max wait time between retries
                    "backoffMultiplier": 2,  # Exponential backoff factor
                    "retryableStatusCodes": [
                        "UNAVAILABLE",
                        "DEADLINE_EXCEEDED",
                    ],  # Retries on network failures
                },
            }
        ]
    }

    service_config_json = json.dumps(service_config)

    channel = grpc.insecure_channel(
        f"{host}:{port}",
        options=[
            ("grpc.max_receive_message_length", learner_service.MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", learner_service.MAX_MESSAGE_SIZE),
            ("grpc.enable_retries", 1),
            ("grpc.service_config", service_config_json),
        ],
    )
    stub = hilserl_pb2_grpc.LearnerServiceStub(channel)
    logging.info("[ACTOR] Learner service client created")
    return stub, channel


def update_policy_parameters(policy: SACPolicy, parameters_queue: Queue, device):
    if not parameters_queue.empty():
        logging.info("[ACTOR] Load new parameters from Learner.")
        bytes_state_dict = get_last_item_from_queue(parameters_queue)
        state_dict = bytes_to_state_dict(bytes_state_dict)
        state_dict = move_state_dict_to_device(state_dict, device=device)
        policy.load_state_dict(state_dict)


def act_with_policy(
    cfg: DictConfig,
    robot: Robot,
    reward_classifier: nn.Module,
    shutdown_event: any,  # Event,
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
):
    """
    Executes policy interaction within the environment.

    This function rolls out the policy in the environment, collecting interaction data and pushing it to a queue for streaming to the learner.
    Once an episode is completed, updated network parameters received from the learner are retrieved from a queue and loaded into the network.

    Args:
        cfg (DictConfig): Configuration settings for the interaction process.
    """

    logging.info("make_env online")

    online_env = make_robot_env(
        robot=robot, reward_classifier=reward_classifier, cfg=cfg
    )

    set_global_seed(cfg.seed)
    device = get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")

    ### Instantiate the policy in both the actor and learner processes
    ### To avoid sending a SACPolicy object through the port, we create a policy intance
    ### on both sides, the learner sends the updated parameters every n steps to update the actor's parameters
    # TODO: At some point we should just need make sac policy
    policy: SACPolicy = make_policy(
        hydra_cfg=cfg,
        # dataset_stats=offline_dataset.meta.stats if not cfg.resume else None,
        # Hack: But if we do online training, we do not need dataset_stats
        dataset_stats=None,
        # TODO: Handle resume training
        device=device,
    )
    policy = torch.compile(policy)
    assert isinstance(policy, nn.Module)

    obs, info = online_env.reset()

    # NOTE: For the moment we will solely handle the case of a single environment
    sum_reward_episode = 0
    list_transition_to_send_to_learner = []
    list_policy_time = []
    episode_intervention = False

    # Create the dual viewer
    dual_viewer = DualMujocoViewer(online_env.model, online_env.data)

    with dual_viewer as viewer:
        for interaction_step in range(cfg.training.online_steps):
            start_time = time.perf_counter()
            viewer.sync()
            if shutdown_event.is_set():
                logging.info("[ACTOR] Shutting down act_with_policy")
                return

            if interaction_step >= cfg.training.online_step_before_learning:
                # Time policy inference and check if it meets FPS requirement
                with TimerManager(
                    elapsed_time_list=list_policy_time,
                    label="Policy inference time",
                    log=False,
                ) as timer:  # noqa: F841
                    action = policy.select_action(batch=obs)
                policy_fps = 1.0 / (list_policy_time[-1] + 1e-9)

                log_policy_frequency_issue(
                    policy_fps=policy_fps, cfg=cfg, interaction_step=interaction_step
                )

                next_obs, reward, done, truncated, info = online_env.step(
                    action.squeeze(dim=0).cpu().numpy()
                )
            else:
                # TODO (azouitine): Make a custom space for torch tensor
                # action = online_env.action_space.sample()
                # TODO (lilkm): why sampling "is_intervention"
                action_robot = online_env.action_space[0].sample()  # Only sample robot action
                action = (action_robot, False)  # Always set intervention to False
                next_obs, reward, done, truncated, info = online_env.step(action)

                # HACK: We have only one env but we want to batch it, it will be resolved with the torch box
                action = (
                    torch.from_numpy(action[0])
                    .to(device, non_blocking=device.type == "cuda")
                    .unsqueeze(dim=0)
                )

            sum_reward_episode += float(reward)

            # NOTE: We overide the action if the intervention is True, because the action applied is the intervention action
            if "is_intervention" in info and info["is_intervention"]:
                # TODO: Check the shape
                # NOTE: The action space for demonstration before hand is with the full action space
                # but sometimes for example we want to deactivate the gripper
                action = info["action_intervention"]
                episode_intervention = True

            # Check for NaN values in observations
            for key, tensor in obs.items():
                if torch.isnan(tensor).any():
                    logging.error(
                        f"[ACTOR] NaN values found in obs[{key}] at step {interaction_step}"
                    )

            list_transition_to_send_to_learner.append(
                Transition(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    done=done,
                    truncated=truncated,  # TODO: (azouitine) Handle truncation properly
                    complementary_info=info,  # TODO Handle information for the transition, is_demonstraction: bool
                )
            )
            # assign obs to the next obs and continue the rollout
            obs = next_obs

            # HACK: We have only one env but we want to batch it, it will be resolved with the torch box
            # Because we are using a single environment we can index at zero
            if done or truncated:
                # TODO: Handle logging for episode information
                logging.info(
                    f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}"
                )

                update_policy_parameters(
                    policy=policy.actor, parameters_queue=parameters_queue, device=device
                )

                if len(list_transition_to_send_to_learner) > 0:
                    push_transitions_to_transport_queue(
                        transitions=list_transition_to_send_to_learner,
                        transitions_queue=transitions_queue,
                    )
                    list_transition_to_send_to_learner = []

                stats = get_frequency_stats(list_policy_time)
                list_policy_time.clear()

                # Send episodic reward to the learner
                interactions_queue.put(
                    python_object_to_bytes(
                        {
                            "Episodic reward": sum_reward_episode,
                            "Interaction step": interaction_step,
                            "Episode intervention": int(episode_intervention),
                            **stats,
                        }
                    )
                )
                sum_reward_episode = 0.0
                episode_intervention = False
                obs, info = online_env.reset()

            if cfg.fps is not None:
                dt_time = time.perf_counter() - start_time
                busy_wait(1 / cfg.fps - dt_time)


def push_transitions_to_transport_queue(transitions: list, transitions_queue):
    """Send transitions to learner in smaller chunks to avoid network issues.

    Args:
        transitions: List of transitions to send
        message_queue: Queue to send messages to learner
        chunk_size: Size of each chunk to send
    """
    transition_to_send_to_learner = []
    for transition in transitions:
        tr = move_transition_to_device(transition=transition, device="cpu")
        for key, value in tr["state"].items():
            if torch.isnan(value).any():
                logging.warning(f"Found NaN values in transition {key}")

        transition_to_send_to_learner.append(tr)

    transitions_queue.put(transitions_to_bytes(transition_to_send_to_learner))


def get_frequency_stats(list_policy_time: list[float]) -> dict[str, float]:
    stats = {}
    list_policy_fps = [1.0 / t for t in list_policy_time]
    if len(list_policy_fps) > 1:
        policy_fps = mean(list_policy_fps)
        quantiles_90 = quantiles(list_policy_fps, n=10)[-1]
        logging.debug(f"[ACTOR] Average policy frame rate: {policy_fps}")
        logging.debug(f"[ACTOR] Policy frame rate 90th percentile: {quantiles_90}")
        stats = {
            "Policy frequency [Hz]": policy_fps,
            "Policy frequency 90th-p [Hz]": quantiles_90,
        }
    return stats


def log_policy_frequency_issue(
    policy_fps: float, cfg: DictConfig, interaction_step: int
):
    if policy_fps < cfg.fps:
        logging.warning(
            f"[ACTOR] Policy FPS {policy_fps:.1f} below required {cfg.fps} at step {interaction_step}"
        )


def establish_learner_connection(
    stub,
    shutdown_event: any,  # Event,
    attempts=30,
):
    for _ in range(attempts):
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down establish_learner_connection")
            return False

        # Force a connection attempt and check state
        try:
            logging.info("[ACTOR] Send ready message to Learner")
            if stub.Ready(hilserl_pb2.Empty()) == hilserl_pb2.Empty():
                return True
        except grpc.RpcError as e:
            logging.error(f"[ACTOR] Waiting for Learner to be ready... {e}")
            time.sleep(2)
    return False


def use_threads(cfg: DictConfig) -> bool:
    return cfg.actor_learner_config.concurrency.actor == "threads"


@hydra.main(version_base="1.2", config_name="default", config_path="../../configs")
def actor_cli(cfg: dict):
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")

    init_logging(log_file="actor.log")
    robot = make_robot(cfg=cfg.robot)

    shutdown_event = setup_process_handlers(use_threads(cfg))

    learner_client, grpc_channel = learner_service_client(
        host=cfg.actor_learner_config.learner_host,
        port=cfg.actor_learner_config.learner_port,
    )

    logging.info("[ACTOR] Establishing connection with Learner")
    if not establish_learner_connection(learner_client, shutdown_event):
        logging.error("[ACTOR] Failed to establish connection with Learner")
        return

    if not use_threads(cfg):
        # If we use multithreading, we can reuse the channel
        grpc_channel.close()
        grpc_channel = None

    logging.info("[ACTOR] Connection with Learner established")

    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()

    concurrency_entity = None
    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from multiprocessing import Process

        concurrency_entity = Process

    receive_policy_process = concurrency_entity(
        target=receive_policy,
        args=(cfg, parameters_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process = concurrency_entity(
        target=send_transitions,
        args=(cfg, transitions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    interactions_process = concurrency_entity(
        target=send_interactions,
        args=(cfg, interactions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process.start()
    interactions_process.start()
    receive_policy_process.start()

    # HACK: FOR MANISKILL we do not have a reward classifier
    # TODO: Remove this once we merge into main
    reward_classifier = None
    if (
        cfg.env.reward_classifier.pretrained_path is not None
        and cfg.env.reward_classifier.config_path is not None
    ):
        reward_classifier = get_classifier(
            pretrained_path=cfg.env.reward_classifier.pretrained_path,
            config_path=cfg.env.reward_classifier.config_path,
        )

    act_with_policy(
        cfg,
        robot,
        reward_classifier,
        shutdown_event,
        parameters_queue,
        transitions_queue,
        interactions_queue,
    )
    logging.info("[ACTOR] Policy process joined")

    logging.info("[ACTOR] Closing queues")
    transitions_queue.close()
    interactions_queue.close()
    parameters_queue.close()

    transitions_process.join()
    logging.info("[ACTOR] Transitions process joined")
    interactions_process.join()
    logging.info("[ACTOR] Interactions process joined")
    receive_policy_process.join()
    logging.info("[ACTOR] Receive policy process joined")

    logging.info("[ACTOR] join queues")
    transitions_queue.cancel_join_thread()
    interactions_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[ACTOR] queues closed")


if __name__ == "__main__":
    actor_cli()
