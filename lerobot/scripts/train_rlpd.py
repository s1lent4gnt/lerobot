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
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from threading import Lock

import hydra
import numpy as np
import torch
from deepdiff import DeepDiff
from omegaconf import DictConfig, ListConfig, OmegaConf
from termcolor import colored
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import GradScaler

from lerobot.common.datasets.factory import make_dataset, resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset
from lerobot.common.datasets.online_buffer import OnlineBuffer, compute_sampler_weights
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.logger import Logger, log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.policy_protocol import PolicyWithUpdate
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    set_global_seed,
)
from lerobot.scripts.eval import eval_policy


def log_train_info(logger: Logger, info, step, cfg, dataset, is_online):
    loss = info["loss"]
    grad_norm = info["grad_norm"]
    lr = info["lr"]
    update_s = info["update_s"]
    dataloading_s = info["dataloading_s"]

    # A sample is an (observation,action) pair, where observation and action
    # can be on multiple timestamps. In a batch, we have `batch_size`` number of samples.
    num_samples = (step + 1) * cfg.training.batch_size
    avg_samples_per_ep = dataset.num_frames / dataset.num_episodes
    num_episodes = num_samples / avg_samples_per_ep
    num_epochs = num_samples / dataset.num_frames
    log_items = [
        f"step:{format_big_number(step)}",
        # number of samples seen during training
        f"smpl:{format_big_number(num_samples)}",
        # number of episodes seen during training
        f"ep:{format_big_number(num_episodes)}",
        # number of time all unique samples are seen
        f"epch:{num_epochs:.2f}",
        f"loss:{loss:.3f}",
        f"grdn:{grad_norm:.3f}",
        f"lr:{lr:0.1e}",
        # in seconds
        f"updt_s:{update_s:.3f}",
        f"data_s:{dataloading_s:.3f}",  # if not ~0, you are bottlenecked by cpu or io
    ]
    logging.info(" ".join(log_items))

    info["step"] = step
    info["num_samples"] = num_samples
    info["num_episodes"] = num_episodes
    info["num_epochs"] = num_epochs
    info["is_online"] = is_online

    logger.log_dict(info, step, mode="train")

def log_eval_info(logger, info, step, cfg, dataset, is_online):
    eval_s = info["eval_s"]
    avg_sum_reward = info["avg_sum_reward"]
    pc_success = info["pc_success"]

    # A sample is an (observation,action) pair, where observation and action
    # can be on multiple timestamps. In a batch, we have `batch_size`` number of samples.
    num_samples = (step + 1) * cfg.training.batch_size
    avg_samples_per_ep = dataset.num_frames / dataset.num_episodes
    num_episodes = num_samples / avg_samples_per_ep
    num_epochs = num_samples / dataset.num_frames
    log_items = [
        f"step:{format_big_number(step)}",
        # number of samples seen during training
        f"smpl:{format_big_number(num_samples)}",
        # number of episodes seen during training
        f"ep:{format_big_number(num_episodes)}",
        # number of time all unique samples are seen
        f"epch:{num_epochs:.2f}",
        f"∑rwrd:{avg_sum_reward:.3f}",
        f"success:{pc_success:.1f}%",
        f"eval_s:{eval_s:.3f}",
    ]
    logging.info(" ".join(log_items))

    info["step"] = step
    info["num_samples"] = num_samples
    info["num_episodes"] = num_episodes
    info["num_epochs"] = num_epochs
    info["is_online"] = is_online

    logger.log_dict(info, step, mode="eval")

def train_sac(cfg: DictConfig, out_dir: str | None = None, job_name: str | None = None):
    if out_dir is None:
        raise NotImplementedError()
    if job_name is None:
        raise NotImplementedError()

    # log metrics to terminal and wandb
    logger = Logger(cfg, out_dir, wandb_job_name=job_name)

    set_global_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Setup environment and datasets
    logging.info("make_dataset")
    offline_dataset = make_dataset(cfg) if cfg.dataset_repo_id else None
    # TODO (michel-aractingi): temporary fix to avoid datasets with task_index key that doesn't exist in online environment
    # i.e., pusht
    if "task_index" in offline_dataset.hf_dataset[0]:
        offline_dataset.hf_dataset = offline_dataset.hf_dataset.remove_columns(["task_index"])

    if isinstance(offline_dataset, MultiLeRobotDataset):
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(offline_dataset.repo_id_to_index , indent=2)}"
        )
    
    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.training.eval_freq > 0:
        logging.info("make_env")
        eval_env = make_env(cfg)

    logging.info("make_policy")
    policy = make_policy(
        hydra_cfg=cfg,
        dataset_stats=offline_dataset.meta.stats if not cfg.resume else None,
        pretrained_policy_name_or_path=str(logger.last_pretrained_model_dir) if cfg.resume else None,
    )
    assert isinstance(policy, nn.Module)
    policy = policy.to(device)

    # Setup optimizers
    actor_optimizer = optim.Adam(policy.actor.parameters(), policy.config.actor_lr)
    critic_optimizer = optim.Adam(policy.critic_ensemble.parameters(), policy.config.critic_lr)
    temperature_optimizer = optim.Adam([policy.log_alpha], policy.config.temperature_lr)

    lr_scheduler = None
    grad_scaler = GradScaler(enabled=cfg.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    log_output_dir(out_dir)
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.training.offline_steps=} ({format_big_number(cfg.training.offline_steps)})")
    logging.info(f"{cfg.training.online_steps=}")
    logging.info(f"{offline_dataset.num_frames=} ({format_big_number(offline_dataset.num_frames)})")
    logging.info(f"{offline_dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Note: this helper will be used in offline and online training loops.
    def evaluate_and_checkpoint_if_needed(step, is_online):
        _num_digits = max(6, len(str(cfg.training.offline_steps + cfg.training.online_steps)))
        step_identifier = f"{step:0{_num_digits}d}"

        if cfg.training.eval_freq > 0 and step % cfg.training.eval_freq == 0:
            logging.info(f"Eval policy at step {step}")
            with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext():
                assert eval_env is not None
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=Path(out_dir) / "eval" / f"videos_step_{step_identifier}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )
            log_eval_info(logger, eval_info["aggregated"], step, cfg, offline_dataset, is_online=is_online)
            if cfg.wandb.enable:
                logger.log_video(eval_info["video_paths"][0], step, mode="eval")
            logging.info("Resume training")

        if cfg.training.save_checkpoint and (
            step % cfg.training.save_freq == 0
            or step == cfg.training.offline_steps + cfg.training.online_steps
        ):
            logging.info(f"Checkpoint policy after step {step}")
            # Note: Save with step as the identifier, and format it to have at least 6 digits but more if
            # needed (choose 6 as a minimum for consistency without being overkill).
            logger.save_checkpoint(
                step,
                policy,
                actor_optimizer,
                lr_scheduler,
                identifier=step_identifier,
            )
            logging.info("Resume training")

    # create dataloader for offline training
    if cfg.training.get("drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            offline_dataset.episode_data_index,
            drop_n_last_frames=cfg.training.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        offline_dataset,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()
    offline_step = 0
    for _ in range(step, cfg.training.offline_steps):
        if offline_step == 0:
            logging.info("Start offline training on a fixed dataset")

        start_time = time.perf_counter()
        batch = next(dl_iter)
        dataloading_s = time.perf_counter() - start_time

        for key in batch:
            batch[key] = batch[key].to(device, non_blocking=True)

        # train_info = update_policy(
        #     policy,
        #     batch,
        #     optimizer,
        #     cfg.training.grad_clip_norm,
        #     grad_scaler=grad_scaler,
        #     lr_scheduler=lr_scheduler,
        #     use_amp=cfg.use_amp,
        # )

        train_info = {
            "loss": 0.0,
            "grad_norm": 0.0,
            "lr": cfg.training.lr,
            "update_s": time.perf_counter() - start_time,
        }

        train_info["dataloading_s"] = dataloading_s

        if step % cfg.training.log_freq == 0:
            log_train_info(logger, train_info, step, cfg, offline_dataset, is_online=False)

        # Note: evaluate_and_checkpoint_if_needed happens **after** the `step`th training update has completed,
        # so we pass in step + 1.
        evaluate_and_checkpoint_if_needed(step + 1, is_online=False)

        step += 1
        offline_step += 1  # noqa: SIM113

    # Create an env dedicated to online episodes collection from policy rollout.
    online_env = make_env(cfg, n_envs=cfg.training.online_rollout_batch_size)
    resolve_delta_timestamps(cfg)
    online_buffer_path = logger.log_dir / "online_buffer"

    # Initialize online buffer
    online_dataset = OnlineBuffer(
        online_buffer_path,
        data_spec={
            **{k: {"shape": v, "dtype": np.dtype("float32")} for k, v in policy.config.input_shapes.items()},
            **{k: {"shape": v, "dtype": np.dtype("float32")} for k, v in policy.config.output_shapes.items()},
            "next.reward": {"shape": (), "dtype": np.dtype("float32")},
            "next.done": {"shape": (), "dtype": np.dtype("?")},
            "next.success": {"shape": (), "dtype": np.dtype("?")},
        },
        buffer_capacity=cfg.training.online_buffer_capacity,
        fps=online_env.unwrapped.metadata["render_fps"],
        delta_timestamps=cfg.training.delta_timestamps,
    )

    # If we are doing online rollouts asynchronously, deepcopy the policy to use for online rollouts (this
    # makes it possible to do online rollouts in parallel with training updates).
    online_rollout_policy = deepcopy(policy) if cfg.training.do_online_rollout_async else policy

    # Create dataloader for online training.
    concat_dataset = torch.utils.data.ConcatDataset([offline_dataset, online_dataset])
    sampler_weights = compute_sampler_weights(
        offline_dataset,
        offline_drop_n_last_frames=cfg.training.get("drop_n_last_frames", 0),
        online_dataset=online_dataset,
        # +1 because online rollouts return an extra frame for the "final observation". Note: we don't have
        # this final observation in the offline datasets, but we might add them in future.
        online_drop_n_last_frames=cfg.training.get("drop_n_last_frames", 0) + 1,
        online_sampling_ratio=cfg.training.online_sampling_ratio,
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        sampler_weights,
        num_samples=len(concat_dataset),
        replacement=True,
    )
    dataloader = torch.utils.data.DataLoader(
        concat_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    dl_iter = cycle(dataloader)

    # Lock and thread pool executor for asynchronous online rollouts. When asynchronous mode is disabled,
    # these are still used but effectively do nothing.
    lock = Lock()
    # Note: 1 worker because we only ever want to run one set of online rollouts at a time. Batch
    # parallelization of rollouts is handled within the job.
    executor = ThreadPoolExecutor(max_workers=1)

    online_step = 0
    online_rollout_s = 0  # time take to do online rollout
    update_online_buffer_s = 0  # time taken to update the online buffer with the online rollout data
    # Time taken waiting for the online buffer to finish being updated. This is relevant when using the async
    # online rollout option.
    await_update_online_buffer_s = 0
    rollout_start_seed = cfg.training.online_env_seed


    while True:
        if online_step == cfg.training.online_steps:
            break

        if online_step == 0:
            logging.info("Start online training by interacting with environment")

        def sample_trajectory_and_update_buffer():
            nonlocal rollout_start_seed
            with lock:
                online_rollout_policy.load_state_dict(policy.state_dict())
            online_rollout_policy.eval()
            start_rollout_time = time.perf_counter()
            with torch.no_grad():
                eval_info = eval_policy(
                    online_env,
                    online_rollout_policy,
                    n_episodes=cfg.training.online_rollout_n_episodes,
                    max_episodes_rendered=min(10, cfg.training.online_rollout_n_episodes),
                    videos_dir=logger.log_dir / "online_rollout_videos",
                    return_episode_data=True,
                    start_seed=(
                        rollout_start_seed := (rollout_start_seed + cfg.training.batch_size) % 1000000
                    ),
                )
            online_rollout_s = time.perf_counter() - start_rollout_time

            with lock:
                start_update_buffer_time = time.perf_counter()
                online_dataset.add_data(eval_info["episodes"])

                # Update the concatenated dataset length used during sampling.
                concat_dataset.cumulative_sizes = concat_dataset.cumsum(concat_dataset.datasets)

                # Update the sampling weights.
                sampler.weights = compute_sampler_weights(
                    offline_dataset,
                    offline_drop_n_last_frames=cfg.training.get("drop_n_last_frames", 0),
                    online_dataset=online_dataset,
                    # +1 because online rollouts return an extra frame for the "final observation". Note: we don't have
                    # this final observation in the offline datasets, but we might add them in future.
                    online_drop_n_last_frames=cfg.training.get("drop_n_last_frames", 0) + 1,
                    online_sampling_ratio=cfg.training.online_sampling_ratio,
                )
                sampler.num_frames = len(concat_dataset)

                update_online_buffer_s = time.perf_counter() - start_update_buffer_time

            return online_rollout_s, update_online_buffer_s

        future = executor.submit(sample_trajectory_and_update_buffer)
        # If we aren't doing async rollouts, or if we haven't yet gotten enough examples in our buffer, wait
        # here until the rollout and buffer update is done, before proceeding to the policy update steps.
        if (
            not cfg.training.do_online_rollout_async
            or len(online_dataset) <= cfg.training.online_buffer_seed_size
        ):
            online_rollout_s, update_online_buffer_s = future.result()

        if len(online_dataset) <= cfg.training.online_buffer_seed_size:
            logging.info(
                f"Seeding online buffer: {len(online_dataset)}/{cfg.training.online_buffer_seed_size}"
            )
            continue

        policy.train()
        for _ in range(cfg.training.online_steps_between_rollouts): # UTD
            with lock:
                start_time = time.perf_counter()
                batch = next(dl_iter)
                dataloading_s = time.perf_counter() - start_time

            for key in batch:
                batch[key] = batch[key].to(cfg.device, non_blocking=True)

            # TODO (lilkm): put this in a policy_update() function 
            # Update critics
            with torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext():
                critics_loss, critics_info = policy.compute_critic_loss(batch)
                
            critic_optimizer.zero_grad()
            grad_scaler.scale(critics_loss).backward()
            grad_scaler.unscale_(critic_optimizer)
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.critic_ensemble.parameters(),
                cfg.training.grad_clip_norm,
                error_if_nonfinite=False,
            )
            
            grad_scaler.step(critic_optimizer)
            grad_scaler.update()

            # Delayed policy updates
            if step % cfg.policy.update_frequency == 0:
                with torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext():
                    actor_loss, actor_info = policy.compute_actor_loss(batch)
                
                actor_optimizer.zero_grad()
                grad_scaler.scale(actor_loss).backward()
                grad_scaler.unscale_(actor_optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.actor.parameters(),
                    cfg.training.grad_clip_norm,
                    error_if_nonfinite=False,
                )

                grad_scaler.step(actor_optimizer)
                grad_scaler.update()

                with torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext():
                    temperature_loss, temp_info = policy.compute_temperature_loss(batch)

                temperature_optimizer.zero_grad()
                grad_scaler.scale(temperature_loss).backward()
                grad_scaler.unscale_(temperature_optimizer)

                # grad_norm = torch.nn.utils.clip_grad_norm_(
                #     policy.log_alpha.exp(),
                #     cfg.training.grad_clip_norm,
                #     error_if_nonfinite=False,
                # )

                grad_scaler.step(temperature_optimizer)
                grad_scaler.update()

                policy.temperature = policy.log_alpha.exp().item()
            
            policy.update()  # Update target networks

            train_info = {
                    "loss": critics_loss.item() + actor_loss.item(),
                    "critic_loss": critics_loss.item(),
                    "actor_loss": actor_loss.item(),
                    "temperature": policy.temperature,
                    "grad_norm": float(grad_norm),
                    "lr": cfg.training.lr,
                    "update_s": time.perf_counter() - start_time,
                }

            train_info["dataloading_s"] = dataloading_s
            train_info["online_rollout_s"] = online_rollout_s
            train_info["update_online_buffer_s"] = update_online_buffer_s
            train_info["await_update_online_buffer_s"] = await_update_online_buffer_s
            with lock:
                train_info["online_buffer_size"] = len(online_dataset)

            if step % cfg.training.log_freq == 0:
                log_train_info(logger, train_info, step, cfg, online_dataset, is_online=True)

            # Note: evaluate_and_checkpoint_if_needed happens **after** the `step`th training update has completed,
            # so we pass in step + 1.
            evaluate_and_checkpoint_if_needed(step + 1, is_online=True)

            step += 1
            online_step += 1

        # If we're doing async rollouts, we should now wait until we've completed them before proceeding
        # to do the next batch of rollouts.
        if future.running():
            start = time.perf_counter()
            online_rollout_s, update_online_buffer_s = future.result()
            await_update_online_buffer_s = time.perf_counter() - start

        if online_step >= cfg.training.online_steps:
            break

    if eval_env:
        eval_env.close()
    logging.info("End of training")

@hydra.main(version_base="1.2", config_name="default", config_path="../configs")
def main(cfg: DictConfig):
    train_sac(
        cfg,
        out_dir=hydra.core.hydra_config.HydraConfig.get().run.dir,
        job_name=hydra.core.hydra_config.HydraConfig.get().job.name,
    )

if __name__ == "__main__":
    main()