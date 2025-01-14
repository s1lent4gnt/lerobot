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

# TODO: (1) better device management

from collections import deque
from typing import Callable, Optional, Sequence, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor

from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.sac.configuration_sac import SACConfig


class SACPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "RL", "SAC"],
):
    name = "sac"

    def __init__(
        self, config: SACConfig | None = None, dataset_stats: dict[str, dict[str, Tensor]] | None = None
    ):
        super().__init__()

        if config is None:
            config = SACConfig()
        self.config = config

        if config.input_normalization_modes is not None:
            self.normalize_inputs = Normalize(
                config.input_shapes, config.input_normalization_modes, dataset_stats
            )
        else:
            self.normalize_inputs = nn.Identity()
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        encoder_critic = SACObservationEncoder(config)
        encoder_actor = SACObservationEncoder(config)
        # Define networks
        critic_nets = []
        for _ in range(config.num_critics):
            critic_net = Critic(
                encoder=encoder_critic,
                network=MLP(
                    input_dim=encoder_critic.output_dim + config.output_shapes["action"][0],
                    **config.critic_network_kwargs,
                ),
            )
            critic_nets.append(critic_net)

        target_critic_nets = []
        for _ in range(config.num_critics):
            target_critic_net = Critic(
                encoder=encoder_critic,
                network=MLP(
                    input_dim=encoder_critic.output_dim + config.output_shapes["action"][0],
                    **config.critic_network_kwargs,
                ),
            )
            target_critic_nets.append(target_critic_net)

        self.critic_ensemble = create_critic_ensemble(critic_nets, config.num_critics)
        self.critic_target = create_critic_ensemble(target_critic_nets, config.num_critics)

        # Add initial weight copying
        for target_critic, critic in zip(self.critic_target, self.critic_ensemble, strict=False):
            target_critic.load_state_dict(critic.state_dict())

        self.actor = Policy(
            encoder=encoder_actor,
            network=MLP(input_dim=encoder_actor.output_dim, **config.actor_network_kwargs),
            action_dim=config.output_shapes["action"][0],
            **config.policy_kwargs,
        )
        if config.target_entropy is None:
            config.target_entropy = -np.prod(config.output_shapes["action"][0]) / 2  # (-dim(A)/2)
        # TODO: fix later device
        # TODO: Handle the case where the temparameter is a fixed
        self.log_alpha = torch.zeros(1, requires_grad=True, device="cuda")
        self.temperature = self.log_alpha.exp().item()

    def reset(self):
        """
        Clear observation and action queues. Should be called on `env.reset()`
        queues are populated during rollout of the policy, they contain the n latest observations and actions
        """

        self._queues = {
            "observation.state": deque(maxlen=1),
            "action": deque(maxlen=1),
        }
        if "observation.image" in self.config.input_shapes:
            self._queues["observation.image"] = deque(maxlen=1)
        if "observation.environment_state" in self.config.input_shapes:
            self._queues["observation.environment_state"] = deque(maxlen=1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation"""
        actions, _, _ = self.actor(batch)
        actions = self.unnormalize_outputs({"action": actions})["action"]
        return actions

    def critic_forward(
        self, observations: dict[str, Tensor], actions: Tensor, use_target: bool = False
    ) -> Tensor:
        """Forward pass through a critic network ensemble

        Args:
            observations: Dictionary of observations
            actions: Action tensor
            use_target: If True, use target critics, otherwise use ensemble critics

        Returns:
            Tensor of Q-values from all critics
        """
        critics = self.critic_target if use_target else self.critic_ensemble
        q_values = torch.stack([critic(observations, actions) for critic in critics])
        return q_values

    def compute_critic_loss(self, batch: dict[str, Tensor]) -> Tuple[Tensor, dict]:
        """Compute critic loss separately"""

        batch = self.normalize_inputs(batch)
        # batch shape is (b, 2, ...) where index 1 returns the current observation and
        # the next observation for calculating the right td index.
        actions = batch["action"][:, 0]
        rewards = batch["next.reward"][:, 0]
        observations = {}
        next_observations = {}
        for k in batch:
            if k.startswith("observation."):
                observations[k] = batch[k][:, 0]
                next_observations[k] = batch[k][:, 1]

        with torch.no_grad():
            next_action_preds, next_log_probs, _ = self.actor(next_observations)

            # 2- compute q targets
            q_targets = self.critic_forward(next_observations, next_action_preds, use_target=True)

            # subsample critics to prevent overfitting if use high UTD (update to date)
            if self.config.num_subsample_critics is not None:
                indices = torch.randperm(self.config.num_critics)
                indices = indices[: self.config.num_subsample_critics]
                q_targets = q_targets[indices]

            # critics subsample size
            min_q, _ = q_targets.min(dim=0)  # Get values from min operation
            if self.config.use_backup_entropy:
                min_q -= self.temperature * next_log_probs
            td_target = rewards + self.config.discount * min_q * ~batch["next.done"]

        # 3- compute predicted qs
        q_preds = self.critic_forward(observations, actions, use_target=False)

        # 4- Calculate loss
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        # You compute the mean loss of the batch for each critic and then to compute the final loss you sum them up
        critics_loss = (
            F.mse_loss(
                input=q_preds,
                target=td_target_duplicate,
                reduction="none",
            ).mean(1)
        ).sum()

        # 5- Update target networks with exponential moving average
        with torch.no_grad():
            for target_critic, critic in zip(self.critic_target, self.critic_ensemble, strict=False):
                for target_param, param in zip(target_critic.parameters(), critic.parameters(), strict=False):
                    target_param.data.copy_(
                        param.data * self.config.critic_target_update_weight
                        + target_param.data * (1.0 - self.config.critic_target_update_weight)
                    )

        info = {
            "td_target_mean": td_target.mean().item(),
            "td_target_max": td_target.max().item(),
        }

        return critics_loss, info

    def compute_actor_loss(self, batch: dict[str, Tensor]) -> Tuple[Tensor, dict]:
        """Compute actor loss separately"""
        batch = self.normalize_inputs(batch)
        observations = {k: batch[k][:, 0] for k in batch if k.startswith("observation.")}
        
        temperature = self.temperature
        actions, log_probs, _ = self.actor(observations)
        with torch.inference_mode():
            q_preds = self.critic_forward(observations, actions, use_target=False)
        min_q_preds = q_preds.min(dim=0)[0]

        actor_loss = ((temperature * log_probs) - min_q_preds).mean()

        info = {
            "mean_log_probs": log_probs.mean().item(),
            "min_log_probs": log_probs.min().item(),
            "max_log_probs": log_probs.max().item(),
            "action_mean": actions.mean().item(),
            "entropy": log_probs,
            "temperature": self.temperature,
        }

        return actor_loss, info

    def compute_temperature_loss(self, entropy: Tensor) -> Tuple[Tensor, dict]:
        """Compute temperature loss separately"""
        temperature_loss = (-self.log_alpha.exp() * (entropy + self.config.target_entropy)).mean()
        self.temperature = self.log_alpha.exp().item()

        info = {}

        return temperature_loss, info

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor | float]:
        """Combined forward for all losses"""
        
        critics_loss, critic_info = self.compute_critic_loss(batch)
        actor_loss, actor_info = self.compute_actor_loss(batch)
        temp_loss, temp_info = self.compute_temperature_loss(batch)
    
        self.temperature = self.log_alpha.exp().item()
        
        return {
            "critics_loss": critics_loss,
            "actor_loss": actor_loss,
            "temperature_loss": temp_loss,
            "loss": critics_loss + actor_loss + temp_loss,
            **critic_info,
            **actor_info,
            **temp_info,
        }


    # def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor | float]:
    #     """Run the batch through the model and compute the loss.

    #     Returns a dictionary with loss as a tensor, and other information as native floats.
    #     """
    #     # We have to actualize the value of the temperature because in the previous
    #     self.temperature = self.log_alpha.exp().item()

    #     # batch = self.normalize_inputs(batch)
    #     # batch shape is (b, 2, ...) where index 1 returns the current observation and
    #     # the next observation for calculating the right td index.
    #     actions = batch["action"][:, 0]
    #     rewards = batch["next.reward"][:, 0]
    #     observations = {}
    #     next_observations = {}
    #     for k in batch:
    #         if k.startswith("observation."):
    #             observations[k] = batch[k][:, 0]
    #             next_observations[k] = batch[k][:, 1]

    #     with torch.no_grad():
    #         next_action_preds, next_log_probs, _ = self.actor(next_observations)

    #         # 2- compute q targets
    #         q_targets = self.critic_forward(next_observations, next_action_preds, use_target=True)

    #         # subsample critics to prevent overfitting if use high UTD (update to date)
    #         if self.config.num_subsample_critics is not None:
    #             indices = torch.randperm(self.config.num_critics)
    #             indices = indices[: self.config.num_subsample_critics]
    #             q_targets = q_targets[indices]

    #         # critics subsample size
    #         min_q, _ = q_targets.min(dim=0)  # Get values from min operation
    #         if self.config.use_backup_entropy:
    #             min_q -= self.temperature * next_log_probs

    #         # td_target = rewards.flatten() + self.config.discount * min_q * ~batch["next.done"].flatten()
    #         td_target = rewards.flatten() + self.config.discount * min_q * (1 - batch["next.done"].float()).flatten()

    #     # 3- compute predicted qs
    #     q_preds = self.critic_forward(observations, actions, use_target=False)

    #     # 4- Calculate loss
    #     # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
    #     # td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
    #     # You compute the mean loss of the batch for each critic and then to compute the final loss you sum them up
    #     # critics_loss = (
    #     #     F.mse_loss(
    #     #         input=q_preds,
    #     #         target=td_target_duplicate,
    #     #         reduction="none",
    #     #     ).mean(1)
    #     # ).sum()

    #     critics_loss = torch.stack([
    #         F.mse_loss(q_preds[i], td_target)
    #         for i in range(q_preds.shape[0])
    #     ]).sum()

    #     temperature = self.temperature
    #     actions, log_probs, _ = self.actor(observations)
    #     with torch.inference_mode():
    #         q_preds = self.critic_forward(observations, actions, use_target=False)
    #     min_q_preds = q_preds.min(dim=0)[0]

    #     actor_loss = ((temperature * log_probs) - min_q_preds).mean()

    #     # calculate temperature loss
    #     with torch.no_grad():
    #         _, log_probs, _ = self.actor(observations)
    #     temperature_loss = (-self.log_alpha.exp() * (log_probs + self.config.target_entropy)).mean()

    #     loss = critics_loss + actor_loss + temperature_loss

    #     return {
    #         "critics_loss": critics_loss,
    #         "actor_loss": actor_loss,
    #         "mean_q_predicts": min_q_preds.mean().item(),
    #         "min_q_predicts": min_q_preds.min().item(),
    #         "max_q_predicts": min_q_preds.max().item(),
    #         "temperature_loss": temperature_loss,
    #         "temperature": temperature,
    #         "mean_log_probs": log_probs.mean().item(),
    #         "min_log_probs": log_probs.min().item(),
    #         "max_log_probs": log_probs.max().item(),
    #         "td_target_mean": td_target.mean().item(),
    #         "td_target_max": td_target.max().item(),
    #         "action_mean": actions.mean().item(),
    #         "entropy": log_probs.mean().item(),
    #         "loss": loss,
    #     }

    def update(self):
        # TODO: implement UTD update
        # First update only critics for utd_ratio-1 times
        # for critic_step in range(self.config.utd_ratio - 1):
        # only update critic and critic target
        # Then update critic, critic target, actor and temperature
        """Update target networks with exponential moving average"""
        with torch.no_grad():
            for target_critic, critic in zip(self.critic_target, self.critic_ensemble, strict=False):
                for target_param, param in zip(target_critic.parameters(), critic.parameters(), strict=False):
                    target_param.data.copy_(
                        param.data * self.config.critic_target_update_weight
                        + target_param.data * (1.0 - self.config.critic_target_update_weight)
                    )


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.ReLU(),
        activate_final: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        self.activate_final = activate_final
        layers = []

        # First layer uses input_dim
        layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # Add activation after first layer
        if dropout_rate is not None and dropout_rate > 0:
            layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.LayerNorm(hidden_dims[0]))
        layers.append(activations if isinstance(activations, nn.Module) else getattr(nn, activations)())

        # Rest of the layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

            if i + 1 < len(hidden_dims) or activate_final:
                if dropout_rate is not None and dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))
                layers.append(nn.LayerNorm(hidden_dims[i]))
                layers.append(
                    activations if isinstance(activations, nn.Module) else getattr(nn, activations)()
                )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module],
        network: nn.Module,
        init_final: Optional[float] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.encoder = encoder
        self.network = network
        self.init_final = init_final

        # Find the last Linear layer's output dimension
        for layer in reversed(network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break

        # Output layer
        if init_final is not None:
            self.output_layer = nn.Linear(out_features, 1)
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            self.output_layer = nn.Linear(out_features, 1)
            orthogonal_init()(self.output_layer.weight)

        self.to(self.device)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        # Move each tensor in observations to device
        observations = {k: v.to(self.device) for k, v in observations.items()}
        actions = actions.to(self.device)

        obs_enc = observations if self.encoder is None else self.encoder(observations)

        inputs = torch.cat([obs_enc, actions], dim=-1)
        x = self.network(inputs)
        value = self.output_layer(x)
        return value.squeeze(-1)


class Policy(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module],
        network: nn.Module,
        action_dim: int,
        log_std_min: float = -5,
        log_std_max: float = 2,
        fixed_std: Optional[torch.Tensor] = None,
        init_final: Optional[float] = None,
        use_tanh_squash: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.encoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.fixed_std = fixed_std.to(self.device) if fixed_std is not None else None
        self.use_tanh_squash = use_tanh_squash

        # Find the last Linear layer's output dimension
        for layer in reversed(network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break

        # Mean layer
        self.mean_layer = nn.Linear(out_features, action_dim)
        if init_final is not None:
            nn.init.uniform_(self.mean_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.mean_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.mean_layer.weight)

        # Standard deviation layer or parameter
        if fixed_std is None:
            self.std_layer = nn.Linear(out_features, action_dim)
            if init_final is not None:
                nn.init.uniform_(self.std_layer.weight, -init_final, init_final)
                nn.init.uniform_(self.std_layer.bias, -init_final, init_final)
            else:
                orthogonal_init()(self.std_layer.weight)

        self.to(self.device)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode observations if encoder exists
        obs_enc = observations if self.encoder is None else self.encoder(observations)

        # Get network outputs
        outputs = self.network(obs_enc)
        means = self.mean_layer(outputs)

        # Compute standard deviations
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
            assert not torch.isnan(log_std).any(), "[ERROR] log_std became NaN after std_layer!"

            if self.use_tanh_squash:
                log_std = torch.tanh(log_std)
                log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
            else:
                log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        else:
            log_std = self.fixed_std.expand_as(means)

        # uses tanh activation function to squash the action to be in the range of [-1, 1]
        normal = torch.distributions.Normal(means, torch.exp(log_std))
        x_t = normal.rsample()  # Reparameterization trick (mean + std * N(0,1))
        log_probs = normal.log_prob(x_t)  # Base log probability before Tanh

        if self.use_tanh_squash:
            actions = torch.tanh(x_t)
            log_probs -= torch.log((1 - actions.pow(2)) + 1e-6)  # Adjust log-probs for Tanh
        else:
            actions = x_t  # No Tanh; raw Gaussian sample

        log_probs = log_probs.sum(-1)  # Sum over action dimensions
        means = torch.tanh(means) if self.use_tanh_squash else means
        return actions, log_probs, means

    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get encoded features from observations"""
        observations = observations.to(self.device)
        if self.encoder is not None:
            with torch.inference_mode():
                return self.encoder(observations)
        return observations


class SACObservationEncoder(nn.Module):
    """Encode image and/or state vector observations.
    TODO(ke-wang): The original work allows for (1) stacking multiple history frames and (2) using pretrained resnet encoders.
    """

    def __init__(self, config: SACConfig, device="cuda"):
        """
        Creates encoders for pixel and/or state modalities.
        """
        super().__init__()
        self.config = config
        self.device = device

        if "observation.image" in config.input_shapes:
            self.image_enc_layers = nn.Sequential(
                nn.Conv2d(
                    config.input_shapes["observation.image"][0], config.image_encoder_hidden_dim, 7, stride=2
                ),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 5, stride=2),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=2),
                nn.ReLU(),
            ).to(self.device)
            dummy_batch = torch.zeros(1, *config.input_shapes["observation.image"])
            with torch.inference_mode():
                out_shape = self.image_enc_layers(dummy_batch).shape[1:]
            self.image_enc_layers.extend(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(out_shape), config.latent_dim),
                    nn.LayerNorm(config.latent_dim),
                    nn.Tanh(),
                ).to(self.device)
            )
        # if "observation.state" in config.input_shapes:
        #     self.state_enc_layers = nn.Sequential(
        #         nn.Linear(config.input_shapes["observation.state"][0], config.latent_dim),
        #         nn.LayerNorm(config.latent_dim),
        #         nn.Tanh(),
        #     )
        # if "observation.environment_state" in config.input_shapes:
        #     self.env_state_enc_layers = nn.Sequential(
        #         nn.Linear(config.input_shapes["observation.environment_state"][0], config.latent_dim),
        #         nn.LayerNorm(config.latent_dim),
        #         nn.Tanh(),
        #     )
        
        # For state and environment_state, we'll use them directly without encoding
        self.has_env_state = "observation.environment_state" in config.input_shapes
        self.has_state = "observation.state" in config.input_shapes
        
        # Calculate total output dimension
        self.total_dim = 0
        if "observation.image" in config.input_shapes:
            self.total_dim += config.latent_dim
        if self.has_env_state:
            self.total_dim += config.input_shapes["observation.environment_state"][0]
        if self.has_state:
            self.total_dim += config.input_shapes["observation.state"][0]

    def forward(self, obs_dict: dict[str, Tensor]) -> Tensor:
        """Encode the image and/or state vector.

        Each modality is encoded into a feature vector of size (latent_dim,) and then a uniform mean is taken
        over all features.
        """
        # feat = []
        # # Concatenate all images along the channel dimension.
        # image_keys = [k for k in self.config.input_shapes if k.startswith("observation.image")]
        # for image_key in image_keys:
        #     feat.append(flatten_forward_unflatten(self.image_enc_layers, obs_dict[image_key]))
        # if "observation.environment_state" in self.config.input_shapes:
        #     feat.append(self.env_state_enc_layers(obs_dict["observation.environment_state"]))
        # if "observation.state" in self.config.input_shapes:
        #     feat.append(self.state_enc_layers(obs_dict["observation.state"]))
        # # TODO(ke-wang): currently average over all features, concatenate all features maybe a better way
        # return torch.stack(feat, dim=0).mean(0)

        features = []
        
        # Handle images if present
        image_keys = [k for k in self.config.input_shapes if k.startswith("observation.image")]
        for image_key in image_keys:
            features.append(flatten_forward_unflatten(self.image_enc_layers, obs_dict[image_key].to(self.device)))
            
        # Add raw environment state
        if self.has_env_state:
            features.append(obs_dict["observation.environment_state"].to(self.device))
            
        # Add raw state
        if self.has_state:
            features.append(obs_dict["observation.state"].to(self.device))
        
        # Concatenate all features
        return torch.cat(features, dim=-1)

    @property
    def output_dim(self) -> int:
        """Returns the dimension of the encoder output"""
        return self.total_dim


def orthogonal_init():
    return lambda x: torch.nn.init.orthogonal_(x, gain=1.0)


def create_critic_ensemble(critics: list[nn.Module], num_critics: int, device: str = "cuda") -> nn.ModuleList:
    """Creates an ensemble of critic networks"""
    assert len(critics) == num_critics, f"Expected {num_critics} critics, got {len(critics)}"
    return nn.ModuleList(critics).to(device)


# borrowed from tdmpc
def flatten_forward_unflatten(fn: Callable[[Tensor], Tensor], image_tensor: Tensor) -> Tensor:
    """Helper to temporarily flatten extra dims at the start of the image tensor.

    Args:
        fn: Callable that the image tensor will be passed to. It should accept (B, C, H, W) and return
            (B, *), where * is any number of dimensions.
        image_tensor: An image tensor of shape (**, C, H, W), where ** is any number of dimensions and
        can be more than 1 dimensions, generally different from *.
    Returns:
        A return value from the callable reshaped to (**, *).
    """
    if image_tensor.ndim == 4:
        return fn(image_tensor)
    start_dims = image_tensor.shape[:-3]
    inp = torch.flatten(image_tensor, end_dim=-4)
    flat_out = fn(inp)
    return torch.reshape(flat_out, (*start_dims, *flat_out.shape[1:]))
