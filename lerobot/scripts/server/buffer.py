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
import functools
import io
import pickle
from typing import Any, Callable, Optional, Sequence, TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class Transition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: float
    next_state: dict[str, torch.Tensor]
    done: bool
    truncated: bool
    complementary_info: dict[str, Any] = None


class BatchTransition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: dict[str, torch.Tensor]
    done: torch.Tensor
    truncated: torch.Tensor


def move_transition_to_device(transition: Transition, device: str = "cpu") -> Transition:
    # Move state tensors to CPU
    device = torch.device(device)
    transition["state"] = {
        key: val.to(device, non_blocking=device.type == "cuda") for key, val in transition["state"].items()
    }

    # Move action to CPU
    transition["action"] = transition["action"].to(device, non_blocking=device.type == "cuda")

    # No need to move reward or done, as they are float and bool

    # No need to move reward or done, as they are float and bool
    if isinstance(transition["reward"], torch.Tensor):
        transition["reward"] = transition["reward"].to(device=device, non_blocking=device.type == "cuda")

    if isinstance(transition["done"], torch.Tensor):
        transition["done"] = transition["done"].to(device, non_blocking=device.type == "cuda")

    if isinstance(transition["truncated"], torch.Tensor):
        transition["truncated"] = transition["truncated"].to(device, non_blocking=device.type == "cuda")

    # Move next_state tensors to CPU
    transition["next_state"] = {
        key: val.to(device, non_blocking=device.type == "cuda")
        for key, val in transition["next_state"].items()
    }

    # If complementary_info is present, move its tensors to CPU
    # if transition["complementary_info"] is not None:
    #     transition["complementary_info"] = {
    #         key: val.to(device, non_blocking=True) for key, val in transition["complementary_info"].items()
    #     }
    return transition


def move_state_dict_to_device(state_dict, device="cpu"):
    """
    Recursively move all tensors in a (potentially) nested
    dict/list/tuple structure to the CPU.
    """
    if isinstance(state_dict, torch.Tensor):
        return state_dict.to(device)
    elif isinstance(state_dict, dict):
        return {k: move_state_dict_to_device(v, device=device) for k, v in state_dict.items()}
    elif isinstance(state_dict, list):
        return [move_state_dict_to_device(v, device=device) for v in state_dict]
    elif isinstance(state_dict, tuple):
        return tuple(move_state_dict_to_device(v, device=device) for v in state_dict)
    else:
        return state_dict


def state_to_bytes(state_dict: dict[str, torch.Tensor]) -> bytes:
    """Convert model state dict to flat array for transmission"""
    buffer = io.BytesIO()

    torch.save(state_dict, buffer)

    return buffer.getvalue()


def bytes_to_state_dict(buffer: bytes) -> dict[str, torch.Tensor]:
    buffer = io.BytesIO(buffer)
    buffer.seek(0)
    return torch.load(buffer)


def python_object_to_bytes(python_object: Any) -> bytes:
    return pickle.dumps(python_object)


def bytes_to_python_object(buffer: bytes) -> Any:
    buffer = io.BytesIO(buffer)
    buffer.seek(0)
    return pickle.load(buffer)


def bytes_to_transitions(buffer: bytes) -> list[Transition]:
    buffer = io.BytesIO(buffer)
    buffer.seek(0)
    return torch.load(buffer)


def transitions_to_bytes(transitions: list[Transition]) -> bytes:
    buffer = io.BytesIO()
    torch.save(transitions, buffer)
    return buffer.getvalue()


def random_crop_vectorized(images: torch.Tensor, output_size: tuple) -> torch.Tensor:
    """
    Perform a per-image random crop over a batch of images in a vectorized way.
    (Same as shown previously.)
    """
    B, C, H, W = images.shape  # noqa: N806
    crop_h, crop_w = output_size

    if crop_h > H or crop_w > W:
        raise ValueError(
            f"Requested crop size ({crop_h}, {crop_w}) is bigger than the image size ({H}, {W})."
        )

    tops = torch.randint(0, H - crop_h + 1, (B,), device=images.device)
    lefts = torch.randint(0, W - crop_w + 1, (B,), device=images.device)

    rows = torch.arange(crop_h, device=images.device).unsqueeze(0) + tops.unsqueeze(1)
    cols = torch.arange(crop_w, device=images.device).unsqueeze(0) + lefts.unsqueeze(1)

    rows = rows.unsqueeze(2).expand(-1, -1, crop_w)  # (B, crop_h, crop_w)
    cols = cols.unsqueeze(1).expand(-1, crop_h, -1)  # (B, crop_h, crop_w)

    images_hwcn = images.permute(0, 2, 3, 1)  # (B, H, W, C)

    # Gather pixels
    cropped_hwcn = images_hwcn[torch.arange(B, device=images.device).view(B, 1, 1), rows, cols, :]
    # cropped_hwcn => (B, crop_h, crop_w, C)

    cropped = cropped_hwcn.permute(0, 3, 1, 2)  # (B, C, crop_h, crop_w)
    return cropped


def random_shift(images: torch.Tensor, pad: int = 4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    _, _, h, w = images.shape
    images = F.pad(input=images, pad=(pad, pad, pad, pad), mode="replicate")
    return random_crop_vectorized(images=images, output_size=(h, w))


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: str = "cuda:0",
        state_keys: Optional[Sequence[str]] = None,
        image_augmentation_function: Optional[Callable] = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
    ):
        """
        Args:
            capacity (int): Maximum number of transitions to store in the buffer.
            device (str): The device where the tensors will be moved when sampling ("cuda:0" or "cpu").
            state_keys (List[str]): The list of keys that appear in `state` and `next_state`.
            image_augmentation_function (Optional[Callable]): A function that takes a batch of images
                and returns a batch of augmented images. If None, a default augmentation function is used.
            use_drq (bool): Whether to use the default DRQ image augmentation style, when sampling in the buffer.
            storage_device: The device (e.g. "cpu" or "cuda:0") where the data will be stored.
                Using "cpu" can help save GPU memory.
            optimize_memory (bool): If True, optimizes memory by not storing duplicate next_states when
                they can be derived from states. This is useful for large datasets where next_state[i] = state[i+1].
        """
        self.capacity = capacity
        self.device = device
        self.storage_device = storage_device
        self.position = 0
        self.size = 0
        self.initialized = False
        self.optimize_memory = optimize_memory

        # Track episode boundaries for memory optimization
        self.episode_ends = torch.zeros(capacity, dtype=torch.bool, device=storage_device)

        # If no state_keys provided, default to an empty list
        self.state_keys = state_keys if state_keys is not None else []

        if image_augmentation_function is None:
            base_function = functools.partial(random_shift, pad=4)
            self.image_augmentation_function = torch.compile(base_function)
        self.use_drq = use_drq

    def _initialize_storage(self, state: dict[str, torch.Tensor], action: torch.Tensor):
        """Initialize the storage tensors based on the first transition."""
        # Determine shapes from the first transition
        state_shapes = {key: val.squeeze(0).shape for key, val in state.items()}
        action_shape = action.squeeze(0).shape

        # Pre-allocate tensors for storage
        self.states = {
            key: torch.empty((self.capacity, *shape), device=self.storage_device)
            for key, shape in state_shapes.items()
        }
        self.actions = torch.empty((self.capacity, *action_shape), device=self.storage_device)
        self.rewards = torch.empty((self.capacity,), device=self.storage_device)

        if not self.optimize_memory:
            # Standard approach: store states and next_states separately
            self.next_states = {
                key: torch.empty((self.capacity, *shape), device=self.storage_device)
                for key, shape in state_shapes.items()
            }
        else:
            # Memory-optimized approach: don't allocate next_states buffer
            # Just create a reference to states for consistent API
            self.next_states = self.states  # Just a reference for API consistency

        self.dones = torch.empty((self.capacity,), dtype=torch.bool, device=self.storage_device)
        self.truncateds = torch.empty((self.capacity,), dtype=torch.bool, device=self.storage_device)

        self.initialized = True

    def __len__(self):
        return self.size

    def add(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: dict[str, torch.Tensor],
        done: bool,
        truncated: bool,
        complementary_info: Optional[dict[str, torch.Tensor]] = None,
    ):
        """Saves a transition, ensuring tensors are stored on the designated storage device."""
        # Initialize storage if this is the first transition
        if not self.initialized:
            self._initialize_storage(state=state, action=action)

        # Store the transition in pre-allocated tensors
        for key in self.states:
            self.states[key][self.position].copy_(state[key].squeeze(dim=0))

            if not self.optimize_memory:
                # Only store next_states if not optimizing memory
                self.next_states[key][self.position].copy_(next_state[key].squeeze(dim=0))

        self.actions[self.position].copy_(action.squeeze(dim=0))
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.truncateds[self.position] = truncated

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> BatchTransition:
        """Sample a random batch of transitions and collate them into batched tensors."""
        if not self.initialized:
            raise RuntimeError("Cannot sample from an empty buffer. Add transitions first.")

        batch_size = min(batch_size, self.size)
        high = max(0, self.size - 1) if self.optimize_memory and self.size < self.capacity else self.size

        # Random indices for sampling - create on the same device as storage
        idx = torch.randint(low=0, high=high, size=(batch_size,), device=self.storage_device)

        # Identify image keys that need augmentation
        image_keys = [k for k in self.states if k.startswith("observation.image")] if self.use_drq else []

        # Create batched state and next_state
        batch_state = {}
        batch_next_state = {}

        # First pass: load all state tensors to target device
        for key in self.states:
            batch_state[key] = self.states[key][idx].to(self.device)

            if not self.optimize_memory:
                # Standard approach - load next_states directly
                batch_next_state[key] = self.next_states[key][idx].to(self.device)
            else:
                # Memory-optimized approach - get next_state from the next index
                next_idx = (idx + 1) % self.capacity
                batch_next_state[key] = self.states[key][next_idx].to(self.device)

        # Apply image augmentation in a batched way if needed
        if self.use_drq and image_keys:
            # Concatenate all images from state and next_state
            all_images = []
            for key in image_keys:
                all_images.append(batch_state[key])
                all_images.append(batch_next_state[key])

            # Batch all images and apply augmentation once
            all_images_tensor = torch.cat(all_images, dim=0)
            augmented_images = self.image_augmentation_function(all_images_tensor)

            # Split the augmented images back to their sources
            for i, key in enumerate(image_keys):
                # State images are at even indices (0, 2, 4...)
                batch_state[key] = augmented_images[i * 2 * batch_size : (i * 2 + 1) * batch_size]
                # Next state images are at odd indices (1, 3, 5...)
                batch_next_state[key] = augmented_images[(i * 2 + 1) * batch_size : (i + 1) * 2 * batch_size]

        # Sample other tensors
        batch_actions = self.actions[idx].to(self.device)
        batch_rewards = self.rewards[idx].to(self.device)
        batch_dones = self.dones[idx].to(self.device).float()
        batch_truncateds = self.truncateds[idx].to(self.device).float()

        return BatchTransition(
            state=batch_state,
            action=batch_actions,
            reward=batch_rewards,
            next_state=batch_next_state,
            done=batch_dones,
            truncated=batch_truncateds,
        )

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        """
        Creates an infinite iterator that yields batches of transitions.
        Will automatically restart when internal iterator is exhausted.

        Args:
            batch_size (int): Size of batches to sample
            async_prefetch (bool): Whether to use asynchronous prefetching with threads (default: True)
            queue_size (int): Number of batches to prefetch (default: 2)

        Yields:
            BatchTransition: Batched transitions
        """
        while True:  # Create an infinite loop
            if async_prefetch:
                # Get the standard iterator
                iterator = self._get_async_iterator(queue_size=queue_size, batch_size=batch_size)
            else:
                iterator = self._get_naive_iterator(batch_size=batch_size, queue_size=queue_size)

            # Yield all items from the iterator
            try:
                yield from iterator
            except StopIteration:
                # Just continue the outer loop to create a new iterator
                pass

    def _get_async_iterator(self, batch_size: int, queue_size: int = 2):
        """
        Creates an iterator that prefetches batches in a background thread.

        Args:
            queue_size (int): Number of batches to prefetch (default: 2)
            batch_size (int): Size of batches to sample (default: 128)

        Yields:
            BatchTransition: Prefetched batch transitions
        """
        import queue
        import threading

        # Use thread-safe queue
        data_queue = queue.Queue(maxsize=queue_size)
        running = [True]  # Use list to allow modification in nested function

        def prefetch_worker():
            while running[0]:
                try:
                    # Sample data and add to queue
                    data = self.sample(batch_size)
                    data_queue.put(data, block=True, timeout=0.5)
                except queue.Full:
                    continue
                except Exception as e:
                    print(f"Prefetch error: {e}")
                    break

        # Start prefetching thread
        thread = threading.Thread(target=prefetch_worker, daemon=True)
        thread.start()

        try:
            while running[0]:
                try:
                    yield data_queue.get(block=True, timeout=0.5)
                except queue.Empty:
                    if not thread.is_alive():
                        break
        finally:
            # Clean up
            running[0] = False
            thread.join(timeout=1.0)

    def _get_naive_iterator(self, batch_size: int, queue_size: int = 2):
        """
        Creates a simple non-threaded iterator that yields batches.

        Args:
            batch_size (int): Size of batches to sample
            queue_size (int): Number of initial batches to prefetch

        Yields:
            BatchTransition: Batch transitions
        """
        import collections

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size)
                queue.append(data)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    @classmethod
    def from_lerobot_dataset(
        cls,
        lerobot_dataset: LeRobotDataset,
        device: str = "cuda:0",
        state_keys: Optional[Sequence[str]] = None,
        capacity: Optional[int] = None,
        action_mask: Optional[Sequence[int]] = None,
        action_delta: Optional[float] = None,
        image_augmentation_function: Optional[Callable] = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
    ) -> "ReplayBuffer":
        """
        Convert a LeRobotDataset into a ReplayBuffer.

        Args:
            lerobot_dataset (LeRobotDataset): The dataset to convert.
            device (str): The device for sampling tensors. Defaults to "cuda:0".
            state_keys (Optional[Sequence[str]]): The list of keys that appear in `state` and `next_state`.
            capacity (Optional[int]): Buffer capacity. If None, uses dataset length.
            action_mask (Optional[Sequence[int]]): Indices of action dimensions to keep.
            action_delta (Optional[float]): Factor to divide actions by.
            image_augmentation_function (Optional[Callable]): Function for image augmentation.
                If None, uses default random shift with pad=4.
            use_drq (bool): Whether to use DrQ image augmentation when sampling.
            storage_device (str): Device for storing tensor data. Using "cpu" saves GPU memory.
            optimize_memory (bool): If True, reduces memory usage by not duplicating state data.

        Returns:
            ReplayBuffer: The replay buffer with dataset transitions.
        """
        if capacity is None:
            capacity = len(lerobot_dataset)

        if capacity < len(lerobot_dataset):
            raise ValueError(
                "The capacity of the ReplayBuffer must be greater than or equal to the length of the LeRobotDataset."
            )

        # Create replay buffer with image augmentation and DrQ settings
        replay_buffer = cls(
            capacity=capacity,
            device=device,
            state_keys=state_keys,
            image_augmentation_function=image_augmentation_function,
            use_drq=use_drq,
            storage_device=storage_device,
            optimize_memory=optimize_memory,
        )

        # Convert dataset to transitions
        list_transition = cls._lerobotdataset_to_transitions(dataset=lerobot_dataset, state_keys=state_keys)

        # Initialize the buffer with the first transition to set up storage tensors
        if list_transition:
            first_transition = list_transition[0]
            first_state = {k: v.to(device) for k, v in first_transition["state"].items()}
            first_action = first_transition["action"].to(device)

            # Apply action mask/delta if needed
            if action_mask is not None:
                if first_action.dim() == 1:
                    first_action = first_action[action_mask]
                else:
                    first_action = first_action[:, action_mask]

            if action_delta is not None:
                first_action = first_action / action_delta

            replay_buffer._initialize_storage(state=first_state, action=first_action)

        # Fill the buffer with all transitions
        for data in list_transition:
            for k, v in data.items():
                if isinstance(v, dict):
                    for key, tensor in v.items():
                        v[key] = tensor.to(storage_device)
                elif isinstance(v, torch.Tensor):
                    data[k] = v.to(storage_device)

            action = data["action"]
            if action_mask is not None:
                if action.dim() == 1:
                    action = action[action_mask]
                else:
                    action = action[:, action_mask]

            if action_delta is not None:
                action = action / action_delta

            replay_buffer.add(
                state=data["state"],
                action=action,
                reward=data["reward"],
                next_state=data["next_state"],
                done=data["done"],
                truncated=False,  # NOTE: Truncation are not supported yet in lerobot dataset
            )

        return replay_buffer

    def to_lerobot_dataset(
        self,
        repo_id: str,
        fps=1,
        root=None,
        task_name="from_replay_buffer",
    ) -> LeRobotDataset:
        """
        Converts all transitions in this ReplayBuffer into a single LeRobotDataset object.
        """
        if self.size == 0:
            raise ValueError("The replay buffer is empty. Cannot convert to a dataset.")

        # Create features dictionary for the dataset
        features = {
            "index": {"dtype": "int64", "shape": [1]},  # global index across episodes
            "episode_index": {"dtype": "int64", "shape": [1]},  # which episode
            "frame_index": {"dtype": "int64", "shape": [1]},  # index inside an episode
            "timestamp": {"dtype": "float32", "shape": [1]},  # for now we store dummy
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        # Add "action"
        sample_action = self.actions[0]
        act_info = guess_feature_info(t=sample_action, name="action")
        features["action"] = act_info

        # Add "reward" and "done"
        features["next.reward"] = {"dtype": "float32", "shape": (1,)}
        features["next.done"] = {"dtype": "bool", "shape": (1,)}

        # Add state keys
        for key in self.states:
            sample_val = self.states[key][0]
            f_info = guess_feature_info(t=sample_val, name=key)
            features[key] = f_info

        # Create an empty LeRobotDataset
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot=None,  # TODO: (azouitine) Handle robot
            robot_type=None,
            features=features,
            use_videos=True,
        )

        # Start writing images if needed
        lerobot_dataset.start_image_writer(num_processes=0, num_threads=3)

        # Convert transitions into episodes and frames
        episode_index = 0
        lerobot_dataset.episode_buffer = lerobot_dataset.create_episode_buffer(episode_index=episode_index)

        frame_idx_in_episode = 0
        for idx in range(self.size):
            actual_idx = (self.position - self.size + idx) % self.capacity

            frame_dict = {}

            # Fill the data for state keys
            for key in self.states:
                frame_dict[key] = self.states[key][actual_idx].cpu()

            # Fill action, reward, done
            frame_dict["action"] = self.actions[actual_idx].cpu()
            frame_dict["next.reward"] = torch.tensor([self.rewards[actual_idx]], dtype=torch.float32).cpu()
            frame_dict["next.done"] = torch.tensor([self.dones[actual_idx]], dtype=torch.bool).cpu()

            # Add task field which is required by LeRobotDataset
            frame_dict["task"] = task_name

            # Add to the dataset's buffer
            lerobot_dataset.add_frame(frame_dict)

            # Move to next frame
            frame_idx_in_episode += 1

            # If we reached an episode boundary, call save_episode, reset counters
            if self.dones[actual_idx] or self.truncateds[actual_idx]:
                lerobot_dataset.save_episode()
                episode_index += 1
                frame_idx_in_episode = 0
                lerobot_dataset.episode_buffer = lerobot_dataset.create_episode_buffer(
                    episode_index=episode_index
                )

        # Save any remaining frames in the buffer
        if lerobot_dataset.episode_buffer["size"] > 0:
            lerobot_dataset.save_episode()

        lerobot_dataset.stop_image_writer()

        return lerobot_dataset

    @staticmethod
    def _lerobotdataset_to_transitions(
        dataset: LeRobotDataset,
        state_keys: Optional[Sequence[str]] = None,
    ) -> list[Transition]:
        """
        Convert a LeRobotDataset into a list of RL (s, a, r, s', done) transitions.

        Args:
            dataset (LeRobotDataset):
                The dataset to convert. Each item in the dataset is expected to have
                at least the following keys:
                {
                    "action": ...
                    "next.reward": ...
                    "next.done": ...
                    "episode_index": ...
                }
                plus whatever your 'state_keys' specify.

            state_keys (Optional[Sequence[str]]):
                The dataset keys to include in 'state' and 'next_state'. Their names
                will be kept as-is in the output transitions. E.g.
                ["observation.state", "observation.environment_state"].
                If None, you must handle or define default keys.

        Returns:
            transitions (List[Transition]):
                A list of Transition dictionaries with the same length as `dataset`.
        """
        if state_keys is None:
            raise ValueError("State keys must be provided when converting LeRobotDataset to Transitions.")

        transitions = []
        num_frames = len(dataset)

        # Check if the dataset has "next.done" key
        sample = dataset[0]
        has_done_key = "next.done" in sample

        # If not, we need to infer it from episode boundaries
        if not has_done_key:
            print("'next.done' key not found in dataset. Inferring from episode boundaries...")

        for i in tqdm(range(num_frames)):
            current_sample = dataset[i]

            # ----- 1) Current state -----
            current_state: dict[str, torch.Tensor] = {}
            for key in state_keys:
                val = current_sample[key]
                current_state[key] = val.unsqueeze(0)  # Add batch dimension

            # ----- 2) Action -----
            action = current_sample["action"].unsqueeze(0)  # Add batch dimension

            # ----- 3) Reward and done -----
            reward = float(current_sample["next.reward"].item())  # ensure float

            # Determine done flag - use next.done if available, otherwise infer from episode boundaries
            if has_done_key:
                done = bool(current_sample["next.done"].item())  # ensure bool
            else:
                # If this is the last frame or if next frame is in a different episode, mark as done
                done = False
                if i == num_frames - 1:
                    done = True
                elif i < num_frames - 1:
                    next_sample = dataset[i + 1]
                    if next_sample["episode_index"] != current_sample["episode_index"]:
                        done = True

            # TODO: (azouitine) Handle truncation (using the same value as done for now)
            truncated = done

            # ----- 4) Next state -----
            # If not done and the next sample is in the same episode, we pull the next sample's state.
            # Otherwise (done=True or next sample crosses to a new episode), next_state = current_state.
            next_state = current_state  # default
            if not done and (i < num_frames - 1):
                next_sample = dataset[i + 1]
                if next_sample["episode_index"] == current_sample["episode_index"]:
                    # Build next_state from the same keys
                    next_state_data: dict[str, torch.Tensor] = {}
                    for key in state_keys:
                        val = next_sample[key]
                        next_state_data[key] = val.unsqueeze(0)  # Add batch dimension
                    next_state = next_state_data

            # ----- Construct the Transition -----
            transition = Transition(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                truncated=truncated,
            )
            transitions.append(transition)

        return transitions


# Utility function to guess shapes/dtypes from a tensor
def guess_feature_info(t: torch.Tensor, name: str):
    """
    Return a dictionary with the 'dtype' and 'shape' for a given tensor or array.
    If it looks like a 3D (C,H,W) shape, we might consider it an 'image'.
    Otherwise default to 'float32' for numeric. You can customize as needed.
    """
    shape = tuple(t.shape)
    # Basic guess: if we have exactly 3 dims and shape[0] in {1, 3}, guess 'image'
    if len(shape) == 3 and shape[0] in [1, 3]:
        return {
            "dtype": "image",
            "shape": shape,
        }
    else:
        # Otherwise treat as numeric
        return {
            "dtype": "float32",
            "shape": shape,
        }


def concatenate_batch_transitions(
    left_batch_transitions: BatchTransition, right_batch_transition: BatchTransition
) -> BatchTransition:
    """NOTE: Be careful it change the left_batch_transitions in place"""
    left_batch_transitions["state"] = {
        key: torch.cat(
            [
                left_batch_transitions["state"][key],
                right_batch_transition["state"][key],
            ],
            dim=0,
        )
        for key in left_batch_transitions["state"]
    }
    left_batch_transitions["action"] = torch.cat(
        [left_batch_transitions["action"], right_batch_transition["action"]], dim=0
    )
    left_batch_transitions["reward"] = torch.cat(
        [left_batch_transitions["reward"], right_batch_transition["reward"]], dim=0
    )
    left_batch_transitions["next_state"] = {
        key: torch.cat(
            [
                left_batch_transitions["next_state"][key],
                right_batch_transition["next_state"][key],
            ],
            dim=0,
        )
        for key in left_batch_transitions["next_state"]
    }
    left_batch_transitions["done"] = torch.cat(
        [left_batch_transitions["done"], right_batch_transition["done"]], dim=0
    )
    left_batch_transitions["truncated"] = torch.cat(
        [left_batch_transitions["truncated"], right_batch_transition["truncated"]],
        dim=0,
    )
    return left_batch_transitions


if __name__ == "__main__":
    pass  # All test code is currently commented out
