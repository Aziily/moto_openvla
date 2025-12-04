"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.motion_tokenizer import MotionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# Make sure the project root (containing the ``latent_motion_tokenizer`` package)
# is on ``sys.path`` so that we can import from it.
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from latent_motion_tokenizer.src.processors.preprocessor_utils import get_rgb_preprocessor

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer = None
    motion_tokenizer: MotionTokenizer = None
    base_tokenizer: PreTrainedTokenizerBase = None
    image_transform: ImageTransform = None
    prompt_builder_fn: Type[PromptBuilder] = None
    predict_stop_token: bool = True
    use_motion_token: bool = False
    future_observation_k: int = 3  # k steps into the future for motion token extraction
    window_size: int = 1  # window_size used in chunk_act_obs, needed to locate current frame
    motion_preprocessor: Any = get_rgb_preprocessor("dinov2")

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name = rlds_batch["dataset_name"]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        
        # Get current frame
        img_primary = rlds_batch["observation"]["image_primary"]
        
        # Handle chunked observations: image_primary shape is (window_size + future_observation_window_size, H, W, C)
        # The frames are arranged as: [t-(window_size-1), ..., t-1, t, t+1, ..., t+future_observation_window_size]
        # Current frame t is at index = window_size - 1 (the last frame in the "past+current" window)
        # Future frame t+k is at index = window_size - 1 + k
        assert img_primary.shape[0] == self.window_size+self.future_observation_k, f"img_primary shape must be ({self.window_size}+{self.future_observation_k}={self.window_size+self.future_observation_k}, H, W, C), but got {img_primary.shape}"
        # print(f"img_primary shape: {img_primary.shape}")
        img_curr = Image.fromarray(img_primary[self.window_size - 1])
        img_next = Image.fromarray(img_primary[self.window_size - 1 + self.future_observation_k])
        
        # Transform images and extract tokens
        if self.use_motion_token and self.motion_tokenizer is not None:
            # Transform images
            pixel_values_curr = self.image_transform(img_curr)
            # pixel_values_next = self.image_transform(img_next)
            motion_pixel_values_curr = self.motion_preprocessor(img_curr)
            motion_pixel_values_next = self.motion_preprocessor(img_next)
            # print(f"pixel_values_curr shape: {pixel_values_curr.shape}")
            # print(f"motion_pixel_values_curr shape: {motion_pixel_values_curr.shape}")
            # print(f"motion_pixel_values_next shape: {motion_pixel_values_next.shape}")
            
            # Extract motion tokens
            motion_tokens_str = self.motion_tokenizer(motion_pixel_values_curr, motion_pixel_values_next)
            output_tokens = motion_tokens_str
            # print(f"motion_tokens_str: {motion_tokens_str}")
            num_output_tokens = self.motion_tokenizer.motion_query_num
        else:
            # Use action tokenizer (original behavior)
            action = rlds_batch["action"][0]
            output_tokens = self.action_tokenizer(action)
            num_output_tokens = len(action)
            pixel_values_curr = self.image_transform(img_curr)

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What {'motion' if self.use_motion_token else 'action'} should the robot take to {lang}?"},
            {"from": "gpt", "value": output_tokens},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = pixel_values_curr

        # [CRITICAL] We do not want to take the loss for anything but the predicted tokens!
        # Count the number of tokens in the output (motion tokens or action tokens)
        labels[: -(num_output_tokens + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        future_observation_k: int = 3,  # k steps into the future for motion token extraction
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        # Check if we need to load future frames for motion tokenizer
        use_motion_token = hasattr(self.batch_transform, 'use_motion_token') and self.batch_transform.use_motion_token
        window_size = 1  # Keep window_size=1 to only load current frame (past frames not needed for motion tokens)
        future_observation_window_size = future_observation_k if use_motion_token else 0
        
        # Update batch_transform with window_size and future_observation_k
        self.batch_transform.window_size = window_size
        self.batch_transform.future_observation_k = future_observation_k
        
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                            # Load current frame (and past if window_size > 1)
                future_action_window_size=0,                        # For action chunking
                future_observation_window_size=future_observation_window_size,  # Load future frames for motion tokens
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
