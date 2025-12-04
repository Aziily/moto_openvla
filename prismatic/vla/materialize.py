"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type, Optional

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.motion_tokenizer import MotionTokenizer
from prismatic.vla.motion_tokenizer_utils import load_latent_motion_tokenizer
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    use_motion_token: bool = False,
    latent_motion_tokenizer_path: Optional[str] = None,
    window_size: int = 1,
    future_observation_k: int = 3,  # k steps into the future for motion token extraction
) -> Tuple[Dataset, Optional[ActionTokenizer], Optional[MotionTokenizer], PaddedCollatorForActionPrediction]:
    """
    Initialize RLDS Dataset (wraps TFDS), ActionTokenizer/MotionTokenizer, and initialize transform/collation functions.
    
    :param use_motion_token: If True, use motion tokenizer instead of action tokenizer
    :param latent_motion_tokenizer_path: Path to latent motion tokenizer checkpoint (required if use_motion_token=True)
    :param future_observation_k: k steps into the future for motion token extraction (default: 1)
    :return: Tuple of (dataset, action_tokenizer, motion_tokenizer, collator)
    """
    if use_motion_token:
        if latent_motion_tokenizer_path is None:
            raise ValueError("latent_motion_tokenizer_path must be provided when use_motion_token=True")
        
        # Load latent motion tokenizer
        latent_motion_tokenizer = load_latent_motion_tokenizer(latent_motion_tokenizer_path)
        motion_tokenizer = MotionTokenizer(latent_motion_tokenizer, tokenizer)
        
        batch_transform = RLDSBatchTransform(
            motion_tokenizer=motion_tokenizer,
            base_tokenizer=tokenizer,
            image_transform=image_transform,
            prompt_builder_fn=prompt_builder_fn,
            predict_stop_token=predict_stop_token,
            use_motion_token=True,
            window_size=window_size,
            future_observation_k=future_observation_k,
        )
        action_tokenizer = None  # Not used when motion_tokenizer is active
    else:
        action_tokenizer = ActionTokenizer(tokenizer)
        motion_tokenizer = None
        batch_transform = RLDSBatchTransform(
            action_tokenizer=action_tokenizer,
            base_tokenizer=tokenizer,
            image_transform=image_transform,
            prompt_builder_fn=prompt_builder_fn,
            predict_stop_token=predict_stop_token,
            use_motion_token=False,
            window_size=window_size,
            future_observation_k=future_observation_k,
        )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    # Build RLDS Iterable Dataset
    # `default_image_resolution` may come in various formats depending on the model config:
    #   - (H, W)
    #   - (C, H, W)
    #   - (S,)  where S is a square side length
    # Normalize it to a (H, W) tuple for the RLDS pipeline / TF resize ops.
    sizes = tuple(default_image_resolution)
    if len(sizes) == 1:
        resize_resolution = (sizes[0], sizes[0])
    elif len(sizes) == 2:
        resize_resolution = sizes
    elif len(sizes) == 3:
        # Assume CHW or similar and drop the channel dimension
        resize_resolution = sizes[1:]
    else:
        # Fallback: take the last two dims as H, W
        resize_resolution = sizes[-2:]

    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=resize_resolution,
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        future_observation_k=future_observation_k,
    )

    return dataset, action_tokenizer, motion_tokenizer, collator
