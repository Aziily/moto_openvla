"""
motion_tokenizer_utils.py

Utility functions for loading and initializing latent motion tokenizer.
"""

import os
import sys
import torch
import omegaconf
import hydra
from pathlib import Path
from typing import Optional

from prismatic.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


def load_latent_motion_tokenizer(pretrained_path: str, device: Optional[str] = None) -> torch.nn.Module:
    """
    Load latent motion tokenizer from checkpoint.
    
    :param pretrained_path: Path to the checkpoint directory containing config.yaml and pytorch_model.bin
    :param device: Device to load the model on (e.g., 'cuda', 'cpu'). If None, uses the model's default.
    :return: Loaded LatentMotionTokenizer model
    """
    config_path = os.path.join(pretrained_path, "config.yaml")
    checkpoint_path = os.path.join(pretrained_path, "pytorch_model.bin")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        overwatch.warning(f"Checkpoint file not found: {checkpoint_path}. Loading model without weights.")
        checkpoint_path = None
    
    # Load config
    config = omegaconf.OmegaConf.load(config_path)

    # Make sure the project root (containing the ``latent_motion_tokenizer`` package)
    # is on ``sys.path`` so that Hydra can resolve targets like
    # ``latent_motion_tokenizer.src.models.latent_motion_tokenizer.LatentMotionTokenizer``.
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    model = hydra.utils.instantiate(config)
    model.config = config
    
    # Load weights if available
    if checkpoint_path is not None:
        missing_keys, unexpected_keys = model.load_state_dict(
            torch.load(checkpoint_path, map_location='cpu'), 
            strict=False
        )
        missing_root_keys = set([k.split(".")[0] for k in missing_keys])
        overwatch.info(f'Loaded checkpoint from {checkpoint_path}')
        overwatch.info(f'Missing keys: {missing_root_keys}')
        overwatch.info(f'Unexpected keys: {unexpected_keys}')
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
    
    model.eval()
    return model

