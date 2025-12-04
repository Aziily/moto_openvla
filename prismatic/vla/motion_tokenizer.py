"""
motion_tokenizer.py

Extension class; wraps latent motion tokenizer to extract motion tokens from image pairs.
"""

from typing import List, Union, Optional
import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
from PIL import Image


class MotionTokenizer:
    def __init__(
        self,
        latent_motion_tokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        codebook_size: int = 128,
    ) -> None:
        """
        Extracts motion tokens from image pairs using latent motion tokenizer.

        :param latent_motion_tokenizer: LatentMotionTokenizer model instance.
        :param base_tokenizer: Base LLM/VLM tokenizer to map motion token IDs to strings.
        :param codebook_size: Size of the motion token codebook (default: 128).
        """
        self.latent_motion_tokenizer = latent_motion_tokenizer
        self.base_tokenizer = base_tokenizer
        self.codebook_size = codebook_size
        
        # Map motion token IDs (0-127) to the last `codebook_size` tokens in the vocabulary
        # Similar to ActionTokenizer, we use the least used tokens at the end
        self.motion_token_begin_idx: int = int(self.base_tokenizer.vocab_size - (self.codebook_size + 1))

    def __call__(
        self, 
        cond_image: Union[Image.Image, torch.Tensor, np.ndarray], 
        target_image: Union[Image.Image, torch.Tensor, np.ndarray]
    ) -> Union[str, List[str]]:
        """
        Extract motion tokens from image pair and convert to string tokens.

        :param cond_image: Condition image (current frame) as PIL Image, Tensor, or numpy array.
        :param target_image: Target image (future frame) as PIL Image, Tensor, or numpy array.
        :return: String representation of motion tokens.
        """
        # Extract motion token IDs
        motion_token_ids = self.extract_motion_token_ids(cond_image, target_image)
        
        # Map motion token IDs to tokenizer vocabulary indices
        # motion_token_ids are in range [0, codebook_size-1]
        # We map them to [vocab_size - codebook_size, vocab_size - 1]
        vocab_indices = self.base_tokenizer.vocab_size - (self.codebook_size - motion_token_ids)
        
        # Handle single element vs. batch
        if len(motion_token_ids.shape) == 1:
            return self.base_tokenizer.decode(vocab_indices.tolist())
        else:
            return self.base_tokenizer.batch_decode(vocab_indices.tolist())

    def extract_motion_token_ids(
        self,
        cond_image: Union[Image.Image, torch.Tensor, np.ndarray],
        target_image: Union[Image.Image, torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """
        Extract motion token IDs directly without converting to strings.

        :param cond_image: Condition image (current frame).
        :param target_image: Target image (future frame).
        :return: Motion token IDs as numpy array, shape (motion_query_num,).
        """
        # Convert to tensors if needed
        if isinstance(cond_image, Image.Image):
            raise ValueError("PIL Images should be converted to tensors before calling extract_motion_token_ids")
        
        if isinstance(cond_image, np.ndarray):
            cond_image = torch.from_numpy(cond_image)
        if isinstance(target_image, np.ndarray):
            target_image = torch.from_numpy(target_image)
        
        # Ensure tensors are on the same device as the model
        device = next(self.latent_motion_tokenizer.parameters()).device
        if isinstance(cond_image, torch.Tensor):
            cond_image = cond_image.to(device)
        if isinstance(target_image, torch.Tensor):
            target_image = target_image.to(device)
        
        # Add batch dimension if needed
        if len(cond_image.shape) == 3:
            cond_image = cond_image.unsqueeze(0)
        if len(target_image.shape) == 3:
            target_image = target_image.unsqueeze(0)
        
        # Extract motion token IDs
        with torch.no_grad():
            motion_token_ids = self.latent_motion_tokenizer(
                cond_pixel_values=cond_image,
                target_pixel_values=target_image,
                return_motion_token_ids_only=True
            )  # Shape: (bs, motion_query_num)
        
        # Convert to numpy and remove batch dimension if single sample
        if isinstance(motion_token_ids, torch.Tensor):
            motion_token_ids = motion_token_ids.cpu().numpy()
        
        # Remove batch dimension if single sample
        if len(motion_token_ids.shape) == 2 and motion_token_ids.shape[0] == 1:
            motion_token_ids = motion_token_ids[0]
        
        return motion_token_ids

    def decode_token_ids_to_motion_tokens(self, motion_token_vocab_ids: np.ndarray) -> np.ndarray:
        """
        Convert tokenizer vocabulary IDs back to motion token IDs.

        :param motion_token_vocab_ids: Token IDs from the tokenizer vocabulary.
        :return: Motion token IDs in range [0, codebook_size-1].
        """
        # Reverse the mapping: vocab_id -> motion_token_id
        motion_token_ids = self.codebook_size - (self.base_tokenizer.vocab_size - motion_token_vocab_ids)
        return np.clip(motion_token_ids, a_min=0, a_max=self.codebook_size - 1)

    @property
    def vocab_size(self) -> int:
        """Number of motion tokens per query."""
        return self.codebook_size

    @property
    def motion_query_num(self) -> int:
        """Number of motion query tokens."""
        return self.latent_motion_tokenizer.m_former.query_num

