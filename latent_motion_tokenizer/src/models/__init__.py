"""
Compatibility wrapper package.

The original latent motion tokenizer project used the namespace
``latent_motion_tokenizer.src.models.*``.  The code in this repository
is flattened under ``latent_motion_tokenizer.src``.

This subpackage re-exports the relevant modules so that checkpoints whose
Hydra configs reference ``latent_motion_tokenizer.src.models.*`` can be
loaded without modification.
"""

from ..latent_motion_tokenizer import LatentMotionTokenizer  # noqa: F401
from ..latent_motion_decoder import LatentMotionDecoder  # noqa: F401
from ..timm_dinov2_model import TimmDinoV2VsionEncoder  # noqa: F401
from ..m_former import MFormer  # noqa: F401
from ..vector_quantizer import VectorQuantizer2  # noqa: F401






