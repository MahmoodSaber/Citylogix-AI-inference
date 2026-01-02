"""
Citylogix AI Inference

Run segmentation inference on road/pavement images using multiple AI model architectures.
"""

__version__ = "0.1.0"

from .config import InferenceConfig, ModelConfig
from .predictor import Predictor

__all__ = [
    "__version__",
    "InferenceConfig",
    "ModelConfig",
    "Predictor",
]
