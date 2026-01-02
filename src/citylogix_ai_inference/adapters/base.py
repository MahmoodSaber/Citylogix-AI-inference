"""
Base class for model adapters.

Ported from streetscan_segmentation/model_tools.py
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch


class ModelAdapter(ABC):
    """
    Abstract base class for model inference.

    All model implementations (PyTorch, ONNX, etc.) must inherit from this class.
    """

    def __init__(self, model_path: str | Path, processor: Any):
        """
        Initialize the model for inference.

        Args:
            model_path: Path to the model file.
            processor: HuggingFace processor for the model.
        """
        self.model_path = Path(model_path)
        self.processor = processor
        self._id2label: dict[int, str] | None = None  # Initialize to avoid AttributeError

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

    @abstractmethod
    def load_model(self) -> Any:
        """
        Load the model from disk.

        Returns:
            Loaded model object.
        """
        raise NotImplementedError("Method `load_model` must be implemented.")

    @abstractmethod
    def preprocess(self, image_data: np.ndarray) -> Any:
        """
        Preprocess the input data before inference.

        Args:
            image_data: Image data with shape (B, C, H, W).

        Returns:
            Preprocessed data ready for model input.
        """
        raise NotImplementedError("Method `preprocess` must be implemented.")

    @abstractmethod
    def postprocess(self, raw_output: Any, target_size: tuple[int, int]) -> list:
        """
        Postprocess the raw output from the model.

        Args:
            raw_output: Raw output from the model.
            target_size: Target size for output (height, width).

        Returns:
            List of prediction masks.
        """
        raise NotImplementedError("Method `postprocess` must be implemented.")

    @abstractmethod
    def run_inference(self, input_data: np.ndarray) -> list:
        """
        Run inference on the input data.

        Args:
            input_data: Input data with shape (B, C, H, W).

        Returns:
            List of prediction masks.
        """
        raise NotImplementedError("Method `run_inference` must be implemented.")

    @property
    def id2label(self) -> dict[int, str] | None:
        """Get id to label mapping if available."""
        return getattr(self, "_id2label", None)

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        # First try from stored num_labels (set during model init)
        if hasattr(self, 'num_labels') and self.num_labels > 0:
            return self.num_labels
        # Fall back to id2label length
        return len(self._id2label) if self._id2label else 0


def post_process_semantic_segmentation(
    class_queries_logits: torch.Tensor,
    masks_queries_logits: torch.Tensor,
    target_sizes: list[tuple[int, int]] | None = None,
) -> list[torch.Tensor]:
    """
    Post-process semantic segmentation outputs.

    This function converts model outputs to semantic segmentation maps.

    Args:
        class_queries_logits: Class logits from the model.
        masks_queries_logits: Mask logits from the model.
        target_sizes: List of target sizes (height, width) for each batch item.

    Returns:
        List of semantic segmentation maps.
    """
    # Remove the null class `[..., :-1]`
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

    # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
    batch_size = class_queries_logits.shape[0]

    # Resize logits and compute semantic segmentation maps
    if target_sizes is not None:
        if batch_size != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )

        semantic_segmentation = []
        for idx in range(batch_size):
            resized_logits = torch.nn.functional.interpolate(
                segmentation[idx].unsqueeze(dim=0),
                size=target_sizes[idx],
                mode="bilinear",
                align_corners=False,
            )
            semantic_map = resized_logits[0].argmax(dim=0)
            semantic_segmentation.append(semantic_map)
    else:
        semantic_segmentation = segmentation.argmax(dim=1)
        semantic_segmentation = [
            semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])
        ]

    return semantic_segmentation


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path | str,
    device: str = "cpu",
) -> dict[int, str] | None:
    """
    Load model checkpoint.

    Args:
        model: PyTorch model to load weights into.
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.

    Returns:
        id2label mapping if available in checkpoint.
    """
    from loguru import logger

    logger.debug(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    pretrained_dict = checkpoint["model_state_dict"]
    model_dict = model.state_dict()

    # Filter out keys that do not match or have incompatible shapes
    filtered_pretrained_dict = {}
    skipped_keys = []
    for k, v in pretrained_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered_pretrained_dict[k] = v
        else:
            skipped_keys.append(k)

    if skipped_keys:
        logger.debug(f"Skipped {len(skipped_keys)} keys with shape mismatch")

    # Overwrite entries in the existing state dict
    model_dict.update(filtered_pretrained_dict)
    model.load_state_dict(model_dict)

    logger.debug(f"Checkpoint loaded: {Path(checkpoint_path).name}")

    # Return id2label if available
    id2label = checkpoint.get("id2label", None)
    if id2label:
        id2label = {int(k): v for k, v in id2label.items()}

    return id2label
