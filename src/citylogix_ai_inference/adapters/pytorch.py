"""
PyTorch model adapter for OneFormer.

Ported from streetscan_segmentation/model_tools.py
"""

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from transformers import (
    AutoModelForUniversalSegmentation,
    AutoProcessor,
    OneFormerProcessor,
)

from .base import ModelAdapter, load_checkpoint


class PyTorchModel(ModelAdapter):
    """
    PyTorch model inference implementation for OneFormer.
    """

    def __init__(
        self,
        model_path: str | Path,
        processor: OneFormerProcessor | AutoProcessor,
        num_labels: int,
        model_base_path: str = "shi-labs/oneformer_coco_swin_large",
        device: str | None = None,
    ):
        """
        Initialize the PyTorch model for inference.

        Args:
            model_path: Path to the PyTorch model checkpoint (.pt file).
            processor: HuggingFace processor for the model.
            num_labels: Number of output classes.
            model_base_path: Base model path or HuggingFace model ID.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        super().__init__(model_path, processor)

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = num_labels
        self.model_base_path = model_base_path

        # Load the model
        self.model = self.load_model()

        # Configure processor
        self.processor.image_processor.num_text = (
            self.model.config.num_queries - self.model.config.text_encoder_n_ctx
        )

        self._id2label: dict[int, str] | None = None

    def load_model(self) -> torch.nn.Module:
        """
        Load the PyTorch model.

        Returns:
            Loaded PyTorch model.
        """
        logger.debug(f"Loading model from {self.model_base_path}")

        # Suppress HuggingFace/transformers warnings about mismatched sizes
        # These are expected when using custom checkpoints with different num_labels
        transformers_logger = logging.getLogger("transformers.modeling_utils")
        original_level = transformers_logger.level
        transformers_logger.setLevel(logging.ERROR)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = AutoModelForUniversalSegmentation.from_pretrained(
                self.model_base_path,
                is_training=True,
                ignore_mismatched_sizes=True,
                num_labels=self.num_labels,
                dropout=0,
            )

        # Restore logging level
        transformers_logger.setLevel(original_level)

        # Load checkpoint weights
        self._id2label = load_checkpoint(
            model=model,
            checkpoint_path=self.model_path,
            device=self.device,
        )

        model.to(self.device)
        model.eval()

        logger.debug(f"Model loaded on {self.device}")

        return model

    def preprocess(self, image_data: np.ndarray) -> dict[str, torch.Tensor]:
        """
        Preprocess input data for PyTorch model.

        Args:
            image_data: Image data with shape (B, C, H, W).

        Returns:
            Preprocessed data as dict of tensors.
        """
        batch_size = image_data.shape[0]
        height = image_data.shape[2]
        width = image_data.shape[3]

        processed = self.processor(
            images=image_data,
            task_inputs=["semantic"] * batch_size,
            segmentation_maps=np.zeros((batch_size, height, width), dtype=np.uint8),
            return_tensors="pt",
        )

        # Move tensors to device
        processed = {
            k: v.to(self.device) for k, v in processed.items() if isinstance(v, torch.Tensor)
        }

        return processed

    def postprocess(
        self, output: Any, target_size: tuple[int, int]
    ) -> list[torch.Tensor]:
        """
        Postprocess output from PyTorch model.

        Args:
            output: Raw output from the model.
            target_size: Target size (height, width).

        Returns:
            List of prediction masks.
        """
        target_sizes = [target_size for _ in output["class_queries_logits"]]

        pred_masks = self.processor.post_process_semantic_segmentation(
            output, target_sizes=target_sizes
        )

        return pred_masks

    @torch.no_grad()
    def run_inference(self, input_data: np.ndarray) -> list[torch.Tensor]:
        """
        Run inference using the PyTorch model.

        Args:
            input_data: Input data with shape (B, C, H, W).

        Returns:
            List of prediction masks.
        """
        input_height = input_data.shape[2]
        input_width = input_data.shape[3]

        preprocessed = self.preprocess(input_data)
        raw_output = self.model(**preprocessed)

        return self.postprocess(raw_output, (input_height, input_width))


def load_class_map_from_checkpoint(model_path: str | Path) -> dict[int, str]:
    """
    Load class map (id2label) from a PyTorch checkpoint.

    Args:
        model_path: Path to the checkpoint file.

    Returns:
        Dictionary mapping class ID to class name.
    """
    checkpoint = torch.load(model_path, map_location="cpu")
    id2label = checkpoint.get("id2label", {})
    return {int(k): v for k, v in id2label.items()}
