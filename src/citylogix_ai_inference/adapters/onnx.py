"""
ONNX model adapter for OneFormer.

Ported from streetscan_segmentation/model_tools.py
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from transformers import AutoProcessor, OneFormerProcessor

from .base import ModelAdapter, post_process_semantic_segmentation


class ONNXModel(ModelAdapter):
    """
    ONNX model inference implementation for OneFormer.
    """

    def __init__(
        self,
        model_path: str | Path,
        processor: OneFormerProcessor | AutoProcessor,
        providers: list[str] | None = None,
    ):
        """
        Initialize the ONNX model for inference.

        Args:
            model_path: Path to the ONNX model file.
            processor: HuggingFace processor for the model.
            providers: List of ONNX execution providers.
                       Defaults to ['CUDAExecutionProvider', 'CPUExecutionProvider'].
        """
        super().__init__(model_path, processor)

        self.providers = providers or [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        # Load the model
        self.session = self.load_model()

        # Try to load id2label from model metadata
        self._id2label = self._load_id2label_from_metadata()

    def load_model(self) -> Any:
        """
        Load the ONNX model.

        Returns:
            ONNX runtime inference session.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX model inference. "
                "Install it with: pip install onnxruntime or pip install onnxruntime-gpu"
            )

        logger.info(f"Available ONNX providers: {ort.get_available_providers()}")

        session = ort.InferenceSession(
            str(self.model_path),
            providers=self.providers,
        )

        logger.info(f"Using ONNX providers: {session.get_providers()}")

        return session

    def _load_id2label_from_metadata(self) -> dict[int, str] | None:
        """
        Try to load id2label mapping from ONNX model metadata.

        Returns:
            id2label dict if found, None otherwise.
        """
        try:
            metadata = self.session.get_modelmeta().custom_metadata_map
            id2label_str = metadata.get("id2label")
            if id2label_str:
                id2label = json.loads(id2label_str)
                return {int(k): v for k, v in id2label.items()}
        except Exception as e:
            logger.warning(f"Could not load id2label from ONNX metadata: {e}")

        return None

    def preprocess(self, image_data: np.ndarray) -> dict[str, np.ndarray]:
        """
        Preprocess input data for ONNX model.

        Args:
            image_data: Image data with shape (B, C, H, W).

        Returns:
            Preprocessed data as dict of numpy arrays.
        """
        batch_size = image_data.shape[0]

        processed = self.processor(
            images=image_data,
            task_inputs=["semantic"] * batch_size,
            return_tensors="np",
        )

        return processed

    def postprocess(
        self, raw_output: list[np.ndarray], target_size: tuple[int, int]
    ) -> list[torch.Tensor]:
        """
        Postprocess output from ONNX model.

        Args:
            raw_output: Raw output from ONNX model [class_logits, mask_logits].
            target_size: Target size (height, width).

        Returns:
            List of prediction masks.
        """
        class_queries_logits = torch.tensor(raw_output[0])
        masks_queries_logits = torch.tensor(raw_output[1])
        batch_size = class_queries_logits.shape[0]

        masks = post_process_semantic_segmentation(
            class_queries_logits,
            masks_queries_logits,
            target_sizes=[target_size] * batch_size,
        )

        return masks

    def run_inference(self, input_data: np.ndarray) -> list[torch.Tensor]:
        """
        Run inference using the ONNX model.

        Args:
            input_data: Input data with shape (B, C, H, W).

        Returns:
            List of prediction masks.
        """
        input_height = input_data.shape[2]
        input_width = input_data.shape[3]

        preprocessed = self.preprocess(input_data)

        # Get numpy arrays for ONNX
        pixel_values = preprocessed["pixel_values"]
        task_inputs = preprocessed["task_inputs"]

        # Handle tensor to numpy conversion if needed
        if hasattr(task_inputs, "cpu"):
            task_inputs = task_inputs.cpu().numpy()

        raw_output = self.session.run(
            None,
            {
                "pixel_values": pixel_values,
                "task_inputs": task_inputs,
            },
        )

        return self.postprocess(raw_output, (input_height, input_width))


def load_class_map_from_onnx(model_path: str | Path) -> dict[int, str] | None:
    """
    Load class map (id2label) from an ONNX model's metadata.

    Args:
        model_path: Path to the ONNX model file.

    Returns:
        Dictionary mapping class ID to class name, or None if not found.
    """
    try:
        import onnx

        model = onnx.load(str(model_path))
        metadata = {prop.key: prop.value for prop in model.metadata_props}

        id2label_str = metadata.get("id2label")
        if id2label_str:
            id2label = json.loads(id2label_str)
            return {int(k): v for k, v in id2label.items()}
    except Exception as e:
        logger.warning(f"Could not load id2label from ONNX model: {e}")

    return None
