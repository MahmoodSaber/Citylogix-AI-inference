"""
Model registry for loading models from configuration.
"""

from pathlib import Path

import torch
from loguru import logger
from transformers import (
    AutoProcessor,
    CLIPTokenizerFast,
    OneFormerImageProcessor,
    OneFormerProcessor,
)

from ..config import InferenceConfig, ModelConfig
from .base import ModelAdapter
from .onnx import ONNXModel
from .pytorch import PyTorchModel, load_class_map_from_checkpoint


class ModelRegistry:
    """
    Registry for loading and managing models from configuration.
    """

    def __init__(self, config: InferenceConfig):
        """
        Initialize model registry.

        Args:
            config: Inference configuration.
        """
        self.config = config
        self._models: dict[str, ModelAdapter] = {}
        self._processors: dict[str, OneFormerProcessor] = {}

    def load_all_models(self) -> dict[str, ModelAdapter]:
        """
        Load all models from configuration.

        Returns:
            Dictionary mapping model name to loaded model adapter.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {len(self.config.models)} models on {device}...")

        for model_config in self.config.models:
            try:
                model = self._load_model(model_config, device)
                self._models[model_config.name] = model
                logger.info(f"Loaded model '{model_config.name}' ({model_config.mode} mode)")
            except Exception as e:
                logger.error(f"Failed to load model '{model_config.name}': {e}")
                raise

        return self._models

    def _load_model(self, model_config: ModelConfig, device: str) -> ModelAdapter:
        """
        Load a single model.

        Args:
            model_config: Model configuration.
            device: Device to load model on.

        Returns:
            Loaded model adapter.
        """
        model_path = Path(model_config.path)

        # Get number of labels from checkpoint
        num_labels = self._get_num_labels(model_path)

        # Create processor
        processor = self._create_processor(model_config, num_labels)

        # Load model based on type
        if model_path.suffix == ".pt":
            return PyTorchModel(
                model_path=model_path,
                processor=processor,
                num_labels=num_labels,
                device=device,
            )
        elif model_path.suffix == ".onnx":
            providers = self._get_onnx_providers(device)
            return ONNXModel(
                model_path=model_path,
                processor=processor,
                providers=providers,
            )
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")

    def _get_num_labels(self, model_path: Path) -> int:
        """
        Get number of labels from model checkpoint.

        Args:
            model_path: Path to model file.

        Returns:
            Number of labels.
        """
        if model_path.suffix == ".pt":
            id2label = load_class_map_from_checkpoint(model_path)
            return len(id2label)
        elif model_path.suffix == ".onnx":
            from .onnx import load_class_map_from_onnx

            id2label = load_class_map_from_onnx(model_path)
            if id2label:
                return len(id2label)
            # Default fallback
            logger.warning(f"Could not determine num_labels for {model_path}, using default 6")
            return 6
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")

    def _create_processor(
        self, model_config: ModelConfig, num_labels: int
    ) -> OneFormerProcessor:
        """
        Create HuggingFace processor for a model.

        Args:
            model_config: Model configuration.
            num_labels: Number of output labels.

        Returns:
            OneFormerProcessor instance.
        """
        processor_size = self.config.get_model_config(model_config, "processor_size")
        norm = self.config.normalization

        processor = OneFormerProcessor(
            image_processor=OneFormerImageProcessor(
                class_info_file="coco_panoptic.json",
                processor_class="OneFormerProcessor",
                size=processor_size,
                image_mean=norm.mean,
                image_std=norm.std,
                num_labels=num_labels,
                ignore_index=255,
            ),
            tokenizer=CLIPTokenizerFast.from_pretrained(
                "shi-labs/oneformer_coco_swin_large"
            ),
            num_labels=num_labels,
        )

        return processor

    def _get_onnx_providers(self, device: str) -> list[str]:
        """
        Get ONNX execution providers based on device.

        Args:
            device: Device string ('cuda' or 'cpu').

        Returns:
            List of provider names.
        """
        if device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def get_model(self, name: str) -> ModelAdapter:
        """
        Get a loaded model by name.

        Args:
            name: Model name.

        Returns:
            Model adapter.

        Raises:
            KeyError: If model not found.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found. Available: {list(self._models.keys())}")
        return self._models[name]

    def get_models_by_mode(self, mode: str) -> dict[str, ModelAdapter]:
        """
        Get all models with a specific mode.

        Args:
            mode: Processing mode ('macro' or 'sliding_window').

        Returns:
            Dictionary of model name to model adapter.
        """
        result = {}
        for model_config in self.config.models:
            if model_config.mode == mode and model_config.name in self._models:
                result[model_config.name] = self._models[model_config.name]
        return result

    def get_model_config(self, name: str) -> ModelConfig:
        """
        Get model configuration by name.

        Args:
            name: Model name.

        Returns:
            Model configuration.
        """
        for model_config in self.config.models:
            if model_config.name == name:
                return model_config
        raise KeyError(f"Model config '{name}' not found")

    def unload_all(self) -> None:
        """Unload all models and free memory."""
        for name in list(self._models.keys()):
            del self._models[name]

        self._models.clear()
        self._processors.clear()

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("All models unloaded")

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        return name in self._models
