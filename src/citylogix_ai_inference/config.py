"""
Configuration system for Citylogix AI Inference.

Uses Pydantic for validation and YAML for configuration files.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class NormalizationConfig(BaseModel):
    """Image normalization parameters."""

    mean: list[float] = Field(
        default=[0.485, 0.456, 0.406],
        description="ImageNet mean values for normalization",
    )
    std: list[float] = Field(
        default=[0.229, 0.224, 0.225],
        description="ImageNet std values for normalization",
    )


class OutputConfig(BaseModel):
    """Output format configuration."""

    coco_json: bool = Field(default=True, description="Export COCO JSON format")
    cvat_xml: bool = Field(default=True, description="Export CVAT XML format")
    fiftyone: bool = Field(default=False, description="Export FiftyOne dataset (optional)")


class ErrorHandlingConfig(BaseModel):
    """Error handling configuration."""

    on_image_error: Literal["skip", "stop"] = Field(
        default="skip",
        description="Action on image processing error: 'skip' to continue, 'stop' to halt",
    )


class ModelConfig(BaseModel):
    """Configuration for a single model."""

    name: str = Field(..., description="Model name identifier")
    path: str = Field(..., description="Path to model file (.pt or .onnx)")
    classes: list[str] = Field(..., description="Classes this model detects")
    mode: Literal["macro", "sliding_window"] = Field(
        default="macro",
        description="Processing mode: 'macro' for full image, 'sliding_window' for crops",
    )
    separate: bool = Field(
        default=False,
        description="Use connected components to separate instances",
    )
    # Per-model overrides (optional)
    processor_size: int | None = Field(default=None, description="Override default processor size")
    crop_top: int | None = Field(default=None, description="Override default crop_top")
    crop_size: int | None = Field(default=None, description="Override default crop_size")
    window_slide_h: int | None = Field(default=None, description="Override default window_slide_h")
    window_slide_v: int | None = Field(default=None, description="Override default window_slide_v")

    @field_validator("path")
    @classmethod
    def validate_path_extension(cls, v: str) -> str:
        """Validate model path has correct extension."""
        if not v.endswith((".pt", ".onnx")):
            raise ValueError(f"Model path must end with .pt or .onnx, got: {v}")
        return v


class DefaultsConfig(BaseModel):
    """Default processing parameters."""

    processor_size: int = Field(default=800, description="Model input size")
    crop_top: int = Field(default=1360, description="Pixels to crop from top (dashcam removal)")
    crop_size: int = Field(default=400, description="Sliding window crop size")
    window_slide_h: int = Field(
        default=-1,
        description="Horizontal slide step (-1 = same as crop_size)",
    )
    window_slide_v: int = Field(
        default=-1,
        description="Vertical slide step (-1 = same as crop_size)",
    )
    batch_size: int = Field(default=5, description="Batch size for inference")
    min_image_width: int = Field(default=1024, description="Minimum image width")
    min_image_height: int = Field(default=2024, description="Minimum image height")


class InferenceConfig(BaseModel):
    """Main configuration for inference pipeline."""

    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    image_patterns: list[str] = Field(
        default=["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"],
        description="Glob patterns for image files",
    )
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    error_handling: ErrorHandlingConfig = Field(default_factory=ErrorHandlingConfig)
    models: list[ModelConfig] = Field(default=[], description="List of models to run")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InferenceConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def get_model_config(self, model: ModelConfig, param: str) -> int:
        """Get effective parameter value for a model (model override or default)."""
        model_value = getattr(model, param, None)
        if model_value is not None:
            return model_value
        return getattr(self.defaults, param)

    def get_effective_window_slide_h(self, model: ModelConfig) -> int:
        """Get effective horizontal window slide for a model."""
        value = self.get_model_config(model, "window_slide_h")
        if value == -1:
            return self.get_model_config(model, "crop_size")
        return value

    def get_effective_window_slide_v(self, model: ModelConfig) -> int:
        """Get effective vertical window slide for a model."""
        value = self.get_model_config(model, "window_slide_v")
        if value == -1:
            return self.get_model_config(model, "crop_size")
        return value

    def validate_models(self) -> list[str]:
        """Validate all model paths exist. Returns list of errors."""
        errors = []
        for model in self.models:
            if not Path(model.path).exists():
                errors.append(f"Model '{model.name}' path not found: {model.path}")
        return errors

    def get_enabled_models(self) -> list[ModelConfig]:
        """Get list of enabled models. Currently returns all models."""
        return self.models
