"""
Main predictor/orchestrator for running inference.

Coordinates folder scanning, model loading, inference, and output generation.
"""

import os
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from loguru import logger
from PIL import Image
from tqdm.auto import tqdm

from .config import InferenceConfig, ModelConfig
from .exporters.coco import COCOExporter, segmentation_map_to_binary_masks
from .exporters.cvat import CVATExporter
from .adapters.registry import ModelRegistry
from .processors.folder_scanner import FolderScanner, ImageInfo
from .processors.image_loader import ImageLoader, ImageValidationError
from .processors.sliding_window import (
    apply_crop_top,
    assemble_crops_with_voting,
    batch_array,
    crop_image_windows,
    pad_mask_top,
)

# Suppress PyTorch meshgrid warning
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")


def get_memory_usage() -> dict | None:
    """Get current memory usage. Returns None if psutil not available."""
    try:
        import psutil
    except ImportError:
        return None

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    result = {
        "ram_used_gb": mem_info.rss / (1024**3),
        "ram_percent": process.memory_percent(),
    }

    # GPU memory if available
    if torch.cuda.is_available():
        result["gpu_used_gb"] = torch.cuda.memory_allocated() / (1024**3)
        result["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    return result


def format_memory_status() -> str:
    """Format memory status for display."""
    try:
        mem = get_memory_usage()
        if mem is None:
            return ""
        status = f"RAM: {mem['ram_used_gb']:.1f}GB ({mem['ram_percent']:.0f}%)"
        if "gpu_used_gb" in mem:
            status += f" | GPU: {mem['gpu_used_gb']:.1f}/{mem['gpu_total_gb']:.1f}GB"
        return status
    except Exception:
        return ""


class ProgressTracker:
    """Track and display detailed progress information."""

    def __init__(self, total_images: int, num_models: int):
        self.total_images = total_images
        self.num_models = num_models
        self.current_image = 0
        self.current_stage = "Initializing"
        self.start_time = time.time()
        self.annotations_found = 0

        # Create progress bar
        self.pbar = tqdm(
            total=total_images,
            desc="Processing",
            unit="img",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            dynamic_ncols=True,
        )

    def update(self, image_name: str, stage: str, annotations: int = 0):
        """Update progress with current status."""
        self.current_stage = stage
        self.annotations_found += annotations

        # Build status string
        elapsed = time.time() - self.start_time
        if self.current_image > 0:
            avg_time = elapsed / self.current_image
            status = f"{image_name[:25]} | {stage} | {avg_time:.1f}s/img"
        else:
            status = f"{image_name[:25]} | {stage}"

        # Add memory info periodically (every 5 images)
        if self.current_image % 5 == 0:
            mem_status = format_memory_status()
            if mem_status:
                status += f" | {mem_status}"

        self.pbar.set_postfix_str(status, refresh=True)

    def next_image(self):
        """Move to next image."""
        self.current_image += 1
        self.pbar.update(1)

    def close(self):
        """Close progress bar and print summary."""
        self.pbar.close()
        elapsed = time.time() - self.start_time

        # Print summary
        print(f"\nProcessed {self.total_images} images in {elapsed:.1f}s ({elapsed/max(1, self.total_images):.1f}s/img)")
        print(f"Found {self.annotations_found} annotations")


class Predictor:
    """
    Main inference orchestrator.

    Handles the complete inference pipeline:
    1. Scan folders for images
    2. Load and validate images
    3. Run inference through all models
    4. Export results in configured formats
    """

    def __init__(self, config: InferenceConfig):
        """
        Initialize predictor.

        Args:
            config: Inference configuration.
        """
        self.config = config
        self.model_registry: ModelRegistry | None = None
        self.image_loader = ImageLoader(
            min_width=config.defaults.min_image_width,
            min_height=config.defaults.min_image_height,
        )

    def setup(self) -> None:
        """Load all models and prepare for inference."""
        import sys

        # Validate model paths
        errors = self.config.validate_models()
        if errors:
            for error in errors:
                logger.error(error)
            raise FileNotFoundError(f"Model validation failed: {len(errors)} errors")

        # Load models with progress indication
        enabled_models = self.config.get_enabled_models()
        print(f"Loading {len(enabled_models)} model(s)...", end=" ", flush=True)

        self.model_registry = ModelRegistry(self.config)
        self.model_registry.load_all_models()

        print("done", flush=True)
        logger.debug(f"Predictor ready with {len(self.model_registry)} models")

    def run(
        self,
        project_path: Path | str,
        output_path: Path | str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict:
        """
        Run inference on a project folder.

        Args:
            project_path: Path to project folder.
            output_path: Path to output folder.
            progress_callback: Optional callback for progress updates (current, total).

        Returns:
            Summary statistics.
        """
        project_path = Path(project_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.model_registry is None:
            self.setup()

        # Scan for images
        scanner = FolderScanner(project_path, self.config.image_patterns)
        images = scanner.scan()

        if not images:
            print("No images found.")
            return {"images_processed": 0, "images_skipped": 0, "annotations": 0}

        print(f"Found {len(images)} images")

        # Validate images
        valid_images, invalid_images = self._validate_images(images)

        if invalid_images:
            print(f"Skipping {len(invalid_images)} invalid images")

        # Initialize exporter
        coco_exporter = COCOExporter(output_path)

        # Process images
        stats = self._process_images(
            valid_images,
            coco_exporter,
            project_path,
            progress_callback,
        )

        # Save outputs
        self._save_outputs(coco_exporter, output_path)

        # Add validation stats
        stats["images_skipped"] = len(invalid_images)

        logger.debug(
            f"Inference complete: {stats['images_processed']} processed, "
            f"{stats['images_skipped']} skipped, {stats['annotations']} annotations"
        )

        return stats

    def _validate_images(
        self, images: list[ImageInfo]
    ) -> tuple[list[ImageInfo], list[tuple[ImageInfo, str]]]:
        """Validate all images and separate valid from invalid."""
        valid: list[ImageInfo] = []
        invalid: list[tuple[ImageInfo, str]] = []
        stop_on_error = self.config.error_handling.on_image_error == "stop"

        for img_info in images:
            is_valid, error = self.image_loader.validate(img_info.path)
            if is_valid:
                valid.append(img_info)
            else:
                invalid.append((img_info, error or "Unknown error"))
                if stop_on_error:
                    raise ImageValidationError(f"Image validation failed: {img_info.path} - {error}")

        return valid, invalid

    def _process_images(
        self,
        images: list[ImageInfo],
        coco_exporter: COCOExporter,
        project_path: Path,
        progress_callback: Callable[[int, int], None] | None,
    ) -> dict:
        """Process all images through all models."""
        total = len(images)
        annotations_count = 0

        # Get models by mode
        macro_models = self.model_registry.get_models_by_mode("macro")
        sliding_window_models = self.model_registry.get_models_by_mode("sliding_window")
        num_models = len(macro_models) + len(sliding_window_models)

        logger.debug(f"Processing with {len(macro_models)} macro + {len(sliding_window_models)} sliding window models")

        # Initialize progress tracker
        progress = ProgressTracker(total, num_models)

        for idx, img_info in enumerate(images):
            image_name = img_info.path.name
            img_annotations = 0

            try:
                # Stage 1: Load image
                progress.update(image_name, "Loading")
                pil_image = self.image_loader.load(img_info.path)
                width, height = pil_image.size

                coco_exporter.add_image(
                    image_id=idx,
                    file_name=str(img_info.relative_path),
                    width=width,
                    height=height,
                )

                # Stage 2: Process with macro models (full image)
                if macro_models:
                    progress.update(image_name, "Macro inference")
                    img_annotations += self._process_macro_models(
                        idx, pil_image, macro_models, coco_exporter
                    )

                # Stage 3: Process with sliding window models
                if sliding_window_models:
                    progress.update(image_name, "Sliding window")
                    img_annotations += self._process_sliding_window_models(
                        idx, pil_image, sliding_window_models, coco_exporter
                    )

                annotations_count += img_annotations
                progress.update(image_name, "Done", img_annotations)
                progress.next_image()

                if progress_callback:
                    progress_callback(idx + 1, total)

            except Exception as e:
                logger.error(f"Error processing {img_info.path}: {e}")
                progress.next_image()
                if self.config.error_handling.on_image_error == "stop":
                    progress.close()
                    raise

        progress.annotations_found = annotations_count
        progress.close()

        return {
            "images_processed": total,
            "annotations": annotations_count,
        }

    def _process_macro_models(
        self,
        image_id: int,
        pil_image: Image.Image,
        models: dict,
        coco_exporter: COCOExporter,
    ) -> int:
        """Process image with macro models (full image inference)."""
        annotations_count = 0

        # Convert image to batch format (1, C, H, W)
        image_array = np.array(pil_image).transpose(2, 0, 1)
        image_batch = np.expand_dims(image_array, axis=0)

        for model_name, model in models.items():
            model_config = self.model_registry.get_model_config(model_name)

            # Get num_classes from model
            num_classes = model.num_classes
            if num_classes == 0:
                raise ValueError(f"Model '{model_name}' has no num_classes defined. Check model checkpoint.")

            # Run inference
            predictions = model.run_inference(image_batch)

            # Convert to binary masks
            pred_array = np.array([p.cpu().numpy() for p in predictions])
            binary_masks = segmentation_map_to_binary_masks(pred_array, num_classes)
            binary_masks = binary_masks.squeeze().numpy()

            # Add annotations for each class
            annotations_count += self._add_model_annotations(
                image_id,
                binary_masks,
                model,
                model_config,
                coco_exporter,
            )

        return annotations_count

    def _process_sliding_window_models(
        self,
        image_id: int,
        pil_image: Image.Image,
        models: dict,
        coco_exporter: COCOExporter,
    ) -> int:
        """Process image with sliding window models."""
        annotations_count = 0
        original_height = pil_image.size[1]

        for model_name, model in models.items():
            model_config = self.model_registry.get_model_config(model_name)

            # Get effective parameters
            crop_top = self.config.get_model_config(model_config, "crop_top")
            crop_size = self.config.get_model_config(model_config, "crop_size")
            window_slide_h = self.config.get_effective_window_slide_h(model_config)
            window_slide_v = self.config.get_effective_window_slide_v(model_config)
            batch_size = self.config.defaults.batch_size

            # Apply crop top
            cropped_image = apply_crop_top(pil_image, crop_top)

            # Get crops
            crops, boxes = crop_image_windows(
                cropped_image, window_slide_h, window_slide_v, crop_size
            )

            # Get num_classes from model
            num_classes = model.num_classes
            if num_classes == 0:
                raise ValueError(f"Model '{model_name}' has no num_classes defined. Check model checkpoint.")

            # Run inference on batches
            pred_semantic_maps = []
            for crop_batch in batch_array(crops, batch_size=batch_size):
                preds = model.run_inference(crop_batch)
                pred_semantic_maps.extend([p.cpu().numpy() for p in preds])

            # Assemble with voting
            cropped_width, cropped_height = cropped_image.size
            pred_mask = assemble_crops_with_voting(
                pred_semantic_maps,
                boxes,
                cropped_height,
                cropped_width,
                num_classes,
            )

            # Convert to binary masks and pad back
            pred_mask_expanded = np.expand_dims(pred_mask, axis=0)
            binary_masks = segmentation_map_to_binary_masks(pred_mask_expanded, num_classes)
            binary_masks = binary_masks.squeeze().numpy()

            # Pad back to original size
            binary_masks = np.array([
                pad_mask_top(mask, crop_top, fill_value=0) for mask in binary_masks
            ])

            # Add annotations
            annotations_count += self._add_model_annotations(
                image_id,
                binary_masks,
                model,
                model_config,
                coco_exporter,
            )

        return annotations_count

    def _add_model_annotations(
        self,
        image_id: int,
        binary_masks: np.ndarray,
        model,
        model_config: ModelConfig,
        coco_exporter: COCOExporter,
    ) -> int:
        """Add annotations from model predictions to COCO exporter."""
        annotations_count = 0
        id2label = model.id2label or {}

        for class_idx, mask in enumerate(binary_masks):
            # Skip background (class 0) and empty masks
            if class_idx == 0 or mask.sum() == 0:
                continue

            class_name = id2label.get(class_idx, f"class_{class_idx}")

            # Only add annotations for classes this model is configured for
            if class_name not in model_config.classes:
                continue

            coco_exporter.add_annotation(
                image_id=image_id,
                category_name=class_name,
                mask=mask,
                separate_instances=model_config.separate,
            )
            annotations_count += 1

        return annotations_count

    def _save_outputs(self, coco_exporter: COCOExporter, output_path: Path) -> None:
        """Save all output formats."""
        coco_json_path = None

        # Save COCO JSON
        if self.config.output.coco_json:
            coco_json_path = coco_exporter.save("predictions.json")
            coco_exporter.save_compressed("predictions_coco.tar.gz")

        # Save CVAT XML
        if self.config.output.cvat_xml:
            cvat_exporter = CVATExporter(output_path)
            # CVAT needs COCO JSON - use saved file or convert from dict directly
            if coco_json_path and coco_json_path.exists():
                cvat_exporter.convert_from_coco(coco_json_path, "predictions.xml")
            else:
                # COCO JSON wasn't saved, convert directly from dict
                cvat_exporter.convert_from_coco_dict(
                    coco_exporter.to_dict(), "predictions.xml"
                )

        # Save FiftyOne (optional)
        if self.config.output.fiftyone:
            self._save_fiftyone(coco_exporter, output_path)

    def _save_fiftyone(self, coco_exporter: COCOExporter, output_path: Path) -> None:
        """Save FiftyOne dataset (optional)."""
        try:
            import fiftyone as fo

            logger.info("Exporting FiftyOne dataset...")

            # This requires the images to be accessible
            # For now, just log that it would be done
            logger.warning("FiftyOne export not fully implemented yet")

        except ImportError:
            logger.warning("FiftyOne not installed, skipping FiftyOne export")

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.model_registry:
            self.model_registry.unload_all()

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
