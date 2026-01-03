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
from collections import defaultdict
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

        logger.debug(f"[Predictor.setup] START - Initializing predictor")

        # Validate model paths
        logger.debug(f"[Predictor.setup] Validating model paths...")
        errors = self.config.validate_models()
        if errors:
            for error in errors:
                logger.error(error)
            raise FileNotFoundError(f"Model validation failed: {len(errors)} errors")
        logger.debug(f"[Predictor.setup] Model paths validated")

        # Load models with progress indication
        enabled_models = self.config.get_enabled_models()
        print(f"Loading {len(enabled_models)} model(s)...", end=" ", flush=True)
        logger.debug(f"[Predictor.setup] Loading {len(enabled_models)} models...")

        self.model_registry = ModelRegistry(self.config)
        self.model_registry.load_all_models()

        print("done", flush=True)
        logger.debug(f"[Predictor.setup] DONE - Predictor ready with {len(self.model_registry)} models")

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
        logger.debug(f"[Predictor.run] START - Project: {project_path}, Output: {output_path}")

        output_path.mkdir(parents=True, exist_ok=True)

        if self.model_registry is None:
            logger.debug(f"[Predictor.run] Model registry not initialized, calling setup()...")
            self.setup()

        # Scan for images
        logger.debug(f"[Predictor.run] Scanning for images...")
        scanner = FolderScanner(project_path, self.config.image_patterns)
        images = scanner.scan()
        logger.debug(f"[Predictor.run] Scan complete, found {len(images)} images")

        if not images:
            print("No images found.")
            return {"images_processed": 0, "images_skipped": 0, "annotations": 0, "tasks_processed": 0}

        # Group images by session/task
        images_by_task = self._group_images_by_task(images)
        num_tasks = sum(len(tasks) for tasks in images_by_task.values())
        print(f"Found {len(images)} images across {num_tasks} tasks in {len(images_by_task)} sessions")

        # Validate images
        logger.debug(f"[Predictor.run] Validating images...")
        valid_images, invalid_images = self._validate_images(images)
        logger.debug(f"[Predictor.run] Validation complete: {len(valid_images)} valid, {len(invalid_images)} invalid")

        if invalid_images:
            print(f"Skipping {len(invalid_images)} invalid images")

        # Re-group valid images by task
        valid_images_by_task = self._group_images_by_task(valid_images)

        # Process images task by task
        logger.debug(f"[Predictor.run] Starting task-by-task processing...")
        stats = self._process_all_tasks(
            valid_images_by_task,
            output_path,
            progress_callback,
        )
        logger.debug(f"[Predictor.run] All tasks processed")

        # Add validation stats
        stats["images_skipped"] = len(invalid_images)

        logger.debug(
            f"[Predictor.run] DONE - {stats['images_processed']} processed, "
            f"{stats['images_skipped']} skipped, {stats['annotations']} annotations, "
            f"{stats['tasks_processed']} tasks"
        )

        return stats

    def _group_images_by_task(
        self, images: list[ImageInfo]
    ) -> dict[str, dict[str, list[ImageInfo]]]:
        """
        Group images by session and task.

        Returns:
            Dict mapping session -> task -> list of images
        """
        grouped: dict[str, dict[str, list[ImageInfo]]] = defaultdict(lambda: defaultdict(list))
        for img in images:
            grouped[img.session][img.task].append(img)
        return grouped

    def _process_all_tasks(
        self,
        images_by_task: dict[str, dict[str, list[ImageInfo]]],
        output_path: Path,
        progress_callback: Callable[[int, int], None] | None,
    ) -> dict:
        """
        Process all tasks, creating separate outputs for each.

        Args:
            images_by_task: Images grouped by session/task.
            output_path: Root output path.
            progress_callback: Optional progress callback.

        Returns:
            Aggregated statistics.
        """
        total_images = sum(
            len(imgs)
            for tasks in images_by_task.values()
            for imgs in tasks.values()
        )
        total_tasks = sum(len(tasks) for tasks in images_by_task.values())

        stats = {
            "images_processed": 0,
            "annotations": 0,
            "tasks_processed": 0,
        }

        # Get models by mode (once, reused for all tasks)
        macro_models = self.model_registry.get_models_by_mode("macro")
        sliding_window_models = self.model_registry.get_models_by_mode("sliding_window")
        num_models = len(macro_models) + len(sliding_window_models)

        logger.debug(f"[Predictor._process_all_tasks] Processing {total_tasks} tasks with {total_images} total images")
        logger.debug(f"[Predictor._process_all_tasks] Models: {len(macro_models)} macro + {len(sliding_window_models)} sliding window")

        # Initialize overall progress tracker
        progress = ProgressTracker(total_images, num_models)
        processed_count = 0

        for session_name, tasks in sorted(images_by_task.items()):
            for task_name, task_images in sorted(tasks.items()):
                logger.debug(f"[Predictor._process_all_tasks] === Processing {session_name}/{task_name} ({len(task_images)} images) ===")

                # Create task output directory mirroring input structure
                task_output_dir = output_path / session_name / task_name
                task_output_dir.mkdir(parents=True, exist_ok=True)

                # Create fresh exporters for this task
                coco_exporter = COCOExporter(task_output_dir)

                # Process images for this task
                task_stats = self._process_task_images(
                    task_images,
                    coco_exporter,
                    macro_models,
                    sliding_window_models,
                    progress,
                )

                # Save outputs for this task
                self._save_outputs(coco_exporter, task_output_dir)

                # Update stats
                stats["images_processed"] += task_stats["images_processed"]
                stats["annotations"] += task_stats["annotations"]
                stats["tasks_processed"] += 1

                processed_count += len(task_images)
                if progress_callback:
                    progress_callback(processed_count, total_images)

                logger.debug(f"[Predictor._process_all_tasks] Task {session_name}/{task_name} complete: {task_stats['annotations']} annotations")

        progress.annotations_found = stats["annotations"]
        progress.close()

        return stats

    def _process_task_images(
        self,
        images: list[ImageInfo],
        coco_exporter: COCOExporter,
        macro_models: dict,
        sliding_window_models: dict,
        progress: "ProgressTracker",
    ) -> dict:
        """
        Process images for a single task.

        Args:
            images: List of images in this task.
            coco_exporter: COCO exporter for this task.
            macro_models: Macro mode models.
            sliding_window_models: Sliding window models.
            progress: Progress tracker.

        Returns:
            Task statistics.
        """
        annotations_count = 0

        for idx, img_info in enumerate(images):
            image_name = img_info.path.name
            img_annotations = 0
            logger.debug(f"[Predictor._process_task_images] Image {idx + 1}/{len(images)}: {image_name}")

            try:
                # Load image
                progress.update(image_name, "Loading")
                pil_image = self.image_loader.load(img_info.path)
                width, height = pil_image.size

                # Use index within task as image_id (resets per task)
                coco_exporter.add_image(
                    image_id=idx,
                    file_name=img_info.filename,  # Just filename, not relative path
                    width=width,
                    height=height,
                )

                # Process with macro models
                if macro_models:
                    progress.update(image_name, "Macro inference")
                    img_annotations += self._process_macro_models(
                        idx, pil_image, macro_models, coco_exporter
                    )

                # Process with sliding window models
                if sliding_window_models:
                    progress.update(image_name, "Sliding window")
                    img_annotations += self._process_sliding_window_models(
                        idx, pil_image, sliding_window_models, coco_exporter
                    )

                annotations_count += img_annotations
                progress.update(image_name, "Done", img_annotations)
                progress.next_image()

            except Exception as e:
                logger.error(f"[Predictor._process_task_images] Error processing {img_info.path}: {e}")
                progress.next_image()
                if self.config.error_handling.on_image_error == "stop":
                    raise

        return {
            "images_processed": len(images),
            "annotations": annotations_count,
        }

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

    def _process_macro_models(
        self,
        image_id: int,
        pil_image: Image.Image,
        models: dict,
        coco_exporter: COCOExporter,
    ) -> int:
        """Process image with macro models (full image inference)."""
        annotations_count = 0
        logger.debug(f"[Predictor._process_macro_models] START - Processing with {len(models)} macro models")

        # Convert image to batch format (1, C, H, W)
        logger.debug(f"[Predictor._process_macro_models] Converting image to batch format...")
        image_array = np.array(pil_image).transpose(2, 0, 1)
        image_batch = np.expand_dims(image_array, axis=0)
        logger.debug(f"[Predictor._process_macro_models] Batch shape: {image_batch.shape}")

        for model_name, model in models.items():
            logger.debug(f"[Predictor._process_macro_models] Running model: {model_name}")
            model_config = self.model_registry.get_model_config(model_name)

            # Get num_classes from model
            num_classes = model.num_classes
            if num_classes == 0:
                raise ValueError(f"Model '{model_name}' has no num_classes defined. Check model checkpoint.")
            logger.debug(f"[Predictor._process_macro_models] num_classes: {num_classes}")

            # Run inference
            logger.debug(f"[Predictor._process_macro_models] Running inference...")
            predictions = model.run_inference(image_batch)
            logger.debug(f"[Predictor._process_macro_models] Inference complete, {len(predictions)} predictions")

            # Convert to binary masks
            logger.debug(f"[Predictor._process_macro_models] Converting to binary masks...")
            pred_array = np.array([p.cpu().numpy() for p in predictions])
            binary_masks = segmentation_map_to_binary_masks(pred_array, num_classes)
            binary_masks = binary_masks.squeeze().numpy()
            logger.debug(f"[Predictor._process_macro_models] Binary masks shape: {binary_masks.shape}")

            # Add annotations for each class
            logger.debug(f"[Predictor._process_macro_models] Adding annotations...")
            annotations_count += self._add_model_annotations(
                image_id,
                binary_masks,
                model,
                model_config,
                coco_exporter,
            )
            logger.debug(f"[Predictor._process_macro_models] Model {model_name} complete")

        logger.debug(f"[Predictor._process_macro_models] DONE - {annotations_count} annotations from macro models")
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
        logger.debug(f"[Predictor._process_sliding_window_models] START - Processing with {len(models)} sliding window models")

        for model_name, model in models.items():
            logger.debug(f"[Predictor._process_sliding_window_models] Running model: {model_name}")
            model_config = self.model_registry.get_model_config(model_name)

            # Get effective parameters
            crop_top = self.config.get_model_config(model_config, "crop_top")
            crop_size = self.config.get_model_config(model_config, "crop_size")
            window_slide_h = self.config.get_effective_window_slide_h(model_config)
            window_slide_v = self.config.get_effective_window_slide_v(model_config)
            batch_size = self.config.defaults.batch_size
            logger.debug(f"[Predictor._process_sliding_window_models] Parameters: crop_top={crop_top}, crop_size={crop_size}, slide_h={window_slide_h}, slide_v={window_slide_v}, batch_size={batch_size}")

            # Apply crop top
            logger.debug(f"[Predictor._process_sliding_window_models] Applying crop_top={crop_top}...")
            cropped_image = apply_crop_top(pil_image, crop_top)
            logger.debug(f"[Predictor._process_sliding_window_models] Cropped image size: {cropped_image.size}")

            # Get crops
            logger.debug(f"[Predictor._process_sliding_window_models] Generating sliding window crops...")
            crops, boxes = crop_image_windows(
                cropped_image, window_slide_h, window_slide_v, crop_size
            )
            logger.debug(f"[Predictor._process_sliding_window_models] Generated {len(crops)} crops")

            # Get num_classes from model
            num_classes = model.num_classes
            if num_classes == 0:
                raise ValueError(f"Model '{model_name}' has no num_classes defined. Check model checkpoint.")
            logger.debug(f"[Predictor._process_sliding_window_models] num_classes: {num_classes}")

            # Run inference on batches
            pred_semantic_maps = []
            batches = list(batch_array(crops, batch_size=batch_size))
            logger.debug(f"[Predictor._process_sliding_window_models] Running inference on {len(batches)} batches...")
            for batch_idx, crop_batch in enumerate(batches):
                logger.debug(f"[Predictor._process_sliding_window_models] Batch {batch_idx + 1}/{len(batches)} (size: {len(crop_batch)})...")
                preds = model.run_inference(crop_batch)
                pred_semantic_maps.extend([p.cpu().numpy() for p in preds])
                logger.debug(f"[Predictor._process_sliding_window_models] Batch {batch_idx + 1} complete")

            logger.debug(f"[Predictor._process_sliding_window_models] All batches complete, {len(pred_semantic_maps)} predictions")

            # Assemble with voting
            cropped_width, cropped_height = cropped_image.size
            logger.debug(f"[Predictor._process_sliding_window_models] Assembling crops with voting...")
            pred_mask = assemble_crops_with_voting(
                pred_semantic_maps,
                boxes,
                cropped_height,
                cropped_width,
                num_classes,
            )
            logger.debug(f"[Predictor._process_sliding_window_models] Voting complete, mask shape: {pred_mask.shape}")

            # Convert to binary masks and pad back
            logger.debug(f"[Predictor._process_sliding_window_models] Converting to binary masks...")
            pred_mask_expanded = np.expand_dims(pred_mask, axis=0)
            binary_masks = segmentation_map_to_binary_masks(pred_mask_expanded, num_classes)
            binary_masks = binary_masks.squeeze().numpy()
            logger.debug(f"[Predictor._process_sliding_window_models] Binary masks shape: {binary_masks.shape}")

            # Pad back to original size
            logger.debug(f"[Predictor._process_sliding_window_models] Padding masks to original size...")
            binary_masks = np.array([
                pad_mask_top(mask, crop_top, fill_value=0) for mask in binary_masks
            ])
            logger.debug(f"[Predictor._process_sliding_window_models] Padded masks shape: {binary_masks.shape}")

            # Add annotations
            logger.debug(f"[Predictor._process_sliding_window_models] Adding annotations...")
            annotations_count += self._add_model_annotations(
                image_id,
                binary_masks,
                model,
                model_config,
                coco_exporter,
            )
            logger.debug(f"[Predictor._process_sliding_window_models] Model {model_name} complete")

        logger.debug(f"[Predictor._process_sliding_window_models] DONE - {annotations_count} annotations from sliding window models")
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
        logger.debug(f"[Predictor._save_outputs] START - Saving outputs to {output_path}")
        coco_json_path = None

        # Save COCO JSON
        if self.config.output.coco_json:
            logger.debug(f"[Predictor._save_outputs] Saving COCO JSON...")
            coco_json_path = coco_exporter.save("predictions.json")
            logger.debug(f"[Predictor._save_outputs] Saving compressed COCO...")
            coco_exporter.save_compressed("predictions_coco.tar.gz")
            logger.debug(f"[Predictor._save_outputs] COCO outputs saved")

        # Save CVAT XML
        if self.config.output.cvat_xml:
            logger.debug(f"[Predictor._save_outputs] Saving CVAT XML...")
            cvat_exporter = CVATExporter(output_path)
            # CVAT needs COCO JSON - use saved file or convert from dict directly
            if coco_json_path and coco_json_path.exists():
                cvat_exporter.convert_from_coco(coco_json_path, "predictions.xml")
            else:
                # COCO JSON wasn't saved, convert directly from dict
                cvat_exporter.convert_from_coco_dict(
                    coco_exporter.to_dict(), "predictions.xml"
                )
            logger.debug(f"[Predictor._save_outputs] CVAT XML saved")

        # Save FiftyOne (optional)
        if self.config.output.fiftyone:
            logger.debug(f"[Predictor._save_outputs] Saving FiftyOne dataset...")
            self._save_fiftyone(coco_exporter, output_path)

        logger.debug(f"[Predictor._save_outputs] DONE - All outputs saved")

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
