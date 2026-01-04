"""
COCO format exporter for segmentation predictions.

Outputs predictions in COCO JSON format with RLE-encoded masks.
"""

import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from pycocotools import mask as mask_utils


class COCOExporter:
    """
    Exports predictions in COCO JSON format.
    """

    def __init__(self, output_dir: Path | str):
        """
        Initialize COCO exporter.

        Args:
            output_dir: Output directory for COCO files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._images: list[dict] = []
        self._annotations: list[dict] = []
        self._categories: dict[str, int] = {}  # name -> id
        self._annotation_id = 0

    def add_category(self, name: str) -> int:
        """
        Add a category if not exists.

        Args:
            name: Category name.

        Returns:
            Category ID.
        """
        if name not in self._categories:
            self._categories[name] = len(self._categories) + 1
        return self._categories[name]

    def add_image(
        self,
        image_id: int,
        file_name: str,
        width: int,
        height: int,
    ) -> None:
        """
        Add image metadata.

        Args:
            image_id: Unique image ID.
            file_name: Image file name (relative path).
            width: Image width.
            height: Image height.
        """
        self._images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        })

    def add_annotation(
        self,
        image_id: int,
        category_name: str,
        mask: np.ndarray,
        separate_instances: bool = False,
    ) -> None:
        """
        Add segmentation annotation.

        Args:
            image_id: Image ID this annotation belongs to.
            category_name: Category name.
            mask: Binary mask (H, W) with values 0 or 1.
            separate_instances: If True, separate connected components into individual annotations.
        """
        if mask.sum() == 0:
            logger.debug(f"[COCOExporter.add_annotation] Skipping empty mask for category '{category_name}'")
            return

        logger.debug(f"[COCOExporter.add_annotation] Adding annotation for image {image_id}, category '{category_name}', mask sum: {mask.sum()}")
        category_id = self.add_category(category_name)

        if separate_instances:
            self._add_separated_annotations(image_id, category_id, mask)
        else:
            self._add_single_annotation(image_id, category_id, mask)

    def _add_single_annotation(
        self,
        image_id: int,
        category_id: int,
        mask: np.ndarray,
    ) -> None:
        """Add a single annotation for the entire mask."""
        logger.debug(f"[COCOExporter._add_single_annotation] Encoding mask with RLE...")
        encoded_mask = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        encoded_mask["counts"] = encoded_mask["counts"].decode("ascii")

        bbox = mask_utils.toBbox(encoded_mask).tolist()
        logger.debug(f"[COCOExporter._add_single_annotation] Annotation ID: {self._annotation_id}, bbox: {bbox}")

        self._annotations.append({
            "id": self._annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": encoded_mask,
            "bbox": bbox,
            "area": float(mask.sum()),
            "iscrowd": 0,
            "score": 1.0,
        })
        self._annotation_id += 1

    def _add_separated_annotations(
        self,
        image_id: int,
        category_id: int,
        mask: np.ndarray,
    ) -> None:
        """Separate connected components and add individual annotations."""
        logger.debug(f"[COCOExporter._add_separated_annotations] Finding connected components...")
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
        )
        logger.debug(f"[COCOExporter._add_separated_annotations] Found {num_labels - 1} components (excluding background)")

        # Skip background (label 0)
        for label_num in range(1, num_labels):
            component_mask = np.zeros_like(mask, dtype=np.uint8)
            component_mask[labels_im == label_num] = 1

            if component_mask.sum() == 0:
                continue

            self._add_single_annotation(image_id, category_id, component_mask)

    def to_dict(self) -> dict:
        """
        Get COCO format dictionary.

        Returns:
            COCO format dictionary.
        """
        categories = [
            {"id": cat_id, "name": cat_name}
            for cat_name, cat_id in sorted(self._categories.items(), key=lambda x: x[1])
        ]

        return {
            "images": self._images,
            "annotations": self._annotations,
            "categories": categories,
        }

    def save(self, filename: str = "predictions.json") -> Path:
        """
        Save COCO JSON to file.

        Args:
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        output_path = self.output_dir / filename
        logger.debug(f"[COCOExporter.save] START - Saving to {output_path}")
        logger.debug(f"[COCOExporter.save] Stats: {len(self._images)} images, {len(self._annotations)} annotations, {len(self._categories)} categories")

        logger.debug(f"[COCOExporter.save] Converting to dict...")
        coco_dict = self.to_dict()

        logger.debug(f"[COCOExporter.save] Writing JSON file...")
        with open(output_path, "w") as f:
            json.dump(coco_dict, f)

        logger.info(f"Saved COCO JSON to {output_path}")
        logger.debug(f"[COCOExporter.save] DONE")
        return output_path

    def save_compressed(self, filename: str = "predictions_coco.tar.gz") -> Path:
        """
        Save COCO JSON as compressed tarball.

        Args:
            filename: Output filename.

        Returns:
            Path to saved file.
        """
        output_path = self.output_dir / filename
        logger.debug(f"[COCOExporter.save_compressed] START - Saving to {output_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "predictions.json"
            logger.debug(f"[COCOExporter.save_compressed] Writing temp JSON...")
            with open(json_path, "w") as f:
                json.dump(self.to_dict(), f)

            logger.debug(f"[COCOExporter.save_compressed] Creating tarball...")
            with tarfile.open(output_path, "w:gz") as tar:
                tar.add(json_path, arcname="predictions.json")

        logger.info(f"Saved compressed COCO to {output_path}")
        logger.debug(f"[COCOExporter.save_compressed] DONE")
        return output_path

    def reset(self) -> None:
        """Reset exporter state for new export."""
        self._images = []
        self._annotations = []
        self._categories = {}
        self._annotation_id = 0

    @property
    def num_images(self) -> int:
        return len(self._images)

    @property
    def num_annotations(self) -> int:
        return len(self._annotations)

    @property
    def categories(self) -> list[str]:
        return list(self._categories.keys())


def segmentation_map_to_binary_masks(
    segmentation_map: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Convert segmentation map to binary masks per class.

    Args:
        segmentation_map: Segmentation map with shape (B, H, W) or (H, W).
        num_classes: Number of classes.

    Returns:
        Binary masks with shape (B, C, H, W) or (C, H, W).
    """
    import torch

    if isinstance(segmentation_map, np.ndarray):
        segmentation_map = torch.from_numpy(segmentation_map)

    if segmentation_map.dim() == 2:
        # (H, W) -> (1, H, W)
        segmentation_map = segmentation_map.unsqueeze(0)

    batch_size, height, width = segmentation_map.shape
    binary_masks = torch.zeros((batch_size, num_classes, height, width), dtype=torch.uint8)

    for c in range(num_classes):
        binary_masks[:, c, :, :] = (segmentation_map == c).to(torch.uint8)

    return binary_masks
