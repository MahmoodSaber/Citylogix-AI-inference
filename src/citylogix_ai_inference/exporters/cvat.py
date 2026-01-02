"""
CVAT XML format exporter for segmentation predictions.

Converts COCO format to CVAT XML format.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from loguru import logger
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from tqdm.auto import tqdm


def encode_uncompressed_rle(mask: np.ndarray) -> list[int]:
    """
    Encodes a binary mask to uncompressed RLE using row-major (C-style) order,
    suitable for CVAT XML representation.

    Args:
        mask: Binary mask array.

    Returns:
        List of RLE counts.
    """
    BACKGROUND_RUN = 0
    FOREGROUND_PIXEL = 1

    flat = mask.flatten(order="C")  # Row-major
    rle = []
    count = 1
    prev = flat[0]

    if prev == FOREGROUND_PIXEL:
        # Add 0-length background run to align with CVAT's expectation
        rle.append(BACKGROUND_RUN)

    for pixel in flat[1:]:
        if pixel == prev:
            count += 1
        else:
            rle.append(count)
            count = 1
            prev = pixel
    rle.append(count)

    return rle


class CVATExporter:
    """
    Exports predictions in CVAT XML format.
    """

    def __init__(self, output_dir: Path | str):
        """
        Initialize CVAT exporter.

        Args:
            output_dir: Output directory for CVAT files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_from_coco(
        self,
        coco_json_path: Path | str,
        output_filename: str = "predictions.xml",
        show_progress: bool = True,
    ) -> Path:
        """
        Convert COCO JSON to CVAT XML format.

        Args:
            coco_json_path: Path to COCO JSON file.
            output_filename: Output XML filename.
            show_progress: Show progress bar.

        Returns:
            Path to output XML file.
        """
        coco = COCO(str(coco_json_path))
        root = ET.Element("annotations")

        img_ids = coco.getImgIds()

        for img_id in tqdm(img_ids, desc="Converting to CVAT", disable=not show_progress):
            img = coco.loadImgs(img_id)[0]

            image_tag = ET.SubElement(
                root,
                "image",
                {
                    "id": str(img["id"]),
                    "name": img["file_name"],
                    "width": str(img["width"]),
                    "height": str(img["height"]),
                },
            )

            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            for ann in anns:
                segm = ann["segmentation"]

                if isinstance(segm, dict) and "counts" in segm:
                    # RLE encoded mask - decompress
                    binary_mask = mask_utils.decode(segm)

                    # Get bounding box of the non-zero region
                    ys, xs = binary_mask.nonzero()
                    if len(xs) == 0 or len(ys) == 0:
                        continue  # Skip empty masks

                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()
                    width = x_max - x_min + 1
                    height = y_max - y_min + 1

                    # Crop the mask to the bounding box
                    cropped_mask = binary_mask[y_min : y_min + height, x_min : x_min + width]

                    rle_list = encode_uncompressed_rle(cropped_mask)
                    rle_str = ", ".join(str(x) for x in rle_list)

                else:
                    raise RuntimeError(
                        f"Expected segmentation dict with RLE. "
                        f"Image ID: {img_id}, File Name: {img['file_name']}, "
                        f"Annotation ID: {ann.get('id', 'N/A')}."
                    )

                # Get category name
                category = coco.loadCats(ann["category_id"])[0]

                ET.SubElement(
                    image_tag,
                    "mask",
                    {
                        "label": category["name"],
                        "source": "auto",
                        "occluded": "0",
                        "left": str(x_min),
                        "top": str(y_min),
                        "width": str(width),
                        "height": str(height),
                        "z_order": "0",
                        "rle": rle_str,
                    },
                )

        output_path = self.output_dir / output_filename
        tree = ET.ElementTree(root)
        tree.write(str(output_path), encoding="utf-8", xml_declaration=True)

        logger.info(f"Saved CVAT XML to {output_path}")
        return output_path

    def convert_from_coco_dict(
        self,
        coco_dict: dict,
        output_filename: str = "predictions.xml",
        show_progress: bool = True,
    ) -> Path:
        """
        Convert COCO dictionary to CVAT XML format.

        Args:
            coco_dict: COCO format dictionary.
            output_filename: Output XML filename.
            show_progress: Show progress bar.

        Returns:
            Path to output XML file.
        """
        import tempfile
        import json

        # Write COCO dict to temp file and use existing converter
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(coco_dict, f)
            temp_path = f.name

        try:
            return self.convert_from_coco(temp_path, output_filename, show_progress)
        finally:
            Path(temp_path).unlink()


def coco_to_cvat(input_json: Path | str, output_xml: Path | str) -> None:
    """
    Convert COCO JSON file to CVAT XML file.

    Args:
        input_json: Path to input COCO JSON file.
        output_xml: Path to output CVAT XML file.
    """
    output_xml = Path(output_xml)
    exporter = CVATExporter(output_xml.parent)
    exporter.convert_from_coco(input_json, output_xml.name)
