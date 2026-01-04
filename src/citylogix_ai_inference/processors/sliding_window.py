"""
Sliding window utilities for processing large images.

Ported from streetscan_segmentation/utils/tools.py
"""

import numpy as np
from loguru import logger
from PIL import Image


def crop_image_windows(
    image: Image.Image,
    window_size_h: int,
    window_size_v: int,
    crop_size: int,
) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    """
    Crops an image into multiple windows with specified crop size and sliding steps,
    ensuring no pixels are missed by adjusting the last window positions.

    Args:
        image: PIL.Image object, the input image to be cropped.
        window_size_h: int, the number of pixels to shift the window horizontally.
        window_size_v: int, the number of pixels to shift the window vertically.
        crop_size: int, the width and height of each cropping window.

    Returns:
        crops: List of numpy arrays with shape (C, H, W) for each crop.
        boxes: List of bounding boxes (x_min, y_min, x_max, y_max) for each crop.
    """
    width, height = image.size
    logger.debug(f"[crop_image_windows] START - Image: {width}x{height}, crop_size: {crop_size}, slide_h: {window_size_h}, slide_v: {window_size_v}")

    crops: list[np.ndarray] = []
    boxes: list[tuple[int, int, int, int]] = []

    # Validate and set default step sizes if necessary
    if window_size_h <= 0:
        window_size_h = crop_size
    if window_size_v <= 0:
        window_size_v = crop_size

    # Calculate all possible x and y positions
    x_positions = list(range(0, width - crop_size + 1, window_size_h))
    y_positions = list(range(0, height - crop_size + 1, window_size_v))

    # Ensure the last window includes the right edge
    if x_positions:
        last_x = x_positions[-1] + crop_size
        if last_x < width:
            x_positions.append(width - crop_size)
    else:
        x_positions = [0]

    # Ensure the last window includes the bottom edge
    if y_positions:
        last_y = y_positions[-1] + crop_size
        if last_y < height:
            y_positions.append(height - crop_size)
    else:
        y_positions = [0]

    # Remove duplicate positions
    x_positions = sorted(list(set(x_positions)))
    y_positions = sorted(list(set(y_positions)))

    total_windows = len(x_positions) * len(y_positions)
    logger.debug(f"[crop_image_windows] Grid: {len(x_positions)} x {len(y_positions)} = {total_windows} windows")

    # Iterate over all positions to crop the image
    for y_min in y_positions:
        for x_min in x_positions:
            x_max = x_min + crop_size
            y_max = y_min + crop_size

            # Final check to ensure we don't exceed image boundaries
            if x_max > width:
                x_max = width
                x_min = max(width - crop_size, 0)
            if y_max > height:
                y_max = height
                y_min = max(height - crop_size, 0)

            box = (x_min, y_min, x_max, y_max)
            crop = image.crop(box)
            crop_array = np.array(crop).transpose(2, 0, 1)  # Convert to CHW format
            crops.append(crop_array)
            boxes.append(box)

    logger.debug(f"[crop_image_windows] DONE - Generated {len(crops)} crops")
    return crops, boxes


def assemble_crops_with_voting(
    predictions: list[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
    image_height: int,
    image_width: int,
    num_classes: int,
) -> np.ndarray:
    """
    Assembles cropped semantic predictions into a full semantic map using a voting mechanism.

    Args:
        predictions: List of NumPy arrays with shape (H, W) containing class labels.
        boxes: List of tuples, each tuple is (left, top, right, bottom).
        image_height: int, height of the original image.
        image_width: int, width of the original image.
        num_classes: int, total number of distinct classes.

    Returns:
        assembled_mask: NumPy array of shape (image_height, image_width) with final class labels.

    Raises:
        ValueError: If predictions and boxes don't match or have invalid coordinates.
    """
    logger.debug(f"[assemble_crops_with_voting] START - {len(predictions)} predictions, image: {image_width}x{image_height}, classes: {num_classes}")

    # Initialize a vote count array with shape (image_height, image_width, num_classes)
    logger.debug(f"[assemble_crops_with_voting] Allocating vote array: {image_height}x{image_width}x{num_classes}...")
    vote_counts = np.zeros((image_height, image_width, num_classes), dtype=np.int32)
    logger.debug(f"[assemble_crops_with_voting] Vote array allocated")

    if len(predictions) != len(boxes):
        raise ValueError(
            f"Number of predictions and boxes do not match. "
            f"Predictions: {len(predictions)}, Boxes: {len(boxes)}"
        )

    logger.debug(f"[assemble_crops_with_voting] Accumulating votes from {len(predictions)} crops...")
    for idx, (pred, box) in enumerate(zip(predictions, boxes)):
        left, top, right, bottom = box

        # Validate box coordinates
        if left < 0 or left >= right or right > image_width:
            raise ValueError(
                f"Box {idx} has invalid horizontal coordinates. "
                f"Conditions violated: 0 <= {left} < {right} <= {image_width}"
            )

        if top < 0 or top >= bottom or bottom > image_height:
            raise ValueError(
                f"Box {idx} has invalid vertical coordinates. "
                f"Conditions violated: 0 <= {top} < {bottom} <= {image_height}"
            )

        # Validate prediction shape matches the box size
        box_width = right - left
        box_height = bottom - top
        if pred.shape != (box_height, box_width):
            raise ValueError(
                f"Prediction {idx} shape {pred.shape} does not match "
                f"box size ({box_height}, {box_width})."
            )

        # One-hot encode the prediction
        # Shape of one_hot: (box_height, box_width, num_classes)
        one_hot = np.eye(num_classes)[pred]  # Converts class labels to one-hot vectors

        # Accumulate the votes
        vote_counts[top:bottom, left:right, :] += one_hot.astype(np.int32)

    logger.debug(f"[assemble_crops_with_voting] Computing argmax for final mask...")
    # Determine the class with the highest vote count for each pixel
    assembled_mask = np.argmax(vote_counts, axis=-1).astype(np.uint8)

    logger.debug(f"[assemble_crops_with_voting] DONE - Assembled mask shape: {assembled_mask.shape}")
    return assembled_mask


def assemble_binary_crops_with_voting(
    predictions: list[np.ndarray],
    boxes: list[tuple[int, int, int, int]],
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """
    Assembles cropped binary masks into a full-sized binary mask using a voting mechanism.

    Args:
        predictions: List of NumPy arrays with shape (H, W, C), binary masks per class.
        boxes: List of tuples, each tuple is (left, top, right, bottom).
        image_height: int, height of the original image.
        image_width: int, width of the original image.

    Returns:
        assembled_mask: NumPy array of shape (image_height, image_width, C) with final binary class votes.

    Raises:
        ValueError: If predictions and boxes don't match.
    """
    if len(predictions) != len(boxes):
        raise ValueError("Number of predictions and boxes do not match.")

    num_classes = predictions[0].shape[-1]
    vote_counts = np.zeros((image_height, image_width, num_classes), dtype=np.int32)

    for idx, (pred, box) in enumerate(zip(predictions, boxes)):
        left, top, right, bottom = box
        box_height = bottom - top
        box_width = right - left

        if pred.shape != (box_height, box_width, num_classes):
            raise ValueError(
                f"Prediction {idx} shape {pred.shape} does not match expected shape "
                f"({box_height}, {box_width}, {num_classes})."
            )

        vote_counts[top:bottom, left:right, :] += pred.astype(np.int32)

    # Majority vote threshold: a pixel gets class c if it was predicted more than 0 times
    assembled_mask = (vote_counts > 0).astype(np.uint8)

    return assembled_mask


def batch_array(array: list | np.ndarray, batch_size: int) -> list[np.ndarray]:
    """
    Batches an array with the batch size as the first dimension.

    Args:
        array: The array to batch.
        batch_size: The size of each batch.

    Returns:
        A list of batches as numpy arrays.
    """
    num_batches = (len(array) + batch_size - 1) // batch_size
    logger.debug(f"[batch_array] Creating {num_batches} batches of size {batch_size} from {len(array)} items")
    return [np.array(array[i : i + batch_size]) for i in range(0, len(array), batch_size)]


def apply_crop_top(image: Image.Image, crop_top: int) -> Image.Image:
    """
    Crop the top portion of an image (e.g., to remove dashcam area).

    Args:
        image: PIL Image to crop
        crop_top: Number of pixels to remove from top

    Returns:
        Cropped PIL Image
    """
    if crop_top <= 0:
        return image

    width, height = image.size
    if crop_top >= height:
        raise ValueError(f"crop_top ({crop_top}) >= image height ({height})")

    return image.crop((0, crop_top, width, height))


def pad_mask_top(mask: np.ndarray, crop_top: int, fill_value: int = 0) -> np.ndarray:
    """
    Pad the top of a mask to restore original image dimensions.

    Args:
        mask: 2D numpy array (H, W) or 3D (C, H, W)
        crop_top: Number of pixels that were cropped from top
        fill_value: Value to fill padded region

    Returns:
        Padded mask with original dimensions
    """
    if crop_top <= 0:
        return mask

    if mask.ndim == 2:
        # 2D mask (H, W)
        return np.pad(mask, ((crop_top, 0), (0, 0)), mode="constant", constant_values=fill_value)
    elif mask.ndim == 3:
        # 3D mask (C, H, W)
        return np.pad(
            mask, ((0, 0), (crop_top, 0), (0, 0)), mode="constant", constant_values=fill_value
        )
    else:
        raise ValueError(f"Unsupported mask dimensions: {mask.ndim}")
