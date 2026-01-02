"""
Image loading and validation utilities.
"""

from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image


class ImageValidationError(Exception):
    """Raised when an image fails validation."""

    pass


class ImageLoader:
    """
    Loads and validates images for inference.
    """

    def __init__(
        self,
        min_width: int = 1024,
        min_height: int = 2024,
    ):
        """
        Initialize image loader.

        Args:
            min_width: Minimum image width
            min_height: Minimum image height
        """
        self.min_width = min_width
        self.min_height = min_height

    def load(self, path: Path | str) -> Image.Image:
        """
        Load and validate an image.

        Args:
            path: Path to image file

        Returns:
            PIL Image in RGB mode

        Raises:
            ImageValidationError: If image fails validation
            FileNotFoundError: If image file not found
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            raise ImageValidationError(f"Failed to open image {path}: {e}")

        # Validate size
        width, height = image.size
        if width < self.min_width or height < self.min_height:
            raise ImageValidationError(
                f"Image {path.name} is {width}x{height}, "
                f"minimum required is {self.min_width}x{self.min_height}"
            )

        return image

    def load_as_array(self, path: Path | str) -> np.ndarray:
        """
        Load image and convert to numpy array.

        Args:
            path: Path to image file

        Returns:
            Numpy array with shape (H, W, C) in RGB format
        """
        image = self.load(path)
        return np.array(image)

    def load_as_chw(self, path: Path | str) -> np.ndarray:
        """
        Load image and convert to CHW format for model input.

        Args:
            path: Path to image file

        Returns:
            Numpy array with shape (C, H, W) in RGB format
        """
        array = self.load_as_array(path)
        return array.transpose(2, 0, 1)

    def load_as_batch(self, path: Path | str) -> np.ndarray:
        """
        Load image and convert to batch format (N, C, H, W).

        Args:
            path: Path to image file

        Returns:
            Numpy array with shape (1, C, H, W)
        """
        chw = self.load_as_chw(path)
        return np.expand_dims(chw, axis=0)

    def validate(self, path: Path | str) -> tuple[bool, str | None]:
        """
        Validate an image without loading it fully.

        Args:
            path: Path to image file

        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(path)

        if not path.exists():
            return False, f"File not found: {path}"

        try:
            with Image.open(path) as img:
                width, height = img.size
                if width < self.min_width or height < self.min_height:
                    return False, (
                        f"Image is {width}x{height}, "
                        f"minimum required is {self.min_width}x{self.min_height}"
                    )
        except Exception as e:
            return False, f"Failed to open image: {e}"

        return True, None

    def get_image_size(self, path: Path | str) -> tuple[int, int]:
        """
        Get image dimensions without loading full image.

        Args:
            path: Path to image file

        Returns:
            Tuple of (width, height)
        """
        with Image.open(path) as img:
            return img.size

    def validate_batch(
        self,
        paths: list[Path],
        stop_on_error: bool = False,
    ) -> tuple[list[Path], list[tuple[Path, str]]]:
        """
        Validate multiple images.

        Args:
            paths: List of image paths
            stop_on_error: If True, stop on first error

        Returns:
            Tuple of (valid_paths, list of (invalid_path, error_message))
        """
        valid: list[Path] = []
        errors: list[tuple[Path, str]] = []

        for path in paths:
            is_valid, error = self.validate(path)
            if is_valid:
                valid.append(path)
            else:
                errors.append((path, error or "Unknown error"))
                if stop_on_error:
                    logger.error(f"Validation failed for {path}: {error}")
                    break
                else:
                    logger.warning(f"Skipping invalid image {path}: {error}")

        return valid, errors
