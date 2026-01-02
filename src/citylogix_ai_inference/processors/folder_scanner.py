"""
Folder scanner for recursive image discovery.

Handles the Project → Session → Task → Images structure.
"""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class ImageInfo:
    """Information about a discovered image."""

    path: Path
    session: str
    task: str
    relative_path: Path  # Relative to project root

    @property
    def filename(self) -> str:
        return self.path.name


class FolderScanner:
    """
    Scans project folder structure for images.

    Expected structure:
        project_folder/
        ├── session_A/
        │   ├── task_1/
        │   │   └── *.jpg
        │   └── task_2/
        │       └── *.jpg
        └── session_B/
            └── task_3/
                └── *.jpg
    """

    def __init__(
        self,
        project_path: Path | str,
        image_patterns: list[str] | None = None,
    ):
        """
        Initialize folder scanner.

        Args:
            project_path: Root project folder path
            image_patterns: Glob patterns for images (default: common image extensions)
        """
        self.project_path = Path(project_path)
        self.image_patterns = image_patterns or [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.JPG",
            "*.JPEG",
            "*.PNG",
        ]

        if not self.project_path.exists():
            raise FileNotFoundError(f"Project path not found: {self.project_path}")

        if not self.project_path.is_dir():
            raise NotADirectoryError(f"Project path is not a directory: {self.project_path}")

    def scan(self) -> list[ImageInfo]:
        """
        Scan project folder and return all discovered images.

        Returns:
            List of ImageInfo objects for all discovered images
        """
        images: list[ImageInfo] = []

        # Get all sessions (first level directories)
        sessions = [d for d in self.project_path.iterdir() if d.is_dir()]

        if not sessions:
            logger.debug(f"No session folders found in {self.project_path}")
            return images

        for session_dir in sorted(sessions):
            session_name = session_dir.name

            # Get all tasks (second level directories)
            tasks = [d for d in session_dir.iterdir() if d.is_dir()]

            if not tasks:
                logger.debug(f"No task folders in session {session_name}")
                continue

            for task_dir in sorted(tasks):
                task_name = task_dir.name

                # Find images in task folder
                task_images = self._find_images_in_folder(task_dir, session_name, task_name)
                images.extend(task_images)

        logger.debug(f"Found {len(images)} images in {len(sessions)} sessions")

        return images

    def _find_images_in_folder(
        self,
        folder: Path,
        session: str,
        task: str,
    ) -> list[ImageInfo]:
        """Find all images in a folder matching the patterns."""
        images: list[ImageInfo] = []

        for pattern in self.image_patterns:
            for image_path in folder.glob(pattern):
                if image_path.is_file():
                    relative_path = image_path.relative_to(self.project_path)
                    images.append(
                        ImageInfo(
                            path=image_path,
                            session=session,
                            task=task,
                            relative_path=relative_path,
                        )
                    )

        # Remove duplicates (in case patterns overlap) and sort
        seen = set()
        unique_images = []
        for img in sorted(images, key=lambda x: x.path):
            if img.path not in seen:
                seen.add(img.path)
                unique_images.append(img)

        return unique_images

    def get_sessions(self) -> list[str]:
        """Get list of session names."""
        return sorted([d.name for d in self.project_path.iterdir() if d.is_dir()])

    def get_tasks(self, session: str) -> list[str]:
        """Get list of task names for a session."""
        session_dir = self.project_path / session
        if not session_dir.exists():
            return []
        return sorted([d.name for d in session_dir.iterdir() if d.is_dir()])

    def get_image_count(self) -> dict[str, dict[str, int]]:
        """
        Get image count per session and task.

        Returns:
            Dict mapping session -> task -> count
        """
        counts: dict[str, dict[str, int]] = {}
        images = self.scan()

        for img in images:
            if img.session not in counts:
                counts[img.session] = {}
            if img.task not in counts[img.session]:
                counts[img.session][img.task] = 0
            counts[img.session][img.task] += 1

        return counts
