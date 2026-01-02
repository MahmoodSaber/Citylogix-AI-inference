"""
CLI entry point for Citylogix AI Inference.

Usage:
    citylogix-infer --project /path/to/project --output /path/to/output --config config.yaml
"""

import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from loguru import logger

from .config import InferenceConfig
from .predictor import Predictor

app = typer.Typer(
    name="citylogix-infer",
    help="Run segmentation inference on road/pavement images.",
    add_completion=False,
)


def setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> None:
    """Configure logging."""
    # Remove default handler
    logger.remove()

    # Console handler
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # File handler (optional)
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
        )


@app.command()
def infer(
    project: Annotated[
        Path,
        typer.Option(
            "--project",
            "-p",
            help="Path to project folder containing session/task/images structure.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to output directory.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose logging.",
        ),
    ] = False,
    log_file: Annotated[
        Optional[Path],
        typer.Option(
            "--log-file",
            help="Path to log file.",
        ),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option(
            "--batch-size",
            "-b",
            help="Override batch size from config.",
        ),
    ] = None,
    on_error: Annotated[
        Optional[str],
        typer.Option(
            "--on-error",
            help="Override error handling: 'skip' or 'stop'.",
        ),
    ] = None,
) -> None:
    """
    Run inference on a project folder.

    \b
    Example:
        citylogix-infer -p /path/to/project -o /path/to/output -c config.yaml
    """
    setup_logging(verbose, log_file)

    logger.info("=" * 60)
    logger.info("Citylogix AI Inference")
    logger.info("=" * 60)
    logger.info(f"Project: {project}")
    logger.info(f"Output:  {output}")
    logger.info(f"Config:  {config}")

    # Load configuration
    try:
        inference_config = InferenceConfig.from_yaml(config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise typer.Exit(code=1)

    # Apply CLI overrides
    if batch_size is not None:
        inference_config.defaults.batch_size = batch_size
        logger.info(f"Override batch_size: {batch_size}")

    if on_error is not None:
        if on_error not in ("skip", "stop"):
            logger.error(f"Invalid --on-error value: {on_error}. Must be 'skip' or 'stop'.")
            raise typer.Exit(code=1)
        inference_config.error_handling.on_image_error = on_error
        logger.info(f"Override on_error: {on_error}")

    # Validate configuration
    if not inference_config.models:
        logger.error("No models configured. Check your config file.")
        raise typer.Exit(code=1)

    logger.info(f"Models configured: {len(inference_config.models)}")
    for model in inference_config.models:
        logger.info(f"  - {model.name} ({model.mode}): {model.classes}")

    # Run inference
    try:
        with Predictor(inference_config) as predictor:
            stats = predictor.run(project, output)

        logger.info("=" * 60)
        logger.info("Inference Complete")
        logger.info("=" * 60)
        logger.info(f"Images processed: {stats['images_processed']}")
        logger.info(f"Images skipped:   {stats['images_skipped']}")
        logger.info(f"Annotations:      {stats['annotations']}")
        logger.info(f"Output saved to:  {output}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def validate_config(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to YAML configuration file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
) -> None:
    """
    Validate a configuration file without running inference.
    """
    setup_logging(verbose=True)

    logger.info(f"Validating config: {config}")

    try:
        inference_config = InferenceConfig.from_yaml(config)
        logger.info("Config syntax: OK")

        # Check model paths
        errors = inference_config.validate_models()
        if errors:
            logger.warning("Model path issues:")
            for error in errors:
                logger.warning(f"  - {error}")
        else:
            logger.info("Model paths: OK")

        # Summary
        logger.info(f"Models: {len(inference_config.models)}")
        logger.info(f"Image patterns: {inference_config.image_patterns}")
        logger.info(f"Defaults:")
        logger.info(f"  - processor_size: {inference_config.defaults.processor_size}")
        logger.info(f"  - crop_top: {inference_config.defaults.crop_top}")
        logger.info(f"  - crop_size: {inference_config.defaults.crop_size}")
        logger.info(f"  - batch_size: {inference_config.defaults.batch_size}")
        logger.info(f"  - min_image_size: {inference_config.defaults.min_image_width}x{inference_config.defaults.min_image_height}")

        logger.info("Configuration is valid!")

    except Exception as e:
        logger.error(f"Configuration error: {e}")
        raise typer.Exit(code=1)


@app.command()
def list_images(
    project: Annotated[
        Path,
        typer.Argument(
            help="Path to project folder.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    pattern: Annotated[
        Optional[str],
        typer.Option(
            "--pattern",
            help="Image pattern (e.g., '*.jpg').",
        ),
    ] = None,
) -> None:
    """
    List all images in a project folder.
    """
    from .processors.folder_scanner import FolderScanner

    setup_logging()

    patterns = [pattern] if pattern else None
    scanner = FolderScanner(project, patterns)

    images = scanner.scan()
    counts = scanner.get_image_count()

    logger.info(f"Found {len(images)} images in {project}")

    for session, tasks in counts.items():
        logger.info(f"  {session}/")
        for task, count in tasks.items():
            logger.info(f"    {task}/: {count} images")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
