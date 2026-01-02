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

    # Console handler - clean format for users
    level = "DEBUG" if verbose else "INFO"
    if verbose:
        # Detailed format for debugging
        fmt = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    else:
        # Clean format for normal use
        fmt = "<level>{message}</level>"

    logger.add(
        sys.stderr,
        format=fmt,
        level=level,
        colorize=True,
    )

    # File handler (optional) - always detailed
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
        )


def get_default_config_path() -> Optional[Path]:
    """Get default config path if it exists."""
    default_paths = [
        Path("config/default.yaml"),
        Path("config.yaml"),
        Path.cwd() / "config" / "default.yaml",
    ]
    for path in default_paths:
        if path.exists():
            return path.resolve()
    return None


def prompt_for_path(prompt_text: str, must_exist: bool = True, is_dir: bool = True) -> Path:
    """Prompt user for a path with validation."""
    while True:
        path_str = typer.prompt(prompt_text)
        path = Path(path_str).resolve()

        if must_exist and not path.exists():
            typer.echo(f"Error: Path does not exist: {path}", err=True)
            continue

        if must_exist and is_dir and not path.is_dir():
            typer.echo(f"Error: Path is not a directory: {path}", err=True)
            continue

        if must_exist and not is_dir and not path.is_file():
            typer.echo(f"Error: Path is not a file: {path}", err=True)
            continue

        return path


@app.command()
def infer(
    project: Annotated[
        Optional[Path],
        typer.Option(
            "--project",
            "-p",
            help="Path to project folder containing session/task/images structure.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Path to output directory.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
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
    If project, output, or config paths are not provided, you will be prompted
    to enter them interactively.

    \b
    Example:
        citylogix-infer infer
        citylogix-infer infer -p /path/to/project -o /path/to/output -c config.yaml
    """
    # Interactive prompts for missing paths
    if project is None:
        project = prompt_for_path("Enter project folder path", must_exist=True, is_dir=True)

    if output is None:
        output = prompt_for_path("Enter output folder path", must_exist=False, is_dir=True)
        # Create output directory if it doesn't exist
        output.mkdir(parents=True, exist_ok=True)

    if config is None:
        default_config = get_default_config_path()
        if default_config:
            use_default = typer.confirm(f"Use default config ({default_config})?", default=True)
            if use_default:
                config = default_config
            else:
                config = prompt_for_path("Enter config file path", must_exist=True, is_dir=False)
        else:
            config = prompt_for_path("Enter config file path", must_exist=True, is_dir=False)

    setup_logging(verbose, log_file)

    # Print header
    typer.echo()
    typer.secho("Citylogix AI Inference", fg=typer.colors.CYAN, bold=True)
    typer.secho("=" * 40, fg=typer.colors.CYAN)
    typer.echo(f"  Project: {project}")
    typer.echo(f"  Output:  {output}")
    typer.echo()

    # Load configuration
    try:
        inference_config = InferenceConfig.from_yaml(config)
    except Exception as e:
        typer.secho(f"Error: Failed to load config: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Apply CLI overrides (silent unless verbose)
    if batch_size is not None:
        inference_config.defaults.batch_size = batch_size
        if verbose:
            logger.debug(f"Override batch_size: {batch_size}")

    if on_error is not None:
        if on_error not in ("skip", "stop"):
            typer.secho(f"Error: Invalid --on-error value: {on_error}. Must be 'skip' or 'stop'.", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        inference_config.error_handling.on_image_error = on_error
        if verbose:
            logger.debug(f"Override on_error: {on_error}")

    # Validate configuration
    enabled_models = inference_config.get_enabled_models()
    if not enabled_models:
        typer.secho("Error: No models enabled. Check your config file.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Show enabled models
    model_names = [m.name for m in enabled_models]
    typer.echo(f"  Models: {', '.join(model_names)}")
    typer.echo()

    # Run inference
    try:
        with Predictor(inference_config) as predictor:
            stats = predictor.run(project, output)

        # Print summary
        typer.echo()
        typer.secho("Complete!", fg=typer.colors.GREEN, bold=True)
        typer.secho("-" * 40, fg=typer.colors.GREEN)
        typer.echo(f"  Images processed: {stats['images_processed']}")
        typer.echo(f"  Images skipped:   {stats['images_skipped']}")
        typer.echo(f"  Annotations:      {stats['annotations']}")
        typer.echo(f"  Output: {output}")
        typer.echo()

    except FileNotFoundError as e:
        typer.secho(f"Error: File not found: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"Error: Inference failed: {e}", fg=typer.colors.RED, err=True)
        if verbose:
            logger.exception("Full traceback:")
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
