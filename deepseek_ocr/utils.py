"""Utility functions for DeepSeek OCR CLI."""

import logging
from pathlib import Path
from typing import List

from PIL import Image

# Supported file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
PDF_EXTENSION = ".pdf"
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | {PDF_EXTENSION}


def setup_logging(level: str = "INFO", verbose: bool = False) -> logging.Logger:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        verbose: Enable verbose output

    Returns:
        Configured logger instance
    """
    log_level = logging.DEBUG if verbose else getattr(logging, level.upper())

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return logging.getLogger("deepseek_ocr")


def is_supported_file(file_path: Path) -> bool:
    """Check if file is a supported format.

    Args:
        file_path: Path to file

    Returns:
        True if file extension is supported
    """
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_image_file(file_path: Path) -> bool:
    """Check if file is an image.

    Args:
        file_path: Path to file

    Returns:
        True if file is an image
    """
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def is_pdf_file(file_path: Path) -> bool:
    """Check if file is a PDF.

    Args:
        file_path: Path to file

    Returns:
        True if file is a PDF
    """
    return file_path.suffix.lower() == PDF_EXTENSION


def collect_files(input_path: Path, recursive: bool = False) -> List[Path]:
    """Collect all supported files from a path.

    Args:
        input_path: File or directory path
        recursive: Recursively search directories

    Returns:
        List of supported file paths

    Raises:
        FileNotFoundError: If input_path doesn't exist
        ValueError: If no supported files found
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    files: List[Path] = []

    if input_path.is_file():
        if is_supported_file(input_path):
            files.append(input_path)
        else:
            raise ValueError(
                f"Unsupported file type: {input_path.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
    elif input_path.is_dir():
        pattern = "**/*" if recursive else "*"
        for file_path in input_path.glob(pattern):
            if file_path.is_file() and is_supported_file(file_path):
                files.append(file_path)

    if not files:
        raise ValueError(f"No supported files found in: {input_path}")

    return sorted(files)


def load_image(image_path: Path) -> Image.Image:
    """Load an image file.

    Args:
        image_path: Path to image file

    Returns:
        PIL Image object

    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image can't be loaded
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path)
        # Convert to RGB if necessary
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing/replacing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem
    """
    # Replace invalid characters with underscore
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Remove leading/trailing spaces and dots
    filename = filename.strip(". ")

    # Ensure filename is not empty
    if not filename:
        filename = "untitled"

    return filename


def ensure_dir(directory: Path) -> Path:
    """Ensure directory exists, create if necessary.

    Args:
        directory: Directory path

    Returns:
        Created/existing directory path
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory
