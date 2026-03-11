"""Utility functions for DeepSeek OCR CLI."""

import logging
import re
from pathlib import Path
from typing import List

from PIL import Image

# Supported file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"}
PDF_EXTENSION = ".pdf"
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | {PDF_EXTENSION}


def setup_logging(level: str = "WARNING", verbose: bool = False) -> logging.Logger:
    log_level = logging.DEBUG if verbose else logging.WARNING

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return logging.getLogger("deepseek_ocr")


def is_supported_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_image_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def is_pdf_file(file_path: Path) -> bool:
    return file_path.suffix.lower() == PDF_EXTENSION


def collect_files(input_path: Path, recursive: bool = False) -> List[Path]:
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
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def resize_image_if_needed(image: Image.Image, max_dimension: int) -> Image.Image:
    """Resize image if it exceeds max_dimension to prevent Ollama timeouts.

    Large images create massive base64 payloads that can cause Ollama to hang.
    This resizes proportionally while preserving aspect ratio.

    Args:
        image: PIL Image to potentially resize
        max_dimension: Maximum allowed width or height. 0 disables resizing.

    Returns:
        Resized image if needed, otherwise the original image
    """
    if max_dimension <= 0:
        return image

    width, height = image.size
    if max(width, height) <= max_dimension:
        return image

    ratio = max_dimension / max(width, height)
    new_size = (int(width * ratio), int(height * ratio))

    logging.getLogger(__name__).info(
        f"Resizing image from {width}x{height} to {new_size[0]}x{new_size[1]} "
        f"(max_dimension={max_dimension})"
    )

    return image.resize(new_size, Image.Resampling.LANCZOS)


def sanitize_filename(filename: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    filename = filename.strip(". ")

    if not filename:
        filename = "untitled"

    return filename


def ensure_dir(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _html_table_to_markdown(html_table: str) -> str:
    rows = []
    row_matches = re.findall(r'<tr[^>]*>(.*?)</tr>', html_table, re.DOTALL | re.IGNORECASE)

    for row_html in row_matches:
        cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row_html, re.DOTALL | re.IGNORECASE)
        cleaned_cells = []
        for cell in cells:
            cell = re.sub(r'<[^>]+>', '', cell)
            cell = ' '.join(cell.split())
            cleaned_cells.append(cell)
        if cleaned_cells:
            rows.append(cleaned_cells)

    if not rows:
        return ""

    md_lines = []
    for idx, row in enumerate(rows):
        md_lines.append("| " + " | ".join(row) + " |")
        if idx == 0:
            md_lines.append("|" + "|".join(["---"] * len(row)) + "|")

    return "\n".join(md_lines)


def clean_ocr_output(text: str) -> str:
    """Remove grounding annotations, convert HTML tables to markdown, decode entities."""
    text = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', text)
    text = re.sub(r'<\|det\|>\[\[.*?\]\]<\|/det\|>', '', text)
    text = re.sub(r'<\|[^|]+\|>', '', text)

    def replace_table(match: re.Match) -> str:
        return _html_table_to_markdown(match.group(0))

    text = re.sub(r'<table[^>]*>.*?</table>', replace_table, text, flags=re.DOTALL | re.IGNORECASE)

    text = re.sub(r'<(sup|sub)>([^<]*)</\1>', r'^\2', text, flags=re.IGNORECASE)
    text = re.sub(r'<center>([^<]*)</center>', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)

    html_entities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&apos;': "'",
        '&nbsp;': ' ',
        '&#39;': "'",
        '&#x27;': "'",
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)

    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    text = text.strip()
    return text
