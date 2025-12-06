"""Model management and inference for DeepSeek-OCR via Ollama."""

import base64
import logging
import re
from pathlib import Path
from typing import Optional, Union

import requests
from PIL import Image

from deepseek_ocr.config import settings

logger = logging.getLogger(__name__)


def _html_table_to_markdown(html_table: str) -> str:
    """Convert a simple HTML table to markdown format."""
    rows = []
    # Extract rows
    row_matches = re.findall(r'<tr[^>]*>(.*?)</tr>', html_table, re.DOTALL | re.IGNORECASE)

    for row_html in row_matches:
        # Extract cells (both td and th)
        cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row_html, re.DOTALL | re.IGNORECASE)
        # Clean cell content
        cleaned_cells = []
        for cell in cells:
            # Remove any nested HTML tags
            cell = re.sub(r'<[^>]+>', '', cell)
            # Clean whitespace
            cell = ' '.join(cell.split())
            cleaned_cells.append(cell)
        if cleaned_cells:
            rows.append(cleaned_cells)

    if not rows:
        return ""

    # Build markdown table
    md_lines = []
    for idx, row in enumerate(rows):
        md_lines.append("| " + " | ".join(row) + " |")
        # Add header separator after first row
        if idx == 0:
            md_lines.append("|" + "|".join(["---"] * len(row)) + "|")

    return "\n".join(md_lines)


def clean_ocr_output(text: str) -> str:
    """Remove grounding annotations and clean HTML from OCR output.

    Strips out <|ref|>...<|/ref|><|det|>[[...]]<|/det|> patterns,
    converts HTML tables to markdown, and cleans up formatting.
    """
    # Remove <|ref|>...<|/ref|> tags
    text = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', text)
    # Remove <|det|>[[...]]<|/det|> bounding boxes
    text = re.sub(r'<\|det\|>\[\[.*?\]\]<\|/det\|>', '', text)
    # Remove any remaining special tokens
    text = re.sub(r'<\|[^|]+\|>', '', text)

    # Convert HTML tables to markdown
    def replace_table(match):
        return _html_table_to_markdown(match.group(0))

    text = re.sub(r'<table[^>]*>.*?</table>', replace_table, text, flags=re.DOTALL | re.IGNORECASE)

    # Remove remaining HTML tags (like <center>, <sup>, etc.)
    # But preserve their content
    text = re.sub(r'<(sup|sub)>([^<]*)</\1>', r'^\2', text, flags=re.IGNORECASE)  # superscript/subscript
    text = re.sub(r'<center>([^<]*)</center>', r'\1', text, flags=re.IGNORECASE)  # center
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)  # line breaks
    text = re.sub(r'<[^>]+>', '', text)  # strip remaining HTML tags

    # Decode common HTML entities
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

    # Clean up multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    # Remove leading/trailing blank lines
    text = text.strip()
    return text

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434"


class ModelManager:
    """Manages DeepSeek-OCR model inference via Ollama."""

    # Default prompts for different OCR tasks
    PROMPTS = {
        "convert": "<|grounding|>Convert the document to markdown.",
        "ocr": "Free OCR.",
        "layout": "<|grounding|>Given the layout of the image.",
        "extract": "Extract the text in the image.",
        "parse": "Parse the figure.",
        "describe_figure": "Describe this figure in detail. What does the chart/graph/diagram show? Explain the axes, data, and key findings.",
    }

    def __init__(
        self,
        model_name: str = "deepseek-ocr",
        ollama_url: Optional[str] = None,
        **kwargs,  # Accept but ignore legacy kwargs for compatibility
    ):
        """Initialize model manager.

        Args:
            model_name: Ollama model name (default: deepseek-ocr)
            ollama_url: Ollama API URL (default: http://localhost:11434)
        """
        self.model_name = model_name
        self.ollama_url = ollama_url or OLLAMA_API_URL
        self.model: bool = False  # Track if "loaded" (connection verified)
        self.device = "ollama"  # For compatibility with processor.py
        self.resolution = getattr(settings, "resolution", None)

        # Ignored legacy kwargs: device, resolution, use_flash_attention, cache_dir
        if kwargs:
            logger.debug(f"Ignoring legacy kwargs: {list(kwargs.keys())}")

        logger.info(f"Initialized ModelManager with Ollama model: {self.model_name}")

    def _check_ollama_running(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def _check_model_available(self) -> bool:
        """Check if deepseek-ocr model is available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(self.model_name in m.get("name", "") for m in models)
            return False
        except Exception:
            return False

    def load_model(self) -> None:
        """Verify Ollama connection and model availability.

        Raises:
            RuntimeError: If Ollama is not running or model not available
        """
        if self.model:
            logger.info("Model already loaded")
            return

        logger.info(f"Connecting to Ollama at {self.ollama_url}")

        if not self._check_ollama_running():
            raise RuntimeError(
                f"Ollama is not running at {self.ollama_url}. "
                "Please start Ollama with: ollama serve"
            )

        if not self._check_model_available():
            raise RuntimeError(
                f"Model '{self.model_name}' not found in Ollama. "
                f"Please pull it with: ollama pull {self.model_name}"
            )

        self.model = True
        logger.info(f"Connected to Ollama, model '{self.model_name}' ready")

    def unload_model(self) -> None:
        """Mark model as unloaded (Ollama manages actual memory)."""
        self.model = False
        logger.info("Model connection closed")

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        import io

        buffer = io.BytesIO()
        # Save as PNG for lossless quality
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def process_image(
        self,
        image: Union[Image.Image, Path, str],
        prompt: Optional[str] = None,
        task: str = "convert",
        return_raw: bool = False,
    ) -> str:
        """Process a single image and return OCR text.

        Args:
            image: PIL Image, path to image, or image URL
            prompt: Custom prompt (if None, uses default for task)
            task: Task type ('convert', 'ocr', 'layout', 'extract', 'parse')
            return_raw: If True, return raw output with bounding boxes

        Returns:
            Extracted text/markdown

        Raises:
            RuntimeError: If model not loaded or inference fails
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first")

        # Load image if path provided
        if isinstance(image, (Path, str)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

        # Get prompt
        if prompt is None:
            prompt = self.PROMPTS.get(task, self.PROMPTS["convert"])

        logger.debug(f"Processing image with prompt: {prompt}")

        try:
            # Convert image to base64
            image_b64 = self._image_to_base64(image)

            # Call Ollama API
            # NOTE: Adding optimization parameters (keep_alive, num_ctx, etc.) caused
            # the model to hang indefinitely. Needs investigation of Ollama API compatibility.
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                },
                timeout=600,  # 10 minute timeout for large/complex images
            )

            if response.status_code != 200:
                raise RuntimeError(f"Ollama API error: {response.text}")

            result = response.json()
            raw_text = result.get("response", "")
            if return_raw:
                return raw_text
            return clean_ocr_output(raw_text)

        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama request timed out (>5 minutes)")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Lost connection to Ollama at {self.ollama_url}. "
                "Is Ollama still running?"
            )
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")

    def process_images_batch(
        self,
        images: list,
        prompt: Optional[str] = None,
        task: str = "convert",
    ) -> list:
        """Process multiple images sequentially.

        Args:
            images: List of PIL Images or paths
            prompt: Custom prompt
            task: Task type

        Returns:
            List of extracted texts
        """
        results = []
        for image in images:
            result = self.process_image(image, prompt=prompt, task=task)
            results.append(result)
        return results

    def describe_figure(self, image: Union[Image.Image, Path, str]) -> str:
        """Describe a figure/chart/graph in the image.

        Args:
            image: PIL Image or path containing a figure

        Returns:
            Description of the figure
        """
        return self.process_image(
            image,
            prompt=self.PROMPTS["describe_figure"],
            task="describe_figure"
        )
