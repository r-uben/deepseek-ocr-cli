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

    def replace_table(match):
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
        **kwargs,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url or OLLAMA_API_URL
        self.model: bool = False
        self.device = "ollama"
        self.resolution = getattr(settings, "resolution", None)

        if kwargs:
            logger.debug(f"Ignoring legacy kwargs: {list(kwargs.keys())}")

        logger.info(f"Initialized ModelManager with Ollama model: {self.model_name}")

    def _check_ollama_running(self) -> bool:
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def _check_model_available(self) -> bool:
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(self.model_name in m.get("name", "") for m in models)
            return False
        except Exception:
            return False

    def load_model(self) -> None:
        """Verify Ollama connection and model availability."""
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
        self.model = False
        logger.info("Model connection closed")

    def _image_to_base64(self, image: Image.Image) -> str:
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def process_image(
        self,
        image: Union[Image.Image, Path, str],
        prompt: Optional[str] = None,
        task: str = "convert",
        return_raw: bool = False,
    ) -> str:
        """Process image and return OCR text."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first")

        if isinstance(image, (Path, str)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

        if prompt is None:
            prompt = self.PROMPTS.get(task, self.PROMPTS["convert"])

        logger.debug(f"Processing image with prompt: {prompt}")

        try:
            image_b64 = self._image_to_base64(image)

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {
                        "num_ctx": 8192,
                        "temperature": 0.1,
                    }
                },
                timeout=1800,
            )

            if response.status_code != 200:
                raise RuntimeError(f"Ollama API error: {response.text}")

            result = response.json()
            raw_text = result.get("response", "")
            if return_raw:
                return raw_text
            return clean_ocr_output(raw_text)

        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama request timed out (>30 minutes)")
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
        results = []
        for image in images:
            result = self.process_image(image, prompt=prompt, task=task)
            results.append(result)
        return results

    def describe_figure(self, image: Union[Image.Image, Path, str]) -> str:
        return self.process_image(
            image,
            prompt=self.PROMPTS["describe_figure"],
            task="describe_figure"
        )
