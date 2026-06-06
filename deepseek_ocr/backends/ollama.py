"""Ollama backend for DeepSeek OCR."""

import base64
import io
import logging
from pathlib import Path

import requests
from PIL import Image

from deepseek_ocr.backends.base import Backend, TransientError
from deepseek_ocr.config import settings
from deepseek_ocr.utils import clean_ocr_output, resize_image_if_needed

logger = logging.getLogger(__name__)

OLLAMA_API_URL = "http://localhost:11434"

# HTTP status codes that indicate transient errors worth retrying
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class OllamaBackend(Backend):
    """Ollama backend for DeepSeek-OCR model inference."""

    def __init__(
        self,
        model_name: str = "deepseek-ocr",
        ollama_url: str | None = None,
        max_dimension: int | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        max_tokens: int | None = None,
    ):
        super().__init__(
            model_name=model_name,
            max_dimension=max_dimension if max_dimension is not None else settings.max_dimension,
            max_retries=max_retries if max_retries is not None else settings.max_retries,
            retry_delay=retry_delay if retry_delay is not None else settings.retry_delay,
        )
        self.max_tokens = max_tokens if max_tokens is not None else settings.max_tokens
        self.ollama_url = ollama_url or settings.ollama_url or OLLAMA_API_URL
        logger.info(f"Initialized OllamaBackend with model: {self.model_name}")

    @property
    def backend_name(self) -> str:
        return "ollama"

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
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _call_ollama_api(self, image_b64: str, prompt: str) -> tuple[str, str | None]:
        """Make the Ollama API call, returning ``(text, finish_reason)``.

        Raises TransientError for retryable failures. Ollama reports the stop
        reason as ``done_reason`` (e.g. ``"length"`` when the response hit
        ``num_predict``); it is threaded out so the processor can flag truncated
        pages instead of recording an incomplete page as a silent success.
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {
                        "num_ctx": 8192,
                        "num_predict": self.max_tokens,
                        "temperature": 0.1,
                    },
                },
                timeout=1800,
            )
        except requests.exceptions.Timeout as e:
            raise TransientError("Ollama request timed out", original=e)
        except requests.exceptions.ConnectionError as e:
            raise TransientError(f"Lost connection to Ollama at {self.ollama_url}", original=e)

        if response.status_code in _TRANSIENT_STATUS_CODES:
            raise TransientError(
                f"Ollama API returned HTTP {response.status_code}: {response.text}"
            )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.text}")

        payload = response.json()
        return payload.get("response", ""), payload.get("done_reason")

    def process_image(
        self,
        image: Image.Image | Path | str,
        prompt: str | None = None,
        task: str = "convert",
        return_raw: bool = False,
    ) -> str:
        """Process image and return OCR text."""
        text, _finish_reason = self.process_image_with_meta(
            image, prompt=prompt, task=task, return_raw=return_raw
        )
        return text

    def process_image_with_meta(
        self,
        image: Image.Image | Path | str,
        prompt: str | None = None,
        task: str = "convert",
        return_raw: bool = False,
    ) -> tuple[str, str | None]:
        """Process image and return ``(text, finish_reason)``."""
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
            prompt = self.get_prompt(task)

        logger.debug(f"Processing image with prompt: {prompt}")

        image = resize_image_if_needed(image, self.max_dimension)
        image_b64 = self._image_to_base64(image)

        raw_text, finish_reason = self._retry(self._call_ollama_api, image_b64, prompt)
        if return_raw:
            return raw_text, finish_reason
        return clean_ocr_output(raw_text), finish_reason
