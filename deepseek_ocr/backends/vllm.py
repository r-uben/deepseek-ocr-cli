"""vLLM backend for DeepSeek OCR (OpenAI-compatible API)."""

import base64
import io
import logging
from pathlib import Path
from typing import Union

from PIL import Image

from deepseek_ocr.backends.base import Backend, TransientError
from deepseek_ocr.config import settings
from deepseek_ocr.model import clean_ocr_output
from deepseek_ocr.utils import resize_image_if_needed

logger = logging.getLogger(__name__)

VLLM_DEFAULT_URL = "http://localhost:8000/v1"


class VLLMBackend(Backend):
    """vLLM backend using OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str = "deepseek-vl2",
        base_url: str | None = None,
        max_dimension: int | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
    ):
        super().__init__(
            model_name=model_name,
            max_dimension=max_dimension if max_dimension is not None else settings.max_dimension,
            max_retries=max_retries if max_retries is not None else settings.max_retries,
            retry_delay=retry_delay if retry_delay is not None else settings.retry_delay,
        )
        self.base_url = base_url or getattr(settings, "vllm_base_url", None) or VLLM_DEFAULT_URL
        self._client = None
        logger.info(f"Initialized VLLMBackend with model: {self.model_name} at {self.base_url}")

    @property
    def backend_name(self) -> str:
        return "vllm"

    def load_model(self) -> None:
        """Initialize OpenAI client and verify vLLM server connection."""
        if self.model:
            logger.info("Model already loaded")
            return

        try:
            import openai
        except ImportError:
            raise RuntimeError(
                "openai package is required for vLLM backend. "
                "Install with: pip install openai"
            )

        logger.info(f"Connecting to vLLM at {self.base_url}")

        self._client = openai.OpenAI(
            base_url=self.base_url,
            api_key="EMPTY",  # vLLM doesn't require a real API key
        )

        # Verify connection by listing models
        try:
            models = self._client.models.list()
            model_ids = [m.id for m in models.data]
            logger.debug(f"Available models: {model_ids}")

            if self.model_name not in model_ids:
                logger.warning(
                    f"Model '{self.model_name}' not found in available models: {model_ids}. "
                    f"Proceeding anyway as vLLM may use different model naming."
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to vLLM at {self.base_url}: {e}. "
                "Is the vLLM server running?"
            )

        self.model = True
        logger.info(f"Connected to vLLM, model '{self.model_name}' ready")

    def unload_model(self) -> None:
        self._client = None
        self.model = False
        logger.info("vLLM client closed")

    def _image_to_base64_url(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URL for OpenAI API."""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _call_vllm_api(self, image_url: str, prompt: str) -> str:
        """Make the vLLM API call, raising TransientError for retryable failures."""
        import openai

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                max_tokens=2048,
                temperature=0.1,
            )
            return response.choices[0].message.content or ""

        except openai.APITimeoutError as e:
            raise TransientError(f"vLLM request timed out: {e}", original=e)
        except openai.APIConnectionError as e:
            raise TransientError(f"Lost connection to vLLM: {e}", original=e)
        except openai.RateLimitError as e:
            raise TransientError(f"vLLM rate limit: {e}", original=e)
        except openai.InternalServerError as e:
            raise TransientError(f"vLLM server error: {e}", original=e)
        except openai.APIStatusError as e:
            # 502, 503, 504 come through as APIStatusError
            if e.status_code in {502, 503, 504}:
                raise TransientError(
                    f"vLLM HTTP {e.status_code}: {e}", original=e
                )
            # Non-transient (400, 404, etc.) — don't retry
            raise RuntimeError(f"vLLM API error (HTTP {e.status_code}): {e}")

    def process_image(
        self,
        image: Union[Image.Image, Path, str],
        prompt: str | None = None,
        task: str = "convert",
        return_raw: bool = False,
    ) -> str:
        """Process image and return OCR text using vLLM."""
        if not self.model or self._client is None:
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
        image_url = self._image_to_base64_url(image)

        raw_text = self._retry(self._call_vllm_api, image_url, prompt)
        if return_raw:
            return raw_text
        return clean_ocr_output(raw_text)
