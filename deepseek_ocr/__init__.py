"""DeepSeek OCR CLI - OCR processing via Ollama or vLLM."""

__version__ = "0.4.1"
__author__ = "Ruben Fernandez Fuertes"
__license__ = "MIT"

from deepseek_ocr.backends import Backend, OllamaBackend, VLLMBackend, create_backend
from deepseek_ocr.processor import OCRProcessor

__all__ = [
    "OCRProcessor",
    "Backend",
    "OllamaBackend",
    "VLLMBackend",
    "create_backend",
]
