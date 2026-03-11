"""DeepSeek OCR CLI - OCR processing via Ollama or vLLM."""

__version__ = "0.3.2"
__author__ = "Ruben Fernandez Fuertes"
__license__ = "MIT"

from deepseek_ocr.backends import Backend, OllamaBackend, VLLMBackend, create_backend
from deepseek_ocr.model import ModelManager  # Backward compatibility
from deepseek_ocr.processor import OCRProcessor

__all__ = [
    "OCRProcessor",
    "ModelManager",  # Backward compatibility
    "Backend",
    "OllamaBackend",
    "VLLMBackend",
    "create_backend",
]
