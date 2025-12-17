"""DeepSeek OCR CLI - Local OCR processing via Ollama."""

__version__ = "0.2.4"
__author__ = "Ruben Fernandez Fuertes"
__license__ = "MIT"

from deepseek_ocr.processor import OCRProcessor
from deepseek_ocr.model import ModelManager

__all__ = ["OCRProcessor", "ModelManager"]
