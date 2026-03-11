"""Backend implementations for DeepSeek OCR."""

from deepseek_ocr.backends.base import Backend
from deepseek_ocr.backends.ollama import OllamaBackend
from deepseek_ocr.backends.vllm import VLLMBackend

__all__ = ["Backend", "OllamaBackend", "VLLMBackend", "create_backend"]


def create_backend(
    backend_type: str = "ollama",
    model_name: str = "deepseek-ocr",
    max_dimension: int | None = None,
    **kwargs,
) -> Backend:
    """Factory function to create the appropriate backend.

    Args:
        backend_type: "ollama" or "vllm"
        model_name: Model name/identifier
        max_dimension: Maximum image dimension for resizing
        **kwargs: Additional backend-specific arguments

    Returns:
        Backend instance
    """
    backend_type = backend_type.lower()

    if backend_type == "ollama":
        return OllamaBackend(
            model_name=model_name,
            max_dimension=max_dimension,
            ollama_url=kwargs.get("ollama_url"),
        )
    elif backend_type == "vllm":
        return VLLMBackend(
            model_name=model_name,
            max_dimension=max_dimension,
            base_url=kwargs.get("vllm_base_url"),
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. Use 'ollama' or 'vllm'.")
