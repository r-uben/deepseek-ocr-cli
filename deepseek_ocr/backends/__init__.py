"""Backend implementations for DeepSeek OCR."""

from deepseek_ocr.backends.base import Backend, TransientError
from deepseek_ocr.backends.ollama import OllamaBackend
from deepseek_ocr.backends.vllm import VLLMBackend

__all__ = ["Backend", "TransientError", "OllamaBackend", "VLLMBackend", "create_backend"]


def create_backend(
    backend_type: str = "ollama",
    model_name: str = "deepseek-ocr",
    max_dimension: int | None = None,
    max_retries: int | None = None,
    retry_delay: float | None = None,
    max_tokens: int | None = None,
    **kwargs,
) -> Backend:
    """Factory function to create the appropriate backend.

    Args:
        backend_type: "ollama" or "vllm"
        model_name: Model name/identifier
        max_dimension: Maximum image dimension for resizing
        max_retries: Maximum number of retries for transient errors
        retry_delay: Base delay in seconds between retries
        max_tokens: Maximum tokens per page response (truncation control)
        **kwargs: Additional backend-specific arguments

    Returns:
        Backend instance
    """
    backend_type = backend_type.lower()

    if backend_type == "ollama":
        return OllamaBackend(
            model_name=model_name,
            max_dimension=max_dimension,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_tokens=max_tokens,
            ollama_url=kwargs.get("ollama_url"),
        )
    elif backend_type == "vllm":
        return VLLMBackend(
            model_name=model_name,
            max_dimension=max_dimension,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_tokens=max_tokens,
            base_url=kwargs.get("vllm_base_url"),
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. Use 'ollama' or 'vllm'.")
