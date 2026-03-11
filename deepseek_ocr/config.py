"""Configuration management for DeepSeek OCR CLI."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DEEPSEEK_OCR_",
        case_sensitive=False,
    )

    # Backend selection
    backend: Literal["ollama", "vllm"] = Field(
        default="ollama",
        description="Backend to use: 'ollama' (local) or 'vllm' (OpenAI-compatible)",
    )

    # Model configuration
    model_name: str = Field(
        default="deepseek-ocr",
        description="Model name (deepseek-ocr for Ollama, deepseek-vl2 for vLLM)",
    )
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API URL",
    )
    vllm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="vLLM OpenAI-compatible API URL",
    )

    # Image preprocessing
    max_dimension: int = Field(
        default=1920,
        description="Maximum image dimension (width or height). Larger images are resized to prevent Ollama timeouts. Set to 0 to disable.",
    )

    # Output configuration
    output_dir: Path = Field(
        default=Path("output"),
        description="Default output directory",
    )
    extract_images: bool = Field(
        default=False,
        description="Extract and save images from documents",
    )
    include_metadata: bool = Field(
        default=True,
        description="Include metadata in output markdown",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output",
    )


# Global settings instance
settings = Settings()
