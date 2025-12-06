"""Configuration management for DeepSeek OCR CLI (Ollama backend)."""

from pathlib import Path

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

    # Model configuration
    model_name: str = Field(
        default="deepseek-ocr",
        description="Ollama model name",
    )
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API URL",
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
