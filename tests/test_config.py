"""Tests for configuration management."""

from deepseek_ocr.config import Settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_settings(self) -> None:
        """Test that default settings are correct."""
        settings = Settings()
        assert settings.model_name == "deepseek-ocr"
        assert settings.backend == "ollama"
        assert settings.ollama_url == "http://localhost:11434"
        assert settings.vllm_base_url == "http://localhost:8000/v1"
        assert settings.max_dimension == 1920
        assert settings.max_tokens == 8192
        assert settings.max_retries == 3
        assert settings.retry_delay == 1.0

    def test_backend_choices(self) -> None:
        """Test that backend only accepts valid choices."""
        settings = Settings(backend="ollama")
        assert settings.backend == "ollama"

        settings = Settings(backend="vllm")
        assert settings.backend == "vllm"
