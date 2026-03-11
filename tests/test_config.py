"""Tests for configuration management."""

from pathlib import Path

from deepseek_ocr.config import Settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_settings(self) -> None:
        """Test that default settings are correct."""
        settings = Settings()
        assert settings.model_name == "deepseek-ocr"
        assert settings.ollama_url == "http://localhost:11434"
        assert settings.max_dimension == 1920
        assert settings.output_dir == Path("output")
        assert settings.extract_images is False
        assert settings.include_metadata is True
