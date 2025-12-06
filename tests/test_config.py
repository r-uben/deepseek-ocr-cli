"""Tests for configuration management."""

import pytest
from pathlib import Path
from deepseek_ocr.config import Resolution, DeviceType, Settings


class TestResolution:
    """Tests for Resolution enum."""

    def test_all_resolutions_exist(self) -> None:
        """Test that all expected resolutions are defined."""
        expected = {"tiny", "small", "base", "large", "gundam"}
        actual = {r.value for r in Resolution}
        assert actual == expected


class TestDeviceType:
    """Tests for DeviceType enum."""

    def test_all_device_types_exist(self) -> None:
        """Test that all expected device types are defined."""
        expected = {"auto", "cpu", "mps", "cuda"}
        actual = {d.value for d in DeviceType}
        assert actual == expected


class TestSettings:
    """Tests for Settings class."""

    def test_default_settings(self) -> None:
        """Test that default settings are correct."""
        settings = Settings()
        assert settings.model_name == "deepseek-ai/DeepSeek-OCR"
        assert settings.resolution == Resolution.BASE
        assert settings.device == DeviceType.AUTO
        assert settings.batch_size >= 1
        assert settings.output_dir == Path("output")

    def test_settings_validation(self) -> None:
        """Test that settings validation works."""
        with pytest.raises(Exception):
            # Batch size must be >= 1
            Settings(batch_size=0)

        with pytest.raises(Exception):
            # Max workers must be >= 1
            Settings(max_workers=0)
