"""Tests for CLI interface."""

import os
import tempfile
from pathlib import Path

from click.testing import CliRunner
from PIL import Image

from deepseek_ocr.cli import cli, _format_size, _run_dry_run


class TestFormatSize:
    """Tests for _format_size helper."""

    def test_bytes(self) -> None:
        assert _format_size(500) == "500.0 B"

    def test_kilobytes(self) -> None:
        assert _format_size(2048) == "2.0 KB"

    def test_megabytes(self) -> None:
        assert _format_size(1048576) == "1.0 MB"

    def test_gigabytes(self) -> None:
        assert _format_size(1073741824) == "1.0 GB"


class TestDryRun:
    """Tests for --dry-run flag."""

    def test_dry_run_single_image(self) -> None:
        """Dry run on a single image file lists it without processing."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            img = Image.new("RGB", (100, 100), color="red")
            img.save(img_path)

            result = runner.invoke(cli, ["process", str(img_path), "--dry-run"])
            assert result.exit_code == 0
            assert "test.png" in result.output
            assert "dry run" in result.output.lower()

    def test_dry_run_directory(self) -> None:
        """Dry run on a directory lists all supported files."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ("a.png", "b.jpg", "c.txt"):
                p = Path(tmpdir) / name
                if name.endswith(".txt"):
                    p.write_text("not an image")
                else:
                    Image.new("RGB", (50, 50)).save(p)

            result = runner.invoke(cli, ["process", tmpdir, "--dry-run"])
            assert result.exit_code == 0
            assert "a.png" in result.output
            assert "b.jpg" in result.output
            assert "c.txt" not in result.output

    def test_dry_run_quiet(self) -> None:
        """Dry run with --quiet outputs only file paths."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            Image.new("RGB", (100, 100)).save(img_path)

            result = runner.invoke(cli, ["process", str(img_path), "--dry-run", "-q"])
            assert result.exit_code == 0
            # Should contain the path, not a table
            assert str(img_path) in result.output
            # Should not contain table headers
            assert "dry run" not in result.output.lower()


class TestQuietFlag:
    """Tests for --quiet / -q flag."""

    def test_quiet_suppresses_banner(self) -> None:
        """With --quiet, the version banner is suppressed."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            Image.new("RGB", (100, 100)).save(img_path)

            result = runner.invoke(cli, ["process", str(img_path), "--dry-run", "--quiet"])
            assert result.exit_code == 0
            assert "deepseek-ocr v" not in result.output


class TestAutoInsertProcess:
    """Tests that the main() entry point auto-inserts 'process' subcommand."""

    def test_help_shows_process_options(self) -> None:
        """--help should mention the process command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output
        assert "--quiet" in result.output
        assert "INPUT_PATH" in result.output

    def test_info_subcommand_accessible(self) -> None:
        """info subcommand should still be accessible."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--help"])
        assert result.exit_code == 0
        assert "system" in result.output.lower() or "configuration" in result.output.lower()
