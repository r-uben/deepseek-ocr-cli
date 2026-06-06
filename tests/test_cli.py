"""Tests for CLI interface."""

import tempfile
from pathlib import Path

import fitz
from click.testing import CliRunner
from PIL import Image

from deepseek_ocr.cli import _format_size, cli


def _make_pdf(path: Path, pages: int = 1) -> None:
    """Write a tiny multi-page PDF for discovery/dry-run tests."""
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=200, height=300)
        page.insert_text((40, 60), f"page {i + 1}")
    doc.save(str(path))
    doc.close()


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
        """Dry run on a batch tree lists each document, excluding unsupported files.

        A directory that mixes PDFs with subdirectories is a BATCH tree, so each
        document is listed individually (the dry-run preview uses the same real
        discovery as the run, so it never diverges).
        """
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # A subdir forces batch-tree semantics (not a single image-dir doc).
            (root / "sub").mkdir()
            _make_pdf(root / "a.pdf")
            _make_pdf(root / "sub" / "b.pdf")
            (root / "c.txt").write_text("not a document")

            result = runner.invoke(cli, ["process", tmpdir, "--dry-run"])
            assert result.exit_code == 0
            assert "a.pdf" in result.output
            assert "b.pdf" in result.output
            assert "c.txt" not in result.output

    def test_dry_run_image_dir_is_one_document(self) -> None:
        """A flat directory of only images is ONE image-dir document (socr case)."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "scan"
            root.mkdir()
            for n in (1, 2):
                Image.new("RGB", (50, 50)).save(root / f"page_{n:04d}.png")

            result = runner.invoke(cli, ["process", str(root), "--dry-run"])
            assert result.exit_code == 0
            # The directory itself is the single document; pages = image count.
            assert "scan" in result.output
            assert "IMG DIR" in result.output

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
