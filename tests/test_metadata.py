"""Tests for metadata tracking."""

import json
from pathlib import Path

from deepseek_ocr.metadata import MetadataManager, _file_checksum


class TestFileChecksum:
    """Tests for file checksum computation."""

    def test_checksum_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        assert _file_checksum(f) == _file_checksum(f)

    def test_checksum_changes_with_content(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello")
        c1 = _file_checksum(f)
        f.write_text("world")
        c2 = _file_checksum(f)
        assert c1 != c2

    def test_checksum_prefix(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("data")
        assert _file_checksum(f).startswith("sha256:")


class TestMetadataManager:
    """Tests for MetadataManager."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        mgr = MetadataManager(tmp_path)
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-fake")

        mgr.record(
            f,
            pages=3,
            processing_time=5.67,
            model="deepseek-ocr",
            backend="ollama",
            output_path="doc.md",
        )

        # Load fresh from disk
        mgr2 = MetadataManager(tmp_path)
        assert "doc.pdf" in mgr2.files
        entry = mgr2.files["doc.pdf"]
        assert entry["status"] == "completed"
        assert entry["pages"] == 3
        assert entry["processing_time"] == 5.67
        assert entry["model"] == "deepseek-ocr"
        assert entry["output_path"] == "doc.md"

    def test_is_processed_true(self, tmp_path: Path) -> None:
        mgr = MetadataManager(tmp_path)
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-fake")

        mgr.record(
            f,
            pages=1,
            processing_time=1.0,
            model="m",
            backend="b",
            output_path="doc.md",
        )

        assert mgr.is_processed(f) is True

    def test_is_processed_false_unknown_file(self, tmp_path: Path) -> None:
        mgr = MetadataManager(tmp_path)
        f = tmp_path / "new.pdf"
        f.write_bytes(b"new content")
        assert mgr.is_processed(f) is False

    def test_is_processed_false_after_content_change(self, tmp_path: Path) -> None:
        mgr = MetadataManager(tmp_path)
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"version1")

        mgr.record(
            f,
            pages=1,
            processing_time=1.0,
            model="m",
            backend="b",
            output_path="doc.md",
        )

        # Modify the file
        f.write_bytes(b"version2")
        assert mgr.is_processed(f) is False

    def test_atomic_write(self, tmp_path: Path) -> None:
        """Verify no .tmp file remains after save."""
        mgr = MetadataManager(tmp_path)
        mgr.save()

        assert (tmp_path / "metadata.json").exists()
        assert not (tmp_path / "metadata.json.tmp").exists()

    def test_load_corrupt_json(self, tmp_path: Path) -> None:
        """Corrupt metadata.json should not crash, just start fresh."""
        meta_file = tmp_path / "metadata.json"
        meta_file.write_text("{bad json", encoding="utf-8")

        mgr = MetadataManager(tmp_path)
        assert mgr.files == {}

    def test_empty_dir_no_metadata(self, tmp_path: Path) -> None:
        mgr = MetadataManager(tmp_path)
        assert mgr.files == {}

    def test_multiple_records(self, tmp_path: Path) -> None:
        mgr = MetadataManager(tmp_path)

        for name in ["a.pdf", "b.pdf", "c.pdf"]:
            f = tmp_path / name
            f.write_bytes(name.encode())
            mgr.record(
                f,
                pages=1,
                processing_time=0.5,
                model="m",
                backend="b",
                output_path=f"{name}.md",
            )

        mgr2 = MetadataManager(tmp_path)
        assert len(mgr2.files) == 3

    def test_metadata_json_structure(self, tmp_path: Path) -> None:
        mgr = MetadataManager(tmp_path)
        f = tmp_path / "test.pdf"
        f.write_bytes(b"test")
        mgr.record(
            f,
            pages=2,
            processing_time=3.14,
            model="deepseek-ocr",
            backend="ollama",
            output_path="test.md",
        )

        raw = json.loads((tmp_path / "metadata.json").read_text())
        assert raw["version"] == "1"
        assert "test.pdf" in raw["files"]
        assert "checksum" in raw["files"]["test.pdf"]
        assert "timestamp" in raw["files"]["test.pdf"]
