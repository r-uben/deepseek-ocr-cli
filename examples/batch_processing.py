#!/usr/bin/env python
"""Batch processing example for DeepSeek OCR CLI library.

A directory tree of documents is walked recursively (the resolved output root is
excluded from discovery, so prior outputs are never re-ingested). Each document is
keyed by its input-relative path, so same-named files in different subdirectories
never collide. Re-running skips already-completed documents (incremental resume).
"""

from pathlib import Path

from deepseek_ocr import create_backend, process


def main() -> None:
    """Demonstrate batch processing of a directory tree."""
    print("DeepSeek OCR - Batch Processing Example\n")

    input_dir = Path("./documents")
    output_dir = Path("./output_batch")

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("Please create ./documents/ and add some PDF/image files")
        return

    print("1. Connecting to Ollama...")
    backend = create_backend(backend_type="ollama", model_name="deepseek-ocr")
    backend.load_model()
    print("   Connected\n")

    print(f"2. Processing documents under: {input_dir}")
    outcome = process(input_dir, backend, output_dir=output_dir)

    total = outcome.completed + outcome.failed + outcome.partial
    print("\n3. Processing Summary:")
    print(f"   Documents processed: {total}")
    print(f"   Completed: {outcome.completed}")
    print(f"   Partial:   {outcome.partial}")
    print(f"   Failed:    {outcome.failed}")
    if outcome.failures:
        print(f"   Failures:  {', '.join(outcome.failures)}")
    print(f"   Output directory: {output_dir}")
    print(f"   Exit code: {outcome.exit_code}\n")

    backend.unload_model()
    print("Done!")


if __name__ == "__main__":
    main()
