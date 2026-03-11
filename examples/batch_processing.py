#!/usr/bin/env python
"""Batch processing example for DeepSeek OCR CLI library."""

from pathlib import Path
from deepseek_ocr import create_backend, OCRProcessor


def main() -> None:
    """Demonstrate batch processing of multiple files."""
    print("DeepSeek OCR - Batch Processing Example\n")

    input_dir = Path("./documents")
    output_dir = Path("./output_batch")

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("Please create ./documents/ and add some PDF/image files")
        return

    # Initialize backend
    print("1. Connecting to Ollama...")
    backend = create_backend(backend_type="ollama", model_name="deepseek-ocr")
    backend.load_model()
    print("   Connected\n")

    # Create processor
    print("2. Setting up processor...")
    processor = OCRProcessor(
        backend=backend,
        output_dir=output_dir,
        extract_images=True,
        include_metadata=True,
    )

    # Process all files in directory
    print(f"3. Processing files in: {input_dir}")
    results = processor.process_batch(
        input_path=input_dir,
        recursive=True,
        show_progress=True,
    )

    # Print summary
    print(f"\n4. Processing Summary:")
    print(f"   Files processed: {len(results)}")
    if results:
        total_pages = sum(r.page_count for r in results)
        total_time = sum(r.processing_time for r in results)
        print(f"   Total pages: {total_pages}")
        print(f"   Total time: {total_time:.2f}s")
        if total_pages > 0:
            print(f"   Average time per page: {total_time/total_pages:.2f}s")
    print(f"   Output directory: {output_dir}\n")

    backend.unload_model()
    print("Done!")


if __name__ == "__main__":
    main()
