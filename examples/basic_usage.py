#!/usr/bin/env python
"""Basic usage example for DeepSeek OCR CLI library."""

from pathlib import Path
from deepseek_ocr import create_backend, OCRProcessor


def main() -> None:
    """Demonstrate basic usage of the library."""
    print("DeepSeek OCR - Basic Usage Example\n")

    # Initialize backend (Ollama by default)
    print("1. Connecting to Ollama...")
    backend = create_backend(backend_type="ollama", model_name="deepseek-ocr")
    backend.load_model()
    print("   Connected\n")

    # Create processor
    print("2. Creating OCR processor...")
    processor = OCRProcessor(
        backend=backend,
        output_dir=Path("./output"),
        include_metadata=True,
        workers=2,
    )

    # Example 1: Process a single image (if available)
    example_image = Path("test_image.jpg")
    if example_image.exists():
        print(f"3. Processing image: {example_image}")
        result = processor.process_file(example_image)
        print(f"   Extracted {len(result.output_text)} characters")
        print(f"   Processing time: {result.processing_time:.2f}s")

        output_path = processor.save_result(result)
        print(f"   Saved to: {output_path}\n")
    else:
        print(f"3. Skipping image processing (no {example_image} found)\n")

    # Example 2: Process a PDF (if available)
    example_pdf = Path("test_document.pdf")
    if example_pdf.exists():
        print(f"4. Processing PDF: {example_pdf}")
        result = processor.process_file(example_pdf)
        print(f"   Processed {result.page_count} pages")
        print(f"   Processing time: {result.processing_time:.2f}s")

        output_path = processor.save_result(result)
        print(f"   Saved to: {output_path}\n")
    else:
        print(f"4. Skipping PDF processing (no {example_pdf} found)\n")

    # Cleanup
    backend.unload_model()
    print("Done!")


if __name__ == "__main__":
    main()
