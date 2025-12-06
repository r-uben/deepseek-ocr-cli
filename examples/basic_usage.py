#!/usr/bin/env python
"""Basic usage example for DeepSeek OCR CLI library (Ollama backend)."""

from pathlib import Path
from deepseek_ocr import ModelManager, OCRProcessor


def main() -> None:
    """Demonstrate basic usage of the library."""
    print("DeepSeek OCR - Basic Usage Example (Ollama)\n")

    # Initialize model manager
    print("1. Initializing model...")
    model_manager = ModelManager(model_name="deepseek-ocr")

    # Load model (connects to Ollama)
    print("2. Connecting to Ollama...")
    model_manager.load_model()
    print("   Connected to Ollama\n")

    # Create processor
    print("3. Creating OCR processor...")
    processor = OCRProcessor(
        model_manager=model_manager,
        output_dir=Path("./output"),
        include_metadata=True,
    )

    # Example 1: Process a single image (if available)
    example_image = Path("test_image.jpg")
    if example_image.exists():
        print(f"4. Processing image: {example_image}")
        result = processor.process_file(example_image)
        print(f"   Extracted {len(result.output_text)} characters")
        print(f"   Processing time: {result.processing_time:.2f}s")

        # Save result
        output_path = processor.save_result(result)
        print(f"   Saved to: {output_path}\n")
    else:
        print(f"4. Skipping image processing (no {example_image} found)\n")

    # Example 2: Process a PDF (if available)
    example_pdf = Path("test_document.pdf")
    if example_pdf.exists():
        print(f"5. Processing PDF: {example_pdf}")
        result = processor.process_file(example_pdf)
        print(f"   Processed {result.page_count} pages")
        print(f"   Processing time: {result.processing_time:.2f}s")

        # Save result
        output_path = processor.save_result(result)
        print(f"   Saved to: {output_path}\n")
    else:
        print(f"5. Skipping PDF processing (no {example_pdf} found)\n")

    # Cleanup
    print("6. Cleaning up...")
    model_manager.unload_model()
    print("   Done!\n")


if __name__ == "__main__":
    main()
