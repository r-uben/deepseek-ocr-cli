#!/usr/bin/env python
"""Basic usage example for DeepSeek OCR CLI library.

Output is routed through the shared ``ocr-output-contract`` package: each document
lands in ``<output_root>/<stem>/<stem>.md`` under ``## Page N`` headers, with a
per-document ``metadata.json`` sidecar and a rolled-up root index. ``process()``
returns a :class:`RunOutcome` whose ``outputs`` lists the written ``.md`` paths.
"""

from pathlib import Path

from deepseek_ocr import create_backend, process


def main() -> None:
    """Demonstrate basic usage of the library."""
    print("DeepSeek OCR - Basic Usage Example\n")

    # Initialize backend (Ollama by default). process() also loads it if needed.
    print("1. Connecting to Ollama...")
    backend = create_backend(backend_type="ollama", model_name="deepseek-ocr")
    backend.load_model()
    print("   Connected\n")

    output_dir = Path("./output")

    # Example 1: Process a single image (if available).
    example_image = Path("test_image.jpg")
    if example_image.exists():
        print(f"2. Processing image: {example_image}")
        outcome = process(example_image, backend, output_dir=output_dir)
        for md_path in outcome.outputs:
            print(f"   Saved to: {md_path}")
        print(f"   completed={outcome.completed} failed={outcome.failed}\n")
    else:
        print(f"2. Skipping image processing (no {example_image} found)\n")

    # Example 2: Process a PDF (if available).
    example_pdf = Path("test_document.pdf")
    if example_pdf.exists():
        print(f"3. Processing PDF: {example_pdf}")
        outcome = process(example_pdf, backend, output_dir=output_dir)
        for md_path in outcome.outputs:
            print(f"   Saved to: {md_path}")
        print(f"   exit_code={outcome.exit_code}\n")
    else:
        print(f"3. Skipping PDF processing (no {example_pdf} found)\n")

    # Cleanup
    backend.unload_model()
    print("Done!")


if __name__ == "__main__":
    main()
