#!/usr/bin/env python
"""Example: Extract and analyze embedded figures from PDFs."""

from pathlib import Path
from deepseek_ocr import ModelManager, OCRProcessor


def main() -> None:
    """Demonstrate figure extraction and analysis."""
    print("DeepSeek OCR - Figure Analysis Example\n")

    # Initialize
    model_manager = ModelManager(model_name="deepseek-ocr")
    model_manager.load_model()

    # Create processor with figure analysis enabled
    processor = OCRProcessor(
        model_manager=model_manager,
        output_dir=Path("./output"),
        analyze_figures=True,  # Enable figure extraction and analysis
        workers=2,  # Parallel processing for speed
    )

    # Process PDF with figures
    pdf_path = Path("test_document.pdf")
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        return

    print(f"Processing: {pdf_path}")
    print("- Extracting text from pages...")
    print("- Extracting embedded figures...")
    print("- Analyzing figures with AI...\n")

    result = processor.process_file(pdf_path)

    print(f"Pages: {result.page_count}")
    print(f"Time: {result.processing_time:.2f}s")

    # Save result
    output_path = processor.save_result(result)
    print(f"Output: {output_path}")

    # Show figures directory
    figures_dir = Path("./output") / f"{pdf_path.stem}_figures"
    if figures_dir.exists():
        figures = list(figures_dir.glob("*"))
        print(f"Figures: {len(figures)} saved to {figures_dir}/")
        for fig in figures:
            print(f"  - {fig.name}")

    model_manager.unload_model()
    print("\nDone!")


if __name__ == "__main__":
    main()
