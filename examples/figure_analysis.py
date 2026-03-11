#!/usr/bin/env python
"""Example: Extract and analyze embedded figures from PDFs."""

from pathlib import Path
from deepseek_ocr import create_backend, OCRProcessor


def main() -> None:
    """Demonstrate figure extraction and analysis."""
    print("DeepSeek OCR - Figure Analysis Example\n")

    backend = create_backend(backend_type="ollama", model_name="deepseek-ocr")
    backend.load_model()

    # Create processor with figure analysis enabled
    processor = OCRProcessor(
        backend=backend,
        output_dir=Path("./output"),
        analyze_figures=True,
        workers=2,
    )

    pdf_path = Path("test_document.pdf")
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        return

    print(f"Processing: {pdf_path}")
    result = processor.process_file(pdf_path)

    print(f"Pages: {result.page_count}")
    print(f"Time: {result.processing_time:.2f}s")

    output_path = processor.save_result(result)
    print(f"Output: {output_path}")

    # Figures are saved in output/doc_name/figures/
    figures_dir = Path("./output") / pdf_path.stem / "figures"
    if figures_dir.exists():
        figures = list(figures_dir.glob("*"))
        print(f"Figures: {len(figures)} saved to {figures_dir}/")
        for fig in figures:
            print(f"  - {fig.name}")

    backend.unload_model()
    print("\nDone!")


if __name__ == "__main__":
    main()
