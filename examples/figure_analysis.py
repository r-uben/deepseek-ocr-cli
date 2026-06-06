#!/usr/bin/env python
"""Example: Extract and analyze embedded figures from PDFs.

With ``analyze_figures=True``, embedded figures are extracted, saved under
``<output_root>/<stem>/figures/figure_<N>_page<P>.png`` (canon naming), and an
AI description is appended to the document markdown.
"""

from pathlib import Path

from deepseek_ocr import create_backend, process


def main() -> None:
    """Demonstrate figure extraction and analysis."""
    print("DeepSeek OCR - Figure Analysis Example\n")

    backend = create_backend(backend_type="ollama", model_name="deepseek-ocr")
    backend.load_model()

    output_dir = Path("./output")
    pdf_path = Path("test_document.pdf")
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        backend.unload_model()
        return

    print(f"Processing: {pdf_path}")
    outcome = process(pdf_path, backend, output_dir=output_dir, analyze_figures=True)

    for md_path in outcome.outputs:
        print(f"Output: {md_path}")
    print(f"completed={outcome.completed} failed={outcome.failed}")

    # Figures are saved in <output_root>/<stem>/figures/.
    figures_dir = output_dir / pdf_path.stem / "figures"
    if figures_dir.exists():
        figures = list(figures_dir.glob("*"))
        print(f"Figures: {len(figures)} saved to {figures_dir}/")
        for fig in figures:
            print(f"  - {fig.name}")

    backend.unload_model()
    print("\nDone!")


if __name__ == "__main__":
    main()
