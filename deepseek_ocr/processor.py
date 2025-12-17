"""Document processing and batch handling for DeepSeek OCR."""

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

from deepseek_ocr.config import settings
from deepseek_ocr.model import ModelManager
from deepseek_ocr.utils import (
    collect_files,
    ensure_dir,
    is_pdf_file,
    load_image,
    sanitize_filename,
)

logger = logging.getLogger(__name__)


@dataclass
class FigureInfo:
    """Container for extracted figure information."""

    page_num: int
    figure_num: int
    image: Image.Image
    width: int
    height: int
    format: str
    context: str = ""
    description: str = ""
    saved_path: Optional[Path] = None


class OCRResult:
    """Container for OCR processing results."""

    def __init__(
        self,
        input_path: Path,
        output_text: str,
        page_count: int = 1,
        processing_time: float = 0.0,
        metadata: Optional[Dict] = None,
    ):
        self.input_path = input_path
        self.output_text = output_text
        self.page_count = page_count
        self.processing_time = processing_time
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_markdown(self, include_metadata: bool = True) -> str:
        lines = []

        if include_metadata:
            lines.append("---")
            lines.append(f"source: {self.input_path}")
            lines.append(f"processed: {self.timestamp.isoformat()}")
            lines.append(f"pages: {self.page_count}")
            lines.append(f"processing_time: {self.processing_time:.2f}s")
            for key, value in self.metadata.items():
                lines.append(f"{key}: {value}")
            lines.append("---")
            lines.append("")

        lines.append(self.output_text)

        return "\n".join(lines)


class OCRProcessor:
    """Handles OCR processing for documents and images."""

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        output_dir: Optional[Path] = None,
        extract_images: bool = False,
        include_metadata: bool = True,
        dpi: int = 200,
        workers: int = 1,
        analyze_figures: bool = False,
    ):
        self.model_manager = model_manager or ModelManager()
        self.output_dir = output_dir or settings.output_dir
        self.extract_images = extract_images or settings.extract_images
        self.include_metadata = include_metadata and settings.include_metadata
        self.dpi = dpi
        self.workers = max(1, workers)  # Ensure at least 1 worker
        self.analyze_figures = analyze_figures

        ensure_dir(self.output_dir)
        logger.info(f"OCRProcessor initialized with output_dir: {self.output_dir}, workers: {self.workers}")

    def _pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        logger.debug(f"Converting PDF to images: {pdf_path} at {self.dpi} DPI")

        try:
            images = []
            pdf_document = fitz.open(pdf_path)

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]

                zoom = self.dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)

                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)

                logger.debug(f"Converted page {page_num + 1}/{len(pdf_document)}")

            pdf_document.close()
            logger.info(f"Converted {len(images)} pages from {pdf_path.name}")

            return images

        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF {pdf_path}: {e}")

    def _save_images(self, images: List[Image.Image], base_name: str) -> Path:
        images_dir = self.output_dir / f"{base_name}_images"
        ensure_dir(images_dir)

        for idx, image in enumerate(images, 1):
            image_path = images_dir / f"page_{idx:04d}.png"
            image.save(image_path, "PNG")
            logger.debug(f"Saved image: {image_path}")

        logger.info(f"Saved {len(images)} images to {images_dir}")
        return images_dir

    def _extract_figures_from_pdf(self, pdf_path: Path) -> List[FigureInfo]:
        """Extract embedded figures/images from a PDF."""
        figures: List[FigureInfo] = []

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                images = page.get_images(full=True)

                for img_idx, img_info in enumerate(images):
                    xref = img_info[0]

                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]

                        # Convert to PIL Image
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        if pil_image.mode != "RGB":
                            pil_image = pil_image.convert("RGB")

                        # Get context from surrounding text
                        context = ""
                        img_rects = page.get_image_rects(xref)
                        if img_rects:
                            rect = img_rects[0]
                            # Expand rect to capture nearby text
                            expanded_rect = fitz.Rect(
                                max(0, rect.x0 - 50),
                                max(0, rect.y0 - 150),
                                rect.x1 + 50,
                                rect.y1 + 150
                            )
                            context = page.get_text("text", clip=expanded_rect).strip()

                        # Fallback to full page text if no local context
                        if not context:
                            context = page_text[:500].strip() if page_text else ""

                        figure = FigureInfo(
                            page_num=page_num + 1,
                            figure_num=img_idx + 1,
                            image=pil_image,
                            width=base_image["width"],
                            height=base_image["height"],
                            format=base_image["ext"],
                            context=context,
                        )
                        figures.append(figure)

                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_idx + 1} from page {page_num + 1}: {e}")
                        continue

            doc.close()
            logger.info(f"Extracted {len(figures)} figures from {pdf_path.name}")

        except Exception as e:
            logger.error(f"Failed to extract figures from {pdf_path}: {e}")

        return figures

    def _save_figures(self, figures: List[FigureInfo], base_name: str) -> Path:
        """Save extracted figures to disk."""
        figures_dir = self.output_dir / f"{base_name}_figures"
        ensure_dir(figures_dir)

        for fig in figures:
            filename = f"page{fig.page_num}_fig{fig.figure_num}.{fig.format}"
            fig_path = figures_dir / filename
            fig.image.save(fig_path)
            fig.saved_path = fig_path
            logger.debug(f"Saved figure: {fig_path}")

        logger.info(f"Saved {len(figures)} figures to {figures_dir}")
        return figures_dir

    def _analyze_single_figure(
        self, figure: FigureInfo
    ) -> Tuple[int, int, str, Optional[str]]:
        """Analyze a single figure. Returns (page_num, fig_num, description, error)."""
        try:
            # Build prompt with context
            if figure.context:
                prompt = (
                    f"This figure appears in a document with the following context:\n"
                    f"---\n{figure.context[:500]}\n---\n\n"
                    f"Describe what this figure shows. Include details about any charts, "
                    f"graphs, diagrams, or visual elements. Explain what it represents "
                    f"in the context of the document."
                )
            else:
                prompt = (
                    "Describe this figure in detail. Include information about any charts, "
                    "graphs, diagrams, tables, or visual elements. Explain what it represents."
                )

            description = self.model_manager.process_image(figure.image, prompt=prompt)
            return (figure.page_num, figure.figure_num, description, None)

        except Exception as e:
            logger.error(f"Failed to analyze figure {figure.figure_num} on page {figure.page_num}: {e}")
            return (figure.page_num, figure.figure_num, "", str(e))

    def _analyze_figures(
        self, figures: List[FigureInfo], show_progress: bool = True
    ) -> List[FigureInfo]:
        """Analyze all figures and populate their descriptions."""
        if not figures:
            return figures

        if self.workers == 1:
            # Sequential processing
            fig_iterator = (
                tqdm(figures, desc="Analyzing figures", unit="fig")
                if show_progress and len(figures) > 1
                else figures
            )
            for fig in fig_iterator:
                _, _, description, error = self._analyze_single_figure(fig)
                fig.description = description if not error else f"[Analysis Error: {error}]"
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(self._analyze_single_figure, fig): fig
                    for fig in figures
                }

                if show_progress and len(figures) > 1:
                    pbar = tqdm(total=len(figures), desc=f"Analyzing figures ({self.workers}w)", unit="fig")
                else:
                    pbar = None

                for future in as_completed(futures):
                    fig = futures[future]
                    _, _, description, error = future.result()
                    fig.description = description if not error else f"[Analysis Error: {error}]"
                    if pbar:
                        pbar.update(1)

                if pbar:
                    pbar.close()

        return figures

    def _figures_to_markdown(self, figures: List[FigureInfo]) -> str:
        """Convert figure analyses to markdown."""
        if not figures:
            return ""

        lines = ["\n\n---\n\n# Figures\n"]

        for fig in figures:
            lines.append(f"\n## Figure {fig.figure_num} (Page {fig.page_num})\n")
            if fig.saved_path:
                lines.append(f"*Saved to: {fig.saved_path.name}*\n")
            lines.append(f"*Size: {fig.width}x{fig.height} ({fig.format})*\n")
            lines.append(f"\n{fig.description}\n")

        return "\n".join(lines)

    def _process_single_page(
        self, page_data: Tuple[int, Image.Image], prompt: Optional[str] = None
    ) -> Tuple[int, str, Optional[str]]:
        """Process a single page image. Returns (page_num, text, error)."""
        page_num, image = page_data
        try:
            output = self.model_manager.process_image(image, prompt=prompt)
            return (page_num, output, None)
        except Exception as e:
            logger.error(f"Failed to process page {page_num}: {e}")
            return (page_num, "", str(e))

    def process_file(
        self, file_path: Path, prompt: Optional[str] = None, show_progress: bool = True
    ) -> OCRResult:
        """Process a single file (image or PDF)."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing file: {file_path}")
        start_time = datetime.now()

        if self.model_manager.model is None:
            self.model_manager.load_model()

        try:
            if is_pdf_file(file_path):
                images = self._pdf_to_images(file_path)
                page_count = len(images)

                if self.extract_images:
                    base_name = sanitize_filename(file_path.stem)
                    self._save_images(images, base_name)

                # Prepare page data: (page_num, image) tuples
                page_data = [(idx, img) for idx, img in enumerate(images, 1)]

                if self.workers == 1:
                    # Sequential processing (original behavior)
                    outputs = []
                    page_iterator = (
                        tqdm(page_data, total=page_count, desc="OCR pages", unit="page")
                        if show_progress and page_count > 1
                        else page_data
                    )
                    for idx, image in page_iterator:
                        logger.debug(f"Processing page {idx}/{page_count}")
                        output = self.model_manager.process_image(image, prompt=prompt)
                        outputs.append(f"## Page {idx}\n\n{output}")
                else:
                    # Parallel processing
                    results_dict: Dict[int, str] = {}
                    errors: List[str] = []

                    with ThreadPoolExecutor(max_workers=self.workers) as executor:
                        futures = {
                            executor.submit(self._process_single_page, pd, prompt): pd[0]
                            for pd in page_data
                        }

                        if show_progress and page_count > 1:
                            pbar = tqdm(total=page_count, desc=f"OCR pages ({self.workers}w)", unit="page")
                        else:
                            pbar = None

                        for future in as_completed(futures):
                            page_num, text, error = future.result()
                            if error:
                                errors.append(f"Page {page_num}: {error}")
                                results_dict[page_num] = f"[OCR Error: {error}]"
                            else:
                                results_dict[page_num] = text
                            if pbar:
                                pbar.update(1)

                        if pbar:
                            pbar.close()

                    if errors:
                        logger.warning(f"Errors on {len(errors)} pages: {errors}")

                    # Reassemble in page order
                    outputs = [
                        f"## Page {i}\n\n{results_dict[i]}"
                        for i in sorted(results_dict.keys())
                    ]

                output_text = "\n\n".join(outputs)

                # Extract and analyze figures if enabled
                if self.analyze_figures:
                    figures = self._extract_figures_from_pdf(file_path)
                    if figures:
                        base_name = sanitize_filename(file_path.stem)
                        self._save_figures(figures, base_name)
                        self._analyze_figures(figures, show_progress=show_progress)
                        output_text += self._figures_to_markdown(figures)

            else:
                image = load_image(file_path)
                output_text = self.model_manager.process_image(image, prompt=prompt)
                page_count = 1

            processing_time = (datetime.now() - start_time).total_seconds()

            metadata = {
                "model": self.model_manager.model_name,
                "backend": "ollama",
            }

            result = OCRResult(
                input_path=file_path,
                output_text=output_text,
                page_count=page_count,
                processing_time=processing_time,
                metadata=metadata,
            )

            logger.info(
                f"Processed {file_path.name} - {page_count} pages in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            raise RuntimeError(f"Failed to process {file_path}: {e}")

    def process_batch(
        self,
        input_path: Path,
        recursive: bool = False,
        prompt: Optional[str] = None,
        show_progress: bool = True,
    ) -> List[OCRResult]:
        """Process multiple files from a directory."""
        files = collect_files(input_path, recursive=recursive)
        logger.info(f"Found {len(files)} files to process")

        if self.model_manager.model is None:
            self.model_manager.load_model()

        results = []
        iterator = tqdm(files, desc="Processing files") if show_progress else files

        for file_path in iterator:
            try:
                result = self.process_file(file_path, prompt=prompt)
                results.append(result)
                self.save_result(result)

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue

        logger.info(f"Successfully processed {len(results)}/{len(files)} files")
        return results

    def save_result(self, result: OCRResult, output_path: Optional[Path] = None) -> Path:
        if output_path is None:
            base_name = sanitize_filename(result.input_path.stem)
            output_path = self.output_dir / f"{base_name}.md"

        ensure_dir(output_path.parent)

        markdown_content = result.to_markdown(include_metadata=self.include_metadata)
        output_path.write_text(markdown_content, encoding="utf-8")

        logger.info(f"Saved result to: {output_path}")
        return output_path
