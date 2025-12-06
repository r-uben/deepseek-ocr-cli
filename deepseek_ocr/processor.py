"""Document processing and batch handling for DeepSeek OCR."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
    ):
        self.model_manager = model_manager or ModelManager()
        self.output_dir = output_dir or settings.output_dir
        self.extract_images = extract_images or settings.extract_images
        self.include_metadata = include_metadata and settings.include_metadata

        ensure_dir(self.output_dir)
        logger.info(f"OCRProcessor initialized with output_dir: {self.output_dir}")

    def _pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """Convert PDF pages to images (200 DPI)."""
        logger.debug(f"Converting PDF to images: {pdf_path}")

        try:
            images = []
            pdf_document = fitz.open(pdf_path)

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]

                zoom = 200 / 72
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

                outputs = []
                page_iterator = (
                    tqdm(enumerate(images, 1), total=page_count, desc="OCR pages", unit="page")
                    if show_progress and page_count > 1
                    else enumerate(images, 1)
                )
                for idx, image in page_iterator:
                    logger.debug(f"Processing page {idx}/{page_count}")
                    output = self.model_manager.process_image(image, prompt=prompt)
                    outputs.append(f"## Page {idx}\n\n{output}")

                output_text = "\n\n".join(outputs)

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
