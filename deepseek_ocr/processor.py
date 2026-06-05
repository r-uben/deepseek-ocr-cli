"""Render inputs to page images, OCR each via a backend, write canonical Markdown.

deepseek is a *local, page-based* engine: it renders a PDF (or reads a directory of
page images) to per-page images and OCRs each page through one of its backends
(ollama / vllm). This module owns *how OCR happens* and deepseek's figure extraction;
the shared ``ocr-output-contract`` package owns *where the bytes go* and what shape
the metadata takes.

The per-page text list this module produces is fed straight into the contract's
:func:`assemble_pages`, so every page of a document lands in ONE
``<root>/<rel/dir>/<stem>/<stem>.md`` under ``## Page N`` headers. An empty/whitespace
page response is a per-page FAILURE recorded in the sidecar, never a silent 0-byte
success reported as ok. The Markdown body carries NO YAML frontmatter — all provenance
lives in the per-document ``metadata.json`` sidecar (canon: single source of truth).
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import fitz  # PyMuPDF
from ocr_output_contract import (
    DocMetadata,
    RootIndex,
    RunOutcome,
    Status,
    assemble_pages,
    doc_dir_for,
    markdown_path_for,
    relative_key,
    resolve_output_root,
    sha256_checksum,
    utc_timestamp,
    write_doc_metadata,
)
from PIL import Image

from deepseek_ocr.utils import (
    IMAGE_EXTENSIONS,
    ensure_dir,
    load_image,
)

if TYPE_CHECKING:
    from deepseek_ocr.backends.base import Backend

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {ext.lower() for ext in IMAGE_EXTENSIONS}
#: Inputs deepseek treats as a single source document to render page-by-page.
_DOC_SUFFIXES = {".pdf"}


@dataclass
class FigureInfo:
    """Container for an extracted embedded figure (opt-in via --analyze-figures)."""

    page_num: int
    figure_num: int
    image: Image.Image
    width: int
    height: int
    format: str
    context: str = ""
    description: str = ""
    saved_path: Path | None = None


@dataclass
class DocResult:
    """Result of OCR'ing one source document (a PDF or an image directory).

    ``pages`` holds the per-page markdown in order; ``page_errors`` maps a
    1-indexed page number to its error string for any page that failed (or whose
    model response was empty). ``status`` maps to the contract enum.
    """

    source: Path
    pages: list[str]
    processing_time: float = 0.0
    page_errors: dict[int, str] = field(default_factory=dict)
    error: str | None = None
    figures_markdown: str = ""

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def status(self) -> Status:
        """``completed`` = every page ok; ``partial`` = some ok; ``failed`` = none."""
        if self.pages and not self.page_errors and not self.error:
            return Status.COMPLETED
        succeeded = self.page_count - len(self.page_errors)
        if succeeded > 0:
            return Status.PARTIAL
        return Status.FAILED


# ---------------------------------------------------------------------------
# Page OCR (image -> page-text; deepseek-owned). Output shape is the contract's job.
# ---------------------------------------------------------------------------


def _render_pdf(pdf_path: Path, dpi: int) -> list[Image.Image]:
    """Render each PDF page to an in-memory RGB PIL image."""
    images: list[Image.Image] = []
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            pix = doc[page_num].get_pixmap(matrix=mat)
            images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return images


def _gather_images(directory: Path) -> list[Path]:
    images = sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"no page images found in {directory}")
    return images


def _ocr_pages(
    source: Path,
    images: list[Image.Image],
    backend: Backend,
    task: str,
    prompt: str | None,
    start: float,
) -> DocResult:
    """OCR an ordered list of page images into a DocResult.

    Each page is an independent backend call so page boundaries and per-page
    failures are real. A page that errors OR returns empty/whitespace text is
    recorded in ``page_errors`` (empty != success) and gets an explicit failure
    marker in its slot, keeping the page count and ``## Page N`` numbering aligned.
    """
    pages: list[str] = []
    page_errors: dict[int, str] = {}
    for idx, image in enumerate(images, start=1):
        try:
            text = backend.process_image(image, prompt=prompt, task=task)
            if not text.strip():
                raise ValueError("empty OCR response (no text returned)")
            pages.append(text)
        except Exception as exc:
            logger.error("OCR failed for page %d of %s: %s", idx, source.name, exc)
            page_errors[idx] = str(exc)
            pages.append(f"*[OCR failed for page {idx}]*")
    return DocResult(
        source=source,
        pages=pages,
        processing_time=time.time() - start,
        page_errors=page_errors,
    )


def _ocr_one_document(
    doc: Path,
    backend: Backend,
    task: str,
    prompt: str | None,
    dpi: int,
) -> tuple[DocResult, list[Image.Image] | None]:
    """OCR one document. Returns the result plus rendered images (for figure reuse)."""
    start = time.time()
    try:
        if doc.is_dir():
            image_paths = _gather_images(doc)
            images = [load_image(p) for p in image_paths]
            return _ocr_pages(doc, images, backend, task, prompt, start), None
        if doc.suffix.lower() in _DOC_SUFFIXES:
            images = _render_pdf(doc, dpi)
            return _ocr_pages(doc, images, backend, task, prompt, start), images
        if doc.suffix.lower() in _IMAGE_SUFFIXES:
            images = [load_image(doc)]
            return _ocr_pages(doc, images, backend, task, prompt, start), None
        raise ValueError(f"unsupported input: {doc} (expected a .pdf, image, or directory)")
    except Exception as exc:
        logger.error("could not process %s: %s", doc, exc)
        return (
            DocResult(source=doc, pages=[], processing_time=time.time() - start, error=str(exc)),
            None,
        )


# ---------------------------------------------------------------------------
# Figure extraction (deepseek-owned, opt-in). Written under the contract doc dir.
# ---------------------------------------------------------------------------


def _extract_figures_from_pdf(pdf_path: Path) -> list[FigureInfo]:
    """Extract embedded figures/images from a PDF."""
    figures: list[FigureInfo] = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                for img_idx, img_info in enumerate(page.get_images(full=True)):
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        pil_image = Image.open(io.BytesIO(base_image["image"]))
                        if pil_image.mode != "RGB":
                            pil_image = pil_image.convert("RGB")
                        context = page_text[:500].strip() if page_text else ""
                        figures.append(
                            FigureInfo(
                                page_num=page_num + 1,
                                figure_num=img_idx + 1,
                                image=pil_image,
                                width=base_image["width"],
                                height=base_image["height"],
                                format=base_image["ext"],
                                context=context,
                            )
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to extract image %d on page %d: %s",
                            img_idx + 1,
                            page_num + 1,
                            exc,
                        )
    except Exception as exc:
        logger.error("Failed to extract figures from %s: %s", pdf_path, exc)
    return figures


def _process_figures(doc: Path, doc_dir: Path, backend: Backend) -> str:
    """Extract, save, and describe embedded figures; return their markdown block.

    Figures land under ``<doc_dir>/figures/figure_<N>_page<P>.png`` (canon naming)
    with resolvable links in the returned markdown.
    """
    figures = _extract_figures_from_pdf(doc)
    if not figures:
        return ""
    figures_dir = ensure_dir(doc_dir / "figures")
    lines = ["\n\n---\n\n# Figures\n"]
    for fig in figures:
        filename = f"figure_{fig.figure_num}_page{fig.page_num}.png"
        fig_path = figures_dir / filename
        fig.image.save(fig_path, "PNG")
        fig.saved_path = fig_path
        try:
            fig.description = backend.describe_figure(fig.image)
        except Exception as exc:
            logger.error("figure description failed (page %d): %s", fig.page_num, exc)
            fig.description = f"[Analysis Error: {exc}]"
        lines.append(f"\n## Figure {fig.figure_num} (Page {fig.page_num})\n")
        lines.append(f"![Figure {fig.figure_num}](./figures/{filename})\n")
        lines.append(f"*Size: {fig.width}x{fig.height} ({fig.format})*\n")
        lines.append(f"\n{fig.description}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output writing (all routed through the ocr-output-contract package)
# ---------------------------------------------------------------------------


def _backend_model(backend: Backend) -> str:
    return getattr(backend, "model_name", "") or ""


def _backend_name(backend: Backend) -> str:
    return getattr(backend, "backend_name", "") or ""


def _doc_checksum(source: Path) -> str:
    """Content checksum for idempotency. Hash a PDF/image's bytes, or an image dir's."""
    if source.is_file():
        return sha256_checksum(source)
    h = hashlib.sha256()
    for img in _gather_images(source):
        h.update(img.name.encode("utf-8"))
        with open(img, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _build_doc_metadata(
    result: DocResult,
    markdown_path: Path,
    output_root: Path,
    backend: Backend,
) -> DocMetadata:
    """Assemble the per-document metadata record from a DocResult."""
    status = result.status
    error = None
    if status is not Status.COMPLETED:
        if result.page_errors:
            error = "; ".join(f"page {n}: {msg}" for n, msg in sorted(result.page_errors.items()))
        elif result.error:
            error = result.error
    return DocMetadata(
        status=status,
        checksum=_doc_checksum(result.source),
        model=_backend_model(backend),
        backend=_backend_name(backend),
        processing_time=result.processing_time,
        timestamp=utc_timestamp(),
        output_path=str(markdown_path.relative_to(output_root)),
        pages=result.page_count,
        error=error,
    )


def _write_document(
    result: DocResult,
    output_root: Path,
    rel_key: str,
    backend: Backend,
    index: RootIndex,
) -> tuple[DocMetadata, Path]:
    """Write the aggregated markdown + BOTH metadata levels for one document.

    Output is always written (even on failure) so failures are recorded with
    ``status=failed`` per the canon. The single ``<stem>/<stem>.md`` aggregates
    every page under ``## Page N`` headers. NO YAML frontmatter — provenance is in
    the sidecar only.
    """
    doc_dir = doc_dir_for(output_root, rel_key)
    doc_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = markdown_path_for(doc_dir, rel_key)

    body = assemble_pages(result.pages) if result.pages else "*[OCR Failed]*\n"
    body += result.figures_markdown
    markdown_path.write_text(body, encoding="utf-8")

    meta = _build_doc_metadata(result, markdown_path, output_root, backend)
    write_doc_metadata(doc_dir, rel_key, meta)
    index.record(rel_key, meta)
    return meta, markdown_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def process(
    source: Path,
    backend: Backend,
    dpi: int = 200,
    task: str = "convert",
    prompt: str | None = None,
    output_dir: Path | None = None,
    reprocess: bool = False,
    analyze_figures: bool = False,
) -> RunOutcome:
    """Process an input (file or directory of documents) through the contract.

    ``source`` is either a single document (a ``.pdf``, an image, or a directory of
    page images treated as ONE document) or a batch directory tree of such documents.
    Output goes to ``resolve_output_root(source, output_dir)`` — default
    ``<input-parent>/ocr/``; ``-o`` overrides; never required.

    Returns a :class:`RunOutcome` whose ``exit_code`` is nonzero if any
    document/page failed (uniform across single-file and batch).
    """
    documents = _discover_documents(source)
    if not documents:
        raise ValueError(f"no documents found at {source}")

    if backend.model is None or backend.model is False:
        backend.load_model()

    output_root = resolve_output_root(source, output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    # A single image-dir document keys on its own folder name (no file stem);
    # a batch tree keys each input document input-relative to the tree root.
    single_image_dir = len(documents) == 1 and documents[0] == source and source.is_dir()
    scan_root = source.parent if (source.is_file() or single_image_dir) else source
    index = RootIndex(output_root)

    outcome = RunOutcome()
    for doc in documents:
        rel_key = relative_key(doc, scan_root)
        if not reprocess and index.is_completed(rel_key, _doc_checksum(doc)):
            logger.info("skip %s (already completed; use --reprocess)", rel_key)
            outcome.add(Status.COMPLETED)
            continue

        result, _images = _ocr_one_document(doc, backend, task, prompt, dpi)

        if (
            analyze_figures
            and result.status is not Status.FAILED
            and doc.suffix.lower() in _DOC_SUFFIXES
        ):
            doc_dir = doc_dir_for(output_root, rel_key)
            doc_dir.mkdir(parents=True, exist_ok=True)
            result.figures_markdown = _process_figures(doc, doc_dir, backend)

        meta, markdown_path = _write_document(result, output_root, rel_key, backend, index)
        outcome.add(
            meta.status,
            detail=None if meta.status is Status.COMPLETED else rel_key,
            output_path=str(markdown_path),
        )

    return outcome


def _discover_documents(source: Path) -> list[Path]:
    """Return the list of source documents under ``source``.

    A bare ``.pdf`` or image is one document. A directory is treated as a SINGLE
    image-dir document when it directly contains page images (socr renders a PDF to
    PNGs and passes the dir); otherwise it is a batch tree and every ``.pdf``/image
    under it (recursively) is one document.
    """
    if source.is_file():
        return [source]
    if source.is_dir():
        has_direct_images = any(
            p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES for p in source.iterdir()
        )
        if has_direct_images:
            return [source]
        supported = _DOC_SUFFIXES | _IMAGE_SUFFIXES
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in supported)
    return []
