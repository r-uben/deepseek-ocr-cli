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
    figure_filename,
    figure_markdown_link,
    figures_dir_for,
    is_truncated,
    is_within_output_root,
    iter_input_files,
    markdown_path_for,
    relative_key,
    resolve_output_root,
    run_fingerprint,
    safe_checksum,
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
    raw: bool = False,
) -> DocResult:
    """OCR an ordered list of page images into a DocResult.

    Each page is an independent backend call so page boundaries and per-page
    failures are real. A page that errors OR returns empty/whitespace text is
    recorded in ``page_errors`` (empty != success) and gets an explicit failure
    marker in its slot, keeping the page count and ``## Page N`` numbering aligned.

    Truncation is a per-page failure too: when the backend reports it stopped on
    a length/token limit (``finish_reason`` in the contract's
    :data:`TRUNCATION_FINISH_REASONS`), the page text is non-empty but
    incomplete, so recording it as ``completed`` would be silent content loss.
    Such a page is recorded in ``page_errors`` (driving status=partial/failed)
    while keeping the recovered-so-far text in its slot rather than discarding it.

    ``raw`` opts out of the backend's ``clean_ocr_output`` post-processing so the
    model's verbatim output (including ``[[d,d,d,d]]`` boxes and ``<...>`` spans)
    is preserved for faithful extraction.
    """
    pages: list[str] = []
    page_errors: dict[int, str] = {}
    for idx, image in enumerate(images, start=1):
        try:
            text, finish_reason = _ocr_one_image(backend, image, prompt, task, raw)
            if not text.strip():
                raise ValueError("empty OCR response (no text returned)")
            # Per-page truncation: 1 image -> 1 page, so the page-shortfall signal
            # does not apply; the finish_reason length-limit signal does. Keep the
            # recovered-so-far text but flag the page so it is not a silent success.
            if is_truncated(finish_reason, parsed_pages=1, actual_pages=1):
                page_errors[idx] = (
                    f"truncated response (finish_reason={finish_reason!r}); "
                    "page content is incomplete"
                )
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


def _ocr_one_image(
    backend: Backend,
    image: Image.Image,
    prompt: str | None,
    task: str,
    raw: bool = False,
) -> tuple[str, object | None]:
    """Call the backend, returning ``(text, finish_reason)``.

    Backends that surface a completion ``finish_reason`` override
    :meth:`Backend.process_image_with_meta`; the default falls back to
    :meth:`Backend.process_image` with ``finish_reason=None`` (no truncation
    signal), keeping the contract intact for engines/mocks that do not report it.
    """
    return backend.process_image_with_meta(image, prompt=prompt, task=task, return_raw=raw)


def _ocr_one_document(
    doc: Path,
    backend: Backend,
    task: str,
    prompt: str | None,
    dpi: int,
    raw: bool = False,
) -> tuple[DocResult, list[Image.Image] | None]:
    """OCR one document. Returns the result plus rendered images (for figure reuse)."""
    start = time.time()
    try:
        if doc.is_dir():
            image_paths = _gather_images(doc)
            images = [load_image(p) for p in image_paths]
            return _ocr_pages(doc, images, backend, task, prompt, start, raw), None
        if doc.suffix.lower() in _DOC_SUFFIXES:
            images = _render_pdf(doc, dpi)
            return _ocr_pages(doc, images, backend, task, prompt, start, raw), images
        if doc.suffix.lower() in _IMAGE_SUFFIXES:
            images = [load_image(doc)]
            return _ocr_pages(doc, images, backend, task, prompt, start, raw), None
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
    figures_dir = ensure_dir(figures_dir_for(doc_dir))
    lines = ["\n\n---\n\n# Figures\n"]
    for fig in figures:
        filename = figure_filename(fig.figure_num, fig.page_num)
        fig_path = figures_dir / filename
        fig.image.save(fig_path, "PNG")
        fig.saved_path = fig_path
        try:
            fig.description = backend.describe_figure(fig.image)
        except Exception as exc:
            logger.error("figure description failed (page %d): %s", fig.page_num, exc)
            fig.description = f"[Analysis Error: {exc}]"
        lines.append(f"\n## Figure {fig.figure_num} (Page {fig.page_num})\n")
        lines.append(figure_markdown_link(fig.figure_num, fig.page_num) + "\n")
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


def _safe_doc_checksum(source: Path) -> str | None:
    """Like :func:`_doc_checksum`, but ``None`` instead of raising on an OSError.

    Mirrors the contract's :func:`safe_checksum` for deepseek's two input shapes:
    a single file (delegating to ``safe_checksum``) and an image directory (which
    may still raise if an image is deleted/unreadable between discovery and
    processing). A ``None`` result means "input unreadable now" — the idempotency
    pre-check must NOT skip (it processes, and the per-doc catch-all in
    :func:`_ocr_one_document` records that one doc ``status=failed`` and the batch
    CONTINUES, rather than an ``OSError`` aborting the whole run — the SYS-02
    "one bad file aborts the batch" failure mode).
    """
    if source.is_file():
        # safe_checksum already returns None on OSError for the single-file case.
        return safe_checksum(source)
    try:
        return _doc_checksum(source)
    except OSError:
        return None


def _build_doc_metadata(
    result: DocResult,
    markdown_path: Path,
    output_root: Path,
    backend: Backend,
    fingerprint: str | None,
) -> DocMetadata:
    """Assemble the per-document metadata record from a DocResult."""
    status = result.status
    error = None
    if status is not Status.COMPLETED:
        if result.page_errors:
            error = "; ".join(f"page {n}: {msg}" for n, msg in sorted(result.page_errors.items()))
        elif result.error:
            error = result.error
    # Tolerant checksum: if the source became unreadable mid-run we still persist a
    # status=failed record (empty checksum) rather than letting the failure-metadata
    # write itself throw and escape the per-doc error boundary.
    checksum = _safe_doc_checksum(result.source) or ""
    return DocMetadata(
        status=status,
        checksum=checksum,
        model=_backend_model(backend),
        backend=_backend_name(backend),
        processing_time=result.processing_time,
        timestamp=utc_timestamp(),
        output_path=str(markdown_path.relative_to(output_root)),
        pages=result.page_count,
        error=error,
        fingerprint=fingerprint,
    )


def _write_document(
    result: DocResult,
    output_root: Path,
    rel_key: str,
    backend: Backend,
    index: RootIndex,
    fingerprint: str | None,
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

    meta = _build_doc_metadata(result, markdown_path, output_root, backend, fingerprint)
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
    raw: bool = False,
) -> RunOutcome:
    """Process an input (file or directory of documents) through the contract.

    ``source`` is either a single document (a ``.pdf``, an image, or a directory of
    page images treated as ONE document) or a batch directory tree of such documents.
    Output goes to ``resolve_output_root(source, output_dir)`` — default
    ``<input-parent>/ocr/``; ``-o`` overrides; never required.

    ``raw`` opts out of ``clean_ocr_output`` so the model's verbatim text is kept.

    Returns a :class:`RunOutcome` whose ``exit_code`` is nonzero if any
    document/page failed (uniform across single-file and batch).
    """
    # Resolve the output root FIRST so discovery can exclude it. The canonical
    # default for a directory input is ``<input>/ocr/`` — INSIDE the scanned tree —
    # so a naive recursive walk would re-ingest the engine's own .md/figure outputs
    # as inputs on the next run. The contract's iter_input_files prunes that subtree.
    output_root = resolve_output_root(source, output_dir)

    documents, scan_root = _discover_documents(source, output_root)
    if not documents:
        raise ValueError(f"no documents found at {source}")

    if backend.model is None or backend.model is False:
        backend.load_model()

    output_root.mkdir(parents=True, exist_ok=True)
    index = RootIndex(output_root)
    # Run fingerprint keys the idempotency cache on everything that changes a
    # document's OUTPUT for a given input. The dedicated task/prompt params handle
    # the prompt selector; ``extra`` carries the remaining RESOLVED output-affecting
    # flags (raw, dpi, max_tokens, analyze_figures) so a re-run with any of them
    # changed reprocesses instead of silently reusing a stale cached result.
    #
    # When a custom --prompt is set the backends IGNORE --task (they only call
    # get_prompt(task) when prompt is None), so task is dropped from the fingerprint
    # to avoid needlessly reprocessing two same-prompt runs that differ only in task.
    fingerprint = run_fingerprint(
        model=_backend_model(backend),
        backend=_backend_name(backend),
        task=None if prompt is not None else task,
        prompt=prompt,
        extra={
            "raw": raw,
            "dpi": dpi,
            "max_tokens": getattr(backend, "max_tokens", None),
            "analyze_figures": analyze_figures,
        },
    )

    outcome = RunOutcome()
    for doc in documents:
        rel_key = relative_key(doc, scan_root)
        # Idempotency pre-check uses the SAFE checksum: an input that became
        # unreadable between discovery and processing yields None, which can never
        # match a recorded checksum, so the doc is NOT skipped — it falls through to
        # _ocr_one_document, whose catch-all records it status=failed and the batch
        # CONTINUES (no whole-run abort: the SYS-02 class the contract guards).
        pre_checksum = _safe_doc_checksum(doc)
        if (
            not reprocess
            and pre_checksum is not None
            and index.is_completed(rel_key, pre_checksum, fingerprint=fingerprint)
        ):
            logger.info("skip %s (already completed; use --reprocess)", rel_key)
            # Quiet mode emits one .md path per processed doc; a cached/resumed doc
            # is still "present", so emit its path too (compute + verify on disk).
            skip_md = markdown_path_for(doc_dir_for(output_root, rel_key), rel_key)
            outcome.add(
                Status.COMPLETED,
                output_path=str(skip_md) if skip_md.exists() else None,
            )
            continue

        result, _images = _ocr_one_document(doc, backend, task, prompt, dpi, raw)

        if (
            analyze_figures
            and result.status is not Status.FAILED
            and doc.suffix.lower() in _DOC_SUFFIXES
        ):
            doc_dir = doc_dir_for(output_root, rel_key)
            doc_dir.mkdir(parents=True, exist_ok=True)
            result.figures_markdown = _process_figures(doc, doc_dir, backend)

        meta, markdown_path = _write_document(
            result, output_root, rel_key, backend, index, fingerprint
        )
        outcome.add(
            meta.status,
            detail=None if meta.status is Status.COMPLETED else rel_key,
            output_path=str(markdown_path),
        )

    return outcome


def discover_documents(source: Path, output_dir: Path | None = None) -> list[Path]:
    """Public preview of what :func:`process` would OCR, in the SAME order.

    Resolves the output root exactly as the real run does and returns the list of
    source documents discovered under ``source`` (with the output-root subtree
    excluded). The CLI ``--dry-run`` uses this so the preview never diverges from
    the real run (the dry-run/real-run divergence the review flagged).
    """
    output_root = resolve_output_root(source, output_dir)
    documents, _scan_root = _discover_documents(source, output_root)
    return documents


def _is_image_dir_document(source: Path, output_root: Path) -> bool:
    """True when ``source`` is unambiguously ONE document made of page images.

    socr renders a PDF to ``page_0001.png ...`` and passes the directory, expecting
    deepseek to OCR it as a single document. That is only unambiguous when the
    directory contains ONLY images: no PDFs and no subdirectories. The moment a PDF
    or a subdirectory is present, ``source`` is a batch tree (the papers-library use
    case) and must be walked recursively — treating it as one image-document would
    silently drop every PDF and subdir (the HIGH bug this guard fixes).

    Classification is OUTPUT-ROOT-AWARE so it is STABLE across runs. With the
    default output root ``<input>/ocr/`` nested inside the scanned directory, the
    first run creates that ``ocr/`` subtree; without this exclusion, a re-run would
    see the new subdir, return False, and silently reclassify the SAME folder as a
    batch tree (the run-1 aggregated ``scan/scan.md`` orphaned, per-image
    ``page_0001_png/`` trees emitted on run 2, broken resume). The resolved output
    root — and its own ``.md``/figure/metadata outputs — are excluded from the
    scan so a first run and a re-run classify the same input identically.
    """
    if not source.is_dir():
        return False
    has_image = False
    for p in source.iterdir():
        # Skip the engine's own output subtree (default root nests in the input):
        # it must not shift classification between runs.
        if is_within_output_root(p, output_root):
            continue
        if p.is_dir():
            return False
        suffix = p.suffix.lower()
        if suffix in _DOC_SUFFIXES:
            return False
        if suffix in _IMAGE_SUFFIXES:
            has_image = True
    return has_image


def _discover_documents(source: Path, output_root: Path) -> tuple[list[Path], Path]:
    """Return ``(documents, scan_root)`` for ``source``, excluding the output root.

    * A bare ``.pdf`` or image file is one document (scan root = its parent).
    * A directory that contains ONLY page images (no PDFs, no subdirs) is a SINGLE
      image-dir document keyed by its folder name (scan root = its parent).
    * Otherwise the directory is a batch tree: every ``.pdf``/image under it is one
      document, discovered via the contract's :func:`iter_input_files`, which prunes
      the resolved ``output_root`` subtree so prior outputs are never re-ingested.
    """
    if source.is_file():
        return [source], source.parent
    if not source.is_dir():
        return [], source
    if _is_image_dir_document(source, output_root):
        # The image-dir document keys on its own folder name (no file stem).
        return [source], source.parent
    suffixes = _DOC_SUFFIXES | _IMAGE_SUFFIXES
    documents = list(iter_input_files(source, output_root, suffixes=suffixes))
    return documents, source
