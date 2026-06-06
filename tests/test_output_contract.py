"""Engine-level conformance: deepseek's REAL output vs the shared contract.

The contract *primitives* (path/key computation, page assembly, metadata writers,
the exit-code policy) are unit-tested inside the ``ocr-output-contract`` package
itself, so they are NOT re-tested here.

What stays here is the engine-side proof: run deepseek's actual processor (with a
mocked backend — no Ollama, no GPU, no network) over real inputs and assert the
produced output tree conforms to the family-wide contract via the package's reusable
:func:`ocr_output_contract.conformance.assert_conforms` harness, including:

* a multi-page PDF aggregating into ONE ``<stem>/<stem>.md`` with both pages;
* an empty/whitespace model response recorded as ``status=failed`` driving a nonzero exit;
* the HIGH fix: the ``.md`` body carries NO YAML frontmatter (provenance -> sidecar only);
* the HIGH fix: ``--task`` actually reaches the backend prompt.
"""

from __future__ import annotations

import json

import fitz
from ocr_output_contract.conformance import ExpectedDoc, assert_conforms
from PIL import Image

from deepseek_ocr.backends.base import Backend
from deepseek_ocr.processor import discover_documents, process


class FakeBackend(Backend):
    """A mocked backend: deterministic per-page text, no network/model."""

    def __init__(self, text="OCR page text", model="deepseek-ocr"):
        super().__init__(model_name=model)
        self.text = text
        self.calls = 0
        self.seen_prompts: list[str] = []

    @property
    def backend_name(self) -> str:
        return "ollama"

    def load_model(self) -> None:
        self.model = True

    def unload_model(self) -> None:
        self.model = False

    def process_image(self, image, prompt=None, task="convert", return_raw=False):
        self.calls += 1
        resolved = prompt if prompt is not None else self.get_prompt(task)
        self.seen_prompts.append(resolved)
        return self.text


def _make_pdf(path, pages=2):
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=300, height=400)
        page.insert_text((40, 60), f"Source page {i + 1}")
    doc.save(path)
    doc.close()


def test_multipage_pdf_conforms_no_data_loss(tmp_path):
    """A 2-page PDF -> one <stem>/<stem>.md with both pages, no per-page folders."""
    pdf = tmp_path / "sample.pdf"
    _make_pdf(pdf, pages=2)
    out = tmp_path / "out"

    counter = {"n": 0}

    class PerPage(FakeBackend):
        def process_image(self, image, prompt=None, task="convert", return_raw=False):
            counter["n"] += 1
            return f"PAGE-{counter['n']}-CONTENT"

    outcome = process(pdf, PerPage(), dpi=120, output_dir=out)
    assert outcome.exit_code == 0

    assert_conforms(
        out,
        [ExpectedDoc(rel_key="sample.pdf", pages=2, status="completed")],
        require_failures_nonzero_exit=outcome.exit_code != 0,
    )

    md = out / "sample" / "sample.md"
    body = md.read_text()
    assert "## Page 1" in body and "## Page 2" in body
    assert "PAGE-1-CONTENT" in body and "PAGE-2-CONTENT" in body
    assert not (out / "sample_p0001").exists()
    assert not (out / "sample_p0002").exists()


def test_no_yaml_frontmatter_in_markdown(tmp_path):
    """HIGH fix: the .md body must NOT carry YAML frontmatter; provenance is in the sidecar."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    outcome = process(pdf, FakeBackend(text="clean body text"), dpi=120, output_dir=out)
    assert outcome.exit_code == 0

    body = (out / "doc" / "doc.md").read_text()
    # No leading YAML frontmatter delimiter, and none of the old provenance keys leaked.
    assert not body.lstrip().startswith("---")
    for key in ("source:", "processed:", "processing_time:", "model:", "backend:"):
        assert key not in body

    # Provenance lives ONLY in the sidecar.
    meta = json.loads((out / "doc" / "metadata.json").read_text())
    assert meta["backend"] == "ollama"
    assert meta["model"] == "deepseek-ocr"
    assert meta["status"] == "completed"


def test_task_reaches_backend_prompt(tmp_path):
    """HIGH fix: --task is honoured -> the task's prompt reaches the backend."""
    pdf = tmp_path / "t.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    be = FakeBackend(text="text")
    process(pdf, be, dpi=120, task="ocr", output_dir=out)

    # "ocr" maps to a distinct prompt from the default "convert".
    assert be.seen_prompts == [Backend.PROMPTS["ocr"]]
    assert be.seen_prompts[0] != Backend.PROMPTS["convert"]


def test_empty_response_fails_and_exits_nonzero(tmp_path):
    """An empty model response -> status=failed, nonzero exit (not a 0-byte success)."""
    pdf = tmp_path / "blank.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    outcome = process(pdf, FakeBackend(text="   "), dpi=120, output_dir=out)
    assert outcome.exit_code != 0

    assert_conforms(
        out,
        [ExpectedDoc(rel_key="blank.pdf", status="failed")],
        require_failures_nonzero_exit=True,
    )
    doc_meta = json.loads((out / "blank" / "metadata.json").read_text())
    assert doc_meta["status"] == "failed"


def test_nested_batch_conforms_no_basename_collision(tmp_path):
    """Two same-basename PDFs in different subdirs both survive (input-relative key)."""
    root = tmp_path / "in"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir(parents=True)
    _make_pdf(root / "a" / "intro.pdf", pages=1)
    _make_pdf(root / "b" / "intro.pdf", pages=1)
    out = tmp_path / "out"

    outcome = process(root, FakeBackend(), dpi=120, output_dir=out)
    assert outcome.exit_code == 0

    assert_conforms(
        out,
        [
            ExpectedDoc(rel_key="a/intro.pdf", pages=1, status="completed"),
            ExpectedDoc(rel_key="b/intro.pdf", pages=1, status="completed"),
        ],
    )


def test_image_dir_document_conforms(tmp_path):
    """A directory of page images is one conforming document keyed by folder name."""
    src = tmp_path / "scan"
    src.mkdir()
    for n in (1, 2):
        Image.new("RGB", (60, 60), "white").save(src / f"page_{n:04d}.png")
    out = tmp_path / "out"

    outcome = process(src, FakeBackend(), dpi=120, output_dir=out)
    assert outcome.exit_code == 0

    assert_conforms(
        out,
        [ExpectedDoc(rel_key="scan", pages=2, status="completed")],
    )


def test_mixed_dir_with_stray_image_processes_all_pdfs(tmp_path):
    """HIGH fix: a batch dir holding PDFs + a stray image processes ALL PDFs.

    Previously any direct image misclassified the whole directory as ONE
    image-document, silently dropping every PDF and subdirectory (the
    papers-library batch use case). Now a directory that contains a PDF (or a
    subdir) is a batch tree: every PDF AND the stray image are OCR'd as their own
    documents, and subdirectories are recursed.
    """
    root = tmp_path / "papers"
    (root / "sub").mkdir(parents=True)
    _make_pdf(root / "paper1.pdf", pages=1)
    _make_pdf(root / "sub" / "paper2.pdf", pages=1)
    Image.new("RGB", (60, 60), "white").save(root / "cover.png")
    out = tmp_path / "out"

    outcome = process(root, FakeBackend(), dpi=120, output_dir=out)
    assert outcome.exit_code == 0
    # 2 PDFs + 1 stray image = 3 documents, none dropped.
    assert outcome.completed == 3

    assert_conforms(
        out,
        [
            ExpectedDoc(rel_key="paper1.pdf", pages=1, status="completed"),
            ExpectedDoc(rel_key="sub/paper2.pdf", pages=1, status="completed"),
            ExpectedDoc(rel_key="cover.png", pages=1, status="completed"),
        ],
    )


def test_rerun_does_not_reingest_default_output_root(tmp_path):
    """HIGH fix: the default <input>/ocr/ output is excluded from re-run discovery.

    With the default (nested) output root, a second run must not re-discover the
    first run's own .md/figure outputs as fresh inputs. Discovery via
    iter_input_files prunes the resolved output-root subtree.
    """
    root = tmp_path / "papers"
    root.mkdir()
    _make_pdf(root / "doc.pdf", pages=1)

    # First run with the DEFAULT output root (<input>/ocr/, inside the tree).
    outcome1 = process(root, FakeBackend(), dpi=120)
    assert outcome1.completed == 1
    assert (root / "ocr").is_dir()  # default root sits inside the scanned tree

    # Second run: the only new document is still the one PDF; the ocr/ outputs
    # (markdown + metadata) must NOT be re-discovered as inputs.
    docs = discover_documents(root)
    assert docs == [root / "doc.pdf"]


def test_truncated_page_recorded_partial_not_completed(tmp_path):
    """MEDIUM fix: a length-truncated non-empty page is partial/failed, not completed.

    The backend reports a length finish_reason; the page text is non-empty but
    incomplete, so recording it as completed would be silent content loss.
    """
    pdf = tmp_path / "dense.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    class TruncatingBackend(FakeBackend):
        def process_image_with_meta(self, image, prompt=None, task="convert", return_raw=False):
            # Non-empty text, but the model stopped on the token limit.
            return "partial dense content cut off", "length"

    outcome = process(pdf, TruncatingBackend(), dpi=120, output_dir=out)
    assert outcome.exit_code != 0

    doc_meta = json.loads((out / "dense" / "metadata.json").read_text())
    assert doc_meta["status"] == "failed"  # single page truncated -> no good pages
    assert "truncat" in doc_meta["error"].lower()


def test_quiet_skip_emits_md_path_on_resume(tmp_path):
    """MEDIUM fix: a cached/resumed doc still contributes its .md path to outputs.

    On the first run the doc is processed and its path emitted; on the second run
    it is skipped (already completed) but must STILL appear in outcome.outputs so
    `deepseek-ocr dir -q` does not silently drop cached docs.
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    outcome1 = process(pdf, FakeBackend(), dpi=120, output_dir=out)
    md = out / "doc" / "doc.md"
    assert outcome1.outputs == [str(md)]

    # Second run: skip branch, but the .md path is still emitted.
    outcome2 = process(pdf, FakeBackend(), dpi=120, output_dir=out)
    assert outcome2.completed == 1
    assert outcome2.outputs == [str(md)]


def test_rerun_under_different_task_reprocesses(tmp_path):
    """The run fingerprint invalidates the cache when --task changes.

    A re-run under a different task must NOT silently reuse the prior output: the
    fingerprint (model/backend/task/prompt) differs, so is_completed returns False.
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    be1 = FakeBackend(text="convert output")
    process(pdf, be1, dpi=120, task="convert", output_dir=out)
    assert be1.calls == 1

    # Same input, different task -> fingerprint differs -> reprocessed (not skipped).
    be2 = FakeBackend(text="ocr output")
    process(pdf, be2, dpi=120, task="ocr", output_dir=out)
    assert be2.calls == 1


def test_image_dir_classification_stable_across_reruns_default_root(tmp_path):
    """MEDIUM fix: an image-dir document classifies identically on run 1 and re-run.

    With the DEFAULT output root (``<input>/ocr/``, nested INSIDE the scanned dir),
    the first run of an image directory creates that ``ocr/`` subtree. Without
    output-root-aware classification, the re-run would see the new subdir, decide
    the folder is no longer a pure image dir, and silently reclassify it as a BATCH
    tree (one aggregated ``scan/scan.md`` on run 1 -> per-image ``page_0001_png/``
    trees on run 2: non-idempotent, orphaned output, spurious index entries, broken
    resume). Classification must be STABLE: a first run and a re-run see the same
    single image-dir document.
    """
    src = tmp_path / "scan"
    src.mkdir()
    for n in (1, 2):
        Image.new("RGB", (60, 60), "white").save(src / f"page_{n:04d}.png")

    # Run 1 (default output root): classified as ONE image-dir document.
    docs_run1 = discover_documents(src)
    assert docs_run1 == [src]

    # Actually run it so the default ocr/ subtree is created inside the scanned dir.
    outcome1 = process(src, FakeBackend(), dpi=120)
    assert outcome1.completed == 1
    assert (src / "ocr").is_dir()  # default root nests inside the scanned tree

    # Re-run classification: the engine's own ocr/ subtree must NOT flip the
    # decision. Still exactly ONE image-dir document, identical to run 1.
    docs_run2 = discover_documents(src)
    assert docs_run2 == docs_run1 == [src]

    # And a real re-run stays idempotent: still one completed document, and NO
    # per-image batch trees were emitted under the output root.
    outcome2 = process(src, FakeBackend(), dpi=120)
    assert outcome2.completed == 1
    out_root = src / "ocr"
    assert (out_root / "scan" / "scan.md").exists()
    assert not (out_root / "page_0001_png").exists()
    assert not (out_root / "page_0002_png").exists()


def test_unreadable_file_recorded_failed_batch_continues(tmp_path):
    """SYS-02 fix: an unreadable input fails ITSELF; the batch keeps going.

    A file that becomes unreadable between discovery and processing must be
    recorded status=failed (via the safe checksum + per-doc catch-all), not raise
    an OSError that aborts the whole batch and drops the good files.
    """
    import os
    import stat

    root = tmp_path / "papers"
    root.mkdir()
    _make_pdf(root / "a_good.pdf", pages=1)
    bad = root / "b_bad.pdf"
    _make_pdf(bad, pages=1)
    _make_pdf(root / "c_good.pdf", pages=1)
    out = tmp_path / "out"

    os.chmod(bad, 0)
    try:
        outcome = process(root, FakeBackend(), dpi=120, output_dir=out)
    finally:
        os.chmod(bad, stat.S_IRUSR | stat.S_IWUSR)

    # The whole batch did NOT abort: two good files completed, the bad one failed.
    assert outcome.completed == 2
    assert outcome.failed == 1
    assert outcome.exit_code != 0

    # The failed doc has a durable status=failed record (no checksum required).
    bad_meta = json.loads((out / "b_bad" / "metadata.json").read_text())
    assert bad_meta["status"] == "failed"


def test_rerun_under_different_raw_flag_reprocesses(tmp_path):
    """The run fingerprint extra invalidates the cache when --raw changes.

    --raw genuinely changes page text (it skips clean_ocr_output), so a re-run with
    a different --raw must reprocess, not silently reuse the prior cleaned output.
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    be1 = FakeBackend(text="output")
    process(pdf, be1, dpi=120, raw=False, output_dir=out)
    assert be1.calls == 1

    # Same input/task/prompt, different --raw -> fingerprint extra differs -> reprocess.
    be2 = FakeBackend(text="output")
    process(pdf, be2, dpi=120, raw=True, output_dir=out)
    assert be2.calls == 1


def test_rerun_same_prompt_different_task_not_reprocessed(tmp_path):
    """LOW fix: --task does NOT over-invalidate the cache when --prompt is set.

    The backends ignore --task when a custom --prompt is given (they only call
    get_prompt(task) when prompt is None), so two runs with the SAME prompt but a
    DIFFERENT task produce identical output and must be cache-skipped, not
    needlessly reprocessed. task is dropped from the fingerprint when prompt is set.
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)
    out = tmp_path / "out"

    be1 = FakeBackend(text="output")
    process(pdf, be1, dpi=120, prompt="my custom prompt", task="convert", output_dir=out)
    assert be1.calls == 1

    # Same prompt, different task -> same fingerprint -> skipped (no reprocess).
    be2 = FakeBackend(text="output")
    process(pdf, be2, dpi=120, prompt="my custom prompt", task="ocr", output_dir=out)
    assert be2.calls == 0
