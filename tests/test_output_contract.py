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
from deepseek_ocr.processor import process


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
