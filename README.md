# DeepSeek OCR CLI

[![PyPI version](https://badge.fury.io/py/deepseek-ocr-cli.svg)](https://badge.fury.io/py/deepseek-ocr-cli)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Command-line tool for OCR using DeepSeek vision models. Supports Ollama (local) and vLLM (GPU server) backends.

## Features

- **Multi-backend**: Ollama (local, free) and vLLM (OpenAI-compatible API)
- Supports PDFs and images (JPG, PNG, WEBP, GIF, BMP, TIFF)
- Canonical output via the shared [`ocr-output-contract`](https://github.com/r-uben/ocr-output-contract): one `<root>/<rel/dir>/<stem>/<stem>.md` per document under `## Page N` headers, dual `metadata.json` (per-doc sidecar + root index), input-relative keying (no basename collisions)
- Batch processing of directory trees with incremental resume (skips already-completed documents; re-runs when the input, model, backend, task, or prompt changes)
- Truncation detection: a length-truncated page is recorded `status=partial/failed`, never a silent `completed`
- Retry with exponential backoff for transient failures
- `--dry-run` to preview the exact documents that will be processed
- Clean markdown output with HTML tables converted to markdown (`--raw` keeps the model's verbatim text)

## Choosing an OCR tool

This is one of five OCR CLI tools with a shared design: clean Markdown output, batch processing, and figure extraction. Pick based on your constraints:

| Tool | Engine | Runs | Cost | Best for |
|------|--------|------|------|----------|
| **deepseek-ocr-cli** (this repo) | DeepSeek vision | Local (Ollama / vLLM) | Free | General-purpose local OCR with multi-backend flexibility |
| [gemini-ocr-cli](https://github.com/r-uben/gemini-ocr-cli) | Google Gemini | Cloud API | Free tier / Pay-per-use | Fast cloud OCR with concurrent processing |
| [marker-ocr-cli](https://github.com/r-uben/marker-ocr-cli) | Marker (Surya + Texify) | Local | Free | Academic papers with equations, tables, complex layouts |
| [mistral-ocr-cli](https://github.com/r-uben/mistral-ocr-cli) | Mistral OCR API | Cloud API | ~$1/1k pages | Structured extraction (tables, headers, footers) |
| [nougat-ocr-cli](https://github.com/r-uben/nougat-ocr-cli) | Meta Nougat | Local (GPU) | Free | Academic papers, GPU-accelerated batch processing |

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running (for Ollama backend)
- `deepseek-ocr` model pulled in Ollama

## Installation

### 1. Install Ollama

```bash
# macOS/Linux
brew install ollama

# Or download from https://ollama.ai
```

### 2. Pull the DeepSeek-OCR model

```bash
ollama pull deepseek-ocr
```

### 3. Install the CLI

```bash
pip install deepseek-ocr-cli
```

## Quick Start

```bash
# Process a single image
deepseek-ocr document.jpg

# Process a PDF
deepseek-ocr paper.pdf

# Process a directory tree (always walked recursively)
deepseek-ocr ./documents/

# Preview the documents that would be processed
deepseek-ocr ./documents/ --dry-run

# Custom output directory
deepseek-ocr doc.pdf -o ./results/

# Use vLLM backend
deepseek-ocr paper.pdf --backend vllm --vllm-url http://gpu-server:8000/v1

# Raise the per-page token budget if dense pages truncate
deepseek-ocr large-document.pdf --max-tokens 16384

# Keep the model's verbatim output (skip the cleaner)
deepseek-ocr paper.pdf --raw

# Extract and analyze embedded figures
deepseek-ocr paper.pdf --analyze-figures

# Quiet mode (paths only, for scripting)
deepseek-ocr paper.pdf -q
```

## CLI Options

```
deepseek-ocr [OPTIONS] INPUT_PATH

Options:
  -o, --output-dir PATH           Output root (default: <input-parent>/ocr/)
  -r, --recursive                 Accepted for compatibility; batch trees are
                                  ALWAYS walked recursively
  --model TEXT                    Model name (default: deepseek-ocr)
  --prompt TEXT                   Custom prompt for OCR (overrides --task)
  --task [convert|ocr|layout|extract|parse]
                                  OCR task type
  --dpi INTEGER                   PDF rendering DPI (default: 200)
  --analyze-figures               Extract and analyze embedded figures with AI
  --raw                           Keep verbatim model output (skip the cleaner)
  --max-tokens INTEGER            Max tokens per page (default: 8192). Raise if
                                  dense pages truncate
  --max-dim INTEGER               Max image dimension (default: 1920, 0 to disable)
  --backend [ollama|vllm]         Backend to use (default: ollama)
  --vllm-url TEXT                 vLLM API URL (default: http://localhost:8000/v1)
  --reprocess                     Force reprocessing of already-done documents
  --dry-run                       Preview documents without processing
  -q, --quiet                     Suppress output, print one .md path per line
  --verbose                       Enable verbose output
  --help                          Show this message and exit.
```

## Commands

### `process` (default)

Process documents and images with OCR. The `process` subcommand is optional:

```bash
deepseek-ocr document.pdf
# equivalent to
deepseek-ocr process document.pdf
```

### `info`

Show system and configuration information.

```bash
deepseek-ocr info
```

## Output Format

Output follows the shared `ocr-output-contract`. The default output root is
`<input-parent>/ocr/` for a single file and `<input>/ocr/` for a directory
(override with `-o`). Each document gets its own folder, mirroring the input
subtree so same-named files in different directories never collide:

```
ocr/
├── metadata.json           # root index, keyed by input-relative path
└── document/
    ├── document.md         # OCR markdown
    ├── metadata.json       # per-document sidecar (provenance)
    └── figures/            # extracted figures (if --analyze-figures)
        └── figure_1_page1.png
```

The markdown body carries **no YAML frontmatter** — all provenance lives in the
JSON sidecars. Pages are separated by `## Page N` headers:

```markdown
## Page 1

[Extracted content...]

## Page 2

[Extracted content...]
```

The per-document `metadata.json` records the ratified schema (`status`,
`checksum`, `model`, `backend`, `processing_time`, `timestamp` (UTC),
`output_path`, `pages`, plus a run `fingerprint`).

### Batch Resume

The root `metadata.json` records every processed document. On re-run, a document
is skipped only when the input is unchanged, its `.md` still exists on disk, and
the run configuration (model, backend, task, prompt) is unchanged. Use
`--reprocess` to force reprocessing.

## Configuration

Create a `.env` file or set environment variables with `DEEPSEEK_OCR_` prefix:

```bash
DEEPSEEK_OCR_BACKEND=ollama
DEEPSEEK_OCR_MODEL_NAME=deepseek-ocr
DEEPSEEK_OCR_OLLAMA_URL=http://localhost:11434
DEEPSEEK_OCR_VLLM_BASE_URL=http://localhost:8000/v1
DEEPSEEK_OCR_MAX_DIMENSION=1920
DEEPSEEK_OCR_MAX_TOKENS=8192
DEEPSEEK_OCR_MAX_RETRIES=3
DEEPSEEK_OCR_RETRY_DELAY=1.0
DEEPSEEK_OCR_LOG_LEVEL=INFO
```

## Programmatic Usage

```python
from pathlib import Path
from deepseek_ocr import create_backend, process

backend = create_backend(backend_type="ollama", model_name="deepseek-ocr")
backend.load_model()

# process() routes all output through the contract and returns a RunOutcome.
outcome = process(Path("document.pdf"), backend, output_dir=Path("./results"))
for md_path in outcome.outputs:
    print("wrote", md_path)
print("exit code:", outcome.exit_code)  # nonzero if any document/page failed

backend.unload_model()
```

## Troubleshooting

### Ollama not running

```bash
ollama serve
```

### Model not found

```bash
ollama pull deepseek-ocr
```

### Check status

```bash
deepseek-ocr info
```

## License

MIT License - see [LICENSE](LICENSE) for details.
