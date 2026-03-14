# DeepSeek OCR CLI

[![PyPI version](https://badge.fury.io/py/deepseek-ocr-cli.svg)](https://badge.fury.io/py/deepseek-ocr-cli)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Command-line tool for OCR using DeepSeek vision models. Supports Ollama (local) and vLLM (GPU server) backends.

## Features

- **Multi-backend**: Ollama (local, free) and vLLM (OpenAI-compatible API)
- Supports PDFs and images (JPG, PNG, WEBP, GIF, BMP, TIFF)
- Per-document output folders with figures
- Batch processing with incremental resume (skips already-processed files)
- Retry with exponential backoff for transient failures
- Parallel page processing for faster PDF OCR
- `--dry-run` to preview files before processing
- Clean markdown output with HTML tables converted to markdown

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

# Process all files in a directory
deepseek-ocr ./documents/ --recursive

# Preview files without processing
deepseek-ocr ./documents/ --dry-run

# Custom output directory
deepseek-ocr doc.pdf -o ./results/

# Use vLLM backend
deepseek-ocr paper.pdf --backend vllm --vllm-url http://gpu-server:8000/v1

# Parallel processing for faster PDF OCR
deepseek-ocr large-document.pdf -w 2

# Extract and analyze embedded figures
deepseek-ocr paper.pdf --analyze-figures

# Quiet mode (paths only, for scripting)
deepseek-ocr paper.pdf -q
```

## CLI Options

```
deepseek-ocr [OPTIONS] INPUT_PATH

Options:
  -o, --output-dir PATH           Output directory for results
  -r, --recursive                 Recursively process directories
  --model TEXT                    Model name (default: deepseek-ocr)
  --prompt TEXT                   Custom prompt for OCR
  --task [convert|ocr|layout|extract|parse]
                                  OCR task type
  --extract-images                Extract and save page images from PDFs
  --no-metadata                   Exclude metadata from output
  --dpi INTEGER                   PDF rendering DPI (default: 200)
  -w, --workers INTEGER           Parallel workers for PDF pages (default: 1)
  --analyze-figures               Extract and analyze embedded figures with AI
  --max-dim INTEGER               Max image dimension (default: 1920, 0 to disable)
  --backend [ollama|vllm]         Backend to use (default: ollama)
  --vllm-url TEXT                 vLLM API URL (default: http://localhost:8000/v1)
  --reprocess                     Force reprocessing of already-done files
  --dry-run                       Preview files without processing
  -q, --quiet                     Suppress output, print paths only
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

Each document gets its own folder:

```
output/
└── document/
    ├── document.md          # OCR markdown
    └── figures/             # Extracted figures (if --analyze-figures)
        └── page1_fig1.png
```

The markdown includes metadata:

```markdown
---
source: /path/to/document.pdf
processed: 2025-12-01T15:30:00
pages: 3
processing_time: 18.45s
model: deepseek-ocr
backend: ollama
---

## Page 1

[Extracted content...]
```

### Batch Resume

Batch processing saves `metadata.json` in the output directory. On re-run, already-processed files are skipped automatically. Use `--reprocess` to force reprocessing.

## Configuration

Create a `.env` file or set environment variables with `DEEPSEEK_OCR_` prefix:

```bash
DEEPSEEK_OCR_BACKEND=ollama
DEEPSEEK_OCR_MODEL_NAME=deepseek-ocr
DEEPSEEK_OCR_OUTPUT_DIR=output
DEEPSEEK_OCR_OLLAMA_URL=http://localhost:11434
DEEPSEEK_OCR_VLLM_BASE_URL=http://localhost:8000/v1
DEEPSEEK_OCR_MAX_DIMENSION=1920
DEEPSEEK_OCR_MAX_RETRIES=3
DEEPSEEK_OCR_RETRY_DELAY=1.0
```

## Programmatic Usage

```python
from pathlib import Path
from deepseek_ocr import create_backend, OCRProcessor

backend = create_backend(backend_type="ollama", model_name="deepseek-ocr")
backend.load_model()

processor = OCRProcessor(
    backend=backend,
    output_dir=Path("./results"),
    workers=2,
)

result = processor.process_file(Path("document.pdf"))
print(result.output_text)

processor.save_result(result)
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
