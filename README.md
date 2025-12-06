# DeepSeek OCR CLI

[![PyPI version](https://badge.fury.io/py/deepseek-ocr-cli.svg)](https://badge.fury.io/py/deepseek-ocr-cli)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Command-line tool for OCR using DeepSeek-OCR via Ollama. Runs locally with no API keys or cloud dependencies.

## Features

- Local processing with no API keys or usage costs
- Powered by Ollama for efficient local inference
- Supports PDFs and images (JPG, PNG, WEBP, GIF, BMP, TIFF)
- Batch processing for multiple files and directories
- Clean markdown output with HTML tables converted to markdown
- Progress tracking for multi-page PDFs
- Terminal interface with progress bars and summary tables

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
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

**Alternative: Install from source**

```bash
git clone https://github.com/r-uben/deepseek-ocr-cli.git
cd deepseek-ocr-cli
pip install -e .
```

## Quick Start

```bash
# Process a single image
deepseek-ocr document.jpg

# Process a PDF
deepseek-ocr paper.pdf

# Process all files in a directory
deepseek-ocr ./documents/ --recursive

# Custom output directory
deepseek-ocr doc.pdf -o ./results/

# Custom prompt
deepseek-ocr form.jpg --prompt "Extract table data in markdown format"

# Extract page images from PDF
deepseek-ocr paper.pdf --extract-images
```

## CLI Options

```
deepseek-ocr [OPTIONS] INPUT_PATH

Options:
  -o, --output-dir PATH           Output directory for results
  -r, --recursive                 Recursively process directories
  --model TEXT                    Ollama model name (default: deepseek-ocr)
  --prompt TEXT                   Custom prompt for OCR
  --task [convert|ocr|layout|extract|parse]
                                  OCR task type
  --extract-images                Extract and save page images from PDFs
  --no-metadata                   Exclude metadata from output
  --verbose                       Enable verbose output
  --help                          Show this message and exit.
```

## Commands

### `process` (default)

Process documents and images with OCR.

```bash
deepseek-ocr process document.pdf
# or simply
deepseek-ocr document.pdf
```

### `info`

Show system and configuration information.

```bash
deepseek-ocr info
```

## Output Format

The CLI generates markdown files with clean, structured output:

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

[Extracted content from page 1...]

## Page 2

[Extracted content from page 2...]
```

### Output Processing

Automatically applied to all OCR results:
- HTML tables converted to markdown tables
- Bounding box annotations removed
- HTML entities decoded
- LaTeX math expressions preserved

## Performance

Typical performance on Apple Silicon M3 Max with 200 DPI, JPEG encoding:
- Simple receipt/form: ~10 seconds
- Standard text pages: ~15-20 seconds per page
- Dense tables/charts: ~30-40 seconds per page
- Very complex pages: Up to 2 minutes (rare)

**Example:** 1-page receipt processed in 11 seconds (tested).

Processing time varies based on content density. The tool uses 200 DPI and JPEG encoding for optimal speed while maintaining quality. Timeout is set to 30 minutes per page for extremely dense documents.

## Configuration

Create a `.env` file to customize settings:

```bash
DEEPSEEK_OCR_MODEL_NAME=deepseek-ocr
DEEPSEEK_OCR_OUTPUT_DIR=output
DEEPSEEK_OCR_EXTRACT_IMAGES=false
DEEPSEEK_OCR_INCLUDE_METADATA=true
DEEPSEEK_OCR_LOG_LEVEL=INFO
OLLAMA_URL=http://localhost:11434
```

## Programmatic Usage

```python
from pathlib import Path
from deepseek_ocr import ModelManager, OCRProcessor

model_manager = ModelManager(model_name="deepseek-ocr")
model_manager.load_model()

processor = OCRProcessor(
    model_manager=model_manager,
    output_dir=Path("./results"),
)

result = processor.process_file(Path("document.pdf"))
print(result.output_text)

processor.save_result(result)

model_manager.unload_model()
```

## Troubleshooting

### Ollama not running

```bash
# Start Ollama
ollama serve
```

### Model not found

```bash
# Pull the model
ollama pull deepseek-ocr
```

### Check status

```bash
deepseek-ocr info
```

## Development

```bash
poetry install

poetry run pytest
poetry run black .
poetry run ruff check .
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Built With

This tool is built on top of:

- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-VL2) - Vision-language model for OCR by DeepSeek AI
- [Ollama](https://ollama.ai/) - Local LLM runtime for running models efficiently
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) - PDF processing library
- [Pillow](https://python-pillow.org/) - Image processing library
- [Click](https://click.palletsprojects.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Terminal formatting and progress bars
