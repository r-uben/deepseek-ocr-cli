# DeepSeek OCR CLI

[![PyPI version](https://badge.fury.io/py/deepseek-ocr-cli.svg)](https://badge.fury.io/py/deepseek-ocr-cli)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A command-line tool for OCR (Optical Character Recognition) using the DeepSeek-OCR model via Ollama. Runs locally with zero cloud dependencies.

## Features

- **Local Processing**: No API keys, no usage costs, complete privacy
- **Ollama Backend**: Simple setup, efficient inference
- **Multiple Formats**: Process PDFs and images (JPG, PNG, WEBP, GIF, BMP, TIFF)
- **Batch Processing**: Handle multiple files and directories
- **Clean Markdown Output**: Tables converted to markdown, HTML stripped
- **Progress Tracking**: Real-time progress bar for multi-page PDFs
- **Rich CLI**: Beautiful terminal interface with progress bars and tables

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- `deepseek-ocr` model pulled in Ollama

## Installation

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Or download from https://ollama.ai/
```

### 2. Pull the DeepSeek-OCR model

```bash
ollama pull deepseek-ocr
```

### 3. Install the CLI

```bash
# Clone the repository
git clone https://github.com/r-uben/deepseek-ocr-cli.git
cd deepseek-ocr-cli

# Install with Poetry
poetry install

# Or with pip
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

### Output Cleaning

The tool automatically:
- Converts HTML tables to markdown tables
- Removes bounding box annotations
- Decodes HTML entities
- Preserves LaTeX math expressions

## Performance

Typical performance on Apple Silicon M3 Pro Max:
- **Simple pages**: 3-8 seconds per page
- **Dense tables/charts**: 15-50 seconds per page
- **Very complex pages**: Up to 7 minutes (rare)
- **Average (mixed content)**: ~20 seconds per page
- **24-page PDF**: ~8-20 minutes

**Note:** Processing time varies significantly based on content density. Sparse pages (cover pages, simple text) process quickly, while dense economic tables or complex layouts take longer. The tool includes a 10-minute timeout per page to handle extreme cases.

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

# Initialize and load model
model_manager = ModelManager(model_name="deepseek-ocr")
model_manager.load_model()

# Create processor
processor = OCRProcessor(
    model_manager=model_manager,
    output_dir=Path("./results"),
)

# Process a file
result = processor.process_file(Path("document.pdf"))
print(result.output_text)

# Save result
processor.save_result(result)

# Cleanup
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
# Install with dev dependencies
poetry install

# Run tests
poetry run pytest

# Format code
poetry run black .

# Lint
poetry run ruff check .
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [DeepSeek](https://github.com/deepseek-ai) for the OCR model
- [Ollama](https://ollama.ai/) for local model serving
