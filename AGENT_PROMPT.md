# Agent Prompt: DeepSeek OCR CLI

## Tool Overview

You have access to `deepseek-ocr`, a command-line tool for extracting text from PDFs and images using local OCR processing via Ollama. The tool runs entirely locally with no API keys required.

## Prerequisites

Before using this tool, verify:
1. Ollama is running: `ollama serve`
2. DeepSeek-OCR model is available: `ollama pull deepseek-ocr`
3. The tool is installed: `deepseek-ocr --version`

Check status with: `deepseek-ocr info`

## Basic Usage

### Single File Processing
```bash
# Process a single file (PDF or image)
deepseek-ocr path/to/document.pdf

# Output is saved to ./output/document.md by default
```

### Batch Processing
```bash
# Process all files in a directory
deepseek-ocr path/to/directory/ --recursive

# Custom output directory
deepseek-ocr path/to/files/ -o ./results/ --recursive
```

### Custom Prompts
```bash
# Use a custom prompt for specific extraction needs
deepseek-ocr document.pdf --prompt "Extract only the tables in markdown format"
deepseek-ocr form.jpg --prompt "Extract form fields and values"
```

## Output Format

The tool generates markdown files with:
- Front matter containing metadata (source, processing time, pages)
- Clean markdown text with HTML tables converted to markdown
- Page-by-page breakdown for multi-page PDFs

Example output structure:
```markdown
---
source: /path/to/document.pdf
processed: 2025-12-06T10:30:00
pages: 3
processing_time: 45.2s
model: deepseek-ocr
backend: ollama
---

## Page 1

[Extracted content...]

## Page 2

[Extracted content...]
```

## Performance Expectations

**Important timing information:**
- Simple pages: 3-8 seconds
- Dense tables/charts: 15-50 seconds
- Very complex pages: Up to 7 minutes (rare)
- 24-page PDF: ~8-20 minutes

**Always inform the user about expected processing time before starting.**

## When to Use This Tool

**Good use cases:**
- Extracting text from scanned documents
- Converting PDFs to markdown
- Processing forms and tables
- Batch processing document archives
- OCR on images containing text

**Not suitable for:**
- Real-time processing (too slow)
- Simple text-based PDFs (use standard PDF text extraction instead)
- Very large batches (>100 pages) without user confirmation

## Best Practices

### 1. Set User Expectations
```
Before processing, tell the user:
"I'll process this [N-page] PDF using local OCR. This will take approximately [X] minutes.
The OCR process runs locally on your machine via Ollama."
```

### 2. Check Prerequisites
Always verify Ollama is running before starting:
```bash
deepseek-ocr info
```

### 3. Handle Errors Gracefully
Common issues:
- Ollama not running → Instruct: `ollama serve`
- Model not found → Instruct: `ollama pull deepseek-ocr`
- File not found → Verify path exists

### 4. Show Progress for Large Files
For multi-page PDFs, the tool shows a progress bar. Let the user know:
```
"Processing... You'll see a progress bar showing pages completed."
```

## Example Workflows

### Workflow 1: Single Document OCR
```bash
# User asks: "Extract text from this scanned PDF"

# 1. Check status
deepseek-ocr info

# 2. Process file
deepseek-ocr path/to/scanned.pdf

# 3. Read result
cat output/scanned.md
```

### Workflow 2: Batch Processing with Custom Output
```bash
# User asks: "Convert all PDFs in /documents to markdown"

# 1. Process batch
deepseek-ocr /documents/ --recursive -o ./markdown_output/

# 2. Inform user of location
echo "Processed files saved to: ./markdown_output/"
```

### Workflow 3: Table Extraction
```bash
# User asks: "Extract the tables from this financial report"

# 1. Use custom prompt for tables
deepseek-ocr report.pdf --prompt "Extract all tables in markdown format, preserve structure and numbers accurately"

# 2. Read and present tables
cat output/report.md
```

## Important Limitations

1. **Processing Speed**: Not real-time. Always warn users about expected duration.
2. **Local Resource Usage**: Uses significant CPU/GPU during processing.
3. **Timeout**: 10-minute timeout per page. Extremely complex pages may timeout.
4. **Quality**: OCR quality depends on image quality and document complexity.

## Error Handling

If the tool fails:

1. **Check Ollama**: `ollama serve` must be running
2. **Check Model**: `ollama list` should show deepseek-ocr
3. **Check File**: Verify file exists and is accessible
4. **Check Disk Space**: Large PDFs need temporary storage

Common error messages and solutions:
- "Model not loaded" → Run `ollama pull deepseek-ocr`
- "Ollama is not running" → Start with `ollama serve`
- "File not found" → Verify file path is correct
- "Timeout" → Document too complex, try splitting it

## Response Templates

### Starting Processing
"I'll process this [N-page] document using DeepSeek OCR via Ollama. This runs locally and will take approximately [X] minutes. Starting now..."

### Completion
"OCR complete! Processed [N] pages in [X] minutes. The markdown output has been saved to [path]. Here's a summary: [brief excerpt]"

### Error
"The OCR process failed: [error]. This likely means [explanation]. To fix: [solution]"

## Advanced Options

```bash
# Extract page images from PDF
deepseek-ocr document.pdf --extract-images

# Exclude metadata from output
deepseek-ocr document.pdf --no-metadata

# Use different Ollama model (if available)
deepseek-ocr document.pdf --model custom-ocr-model

# Different task types
deepseek-ocr image.jpg --task extract  # Extract text only
deepseek-ocr doc.pdf --task layout     # Preserve layout structure
```

## Summary

- **What**: Local OCR tool using Ollama and DeepSeek-OCR model
- **When**: Converting scanned documents/images to markdown
- **Speed**: Slow (seconds to minutes per page)
- **Output**: Clean markdown with tables converted
- **Prerequisite**: Ollama must be running with deepseek-ocr model

Always inform users about processing time expectations and verify prerequisites before starting.
