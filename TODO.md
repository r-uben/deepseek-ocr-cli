# TODO — deepseek-ocr-cli v0.4.0

Improvements inspired by mistral-ocr-cli.

## Phase 1: Ship the backend refactor
- [x] Backend abstraction layer (backends/)
- [x] Commit and test with Ollama

## Phase 2: Core improvements
- [x] **Per-document output folders** — `output/doc_name/{doc_name.md, figures/}`
- [x] **Incremental metadata** — `metadata.json` tracks processed files, skip on re-run, `--reprocess` to force
- [x] **Retry with exponential backoff** — for transient Ollama/vLLM failures (429, 500, timeout)
- [x] **`--dry-run` flag** — preview files without processing

## Phase 3: UX polish
- [x] **Simplified CLI** — `deepseek-ocr INPUT [OPTIONS]` directly (auto-inserts `process`)
- [x] **`--quiet` / `-q` flag** — suppress non-error output, only print paths
- [x] **`--reprocess` flag** — force reprocessing of already-done files

## Phase 4: Robustness (future)
- [ ] **Atomic metadata writes** — tmp file + rename to prevent corruption on crash
- [ ] **File size limits** — warn/skip files over configurable max size
