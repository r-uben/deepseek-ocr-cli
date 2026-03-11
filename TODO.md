# TODO — deepseek-ocr-cli v0.4.0

Improvements inspired by mistral-ocr-cli, prioritized by impact.

## Phase 1: Ship the backend refactor
- [x] Backend abstraction layer (backends/)
- [ ] **Commit the WIP changes** — clean up, test with Ollama, commit

## Phase 2: Core improvements
- [ ] **Per-document output folders** — `output/doc_name/{doc_name.md, figures/}` instead of flat files
- [ ] **Incremental metadata** — `metadata.json` tracks processed files, skip on re-run, `--reprocess` to force
- [ ] **Retry with exponential backoff** — for transient Ollama/vLLM failures (429, 500, timeout)
- [ ] **`--dry-run` flag** — preview files without processing

## Phase 3: UX polish
- [ ] **Drop `process` subcommand** — `deepseek-ocr INPUT [OPTIONS]` directly (keep `info` as subcommand)
- [ ] **`--quiet` flag** — suppress non-error output
- [ ] **`--reprocess` flag** — force reprocessing of already-done files

## Phase 4: Robustness
- [ ] **Atomic metadata writes** — tmp file + rename to prevent corruption on crash
- [ ] **File size limits** — warn/skip files over configurable max size
- [ ] **Better error messages** — catch common issues (Ollama not running, model not pulled)
