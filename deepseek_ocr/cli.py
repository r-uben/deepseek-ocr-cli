"""Command-line interface for DeepSeek OCR.

Output is written through the shared ``ocr-output-contract`` package so deepseek's
output is byte-structure-identical to the rest of the engine family: default root
``<input-parent>/ocr/`` (``-o`` overrides, never required), one
``<root>/<rel/dir>/<stem>/<stem>.md`` per document under ``## Page N`` headers, dual
metadata sidecars, and a nonzero exit on any failure.
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from deepseek_ocr import __version__
from deepseek_ocr.backends import create_backend
from deepseek_ocr.config import settings
from deepseek_ocr.processor import process as run_process
from deepseek_ocr.utils import collect_files, is_pdf_file, setup_logging

console = Console()
err_console = Console(stderr=True)


def print_banner(quiet: bool = False) -> None:
    if not quiet:
        console.print(f"[dim]deepseek-ocr v{__version__}[/dim]")


def _format_size(size_bytes: float) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _get_pdf_page_count(path: Path) -> int:
    """Get page count from a PDF without rendering."""
    import fitz

    doc = fitz.open(path)
    count = len(doc)
    doc.close()
    return count


def _run_dry_run(input_path: Path, recursive: bool, quiet: bool) -> None:
    """List files that would be processed without actually processing them."""
    files = collect_files(input_path, recursive=recursive)

    if quiet:
        for f in files:
            console.print(str(f))
        return

    table = Table(title="Files to process (dry run)", show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("File", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Size", justify="right")
    table.add_column("Pages", justify="right")

    total_size = 0
    total_pages = 0

    for idx, f in enumerate(files, 1):
        size = f.stat().st_size
        total_size += size

        if is_pdf_file(f):
            try:
                pages = _get_pdf_page_count(f)
            except Exception:
                pages = 0
            file_type = "PDF"
        else:
            pages = 1
            file_type = f.suffix.upper().lstrip(".")

        total_pages += pages
        table.add_row(str(idx), f.name, file_type, _format_size(size), str(pages))

    console.print(table)
    console.print(
        f"\n[bold]{len(files)}[/bold] files, "
        f"[bold]{_format_size(total_size)}[/bold] total, "
        f"[bold]{total_pages}[/bold] pages"
    )


@click.group()
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """DeepSeek OCR CLI - OCR processing via Ollama or vLLM.

    Process documents and images directly:

    \b
        deepseek-ocr document.pdf
        deepseek-ocr ./papers/ --recursive
        deepseek-ocr document.pdf --dry-run

    Output goes to <input-parent>/ocr/ by default (-o overrides; never required).
    Supports Ollama (local, default) and vLLM (OpenAI-compatible) backends.
    """
    ctx.ensure_object(dict)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output root (default: <input-parent>/ocr/). Writes <stem>/<stem>.md per document.",
)
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    help="Recursively process directories (reserved; batch trees are walked recursively).",
)
@click.option(
    "--model",
    "model_name",
    type=str,
    default="deepseek-ocr",
    help="Ollama model name (default: deepseek-ocr).",
)
@click.option(
    "--prompt",
    type=str,
    default=None,
    help="Custom prompt for OCR (overrides --task).",
)
@click.option(
    "--task",
    type=click.Choice(["convert", "ocr", "layout", "extract", "parse"]),
    default="convert",
    show_default=True,
    help="OCR task type; selects the backend prompt unless --prompt is given.",
)
@click.option(
    "--dpi",
    type=int,
    default=200,
    show_default=True,
    help="PDF rendering DPI (higher=slower but better quality).",
)
@click.option(
    "--analyze-figures",
    is_flag=True,
    help="Extract and describe embedded figures/images from PDFs.",
)
@click.option(
    "--max-dim",
    "max_dimension",
    type=int,
    default=None,
    help="Maximum image dimension. Larger images are resized to prevent timeouts. 0 disables.",
)
@click.option(
    "--backend",
    type=click.Choice(["ollama", "vllm"]),
    default=None,
    help="Backend: 'ollama' (local, default) or 'vllm'. Or set DEEPSEEK_OCR_BACKEND.",
)
@click.option(
    "--vllm-url",
    "vllm_base_url",
    type=str,
    default=None,
    help="vLLM API URL (default: http://localhost:8000/v1). Or set DEEPSEEK_OCR_VLLM_BASE_URL.",
)
@click.option(
    "--reprocess",
    is_flag=True,
    help="Re-OCR documents already recorded completed.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="List files that would be processed without running OCR.",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Suppress non-error output. Print one output .md path per line (for scripting).",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output.",
)
@click.pass_context
def process(
    ctx: click.Context,
    input_path: Path,
    output_dir: Path | None,
    recursive: bool,
    model_name: str,
    prompt: str | None,
    task: str,
    dpi: int,
    analyze_figures: bool,
    max_dimension: int | None,
    backend: str | None,
    vllm_base_url: str | None,
    reprocess: bool,
    dry_run: bool,
    quiet: bool,
    verbose: bool,
) -> None:
    """Process documents and images with OCR.

    INPUT_PATH can be a single file, a directory of page images (one document), or
    a tree of documents (batch). Supported: PDF, JPG, PNG, WEBP, GIF, BMP, TIFF.

    \b
    Examples:
        deepseek-ocr document.pdf
        deepseek-ocr ./documents/ --recursive
        deepseek-ocr paper.pdf --dry-run
        deepseek-ocr image.jpg --prompt "Extract all text"
        deepseek-ocr paper.pdf -q | xargs ls -la
    """
    setup_logging(level=settings.log_level, verbose=verbose)

    if dry_run:
        print_banner(quiet=quiet)
        try:
            _run_dry_run(input_path, recursive=recursive, quiet=quiet)
        except Exception as e:
            err_console.print(f"[red]error:[/red] {e}")
            sys.exit(1)
        return

    print_banner(quiet=quiet)

    # Resolve backend from CLI flag or config.
    backend_type = backend or settings.backend

    # For vLLM, default model is deepseek-vl2 unless explicitly specified.
    if backend_type == "vllm" and model_name == "deepseek-ocr":
        model_name = "deepseek-vl2"

    try:
        backend_instance = create_backend(
            backend_type=backend_type,
            model_name=model_name,
            max_dimension=max_dimension,
            ollama_url=settings.ollama_url,
            vllm_base_url=vllm_base_url or settings.vllm_base_url,
        )

        outcome = run_process(
            input_path,
            backend_instance,
            dpi=dpi,
            task=task,
            prompt=prompt,
            output_dir=output_dir,
            reprocess=reprocess,
            analyze_figures=analyze_figures,
        )

        backend_instance.unload_model()
    except Exception as e:
        err_console.print(f"[red]error:[/red] {e}")
        sys.exit(1)

    total = outcome.completed + outcome.failed + outcome.partial
    if quiet:
        for path in outcome.outputs:
            console.print(path)
    else:
        console.print(f"[dim]deepseek-ocr: backend={backend_type} dpi={dpi}[/dim]")
        console.print(f"  {outcome.completed}/{total} document(s) completed")
        if outcome.has_failures:
            err_console.print(
                f"  {outcome.failed} failed, {outcome.partial} partial: "
                + ", ".join(outcome.failures)
            )

    # Uniform exit policy (canon SYS-02): nonzero if any document/page failed.
    if outcome.exit_code != 0:
        sys.exit(outcome.exit_code)


@cli.command()
def info() -> None:
    """Show system and configuration information."""
    print_banner()

    sys_table = Table(title="System Information")
    sys_table.add_column("Component", style="cyan")
    sys_table.add_column("Status", style="green")

    sys_table.add_row("Python", f"{sys.version_info.major}.{sys.version_info.minor}")
    sys_table.add_row("Backend", settings.backend)

    from deepseek_ocr.backends.ollama import OllamaBackend

    ollama_backend = OllamaBackend()
    ollama_running = ollama_backend._check_ollama_running()
    ollama_model_available = ollama_backend._check_model_available() if ollama_running else False

    sys_table.add_row("Ollama URL", settings.ollama_url)
    sys_table.add_row("Ollama Running", "Yes" if ollama_running else "No")
    sys_table.add_row("deepseek-ocr Model", "Available" if ollama_model_available else "Not found")
    sys_table.add_row("vLLM URL", settings.vllm_base_url)

    console.print(sys_table)

    settings_table = Table(title="Current Settings")
    settings_table.add_column("Setting", style="cyan")
    settings_table.add_column("Value", style="yellow")

    settings_table.add_row("Model", settings.model_name)
    settings_table.add_row("Output root default", "<input-parent>/ocr/")
    settings_table.add_row("Max Image Dimension", str(settings.max_dimension))

    console.print(settings_table)

    console.print("\n[bold]Supported Formats:[/bold]")
    console.print("Images: JPG, PNG, WEBP, GIF, BMP, TIFF")
    console.print("Documents: PDF\n")

    console.print("[bold]Backend Options:[/bold]")
    console.print("  --backend ollama  Local processing (default)")
    console.print("  --backend vllm    OpenAI-compatible API (GPU server)\n")

    if settings.backend == "ollama":
        if not ollama_running:
            console.print("[yellow]Ollama is not running. Start with: ollama serve[/yellow]\n")
        elif not ollama_model_available:
            console.print("[yellow]Model not found. Pull with: ollama pull deepseek-ocr[/yellow]\n")


def main() -> None:
    """Entry point. Auto-inserts 'process' when first arg is a file/directory path."""
    argv = sys.argv[1:]

    if argv:
        known_subcommands = {"process", "info"}

        first_non_option_index = None
        for idx, arg in enumerate(argv):
            if not arg.startswith("-"):
                first_non_option_index = idx
                break

        if first_non_option_index is not None:
            candidate = argv[first_non_option_index]

            if candidate not in known_subcommands and Path(candidate).exists():
                argv = argv[:first_non_option_index] + ["process"] + argv[first_non_option_index:]
                sys.argv = [sys.argv[0], *argv]

    cli(obj={})


if __name__ == "__main__":
    main()
