"""Command-line interface for DeepSeek OCR."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from deepseek_ocr import __version__
from deepseek_ocr.config import settings
from deepseek_ocr.model import ModelManager
from deepseek_ocr.processor import OCRProcessor
from deepseek_ocr.utils import collect_files, is_pdf_file, setup_logging

console = Console()
err_console = Console(stderr=True)


def print_banner(quiet: bool = False) -> None:
    if not quiet:
        console.print(f"[dim]deepseek-ocr v{__version__}[/dim]")


def _format_size(size_bytes: int) -> str:
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

    Supports Ollama (local, default) and vLLM (OpenAI-compatible API) backends.
    """
    ctx.ensure_object(dict)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Output directory for results",
)
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    help="Recursively process directories",
)
@click.option(
    "--model",
    "model_name",
    type=str,
    default="deepseek-ocr",
    help="Ollama model name (default: deepseek-ocr)",
)
@click.option(
    "--prompt",
    type=str,
    help="Custom prompt for OCR",
)
@click.option(
    "--task",
    type=click.Choice(["convert", "ocr", "layout", "extract", "parse"]),
    default="convert",
    help="OCR task type",
)
@click.option(
    "--extract-images",
    is_flag=True,
    help="Extract and save page images from PDFs",
)
@click.option(
    "--no-metadata",
    is_flag=True,
    help="Exclude metadata from output",
)
@click.option(
    "--dpi",
    type=int,
    default=200,
    help="PDF rendering DPI (default: 200, higher=slower but better quality)",
)
@click.option(
    "-w",
    "--workers",
    type=int,
    default=1,
    help="Parallel workers for PDF pages (default: 1). Note: Ollama processes sequentially, so >1 workers mainly overlap I/O, not GPU inference.",
)
@click.option(
    "--analyze-figures",
    is_flag=True,
    help="Extract and analyze embedded figures/images from PDFs with AI descriptions.",
)
@click.option(
    "--max-dim",
    "max_dimension",
    type=int,
    default=None,
    help="Maximum image dimension (width or height). Larger images are resized to prevent timeouts. Default: 1920. Set to 0 to disable.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="List files that would be processed without running OCR. Shows file count, sizes, and page counts.",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Suppress non-error output. Only print file paths on success (useful for scripting).",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.pass_context
def process(
    ctx: click.Context,
    input_path: Path,
    output_dir: Optional[Path],
    recursive: bool,
    model_name: str,
    prompt: Optional[str],
    task: str,
    extract_images: bool,
    no_metadata: bool,
    dpi: int,
    workers: int,
    analyze_figures: bool,
    max_dimension: Optional[int],
    dry_run: bool,
    quiet: bool,
    verbose: bool,
) -> None:
    """Process documents and images with OCR.

    INPUT_PATH can be a single file or a directory containing multiple files.

    Supported formats: PDF, JPG, PNG, WEBP, GIF, BMP, TIFF

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

    try:
        model_manager = ModelManager(model_name=model_name, max_dimension=max_dimension)
        model_manager.load_model()

        processor_kwargs = {
            "model_manager": model_manager,
            "extract_images": extract_images,
            "include_metadata": not no_metadata,
            "dpi": dpi,
            "workers": workers,
            "analyze_figures": analyze_figures,
        }
        if output_dir:
            processor_kwargs["output_dir"] = output_dir

        processor = OCRProcessor(**processor_kwargs)

        if input_path.is_file():
            result = processor.process_file(input_path, prompt=prompt, show_progress=not verbose and not quiet)
            output_path = processor.save_result(result)
            if quiet:
                console.print(str(output_path))
            else:
                console.print(f"[dim]->[/dim] {output_path}")
        else:
            results = processor.process_batch(
                input_path,
                recursive=recursive,
                prompt=prompt,
                show_progress=not verbose and not quiet,
            )

            if quiet:
                for result in results:
                    base_name = result.input_path.stem
                    output_path = processor.output_dir / f"{base_name}.md"
                    console.print(str(output_path))
            else:
                table = Table(show_header=True, header_style="bold")
                table.add_column("File", style="cyan")
                table.add_column("Pages", justify="right")
                table.add_column("Time (s)", justify="right")

                for result in results:
                    table.add_row(
                        result.input_path.name,
                        str(result.page_count),
                        f"{result.processing_time:.2f}",
                    )

                console.print(table)

        model_manager.unload_model()

    except Exception as e:
        err_console.print(f"[red]error:[/red] {e}")
        sys.exit(1)


@cli.command()
def info() -> None:
    """Show system and configuration information."""
    print_banner()

    sys_table = Table(title="System Information")
    sys_table.add_column("Component", style="cyan")
    sys_table.add_column("Status", style="green")

    sys_table.add_row("Python", f"{sys.version_info.major}.{sys.version_info.minor}")
    sys_table.add_row("Ollama URL", settings.ollama_url)

    model_manager = ModelManager()
    ollama_running = model_manager._check_ollama_running()
    model_available = model_manager._check_model_available() if ollama_running else False

    sys_table.add_row("Ollama Running", "Yes" if ollama_running else "No")
    sys_table.add_row("deepseek-ocr Model", "Available" if model_available else "Not found")

    console.print(sys_table)

    settings_table = Table(title="Current Settings")
    settings_table.add_column("Setting", style="cyan")
    settings_table.add_column("Value", style="yellow")

    settings_table.add_row("Model", settings.model_name)
    settings_table.add_row("Output Directory", str(settings.output_dir))
    settings_table.add_row("Extract Images", str(settings.extract_images))
    settings_table.add_row("Max Image Dimension", str(settings.max_dimension))

    console.print(settings_table)

    console.print("\n[bold]Supported Formats:[/bold]")
    console.print("Images: JPG, PNG, WEBP, GIF, BMP, TIFF")
    console.print("Documents: PDF\n")

    if not ollama_running:
        console.print("[yellow]Ollama is not running. Start with: ollama serve[/yellow]\n")
    elif not model_available:
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
                argv = (
                    argv[:first_non_option_index]
                    + ["process"]
                    + argv[first_non_option_index:]
                )
                sys.argv = [sys.argv[0], *argv]

    cli(obj={})


if __name__ == "__main__":
    main()
