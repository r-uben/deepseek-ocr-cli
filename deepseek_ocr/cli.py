"""Command-line interface for DeepSeek OCR."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from deepseek_ocr import __version__
from deepseek_ocr.backends import create_backend
from deepseek_ocr.config import settings
from deepseek_ocr.processor import OCRProcessor
from deepseek_ocr.utils import setup_logging

console = Console()


def print_banner() -> None:
    console.print(f"[dim]deepseek-ocr v{__version__}[/dim]")


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """DeepSeek OCR CLI - OCR processing via Ollama or vLLM.

    Supports Ollama (local, default) and vLLM (OpenAI-compatible API) backends.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(level=settings.log_level, verbose=verbose)


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
    "--backend",
    type=click.Choice(["ollama", "vllm"]),
    default=None,
    help="Backend to use: 'ollama' (local, default) or 'vllm' (OpenAI-compatible). Can also set DEEPSEEK_OCR_BACKEND env var.",
)
@click.option(
    "--vllm-url",
    "vllm_base_url",
    type=str,
    default=None,
    help="vLLM API URL (default: http://localhost:8000/v1). Can also set DEEPSEEK_OCR_VLLM_BASE_URL env var.",
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
    backend: Optional[str],
    vllm_base_url: Optional[str],
) -> None:
    """Process documents and images with OCR.

    INPUT_PATH can be a single file or a directory containing multiple files.

    Supported formats: PDF, JPG, PNG, WEBP, GIF, BMP, TIFF

    Examples:

        # Process a single PDF
        deepseek-ocr process document.pdf

        # Process all files in a directory
        deepseek-ocr process ./documents/ --recursive

        # Use custom prompt
        deepseek-ocr process image.jpg --prompt "Extract all text"

        # Extract page images from PDF
        deepseek-ocr process paper.pdf --extract-images
    """
    print_banner()

    # Resolve backend from CLI flag or config
    backend_type = backend or settings.backend

    # For vLLM, default model is deepseek-vl2 unless explicitly specified
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
        backend_instance.load_model()

        processor_kwargs = {
            "backend": backend_instance,
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
            result = processor.process_file(input_path, prompt=prompt, show_progress=not ctx.obj["verbose"])
            output_path = processor.save_result(result)
            console.print(f"→ {output_path}")
        else:
            results = processor.process_batch(
                input_path,
                recursive=recursive,
                prompt=prompt,
                show_progress=not ctx.obj["verbose"],
            )

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

        backend_instance.unload_model()

    except Exception as e:
        console.print(f"[red]error:[/red] {e}")
        sys.exit(1)


@cli.command()
def info() -> None:
    """Show system and configuration information."""
    print_banner()

    sys_table = Table(title="System Information")
    sys_table.add_column("Component", style="cyan")
    sys_table.add_column("Status", style="green")

    sys_table.add_row("Python", f"{sys.version_info.major}.{sys.version_info.minor}")
    sys_table.add_row("Backend", settings.backend)

    # Check Ollama status
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
    settings_table.add_row("Output Directory", str(settings.output_dir))
    settings_table.add_row("Extract Images", str(settings.extract_images))
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
    """Entry point. Supports shorthand `deepseek-ocr INPUT_PATH` by auto-inserting 'process' subcommand."""
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
