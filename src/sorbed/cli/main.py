"""The ``sorbed`` command-line interface.

Commands share the exact pipeline the API uses; no analysis logic lives here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from sorbed.config.settings import get_settings
from sorbed.domain.analysis import WoundAnalysis
from sorbed.io import DecoderRegistry, load_image, sniff_format
from sorbed.pipeline import AnalyzeOptions, analyze_image
from sorbed.report.builder import DEFAULT_ARTIFACTS, write_report
from sorbed.version import __version__

app = typer.Typer(
    add_completion=False,
    help="Explainable pressure-injury (bedsore) image analysis. Decision support, "
    "not a diagnosis — every result must be reviewed by a clinician.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


@app.command()
def analyze(
    image: Annotated[Path, typer.Argument(help="Path to the wound image (any supported format).")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")] = Path("reports"),
    mm_per_px: Annotated[
        float | None, typer.Option("--mm-per-px", help="Known scale in millimeters per pixel.")
    ] = None,
    marker_mm: Annotated[
        float | None, typer.Option("--marker-mm", help="ArUco marker side length in mm.")
    ] = None,
    coin_mm: Annotated[
        float | None, typer.Option("--coin-mm", help="Reference coin diameter in mm.")
    ] = None,
    fmt: Annotated[
        str, typer.Option("--format", "-f", help="Comma list of artifacts, or 'all'.")
    ] = "all",
    json_out: Annotated[
        bool, typer.Option("--json", help="Print the analysis JSON to stdout.")
    ] = False,
) -> None:
    """Analyze one wound image and write a report."""
    if not image.exists():
        err_console.print(f"[red]No such file:[/red] {image}")
        raise typer.Exit(2)

    artifacts = DEFAULT_ARTIFACTS if fmt.strip() == "all" else tuple(
        p.strip() for p in fmt.split(",") if p.strip()
    )
    options = AnalyzeOptions(
        mm_per_px=mm_per_px, marker_length_mm=marker_mm, coin_diameter_mm=coin_mm
    )

    try:
        bundle = analyze_image(image, options=options)
    except Exception as exc:  # surface a clean message, not a traceback
        err_console.print(f"[red]Analysis failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    report = write_report(bundle, out, artifacts=artifacts)
    if json_out:
        console.print_json(bundle.analysis.model_dump_json())
    else:
        _print_summary(bundle.analysis)
        console.print(f"\n[green]Wrote {len(report.artifacts)} artifact(s) to[/green] {out}/")


@app.command()
def inspect(
    image: Annotated[Path, typer.Argument(help="Image to inspect (no grading).")],
) -> None:
    """Decode an image and report its metadata and detected scale, without grading."""
    if not image.exists():
        err_console.print(f"[red]No such file:[/red] {image}")
        raise typer.Exit(2)
    data = image.read_bytes()
    fmt = sniff_format(data, filename=image.name)
    loaded = load_image(data)
    table = Table(title=f"{image.name}", show_header=False)
    table.add_row("Detected format", fmt.value)
    table.add_row("Dimensions", f"{loaded.metadata.width_px} × {loaded.metadata.height_px} px")
    table.add_row("Bit depth", str(loaded.metadata.bit_depth))
    table.add_row("Calibration", loaded.calibration.status.value)
    if loaded.calibration.mm_per_px:
        table.add_row("Scale", f"{loaded.calibration.mm_per_px:.4f} mm/px")
    if loaded.metadata.retained_tags:
        table.add_row("Retained tags", json.dumps(loaded.metadata.retained_tags))
    console.print(table)


@app.command()
def formats() -> None:
    """List image formats and whether a decoder for each is available."""
    table = Table("Format", "Available")
    for fmt, ok in DecoderRegistry().supported_formats().items():
        table.add_row(fmt, "[green]yes[/green]" if ok else "[yellow]needs extras[/yellow]")
    console.print(table)


@app.command()
def schema(
    out: Annotated[Path | None, typer.Option("--out", help="Write schema here instead of stdout.")]
    = None,
) -> None:
    """Print the JSON Schema of the analysis result contract."""
    doc = json.dumps(WoundAnalysis.model_json_schema(), indent=2)
    if out is not None:
        out.write_text(doc, encoding="utf-8")
        console.print(f"[green]Wrote schema to[/green] {out}")
    else:
        console.print_json(doc)


@app.command()
def config() -> None:
    """Show the effective configuration and its digest."""
    settings = get_settings()
    console.print_json(settings.model_dump_json())
    console.print(f"config digest: {settings.digest()[:16]}")


@app.command()
def version() -> None:
    """Print the Sorbed version."""
    console.print(__version__)


def _print_summary(analysis: WoundAnalysis) -> None:
    d = analysis.decision
    g = analysis.metrics.geometry
    grade = d.stage.value.replace("_", " ").title()
    conf = "withheld" if d.abstained else f"{d.confidence * 100:.0f}%"
    console.print(f"\n[bold]Provisional grade:[/bold] {grade}  ([cyan]{conf}[/cyan])")
    if g.area_cm2 is not None:
        console.print(f"Area: {g.area_cm2:.2f} cm²   Size: {g.length_mm:.0f} × {g.width_mm:.0f} mm")
    else:
        console.print(f"Area: {g.area_px:.0f} px (uncalibrated)")
    tissue = ", ".join(
        f"{c.value.replace('_', ' ')} {f * 100:.0f}%"
        for c, f in sorted(analysis.metrics.tissue.fractions.items(), key=lambda kv: -kv[1])
        if f >= 0.02
    )
    console.print(f"Tissue: {tissue}")
    palette = {"info": "blue", "warning": "yellow", "critical": "red"}
    for c in d.caveats:
        color = palette.get(c.severity.value, "white")
        console.print(f"[{color}]! {c.message}[/{color}]")
    console.print(f"\n[dim]{d.narrative}[/dim]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
