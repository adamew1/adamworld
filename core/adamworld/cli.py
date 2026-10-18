"""AdamWorld command-line interface.

adamworld generate --prompt "..." | --image a.jpg | --panorama p.jpg
adamworld expand world/ --direction north --prompt "..."
adamworld edit   world/ --region ... --regenerate|--delete --style "..."
adamworld export world/ --format ply|spz|glb --out out.ply
adamworld serve
"""

from __future__ import annotations

import shutil
from pathlib import Path

import typer
from rich.console import Console

from adamworld.config import Settings
from adamworld.pipeline import Pipeline
from adamworld.profiling import Profiler
from adamworld.types import GenerationRequest

app = typer.Typer(add_completion=False, help="Generate and explore 3D worlds from prompts/images.")
console = Console()


@app.command()
def generate(
    prompt: str = typer.Option(None, "--prompt", "-p", help="Text prompt."),
    image: list[Path] = typer.Option(None, "--image", "-i", help="Seed image(s)."),
    panorama: Path = typer.Option(None, "--panorama", help="User-supplied equirectangular pano."),
    out: Path = typer.Option(Path("worlds/world"), "--out", "-o", help="Output world dir."),
    backend: str = typer.Option("default", help="default | mock | flux-pano | large"),
    resolution: int = typer.Option(1024, help="Working resolution."),
    seed: int = typer.Option(0, help="Deterministic seed."),
    offload: bool = typer.Option(False, help="Enable CPU offload / sequential loading."),
    profile: bool = typer.Option(False, help="Print per-stage time + peak VRAM."),
    force: bool = typer.Option(False, help="Ignore cache and recompute."),
) -> None:
    if not prompt and not image and not panorama:
        raise typer.BadParameter("Provide --prompt, --image, or --panorama.")
    req = GenerationRequest(
        prompt=prompt,
        image_paths=list(image or []),
        panorama_path=panorama,
        seed=seed,
        resolution=resolution,
        backend=backend,
    )
    profiler = Profiler()
    settings = Settings(offload=offload)

    def _progress(stage: str, frac: float) -> None:
        console.print(f"[dim]{frac * 100:5.1f}%[/dim] {stage}")

    pipeline = Pipeline(settings=settings, profiler=profiler, force=force, progress=_progress)
    console.print(f"[bold]Generating[/bold] ({req.kind()}, backend={backend}) -> {out}")
    world = pipeline.generate(req, out)
    meta = world.meta()
    console.print(f"[green]Done[/green] world [bold]{meta.id}[/bold] at {world.root}")
    console.print(f"  splats: {world.splats_ply}")
    console.print(f"  bounds: {meta.bounds.aabb_min} .. {meta.bounds.aabb_max}")
    if profile:
        console.print("\n[bold]Profile[/bold]")
        console.print(profiler.table())


@app.command()
def export(
    world: Path = typer.Argument(..., help="World directory."),
    format: str = typer.Option("ply", "--format", "-f", help="ply | spz | glb"),
    out: Path = typer.Option(None, "--out", "-o", help="Output file."),
) -> None:
    mapping = {"ply": "splats.ply", "spz": "splats.spz", "glb": "mesh.glb"}
    if format not in mapping:
        raise typer.BadParameter("format must be ply, spz, or glb")
    src = world / mapping[format]
    if not src.exists():
        console.print(f"[red]No {format} in {world}[/red] (spz/glb land in M6).")
        raise typer.Exit(1)
    dst = out or Path(f"{world.name}.{format}")
    shutil.copyfile(src, dst)
    console.print(f"[green]Exported[/green] {dst}")


@app.command()
def expand(
    world: Path = typer.Argument(...),
    direction: str = typer.Option("north", help="north|south|east|west or a camera."),
    prompt: str = typer.Option("", help="What to generate at the new edge."),
) -> None:
    console.print("[yellow]World expansion lands in M4.[/yellow]")
    raise typer.Exit(1)


@app.command()
def edit(
    world: Path = typer.Argument(...),
    region: str = typer.Option(None, help="x0,y0,z0,x1,y1,z1 AABB."),
    regenerate: bool = typer.Option(False),
    delete: bool = typer.Option(False),
    style: str = typer.Option("", help="Style prompt for regeneration."),
) -> None:
    console.print("[yellow]Region editing lands in M5.[/yellow]")
    raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    worlds: Path = typer.Option(Path("worlds"), help="Worlds storage dir."),
) -> None:
    import uvicorn

    from adamworld.api import create_app

    console.print(f"[bold]AdamWorld API[/bold] on http://{host}:{port}  (worlds: {worlds})")
    uvicorn.run(create_app(worlds_root=worlds), host=host, port=port)


if __name__ == "__main__":
    app()
