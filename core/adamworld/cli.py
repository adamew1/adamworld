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
    layout: Path = typer.Option(None, "--layout", help="Block-out layout JSON (stage 8)."),
    allow_noncommercial: bool = typer.Option(
        False, "--allow-noncommercial", help="Opt into non-commercial models (DAv2 L, FLUX NC)."
    ),
) -> None:
    if layout is not None:
        from adamworld.world.schema import Layout

        lay = Layout.model_validate_json(Path(layout).read_text(encoding="utf-8-sig"))
        pipe = Pipeline(settings=Settings(offload=offload, allow_noncommercial=allow_noncommercial))
        world = pipe.generate_from_layout(
            lay, out, backend=backend, resolution=resolution, seed=seed
        )
        console.print(
            f"[green]Done[/green] layout world at {world.root} ({len(lay.blocks)} blocks)"
        )
        return
    if not prompt and not image and not panorama:
        raise typer.BadParameter("Provide --prompt, --image, --panorama, or --layout.")
    req = GenerationRequest(
        prompt=prompt,
        image_paths=list(image or []),
        panorama_path=panorama,
        seed=seed,
        resolution=resolution,
        backend=backend,
    )
    profiler = Profiler()
    settings = Settings(offload=offload, allow_noncommercial=allow_noncommercial)

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
    if format == "glb" and not src.exists():
        from adamworld.meshing import mesh_world
        from adamworld.world import World

        console.print("Building mesh.glb from cached depth ...")
        ref = mesh_world(World.load(world))
        console.print(f"  mesh: {ref.vertices} verts, {ref.faces} faces")
    if not src.exists():
        console.print(f"[red]No {format} in {world}[/red]")
        raise typer.Exit(1)
    dst = out or Path(f"{world.name}.{format}")
    shutil.copyfile(src, dst)
    console.print(f"[green]Exported[/green] {dst}")


@app.command()
def mesh(
    world: Path = typer.Argument(..., help="World directory."),
    max_rows: int = typer.Option(96, help="Mesh grid resolution (rows)."),
) -> None:
    from adamworld.meshing import mesh_world
    from adamworld.world import World

    ref = mesh_world(World.load(world), max_rows=max_rows)
    console.print(f"[green]Wrote[/green] {ref.path}  ({ref.vertices} verts, {ref.faces} faces)")


@app.command()
def expand(
    world: Path = typer.Argument(..., help="Existing world directory."),
    direction: str = typer.Option("north", help="north|south|east|west|up|down or x,y,z."),
    prompt: str = typer.Option("", help="What to generate at the new edge."),
    distance: float = typer.Option(None, help="Override the step distance."),
) -> None:
    from adamworld.splat import GaussianCloud
    from adamworld.world import World

    w = World.load(world)
    profiler = Profiler()
    pipe = Pipeline(
        profiler=profiler, progress=lambda s, f: console.print(f"[dim]{f * 100:5.1f}%[/dim] {s}")
    )
    pipe.expand(w, direction=direction, prompt=prompt, distance=distance)
    n = len(GaussianCloud.load_ply(w.splats_ply))
    console.print(f"[green]Expanded[/green] {direction}: {n} splats, {len(w.meta().tiles)} tiles")


@app.command()
def edit(
    world: Path = typer.Argument(...),
    region: str = typer.Option(..., help="x0,y0,z0,x1,y1,z1 AABB."),
    regenerate: bool = typer.Option(False, help="Regenerate the region (GPU)."),
    delete: bool = typer.Option(False, help="Delete splats in the region."),
    style: str = typer.Option("", help="Style prompt for regeneration."),
) -> None:
    from adamworld.splat import GaussianCloud
    from adamworld.world import World

    vals = [float(x) for x in region.split(",")]
    if len(vals) != 6:
        raise typer.BadParameter("--region must be x0,y0,z0,x1,y1,z1")
    lo = (min(vals[0], vals[3]), min(vals[1], vals[4]), min(vals[2], vals[5]))
    hi = (max(vals[0], vals[3]), max(vals[1], vals[4]), max(vals[2], vals[5]))
    w = World.load(world)
    before = len(GaussianCloud.load_ply(w.splats_ply))
    Pipeline().edit_region(w, lo, hi, delete=delete, regenerate=regenerate, style=style)
    after = len(GaussianCloud.load_ply(w.splats_ply))
    console.print(f"[green]Edited[/green] region: {before} -> {after} splats")


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
