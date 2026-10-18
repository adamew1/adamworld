"""Write docs/BENCHMARKS.md from a profiled run.

On a T4/P100 this records real per-stage time + peak VRAM for the default
backend. On CPU (no GPU) it still runs with the mock backend and reports 0 VRAM,
which is enough to validate the harness. Run:  python scripts/benchmark.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from adamworld.config import default_settings
from adamworld.pipeline import Pipeline
from adamworld.profiling import Profiler
from adamworld.types import GenerationRequest

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "docs" / "BENCHMARKS.md"


def main() -> None:
    settings = default_settings()
    backend = "default" if settings.device == "cuda" else "mock"
    rows = []
    for res in (256, 512):
        profiler = Profiler()
        pipeline = Pipeline(settings=settings, profiler=profiler)
        with tempfile.TemporaryDirectory() as td:
            pipeline.generate(
                GenerationRequest(
                    prompt="a quiet forest clearing", resolution=res, backend=backend
                ),
                Path(td) / "w",
            )
        rows.append((res, profiler))

    lines = [
        "# BENCHMARKS.md",
        "",
        f"Device: **{settings.device}**  ·  backend: **{backend}**",
        "",
        "> On CPU the mock backend runs and VRAM reads 0. Re-run on a T4/P100 with the",
        "> `[gpu]` extra for real default-backend numbers.",
        "",
    ]
    for res, profiler in rows:
        lines.append(f"## resolution {res}")
        lines.append("```")
        lines.append(profiler.table())
        lines.append("```")
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
