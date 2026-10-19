"""Resource profiling: wall time and peak VRAM per stage, plus the OOM helper.

Every stage is wrapped by ``Profiler.track`` so ``--profile`` and the benchmark
script can report T4 numbers. VRAM reporting is a no-op on CPU.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StageProfile:
    name: str
    seconds: float
    peak_vram_mb: float  # 0.0 on CPU / when torch is unavailable


class OutOfMemoryError(RuntimeError):
    """Raised on CUDA OOM with an actionable remedy in the message."""


def _cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _reset_peak() -> None:
    if _cuda_available():
        import torch  # noqa: PLC0415

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _peak_mb() -> float:
    """Peak VRAM for the stage as the allocator's high-water *reserved* bytes.

    We report ``max_memory_reserved`` (not ``max_memory_allocated``): reserved is
    the memory the caching allocator actually held from the GPU, which reflects the
    real footprint. With ``enable_sequential_cpu_offload`` this is legitimately
    low, because model weights are streamed from CPU layer-by-layer rather than
    kept resident — that is the point of offload, not a measurement error.
    """
    if _cuda_available():
        import torch  # noqa: PLC0415

        torch.cuda.synchronize()
        return float(torch.cuda.max_memory_reserved()) / (1024 * 1024)
    return 0.0


@dataclass
class Profiler:
    records: list[StageProfile] = field(default_factory=list)

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        _reset_peak()
        start = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - start
            self.records.append(StageProfile(name, seconds, _peak_mb()))

    def table(self) -> str:
        if not self.records:
            return "(no stages profiled)"
        name_w = max(len(r.name) for r in self.records + [StageProfile("stage", 0, 0)])
        lines = [f"{'stage'.ljust(name_w)}   time(s)   peak VRAM MB (reserved)"]
        lines.append("-" * (name_w + 35))
        for r in self.records:
            lines.append(f"{r.name.ljust(name_w)}   {r.seconds:7.2f}   {r.peak_vram_mb:12.1f}")
        total = sum(r.seconds for r in self.records)
        peak = max((r.peak_vram_mb for r in self.records), default=0.0)
        lines.append("-" * (name_w + 35))
        lines.append(f"{'TOTAL'.ljust(name_w)}   {total:7.2f}   {peak:12.1f}")
        return "\n".join(lines)


def oom_message(stage: str, resolution: int, backend: str) -> str:
    return (
        f"CUDA out of memory in stage '{stage}' at resolution {resolution}px "
        f"(backend='{backend}'). Try one of:\n"
        f"  - lower resolution:   --resolution {max(256, resolution // 2)}\n"
        f"  - enable CPU offload: --offload\n"
        f"  - use a bigger GPU:   --backend large   (needs 24-80 GB)\n"
        f"  - use the CPU mock:   --backend mock     (no GPU, placeholder quality)"
    )
