# adamworld (core)

The Python core of **AdamWorld** — the generation pipeline (stages 1–8), world
format, caching/profiling, CLI, and FastAPI backend.

See the repository root `README.md` for the project overview and `docs/` for the
architecture, research, licenses, and plan.

```bash
pip install -e ".[dev]"      # CPU dev + tests (mock backends)
pip install -e ".[gpu]"      # + CUDA models (T4/P100): torch, diffusers, gsplat, ...
pytest -q                    # CPU smoke + unit tests
adamworld generate --prompt "a quiet forest clearing" --backend mock --out ../worlds/demo
adamworld serve
```
