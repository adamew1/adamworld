# Contributing to AdamWorld

Thanks for your interest! AdamWorld is a clean-room, license-clean project. A few ground rules keep it that way.

## Clean-room & licensing rules (non-negotiable)

1. **Only open inputs.** Contribute only code you wrote or that comes from compatibly-licensed open source. Never paste leaked, decompiled, or scraped proprietary code, weights, or assets.
2. **No proprietary outputs as training/eval data.** Do not use closed world-model APIs (e.g. commercial "prompt→world" services) to produce data for training or benchmarking.
3. **License every model/dependency.** Any new model or library must be recorded in `docs/LICENSES.md` with its license. Non-commercial or "community"/use-restricted licenses may only be added as an **optional, flagged backend**, never as a default. Making a non-Apache model a *default* requires maintainer sign-off.
4. **No trademarks.** Don't use "Marble", "World Labs", "Chisel", or their logos in names, UI, docs, or packages.

## Development setup

```bash
cd core
python -m venv ../.venv && . ../.venv/Scripts/activate
pip install -e ".[dev]"
pre-commit install
pytest -q
```

Web:

```bash
cd web && npm install && npm run lint && npm run build
```

## Standards

- **Python:** ruff (lint + format) and mypy must pass. Public functions are typed. Line length 100.
- **TypeScript:** eslint must pass; `npm run build` must succeed.
- **Tests stay GPU-free.** Unit tests mock models. There is exactly one CPU smoke end-to-end test (`tests/test_smoke_e2e.py`) that runs the whole pipeline with `--backend mock` at tiny resolution. New stages must ship a `mock` backend and unit tests.
- **Every stage logs peak VRAM + wall time** and supports `--profile`.
- **Cache everything** by content hash; a re-run must skip completed stages.
- **Fail with a fix.** User-facing errors (especially OOM) must suggest a concrete remedy.
- **Never commit weights or generated worlds.** They are git-ignored and downloaded/generated on demand.

## Pull requests

- Keep PRs focused; update `docs/PROGRESS.md` and relevant docs in the same PR.
- CI (CPU-only: ruff, mypy, eslint, pytest smoke, web build) must be green.
- By contributing you agree your contribution is licensed under Apache-2.0.
