# AdamWorld dev tasks — run `just <task>`. (If you don't have `just`, the commands
# below are plain shell and can be run directly.)

set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# List tasks
default:
    @just --list

# Install the Python core in editable mode with dev deps
install:
    cd core && pip install -e ".[dev]"

# Install with GPU extras (needs CUDA)
install-gpu:
    cd core && pip install -e ".[gpu,dev]"

# Install web deps
web-install:
    cd web && npm install

# Lint everything
lint:
    cd core && ruff check . && ruff format --check .
    cd web && npm run lint

# Auto-format
fmt:
    cd core && ruff check --fix . && ruff format .

# Type-check Python
typecheck:
    cd core && mypy adamworld

# Run Python tests (CPU)
test:
    cd core && pytest -q

# Build the web viewer
web-build:
    cd web && npm run build

# Full CI-equivalent check (CPU)
ci: lint typecheck test web-build

# Generate a demo world with mock backends (no GPU)
demo:
    cd core && adamworld generate --prompt "a quiet forest clearing" --backend mock --out ../worlds/demo

# Run the backend API
serve:
    cd core && adamworld serve

# Run the web dev server
web-dev:
    cd web && npm run dev

# Write T4 benchmark table (needs GPU)
bench:
    cd core && python scripts/benchmark.py
