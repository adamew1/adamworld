"""Poll the adamworld-generate kernel until it finishes, then fetch its output.

Separated from run_on_kaggle so it survives the pusher being stopped. Run:
    PYTHONUTF8=1 python notebooks/_wait_fetch.py [out_subdir]
"""

from __future__ import annotations

import os
import shutil
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi

KREF = "adamlahchimi/adamworld-generate"
OUT = os.path.join(".kaggle_run", sys.argv[1] if len(sys.argv) > 1 else "full")


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    for _ in range(80):  # ~40 min max
        try:
            s = str(api.kernels_status(KREF))
        except Exception as exc:
            s = f"(status error: {exc})"
        print("status", s, flush=True)
        if any(k in s.lower() for k in ("complete", "error", "cancel")):
            break
        time.sleep(30)

    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)
    try:
        api.kernels_output(KREF, OUT)
    except Exception as exc:
        print("fetch partial:", exc, flush=True)
    print("DONE files:", os.listdir(OUT), flush=True)


if __name__ == "__main__":
    main()
