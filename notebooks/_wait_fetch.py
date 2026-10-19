"""Poll the adamworld-generate kernel and fetch its output when it finishes.

Prints only when the status CHANGES (QUEUED -> RUNNING -> COMPLETE) so it can be
driven by Monitor without spamming. On a terminal state it fetches the output and
prints a DONE line. Run:  PYTHONUTF8=1 python notebooks/_wait_fetch.py [out_subdir]
"""

from __future__ import annotations

import os
import shutil
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi

KREF = "adamlahchimi/adamworld-generate"
OUT = os.path.join(".kaggle_run", sys.argv[1] if len(sys.argv) > 1 else "full")
TERMINAL = ("complete", "error", "cancel")


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    last = ""
    for _ in range(140):  # ~35 min at 15s
        try:
            s = str(api.kernels_status(KREF))
        except Exception as exc:
            s = f"(status error: {exc})"
        if s != last:
            print("status:", s, flush=True)
            last = s
        if any(k in s.lower() for k in TERMINAL):
            break
        time.sleep(15)

    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)
    try:
        api.kernels_output(KREF, OUT)
    except Exception as exc:
        print("fetch partial:", exc, flush=True)
    print("DONE files:", os.listdir(OUT), flush=True)


if __name__ == "__main__":
    main()
