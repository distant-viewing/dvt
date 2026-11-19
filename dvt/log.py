"""The line a stage prints when it starts, skips or finishes."""

import time
from pathlib import Path


def log_status(od: Path, step_name: str, status: str) -> None:
    print(f"[{od.name}:{step_name}] {status}")


def log_finished(od: Path, step_name: str, t0: float) -> None:
    elapsed = time.monotonic() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"[{od.name}:{step_name}] Finished ({h:02d}:{m:02d}:{s:02d})")
