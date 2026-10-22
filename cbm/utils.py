"""Flat utility set: file hashing, process ETA titles, atomic writes, pool draining.

Everything here is dataset-neutral plumbing shared by the ThyroidXL pipeline
scripts; no domain logic lives in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from concurrent.futures import Future, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import Any

import pandas as pd

# --------------------------------------------------------------------- hashing


def sha256_file(path: Path | str) -> str:
    """Return a streaming SHA256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_config_hash(config: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON dump (sorted keys, compact separators)."""
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# --------------------------------------------------------------- process title

_SHOULD_UPDATE_TITLE = False


def _rounded_end_time(seconds_remaining: float, *, now: datetime | None = None) -> datetime:
    current = now or datetime.now().astimezone()
    estimated = current + timedelta(seconds=max(float(seconds_remaining), 0.0))
    rounded = estimated.replace(second=0, microsecond=0)
    remainder = rounded.minute % 10
    if remainder or estimated.second or estimated.microsecond:
        rounded += timedelta(minutes=10 - remainder if remainder else 10)
    return rounded


def start_estimated_process(estimated_minutes: float) -> str:
    """Enable ETA titles and set the initial process title."""
    global _SHOULD_UPDATE_TITLE

    _SHOULD_UPDATE_TITLE = True
    return update_estimated_process(float(estimated_minutes) * 60.0)


def update_estimated_process(seconds_remaining: float) -> str:
    """Set ``CBM_YYYYMMDD_HHMM`` when ETA title updates are enabled."""
    if not _SHOULD_UPDATE_TITLE:
        return ""
    try:
        from setproctitle import setproctitle
    except ModuleNotFoundError:
        # Local lightweight environments may intentionally omit this optional
        # monitoring dependency; preprocessing itself must still be runnable.
        return ""

    title = f"CBM_{_rounded_end_time(seconds_remaining):%Y%m%d_%H%M}"
    setproctitle(title)
    return title


# ------------------------------------------------------------ atomic artifacts


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a CSV through a temporary file and an atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write indented JSON through a temporary file and an atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


# --------------------------------------------------------------- pool draining


def drain_futures(
    futures: Sequence[Future],
    *,
    label: str,
    update_every: int = 10,
    print_every: int = 100,
) -> list[Any]:
    """Collect ``as_completed`` results while updating the process ETA title.

    Timing starts when draining begins (after submission), so the estimate
    covers per-case work rather than pool startup. The remaining-time process
    title is refreshed every ``update_every`` completions and a progress line
    is printed every ``print_every`` completions.
    """
    total = len(futures)
    started = monotonic()
    results: list[Any] = []
    remaining = 0.0
    for completed, future in enumerate(as_completed(futures), start=1):
        results.append(future.result())
        if completed % update_every == 0 or completed == total:
            elapsed = monotonic() - started
            remaining = elapsed / completed * (total - completed)
            update_estimated_process(remaining)
        if completed % print_every == 0 or completed == total:
            print(f"[{completed}/{total}] {label} eta={remaining:.1f}s", flush=True)
    update_estimated_process(0)
    return results
