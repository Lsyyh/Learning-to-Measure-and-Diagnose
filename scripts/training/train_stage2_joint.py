#!/usr/bin/env python3
"""Thin CLI wrapper for the Stage-2 joint-consistent trainer of record.

One cohort/seed per invocation over the frozen per-cohort config
(``stage2_joint_final.yaml``); the full recipe lives in
``cbm.train.stage2_joint``. Trains on the final train split only, monitors
the validation regression quality for checkpoint selection, and never reads
either cohort's test partition.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cbm.train.stage1 import CUBLAS_WORKSPACE_CONFIG  # noqa: E402
from cbm.train.stage2_joint import main  # noqa: E402

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)

if __name__ == "__main__":
    main(root=ROOT)
