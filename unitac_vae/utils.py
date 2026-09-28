"""Small helpers shared by scripts and library code."""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import torch

DATA_ROOT_ENV = "UNITAC_DATA_ROOT"


def get_device(preferred: Optional[str] = None) -> torch.device:
    if preferred and preferred != "auto":
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def resolve_data_root(cli_value: Optional[str]) -> str:
    """Precedence: CLI flag > $UNITAC_DATA_ROOT."""
    root = cli_value or os.environ.get(DATA_ROOT_ENV)
    if not root:
        raise SystemExit(f"No dataset root given. Pass --data-root or set ${DATA_ROOT_ENV}.")
    return os.path.expanduser(root)
