"""Data API: released-format loader and the bundled single-sample file."""
from __future__ import annotations

from .paired import PairedTactileDataset, load_sample


def load_dataset(root: str, **kw) -> PairedTactileDataset:
    """All paired multi-sensor samples under ``root`` (flat schema); ``objects=`` narrows to some objects."""
    return PairedTactileDataset(root, **kw)


__all__ = ["PairedTactileDataset", "load_dataset", "load_sample"]
