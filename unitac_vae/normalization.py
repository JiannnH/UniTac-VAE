"""Z-score normalization for sensor signals and depth maps.

Stats formats (JSON-serialisable):
    sensor_stats = {"gelsight": {"mean": [3], "std": [3]}, ..., "xela": {...}}
    depth_stats  = {"depth_map": {"mean": float, "std": float}}
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from .utils import load_json

EPS = 1e-6


def _broadcast(stats: Dict, like: torch.Tensor):
    mean = torch.as_tensor(stats["mean"], device=like.device, dtype=like.dtype)
    std = torch.as_tensor(stats["std"], device=like.device, dtype=like.dtype)
    if like.dim() == 4:            # vision (B, C, H, W)
        mean, std = mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)
    elif like.dim() == 3:          # taxel (B, N, C)
        mean, std = mean.view(1, 1, -1), std.view(1, 1, -1)
    elif like.dim() == 2:          # single taxel sample (N, C)
        mean, std = mean.view(1, -1), std.view(1, -1)
    return mean, std


def normalize_signal(x: torch.Tensor, stats: Dict) -> torch.Tensor:
    mean, std = _broadcast(stats, x)
    return (x - mean) / (std + EPS)


def denormalize_signal(x: torch.Tensor, stats: Dict) -> torch.Tensor:
    mean, std = _broadcast(stats, x)
    return x * std + mean


def _depth_ms(depth_stats: Dict, like: torch.Tensor):
    s = depth_stats.get("depth_map", depth_stats)
    mean = torch.as_tensor(s["mean"], device=like.device, dtype=like.dtype)
    std = torch.as_tensor(s["std"], device=like.device, dtype=like.dtype).clamp(min=EPS)
    return mean, std


def normalize_depth(x: torch.Tensor, depth_stats: Optional[Dict]) -> torch.Tensor:
    if depth_stats is None:
        return x
    mean, std = _depth_ms(depth_stats, x)
    return (x - mean) / std


def denormalize_depth(x: torch.Tensor, depth_stats: Optional[Dict]) -> torch.Tensor:
    if depth_stats is None:
        return x
    mean, std = _depth_ms(depth_stats, x)
    return x * std + mean


def load_stats(path: str) -> Dict:
    return load_json(path)
