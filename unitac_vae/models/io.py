"""Checkpoint loading helpers."""
from __future__ import annotations

import os
from typing import Dict, Iterable

import torch

from ..constants import (DEPTH_CKPT_NAME, DEPTH_STATS_NAME, IMAGE_HW, SENSOR_CKPT_NAME,
                         SENSOR_STATS_NAME, SENSORS)
from ..normalization import load_stats
from .depth_vae import DepthVAE
from .sensor_vae import build_sensor_vae


def _load_state(path: str, device):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location=device, weights_only=True)
    return obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj


def load_depth_vae(path: str, latent_dim: int = 64, device="cpu") -> DepthVAE:
    model = DepthVAE(latent_dim=latent_dim)
    model.load_state_dict(_load_state(path, device), strict=True)
    return model.to(device).eval()


def load_sensor_vae(sensor: str, path: str, latent_dim: int = 64, device="cpu", image_hw=IMAGE_HW):
    model = build_sensor_vae(sensor, latent_dim, image_hw)
    model.load_state_dict(_load_state(path, device), strict=True)
    return model.to(device).eval()


def load_release_checkpoints(ckpt_dir: str, device="cpu", latent_dim: int = 64,
                             sensors: Iterable[str] = SENSORS) -> Dict:
    """Load ``depth_vae.pth``, ``<sensor>_vae.pth`` and the two stats JSONs from one directory."""
    return {"depth": load_depth_vae(os.path.join(ckpt_dir, DEPTH_CKPT_NAME), latent_dim, device),
            "sensors": {s: load_sensor_vae(s, os.path.join(ckpt_dir, SENSOR_CKPT_NAME.format(sensor=s)),
                                           latent_dim, device) for s in sensors},
            "depth_stats": load_stats(os.path.join(ckpt_dir, DEPTH_STATS_NAME)),
            "sensor_stats": load_stats(os.path.join(ckpt_dir, SENSOR_STATS_NAME))}
