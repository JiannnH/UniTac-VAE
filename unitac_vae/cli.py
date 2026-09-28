"""Argument helpers shared by the scripts (dataset root, checkpoint selection)."""
from __future__ import annotations

import argparse
import os
from typing import Dict

from .constants import (DEPTH_CKPT_NAME, DEPTH_STATS_NAME, SENSOR_CKPT_NAME, SENSOR_STATS_NAME, SENSORS)
from .data import load_dataset
from .models.io import load_depth_vae, load_sensor_vae
from .normalization import load_stats
from .utils import DATA_ROOT_ENV, resolve_data_root


def add_data_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("data")
    g.add_argument("--data-root", help=f"released dataset root (default: ${DATA_ROOT_ENV})")


def paired_dataset(args, **kw):
    return load_dataset(resolve_data_root(args.data_root), **kw)


def add_checkpoint_args(p: argparse.ArgumentParser, default_dir="weights") -> None:
    g = p.add_argument_group("weights")
    g.add_argument("--weights", default=default_dir,
                   help="directory with depth_vae.pth, <sensor>_vae.pth, depth_stats.json, sensor_stats.json")
    g.add_argument("--depth-ckpt", help="override: depth VAE state dict")
    g.add_argument("--sensor-ckpt-dir", help="override: directory of sensor VAE state dicts")
    g.add_argument("--sensor-ckpt-pattern", default=SENSOR_CKPT_NAME,
                   help="file name pattern inside --sensor-ckpt-dir, e.g. best_{sensor}_vae.pth")
    g.add_argument("--depth-stats", help="override: depth_stats.json")
    g.add_argument("--sensor-stats", help="override: sensor_stats.json")
    g.add_argument("--latent-dim", type=int, default=64)


def load_models(args, device) -> Dict:
    d = args.weights
    depth_path = args.depth_ckpt or os.path.join(d, DEPTH_CKPT_NAME)
    sdir = args.sensor_ckpt_dir or d
    return {"depth": load_depth_vae(depth_path, args.latent_dim, device),
            "sensors": {s: load_sensor_vae(s, os.path.join(sdir, args.sensor_ckpt_pattern.format(sensor=s)),
                                           args.latent_dim, device) for s in SENSORS},
            "depth_stats": load_stats(args.depth_stats or os.path.join(d, DEPTH_STATS_NAME)),
            "sensor_stats": load_stats(args.sensor_stats or os.path.join(d, SENSOR_STATS_NAME))}
