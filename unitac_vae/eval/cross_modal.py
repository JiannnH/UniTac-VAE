"""Cross-modal inference through the shared latent space (encode -> latent -> decode)."""
from __future__ import annotations

from typing import Dict

import torch

from ..constants import DEPTH
from ..normalization import denormalize_depth, denormalize_signal, normalize_depth, normalize_signal


@torch.no_grad()
def encode(modality: str, x: torch.Tensor, models: Dict, depth_vae, sensor_stats: Dict, depth_stats: Dict):
    """Encode a batch of raw signals (or raw depth maps) to the latent mean."""
    if modality == DEPTH:
        mu, _ = depth_vae.encoder(normalize_depth(x, depth_stats).unsqueeze(1))
    else:
        mu, _ = models[modality].encoder(normalize_signal(x, sensor_stats[modality]))
    return mu


@torch.no_grad()
def decode(modality: str, z: torch.Tensor, models: Dict, depth_vae, sensor_stats: Dict, depth_stats: Dict):
    """Decode latents to raw-unit signals (or depth maps in mm)."""
    if modality == DEPTH:
        return denormalize_depth(depth_vae.decoder(z).squeeze(1), depth_stats)
    return denormalize_signal(models[modality].decoder(z), sensor_stats[modality])


@torch.no_grad()
def translate(x: torch.Tensor, src: str, tgt: str, models: Dict, depth_vae, sensor_stats: Dict, depth_stats: Dict):
    return decode(tgt, encode(src, x, models, depth_vae, sensor_stats, depth_stats), models, depth_vae,
                  sensor_stats, depth_stats)


def find_sample(loader_or_dataset, sample_id: str) -> Dict:
    """Return the flat sample with the given id (single sample, no batch axis)."""
    ds = getattr(loader_or_dataset, "dataset", loader_or_dataset)
    if hasattr(ds, "index"):
        hit = ds.index.index[ds.index["sample_id"] == sample_id]
        if len(hit):
            return ds[int(hit[0])]
    for i in range(len(ds)):
        s = ds[i]
        if s["sample_id"] == sample_id:
            return s
    raise KeyError(f"sample id {sample_id!r} not found")
