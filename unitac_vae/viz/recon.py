"""Reconstruction / cross-modal figures."""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..constants import DEPTH, SENSOR_DISPLAY_NAMES, SENSORS, TAXEL_SENSORS, VISION_SENSORS
from ..eval.cross_modal import translate
from .tactile import plot_tactile_on_axis

DEPTH_CMAP = "viridis"


def _show_depth(ax, dm, vmin, vmax):
    ax.imshow(np.asarray(dm)[::-1, :], cmap=DEPTH_CMAP, vmin=vmin, vmax=vmax)


def _show_image(ax, img):
    ax.imshow(np.clip(np.transpose(np.asarray(img), (1, 2, 0)), 0, 1))


def plot_sample(sample: Dict, depth_vmin=None, depth_vmax=None):
    """One flat sample: top row native signals, bottom row each sensor's depth map."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for c, s in enumerate(SENSORS):
        d = sample[s]
        sig = d["signal"].numpy() if torch.is_tensor(d["signal"]) else d["signal"]
        if s in VISION_SENSORS:
            _show_image(axes[0, c], sig)
            axes[0, c].axis("off")
        else:
            plot_tactile_on_axis(axes[0, c], sig, s)
        axes[0, c].set_title(SENSOR_DISPLAY_NAMES[s])
        dm = d["depth_map"].numpy() if torch.is_tensor(d["depth_map"]) else d["depth_map"]
        lo = dm.min() if depth_vmin is None else depth_vmin
        hi = 0.0 if depth_vmax is None else depth_vmax
        _show_depth(axes[1, c], dm, lo, hi)
        axes[1, c].axis("off")
        axes[1, c].set_title(f"{SENSOR_DISPLAY_NAMES[s]} depth map (mm)")
    fig.suptitle(sample.get("sample_id", ""))
    fig.tight_layout()
    return fig


@torch.no_grad()
def plot_cross_modal_grid(sample: Dict, models: Dict, depth_vae, sensor_stats: Dict, depth_stats: Dict,
                          device="cpu", depth_reference: str = "gelsight", depth_vmin: float = -3.0,
                          depth_vmax: float = 0.0, sensors=SENSORS, annotate_taxels: bool = False):
    """Rows: source modality (with its ground truth in column 0); columns: decoded targets."""
    mods = list(sensors) + [DEPTH]

    def gt(m):
        if m == DEPTH:
            return sample[depth_reference]["depth_map"]
        return sample[m]["signal"]

    gen = {s: {t: translate(gt(s).unsqueeze(0).to(device).float(), s, t, models, depth_vae, sensor_stats,
                            depth_stats)[0].cpu().numpy() for t in mods} for s in mods}
    scales = {}
    for t in TAXEL_SENSORS:
        vals = [gt(t).numpy()[..., 2]] + [gen[s][t][..., 2] for s in mods]
        cat = np.concatenate([v.ravel() for v in vals])
        scales[t] = (cat.min(), cat.max())
    fig, axes = plt.subplots(len(mods), len(mods) + 1, figsize=(3 * (len(mods) + 1), 2.6 * len(mods)))
    for r, s in enumerate(mods):
        ax = axes[r, 0]
        arr = gt(s).numpy()
        if s == DEPTH:
            _show_depth(ax, arr, depth_vmin, depth_vmax)
        elif s in VISION_SENSORS:
            _show_image(ax, arr)
        else:
            plot_tactile_on_axis(ax, arr, s, "", *scales[s], annotate=annotate_taxels)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(SENSOR_DISPLAY_NAMES[s], fontsize=11, fontweight="bold")
        if r == 0:
            ax.set_title("Input (GT)", fontsize=11)
        for c, t in enumerate(mods):
            ax = axes[r, c + 1]
            arr = gen[s][t]
            if t == DEPTH:
                _show_depth(ax, arr, depth_vmin, depth_vmax)
            elif t in VISION_SENSORS:
                _show_image(ax, arr)
            else:
                plot_tactile_on_axis(ax, arr, t, "", *scales[t], annotate=annotate_taxels)
            ax.axis("off")
            if r == 0:
                ax.set_title(SENSOR_DISPLAY_NAMES[t], fontsize=11)
    fig.suptitle(f"{sample.get('sample_id', '')}: source (rows) -> target (columns)")
    fig.tight_layout()
    return fig
