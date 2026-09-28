"""Taxel-array rendering (circle size/colour = normal component, arrow = shear)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..constants import TAXEL_LAYOUT


def plot_tactile_on_axis(ax, forces_xyz, sensor: str, title: str = "", vmin=None, vmax=None,
                         annotate: bool = True):
    ax.clear()
    if forces_xyz is None or np.size(forces_xyz) == 0 or sensor not in TAXEL_LAYOUT:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    lay = TAXEL_LAYOUT[sensor]
    cmap = plt.get_cmap(lay["cmap"])
    f = np.asarray(forces_xyz)[lay["order"]]
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    vmin = fz.min() if vmin is None else vmin
    vmax = fz.max() if vmax is None else vmax
    norm = plt.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-6)
    rows, cols = lay["rows"], lay["cols"]
    k = 0
    for i in range(rows):
        for j in range(cols):
            x, y = j, rows - 1 - i
            c = cmap(norm(fz[k]))
            ax.add_patch(plt.Circle((x, y), lay["radius"], color=c, alpha=0.85, ec="k", lw=0.5))
            if annotate:
                label = f"{fz[k]:.0f}" if abs(fz[k]) >= 100 else f"{fz[k]:.2f}"
                ax.text(x, y, label, ha="center", va="center", fontsize=4 if len(label) > 4 else 5,
                        color="white" if norm(fz[k]) > 0.6 else "black")
            if np.hypot(fx[k], fy[k]) > 0.01:
                ax.arrow(x, y, fx[k] * lay["shear_scale"], -fy[k] * lay["shear_scale"],
                         head_width=0.1, fc="k", ec="k")
            k += 1
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
