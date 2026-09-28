#!/usr/bin/env python
"""Translate one sample between every pair of modalities and draw the grid.

Bundled example (no dataset needed):

    python scripts/cross_modal_demo.py --out demo.png --also-plot-sample

Any sample of the released dataset:

    python scripts/cross_modal_demo.py --data-root DATA --sample-id hammer_point_82_target_10.0N --out demo.png
"""
import argparse
import os
import sys

# Make the demo runnable from a plain clone (no installation needed).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unitac_vae.cli import add_checkpoint_args, add_data_args, load_models, paired_dataset
from unitac_vae.data import load_sample
from unitac_vae.eval import find_sample
from unitac_vae.utils import get_device
from unitac_vae.viz import plot_cross_modal_grid, plot_sample

DEFAULT_SAMPLE = os.path.join(REPO_ROOT, "examples", "largemarker_point_24_target_10.0N.npz")
DEFAULT_CKPT = os.path.join(REPO_ROOT, "weights")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", default=DEFAULT_SAMPLE, help="single-sample .npz (default: bundled example)")
    p.add_argument("--sample-id", help="sample id inside the released dataset (needs --data-root)")
    p.add_argument("--out", default="cross_modal_demo.png")
    p.add_argument("--also-plot-sample", action="store_true", help="save a second figure of the raw sample")
    p.add_argument("--depth-reference", default="gelsight")
    p.add_argument("--device", default="auto")
    add_data_args(p)
    add_checkpoint_args(p, default_dir=DEFAULT_CKPT)
    a = p.parse_args()
    dev = get_device(a.device)
    m = load_models(a, dev)
    sample = find_sample(paired_dataset(a), a.sample_id) if a.sample_id else load_sample(a.sample)
    fig = plot_cross_modal_grid(sample, m["sensors"], m["depth"], m["sensor_stats"], m["depth_stats"], dev,
                                depth_reference=a.depth_reference)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", a.out)
    if a.also_plot_sample:
        out2 = a.out.rsplit(".", 1)[0] + "_sample.png"
        plot_sample(sample).savefig(out2, dpi=150, bbox_inches="tight")
        print("wrote", out2)


if __name__ == "__main__":
    main()
