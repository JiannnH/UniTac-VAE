"""Reader for the released paired dataset (one HDF5 file per object)."""
from __future__ import annotations

import os
from typing import Dict, Iterable, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..constants import SENSORS, SIGNAL_NAMES


class PairedTactileDataset(Dataset):
    """Flat-schema samples from ``<root>/paired/{index.csv, <object>.h5}``.

    Files are opened lazily per worker process so the dataset is safe with
    ``num_workers > 0``.
    """

    def __init__(self, root: str, objects: Optional[Iterable[str]] = None,
                 sensors: Optional[Iterable[str]] = None, image_dtype=torch.float32,
                 load_mesh_points: bool = False):
        self.root = root
        self.dir = os.path.join(root, "paired")
        index = pd.read_csv(os.path.join(self.dir, "index.csv"))
        if objects is not None:
            index = index[index["object"].isin(list(objects))]
        self.index = index.reset_index(drop=True)
        self.sensors = tuple(sensors) if sensors is not None else SENSORS
        self.image_dtype = image_dtype
        self.load_mesh_points = load_mesh_points
        self._files: Dict[str, h5py.File] = {}

    def _h5(self, fname: str) -> h5py.File:
        if fname not in self._files:
            self._files[fname] = h5py.File(os.path.join(self.dir, fname), "r")
        return self._files[fname]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict:
        row = self.index.iloc[i]
        f = self._h5(row["file"])
        r = int(row["row"])
        out = {"sample_id": str(row["sample_id"]), "object": str(row["object"]),
               "point_name": str(row["point_name"]),
               "force_N": torch.tensor(float(row["force_N"]), dtype=torch.float32)}
        for s in self.sensors:
            g = f[s]
            sig = torch.from_numpy(np.asarray(g[SIGNAL_NAMES[s]][r]))
            sig = sig.to(self.image_dtype) if s in ("gelsight", "digit") else sig.float()
            d = {"signal": sig,
                 "depth_map": torch.from_numpy(np.asarray(g["depth_map"][r])).float(),
                 "pose": torch.from_numpy(np.asarray(g["pose"][r])).float(),
                 "ft_forces": torch.from_numpy(np.asarray(g["ft_forces"][r])).float()}
            if self.load_mesh_points:
                d["mesh_points"] = torch.from_numpy(np.asarray(g["mesh_points"][r])).float()
            out[s] = d
        return out

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = {}
        return state



# --- the bundled single-sample .npz -------------------------------------------
_FIELDS = ("signal", "depth_map", "pose", "ft_forces")


def load_sample(path: str) -> Dict:
    """Read a single-sample ``.npz`` (one key per ``<sensor>/<field>``) into the flat sample schema."""
    z = np.load(path)
    out = {"sample_id": str(z["sample_id"]), "object": str(z["object"]), "point_name": str(z["point_name"]),
           "force_N": torch.tensor(float(z["force_N"]), dtype=torch.float32)}
    for s in SENSORS:
        out[s] = {k: torch.from_numpy(np.asarray(z[f"{s}/{k}"])).float() for k in _FIELDS}
    return out
