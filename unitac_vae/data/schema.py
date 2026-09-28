"""The in-memory sample schema shared by the loader, the single-sample files and the plots.

A *flat sample* is::

    {
      "sample_id": "hammer_point_01_target_5.0N",
      "object": "hammer", "point_name": "hammer_point_01", "force_N": 5.0,
      "gelsight": {"signal": (3,320,240), "depth_map": (50,50), "pose": (6,), "ft_forces": (6,)},
      "digit":    {...same...},
      "papill":   {"signal": (9,3),  "depth_map": (50,50), "pose": (6,), "ft_forces": (6,)},
      "xela":     {"signal": (24,3), "depth_map": (50,50), "pose": (6,), "ft_forces": (6,)},
    }

``mesh_points`` (1024, 3) is added per sensor when requested. Batches are the
default-collated form (leading batch axis on tensors, lists for strings).
Each sensor carries its *own* pose-conditioned depth map.
"""
