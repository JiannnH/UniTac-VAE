"""Project-wide constants: sensor names, geometry, signal names and units, taxel layouts.

Everything that the dataset schema, the models and the plots agree on lives here so
that there is a single source of truth.
"""

# --- Sensors -----------------------------------------------------------------
SENSORS = ("gelsight", "digit", "papill", "xela")
VISION_SENSORS = ("gelsight", "digit")
TAXEL_SENSORS = ("papill", "xela")
DEPTH = "depth"                      # the shared depth-map modality (depth-map VAE)

SENSOR_DISPLAY_NAMES = {
    "gelsight": "GelSight Mini",
    "digit": "DIGIT",
    "papill": "PapillArray",
    "xela": "uSkin 46",
    DEPTH: "Depth Map",
}

# --- Objects -----------------------------------------------------------------
OBJECTS = ("hammer", "scissor", "largemarker", "windex")

# --- Geometry ----------------------------------------------------------------
IMAGE_HW = (320, 240)          # vision sensors, (H, W) after preprocessing
GRID_SIZE = 50                 # depth map resolution (GRID_SIZE x GRID_SIZE)
DEPTH_REGION_MM = 40.0         # depth map covers a 40 x 40 mm patch

# --- Signals -----------------------------------------------------------------
SIGNAL_NAMES = {               # dataset field name of each sensor's native signal
    "gelsight": "image",
    "digit": "image",
    "papill": "forces_xyz",
    "xela": "taxel_raw",
}
SIGNAL_UNITS = {
    "gelsight": "RGB, per-channel min-max stretched to [0, 1]",
    "digit": "RGB, per-channel min-max stretched to [0, 1]",
    "papill": "pillar force (fx, fy, fz) in newtons, 9 pillars",
    "xela": "raw magnetic taxel readings (x, y, z), baseline-subtracted counts, 24 taxels",
}

# --- Taxel plotting layouts ---------------------------------------------------
# order: index permutation from sensor channel order to a row-major grid.
TAXEL_LAYOUT = {
    "papill": dict(order=[8, 5, 2, 7, 4, 1, 6, 3, 0], rows=3, cols=3,
                   radius=0.35, cmap="Blues", shear_scale=0.5),
    "xela": dict(order=[23, 17, 11, 5, 22, 16, 10, 4, 21, 15, 9, 3,
                        20, 14, 8, 2, 19, 13, 7, 1, 18, 12, 6, 0],
                 rows=6, cols=4, radius=0.25, cmap="Reds", shear_scale=0.001),
}

# --- Release checkpoint file names --------------------------------------------
DEPTH_CKPT_NAME = "depth_vae.pth"
SENSOR_CKPT_NAME = "{sensor}_vae.pth"
DEPTH_STATS_NAME = "depth_stats.json"
SENSOR_STATS_NAME = "sensor_stats.json"
