import json
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.json"


def load_config(path=None):
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    if not np.isfinite(cfg["cube_size_mm"]) or cfg["cube_size_mm"] <= 0:
        raise ValueError("cube_size_mm must be positive")
    axes = [tuple(c["axis"]) for c in cfg["colors"].values()]
    required = {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)}
    if len(axes) != 6 or set(axes) != required:
        raise ValueError("colors must map bijectively onto the six signed coordinate axes")
    K = np.asarray(cfg["camera"]["matrix"], float)
    if K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.isfinite(K).all():
        raise ValueError("invalid camera matrix")
    if not np.allclose(K[2], [0, 0, 1]) or K[0, 1] != 0 or K[1, 0] != 0:
        raise ValueError("camera matrix must be an OpenCV zero-skew pinhole matrix")
    dist = np.asarray(cfg["camera"]["distortion"], float)
    if dist.ndim != 1 or len(dist) not in (0, 4, 5, 8, 12, 14) or not np.isfinite(dist).all():
        raise ValueError("invalid OpenCV distortion coefficients")
    if len(cfg["camera"]["reference_size"]) != 2 or min(cfg["camera"]["reference_size"]) <= 0:
        raise ValueError("invalid camera.reference_size")
    for color in cfg["colors"].values():
        rgb = np.asarray(color["rgb"], float)
        if rgb.shape != (3,) or not np.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 255).any():
            raise ValueError("rgb must contain three values in [0,255]")
    return cfg


def camera_for_image(cfg, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    camera = cfg["camera"]
    rw, rh = camera["reference_size"]
    if (width, height) != (rw, rh) and not camera.get("allow_resize", False):
        raise ValueError("Image dimensions differ from camera.reference_size")
    sx, sy = width / rw, height / rh
    if camera.get("calibrated") and abs(sx - sy) > 1e-5:
        raise ValueError(
            "Calibrated image aspect ratio changed; supply intrinsics for the actual crop/rotation"
        )
    K = np.asarray(camera["matrix"], float).copy()
    K[0] *= sx
    K[1] *= sy
    # Pixel-center convention under cv2.resize.
    K[0, 2] += 0.5 * sx - 0.5
    K[1, 2] += 0.5 * sy - 0.5
    return K, np.asarray(camera["distortion"], float)
