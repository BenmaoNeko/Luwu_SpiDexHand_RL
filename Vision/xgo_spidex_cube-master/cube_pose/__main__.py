import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .config import load_config
from .estimator import CubePoseEstimator, draw_overlay


def main():
    parser = argparse.ArgumentParser(
        description="Cube 6D pose / articulated single-layer pose from RGB images"
    )
    parser.add_argument(
        "input", type=Path, help="image or directory; auto task uses 1_ / 2_ prefixes"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", choices=["auto", "1", "2"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--angle-reference-deg", type=float)
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = (
        sorted(
            p for p in args.input.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if args.input.is_dir()
        else [args.input]
    )
    if not paths:
        parser.error("No input images")
    args.output.mkdir(parents=True, exist_ok=True)
    estimator = CubePoseEstimator(cfg)
    summary = []
    for path in paths:
        if args.task == "auto":
            if not path.name.startswith(("1_", "2_")):
                parser.error("Use --task 1 or --task 2 for filenames without 1_/2_ prefix")
            task = int(path.name[0])
        else:
            task = int(args.task)
        start = time.perf_counter()
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            parser.error(f"Cannot read {path}")
        result, debug = estimator.estimate(image, task, args.angle_reference_deg)
        result["input"] = str(path.resolve())
        result["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        out = args.output / path.stem
        out.with_suffix(".json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        overlay = draw_overlay(result, debug)
        cv2.imencode(".jpg", overlay)[1].tofile(str(out) + "_overlay.jpg")
        summary.append(result)
        layer = result.get("layer") or result.get("candidate_layer")
        print(
            f"{path.name}: {result['status']}, {result.get('quality', {}).get('edge_support_fraction', 0):.2f} support, {result['elapsed_seconds']:.1f}s"
            + (f", {layer['angle_deg_mod90']:.1f} deg" if layer else ""),
            flush=True,
        )
    if len(paths) > 1:
        (args.output / "results.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
