"""One training entry point for the baseline and repository model variants."""

import argparse
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_ROOT / "ultralytics"))
for cache_path in (os.environ["MPLCONFIGDIR"], os.environ["XDG_CACHE_HOME"], os.environ["YOLO_CONFIG_DIR"]):
    Path(cache_path).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402

MODEL_CONFIGS = {
    "baseline": "yolov8n.pt",
    "p2": REPO_ROOT / "model_configs" / "yolov8-p2.yaml",
    "p2-nop5": REPO_ROOT / "model_configs" / "yolov8-p2_woP5.yaml",
    "fusion": REPO_ROOT / "model_configs" / "yolov8-p2_woP5_sandwich.yaml",
    "final": REPO_ROOT / "model_configs" / "yolov8-p2_woP5_sandwich_CBAM.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_CONFIGS, default="final")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Use 'cpu' without a GPU, or '0' for the first available GPU.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data",
        default=str(REPO_ROOT / "ultralytics" / "cfg" / "datasets" / "VisDrone.yaml"),
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load matching YOLOv8n pretrained weights for a custom model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(MODEL_CONFIGS[args.model]))

    if args.pretrained and args.model != "baseline":
        model.load("yolov8n.pt")

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=30,
        project=str(REPO_ROOT / "runs" / "coursework"),
        name=args.model,
    )


if __name__ == "__main__":
    main()

