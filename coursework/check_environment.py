"""Verify that the repository's modified Ultralytics code can build every model."""

from pathlib import Path
import os
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model_configs"
CACHE_ROOT = REPO_ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_ROOT / "ultralytics"))
for cache_path in (os.environ["MPLCONFIGDIR"], os.environ["XDG_CACHE_HOME"], os.environ["YOLO_CONFIG_DIR"]):
    Path(cache_path).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def main() -> None:
    print(f"Ultralytics version: {ultralytics.__version__}")
    print(f"Ultralytics path: {Path(ultralytics.__file__).resolve()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    local_package = REPO_ROOT / "ultralytics" / "__init__.py"
    imported_package = Path(ultralytics.__file__).resolve()
    if imported_package != local_package.resolve():
        raise RuntimeError(
            "Imported the installed Ultralytics package instead of this repository. "
            "Run this script from the repository root."
        )

    for config_path in sorted(MODEL_DIR.glob("*.yaml")):
        model = YOLO(str(config_path))
        parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
        print(f"OK: {config_path.name} ({parameter_count:,} parameters)")

    print("Environment check passed.")


if __name__ == "__main__":
    main()

