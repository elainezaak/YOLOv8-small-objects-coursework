"""Verify dependency versions, acceleration, and all custom model configs."""

import argparse
from pathlib import Path
import os
import platform
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
import torchvision  # noqa: E402
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless PyTorch can use an NVIDIA CUDA device.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Operating system: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    print(f"Ultralytics version: {ultralytics.__version__}")
    print(f"Ultralytics path: {Path(ultralytics.__file__).resolve()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    if torch.cuda.is_available():
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        for index in range(torch.cuda.device_count()):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")

    expected_versions = {
        "Python": (platform.python_version(), "3.10"),
        "PyTorch": (torch.__version__.split("+")[0], "2.2.2"),
        "Torchvision": (torchvision.__version__.split("+")[0], "0.17.2"),
        "Ultralytics": (ultralytics.__version__, "8.1.20"),
    }
    mismatches = [
        f"{name}: expected {expected}, found {actual}"
        for name, (actual, expected) in expected_versions.items()
        if (name == "Python" and not actual.startswith(expected))
        or (name != "Python" and actual != expected)
    ]
    if mismatches:
        raise RuntimeError("Version check failed:\n" + "\n".join(mismatches))
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required, but PyTorch cannot access a CUDA GPU.")

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

