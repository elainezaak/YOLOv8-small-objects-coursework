"""Convert an extracted VisDrone DET dataset to YOLO format and validate it."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
# 默认使用仓库内部的数据目录，便于把代码和本地数据一起交给实验室服务器。
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "VisDrone"
SPLITS = {
    "VisDrone2019-DET-train": 6471,
    "VisDrone2019-DET-val": 548,
    "VisDrone2019-DET-test-dev": 1610,
}
CLASS_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)


@dataclass
class SplitStats:
    images: int = 0
    annotations: int = 0
    labels: int = 0
    objects: int = 0
    ignored: int = 0
    invalid_boxes: int = 0
    duplicate_boxes: int = 0
    empty_labels: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Extracted VisDrone root (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--yaml-name",
        default="VisDrone-local.yaml",
        help="Name of the generated local dataset YAML.",
    )
    return parser.parse_args()


def convert_split(split_dir: Path, expected_images: int) -> SplitStats:
    images_dir = split_dir / "images"
    annotations_dir = split_dir / "annotations"
    labels_dir = split_dir / "labels"

    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise FileNotFoundError(
            f"Missing images/ or annotations/ under {split_dir}. "
            "Extract the VisDrone ZIP files first."
        )

    image_paths = sorted(images_dir.glob("*.jpg"))
    annotation_paths = sorted(annotations_dir.glob("*.txt"))
    if len(image_paths) != expected_images:
        raise RuntimeError(
            f"{split_dir.name}: expected {expected_images} images, found {len(image_paths)}"
        )
    if len(annotation_paths) != len(image_paths):
        raise RuntimeError(
            f"{split_dir.name}: found {len(image_paths)} images but "
            f"{len(annotation_paths)} annotation files"
        )

    labels_dir.mkdir(parents=True, exist_ok=True)
    stats = SplitStats(images=len(image_paths), annotations=len(annotation_paths))

    for annotation_path in tqdm(annotation_paths, desc=f"Converting {split_dir.name}"):
        image_path = images_dir / f"{annotation_path.stem}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found for {annotation_path.name}: {image_path}")

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        yolo_lines: list[str] = []
        seen_lines: set[str] = set()
        for line_number, raw_line in enumerate(
            annotation_path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            row = [value.strip() for value in raw_line.split(",")]
            if len(row) < 6:
                raise ValueError(
                    f"{annotation_path}:{line_number}: expected at least 6 columns"
                )

            if row[4] == "0":
                stats.ignored += 1
                continue

            class_id = int(row[5]) - 1
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(
                    f"{annotation_path}:{line_number}: invalid category {row[5]}"
                )

            left, top, width, height = map(float, row[:4])
            if width <= 0 or height <= 0:
                stats.invalid_boxes += 1
                continue
            box = (
                (left + width / 2) / image_width,
                (top + height / 2) / image_height,
                width / image_width,
                height / image_height,
            )
            if any(value < 0 or value > 1 for value in box):
                raise ValueError(
                    f"{annotation_path}:{line_number}: invalid normalized box {box}"
                )

            yolo_line = f"{class_id} " + " ".join(f"{value:.6f}" for value in box)
            if yolo_line in seen_lines:
                stats.duplicate_boxes += 1
                continue
            seen_lines.add(yolo_line)
            yolo_lines.append(yolo_line)

        label_path = labels_dir / f"{annotation_path.stem}.txt"
        label_path.write_text(
            "\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8"
        )
        stats.labels += 1
        stats.objects += len(yolo_lines)
        stats.empty_labels += not yolo_lines

    return stats


def write_local_yaml(dataset_dir: Path, yaml_name: str) -> Path:
    yaml_path = dataset_dir / yaml_name
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        f"path: {dataset_dir.resolve()}\n"
        "train: VisDrone2019-DET-train/images\n"
        "val: VisDrone2019-DET-val/images\n"
        "test: VisDrone2019-DET-test-dev/images\n\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()

    total_objects = 0
    for split_name, expected_images in SPLITS.items():
        stats = convert_split(dataset_dir / split_name, expected_images)
        total_objects += stats.objects
        print(
            f"OK {split_name}: {stats.images} images, {stats.labels} labels, "
            f"{stats.objects} objects, {stats.ignored} ignored, "
            f"{stats.invalid_boxes} invalid boxes skipped, "
            f"{stats.duplicate_boxes} duplicates removed, "
            f"{stats.empty_labels} empty label files"
        )

    yaml_path = write_local_yaml(dataset_dir, args.yaml_name)
    print(f"Dataset validation passed: {sum(SPLITS.values())} images, {total_objects} objects")
    print(f"Local dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
