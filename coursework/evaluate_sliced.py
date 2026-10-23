"""Evaluate predictions produced by ``sliced_inference.py`` on original images.

The sliced inference script is responsible for prediction and cross-tile NMS.
This module only reads its per-image JSON files, matches boxes to the original
YOLO labels, and computes metrics with the same AP implementation used by the
local Ultralytics code.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


# 支持的图片后缀；评估时通过标签文件名查找对应的原图。
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# VisDrone 的 10 个类别名称，类别编号必须和标签转换脚本保持一致。
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


def build_iou_thresholds(start: float = 0.50, end: float = 0.95, step: float = 0.05) -> np.ndarray:
    """Return inclusive IoU thresholds, avoiding floating-point drift."""
    if not 0 <= start <= end <= 1:
        raise ValueError("IoU thresholds must satisfy 0 <= start <= end <= 1")
    if step <= 0:
        raise ValueError("IoU step must be positive")
    # 用整数个数生成阈值，避免 0.95 这类浮点数出现精度误差。
    count = int(round((end - start) / step))
    thresholds = start + np.arange(count + 1, dtype=np.float64) * step
    return np.round(thresholds, 10)


def _find_image(images_dir: Path, stem: str) -> Path:
    """Find an image by stem so the evaluator supports common image suffixes."""
    # 标签和预测文件只保存 stem，因此需要在图片目录中补回实际后缀。
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image found for label stem '{stem}' under {images_dir}")


def _image_size(image_path: Path) -> tuple[int, int]:
    """Read image dimensions without adding another image dependency."""
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    # OpenCV 的 shape 顺序是 height, width, channels，返回时改成 width, height。
    height, width = image.shape[:2]
    return width, height


def load_yolo_labels(label_path: Path, image_width: int, image_height: int) -> list[tuple[int, float, float, float, float]]:
    """Load normalized YOLO labels and convert them to absolute ``xyxy`` boxes."""
    targets = []
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label file: {label_path}")
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        if len(fields) < 5:
            raise ValueError(f"{label_path}:{line_number}: expected at least 5 fields")
        # YOLO 标签格式为 class, center_x, center_y, width, height，坐标均已归一化。
        class_id, center_x, center_y, box_width, box_height = map(float, fields[:5])
        if class_id < 0 or class_id != int(class_id):
            raise ValueError(f"{label_path}:{line_number}: invalid class id {class_id}")
        values = (center_x, center_y, box_width, box_height)
        if any(value < 0 or value > 1 for value in values) or box_width <= 0 or box_height <= 0:
            raise ValueError(f"{label_path}:{line_number}: invalid normalized box {values}")
        # 评估匹配需要像素坐标，因此将归一化 xywh 转换成绝对 xyxy。
        targets.append(
            (
                int(class_id),
                (center_x - box_width / 2) * image_width,
                (center_y - box_height / 2) * image_height,
                (center_x + box_width / 2) * image_width,
                (center_y + box_height / 2) * image_height,
            )
        )
    return targets


def load_prediction_json(prediction_path: Path) -> list[dict[str, float | int]]:
    """Load one sliced-inference JSON file and validate its box fields."""
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    boxes = payload.get("boxes", [])
    if not isinstance(boxes, list):
        raise ValueError(f"{prediction_path}: 'boxes' must be a list")
    predictions = []
    # JSON 中的框已经由切片推理器映射回原图坐标，不需要再次加切片偏移。
    for index, box in enumerate(boxes):
        required = ("x1", "y1", "x2", "y2", "confidence", "class_id")
        if any(key not in box for key in required):
            raise ValueError(f"{prediction_path}: box {index} is missing a required field")
        x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
        confidence = float(box["confidence"])
        class_id = int(box["class_id"])
        # 忽略宽度或高度为零的无效框，避免影响 IoU 计算。
        if x2 <= x1 or y2 <= y1:
            continue
        predictions.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence,
                "class_id": class_id,
            }
        )
    return predictions


def _iou(box: tuple[float, float, float, float], targets: np.ndarray) -> np.ndarray:
    """Compute IoU between one ``xyxy`` box and an ``N x 4`` target array."""
    if len(targets) == 0:
        return np.zeros(0, dtype=np.float64)
    # 向量化计算一个预测框与多个真实框的交并比。
    left = np.maximum(box[0], targets[:, 0])
    top = np.maximum(box[1], targets[:, 1])
    right = np.minimum(box[2], targets[:, 2])
    bottom = np.minimum(box[3], targets[:, 3])
    intersection = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    area_box = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area_targets = np.maximum(0, targets[:, 2] - targets[:, 0]) * np.maximum(0, targets[:, 3] - targets[:, 1])
    return intersection / np.maximum(area_box + area_targets - intersection, 1e-16)


def _match_image(
    predictions: list[dict[str, float | int]],
    targets: list[tuple[int, float, float, float, float]],
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match one image at every IoU threshold using confidence ordering."""
    # 先按置信度从高到低排序，符合 PR 曲线和 AP 的计算要求。
    predictions = sorted(predictions, key=lambda item: float(item["confidence"]), reverse=True)
    tp = np.zeros((len(predictions), len(thresholds)), dtype=bool)
    confidences = np.asarray([float(item["confidence"]) for item in predictions], dtype=np.float64)
    predicted_classes = np.asarray([int(item["class_id"]) for item in predictions], dtype=np.int64)
    target_classes = np.asarray([int(item[0]) for item in targets], dtype=np.int64)

    target_array = np.asarray([item[1:] for item in targets], dtype=np.float64).reshape(-1, 4)
    # 每个 IoU 阈值独立匹配，且同一个真实框在同一阈值下只能匹配一次。
    for threshold_index, threshold in enumerate(thresholds):
        matched_targets: set[int] = set()
        for prediction_index, prediction in enumerate(predictions):
            # 预测框只能与相同类别的真实框匹配。
            same_class = np.flatnonzero(target_classes == int(prediction["class_id"]))
            same_class = np.asarray([index for index in same_class if index not in matched_targets], dtype=np.int64)
            if len(same_class) == 0:
                continue
            box = tuple(float(prediction[key]) for key in ("x1", "y1", "x2", "y2"))
            overlaps = _iou(box, target_array[same_class])
            best_local = int(np.argmax(overlaps))
            if overlaps[best_local] >= threshold:
                matched_targets.add(int(same_class[best_local]))
                tp[prediction_index, threshold_index] = True
    return tp, confidences, predicted_classes, target_classes


def compute_metrics(
    records: list[tuple[str, list[dict[str, float | int]], list[tuple[int, float, float, float, float]]]],
    thresholds: np.ndarray,
) -> tuple[dict, list[dict]]:
    """Compute Ultralytics-compatible AP metrics from per-image records."""
    true_positive_parts = []
    confidence_parts = []
    predicted_class_parts = []
    target_class_parts = []
    # 先逐张原图匹配，再把所有图片的结果拼接起来计算全数据集 PR 曲线。
    for _, predictions, targets in records:
        tp, confidence, predicted_classes, target_classes = _match_image(predictions, targets, thresholds)
        true_positive_parts.append(tp)
        confidence_parts.append(confidence)
        predicted_class_parts.append(predicted_classes)
        target_class_parts.append(target_classes)

    true_positives = np.concatenate(true_positive_parts, axis=0) if true_positive_parts else np.zeros((0, len(thresholds)), dtype=bool)
    confidences = np.concatenate(confidence_parts) if confidence_parts else np.zeros(0, dtype=np.float64)
    predicted_classes = np.concatenate(predicted_class_parts) if predicted_class_parts else np.zeros(0, dtype=np.int64)
    target_classes = np.concatenate(target_class_parts) if target_class_parts else np.zeros(0, dtype=np.int64)
    if len(target_classes) == 0:
        raise ValueError("No ground-truth boxes were found")

    # 复用仓库内 Ultralytics 的 AP 实现，保证指标口径与标准验证结果一致。
    from ultralytics.utils.metrics import ap_per_class

    names = {index: name for index, name in enumerate(CLASS_NAMES)}
    _, _, precision, recall, f1, ap, unique_classes, *_ = ap_per_class(
        true_positives,
        confidences,
        predicted_classes,
        target_classes,
        names=names,
    )
    # 第 0 列对应 IoU=0.50；全部列的平均值对应 mAP50-95。
    map50 = float(ap[:, 0].mean()) if len(ap) else 0.0
    map50_95 = float(ap.mean()) if len(ap) else 0.0
    per_class = []
    for index, class_id in enumerate(unique_classes.tolist()):
        per_class.append(
            {
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id),
                "ground_truth": int(np.sum(target_classes == class_id)),
                "predictions": int(np.sum(predicted_classes == class_id)),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "AP50": float(ap[index, 0]),
                "AP50-95": float(ap[index].mean()),
            }
        )
    metrics = {
        "images": len(records),
        "ground_truth_boxes": int(len(target_classes)),
        "predictions": int(len(predicted_classes)),
        "iou_thresholds": [float(value) for value in thresholds],
        "mAP50": map50,
        "mAP50-95": map50_95,
        "precision": float(precision.mean()) if len(precision) else 0.0,
        "recall": float(recall.mean()) if len(recall) else 0.0,
        "f1": float(f1.mean()) if len(f1) else 0.0,
    }
    return metrics, per_class


def evaluate_dataset(predictions_dir: Path, labels_dir: Path, images_dir: Path, thresholds: np.ndarray) -> tuple[dict, list[dict]]:
    """Load every original image/label pair and evaluate its merged predictions."""
    # 以真实标签为遍历基准，确保没有预测文件的图片也会被计入漏检。
    label_paths = sorted(labels_dir.glob("*.txt"))
    if not label_paths:
        raise FileNotFoundError(f"No YOLO label files found under {labels_dir}")
    records = []
    for label_path in label_paths:
        image_path = _find_image(images_dir, label_path.stem)
        image_width, image_height = _image_size(image_path)
        targets = load_yolo_labels(label_path, image_width, image_height)
        prediction_path = predictions_dir / f"{label_path.stem}.json"
        # 缺少预测文件时按空预测处理，表示该图片所有目标均未检出。
        predictions = load_prediction_json(prediction_path) if prediction_path.is_file() else []
        records.append((label_path.stem, predictions, targets))
    return compute_metrics(records, thresholds)


def write_outputs(output_dir: Path, metrics: dict, per_class: list[dict]) -> None:
    """Write machine-readable overall and per-class reports."""
    # 同时输出 JSON 和 CSV，分别方便程序读取和报告/表格整理。
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps({**metrics, "per_class": per_class}, indent=2), encoding="utf-8"
    )
    fieldnames = ["class_id", "class_name", "ground_truth", "predictions", "precision", "recall", "f1", "AP50", "AP50-95"]
    with (output_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_class)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # predictions 是切片推理生成的 JSON；labels 和 images 必须来自同一个数据划分。
    parser.add_argument("--predictions", required=True, type=Path, help="Directory containing sliced-inference JSON predictions")
    parser.add_argument("--labels", required=True, type=Path, help="Directory containing original-image YOLO labels")
    parser.add_argument("--images", required=True, type=Path, help="Directory containing the original images")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for metrics.json and CSV reports")
    parser.add_argument("--iou-start", type=float, default=0.50)
    parser.add_argument("--iou-end", type=float, default=0.95)
    parser.add_argument("--iou-step", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 默认使用 COCO/Ultralytics 常用的 IoU=0.50:0.05:0.95。
    thresholds = build_iou_thresholds(args.iou_start, args.iou_end, args.iou_step)
    metrics, per_class = evaluate_dataset(
        args.predictions.expanduser().resolve(),
        args.labels.expanduser().resolve(),
        args.images.expanduser().resolve(),
        thresholds,
    )
    write_outputs(args.output_dir.expanduser().resolve(), metrics, per_class)
    print(
        f"Evaluated {metrics['images']} images; "
        f"mAP50={metrics['mAP50']:.4f}, mAP50-95={metrics['mAP50-95']:.4f}; "
        f"results saved to {args.output_dir.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
