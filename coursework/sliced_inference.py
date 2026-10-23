"""Run tiled inference on large images and merge detections in image coordinates.

This script intentionally keeps tiled inference outside the training pipeline. It
can therefore be tested with synthetic detections before the final ``best.pt`` is
available.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_ROOT / "ultralytics"))
for cache_path in (os.environ["MPLCONFIGDIR"], os.environ["YOLO_CONFIG_DIR"]):
    Path(cache_path).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO_ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# 将大图按固定尺寸和重叠率划分成多个切片，保证边缘区域也被覆盖。
def generate_tiles(
    image_width: int,
    image_height: int,
    tile_size: int = 640,
    overlap: float = 0.25,
) -> list[tuple[int, int, int, int]]:
    """Return ``(x1, y1, x2, y2)`` crops covering the complete image."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")

    # 计算一个方向上的所有起始坐标，并强制补上最后一个起点。
    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        stride = max(1, int(tile_size * (1 - overlap)))
        values = list(range(0, length - tile_size + 1, stride))
        last = length - tile_size
        if values[-1] != last:
            values.append(last)
        return values

    # 横向和纵向分别生成起点，再组合成二维切片坐标。
    x_starts = starts(image_width)
    y_starts = starts(image_height)
    tiles = []
    for y1 in y_starts:
        for x1 in x_starts:
            x2 = min(x1 + tile_size, image_width)
            y2 = min(y1 + tile_size, image_height)
            tiles.append((x1, y1, x2, y2))
    return tiles


# 将切片内的局部检测框加上切片偏移，转换为原图坐标。
def map_boxes_to_image(
    boxes,
    offset: tuple[int, int],
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Map tile-local ``xyxy, confidence, class`` detections to image coordinates."""
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError("boxes must have shape (N, 6): xyxy, confidence, class")
    # 复制输入，避免修改模型原始输出。
    mapped = boxes.clone()
    mapped[:, [0, 2]] += offset[0]
    mapped[:, [1, 3]] += offset[1]
    # Advanced indexing returns a copy in PyTorch, so clamp each column in place.
    mapped[:, 0].clamp_(0, image_width)
    mapped[:, 2].clamp_(0, image_width)
    mapped[:, 1].clamp_(0, image_height)
    mapped[:, 3].clamp_(0, image_height)
    # 过滤裁剪后宽或高为 0 的无效框。
    return mapped[(mapped[:, 2] > mapped[:, 0]) & (mapped[:, 3] > mapped[:, 1])]


# 对所有切片映射回来的框执行按类别 NMS，去掉重叠重复框。
def merge_detections(
    detections,
    iou_threshold: float = 0.50,
    max_det: int = 300,
) -> torch.Tensor:
    """Apply class-aware NMS to image-coordinate detections."""
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError("detections must have shape (N, 6): xyxy, confidence, class")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must satisfy 0 <= iou_threshold <= 1")
    if max_det <= 0:
        raise ValueError("max_det must be positive")
    if detections.shape[0] == 0:
        return detections.reshape(0, 6)

    import torch
    from torchvision.ops import nms

    kept_indices = []
    # 不同类别的框不能互相抑制，因此逐类别执行 NMS。
    for class_id in detections[:, 5].unique(sorted=True):
        class_indices = torch.where(detections[:, 5] == class_id)[0]
        keep = nms(detections[class_indices, :4], detections[class_indices, 4], iou_threshold)
        kept_indices.append(class_indices[keep])
    # 合并各类别保留的索引，再按置信度排序并限制总框数。
    indices = torch.cat(kept_indices)
    indices = indices[torch.argsort(detections[indices, 4], descending=True)[:max_det]]
    return detections[indices]


# 将原图像素坐标框转换为 YOLO 标签格式的归一化文本。
def xyxy_to_yolo(detections, image_width: int, image_height: int) -> list[str]:
    """Convert image-coordinate detections to normalized YOLO text lines."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    lines = []
    # YOLO 标签顺序为：类别、中心点 x/y、宽、高、置信度。
    for x1, y1, x2, y2, confidence, class_id in detections.detach().cpu().tolist():
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2 / image_width
        center_y = (y1 + y2) / 2 / image_height
        lines.append(
            f"{int(class_id)} {center_x:.6f} {center_y:.6f} "
            f"{width / image_width:.6f} {height / image_height:.6f} {confidence:.6f}"
        )
    return lines


# 调用 Ultralytics 对单个切片进行检测，统一返回 CPU 上的六列框数据。
def _predict_tile(
    model,
    crop: np.ndarray,
    imgsz: int,
    conf: float,
    predict_iou: float,
    max_det: int,
    device: str,
) -> torch.Tensor:
    """Run one tile through Ultralytics and return a CPU ``(N, 6)`` tensor."""
    import torch

    results = model.predict(
        source=crop,
        imgsz=imgsz,
        conf=conf,
        iou=predict_iou,
        max_det=max_det,
        device=device,
        save=False,
        verbose=False,
    )
    # 没有检测结果时返回空的 (0, 6) 张量，便于后续拼接。
    if not results or results[0].boxes is None:
        return torch.zeros((0, 6), dtype=torch.float32)
    return results[0].boxes.data.detach().cpu().float()


# 完成单张原图的完整流程：切片、逐块推理、坐标回映射和全图去重。
def predict_image(
    model,
    image: np.ndarray,
    *,
    imgsz: int = 640,
    tile_size: int = 640,
    overlap: float = 0.25,
    conf: float = 0.25,
    predict_iou: float = 0.70,
    merge_iou: float = 0.50,
    max_det: int = 300,
    device: str = "cpu",
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Predict one BGR image and return merged boxes plus timing metadata."""
    import numpy as np
    import torch

    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR array with shape (H, W, 3)")
    height, width = image.shape[:2]
    # 先确定切片位置，再逐块调用模型，避免一次性缩放整张大图。
    tiles = generate_tiles(width, height, tile_size, overlap)
    detections = []
    inference_start = time.perf_counter()
    for x1, y1, x2, y2 in tiles:
        local_boxes = _predict_tile(
            model,
            image[y1:y2, x1:x2],
            imgsz,
            conf,
            predict_iou,
            max_det,
            device,
        )
        if len(local_boxes):
            # 模型输出仍是切片局部坐标，需要加回切片左上角偏移。
            detections.append(map_boxes_to_image(local_boxes, (x1, y1), width, height))
    inference_ms = (time.perf_counter() - inference_start) * 1000
    # 汇总所有切片结果；同一目标可能在重叠区域出现多次。
    all_boxes = (
        torch.cat(detections, dim=0)
        if detections
        else torch.zeros((0, 6), dtype=torch.float32)
    )
    # 在原图坐标系中执行最终 NMS，并记录推理统计信息。
    merged = merge_detections(all_boxes, merge_iou, max_det)
    timing = {
        "tiles": len(tiles),
        "detections_before_nms": int(len(all_boxes)),
        "detections_after_nms": int(len(merged)),
        "inference_ms": inference_ms,
        "mean_tile_ms": inference_ms / len(tiles),
    }
    return merged, timing


# 解析输入源，支持单张图片或包含图片的目录。
def _image_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"Image source does not exist: {source}")


# 在原图上绘制检测框和类别置信度，用于人工检查结果。
def _draw_detections(image, detections, names):
    import cv2

    output = image.copy()
    for x1, y1, x2, y2, confidence, class_id in detections.cpu().tolist():
        class_index = int(class_id)
        if isinstance(names, dict):
            class_name = names.get(class_index, str(class_index))
        elif names and class_index < len(names):
            class_name = names[class_index]
        else:
            class_name = str(class_index)
        label = f"{class_name} {confidence:.2f}"
        left, top, right, bottom = map(int, (x1, y1, x2, y2))
        cv2.rectangle(output, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(output, label, (left, max(18, top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return output


# 批量处理图片，并保存可视化、YOLO 标签、JSON 和计时汇总。
def run_directory(model, args: argparse.Namespace) -> None:
    """Run the CLI workflow and save images, labels, JSON, and timing CSV."""
    import cv2

    source = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    image_dir = output_dir / "images"
    prediction_dir = output_dir / "predictions"
    # 预测图片和结构化预测结果分别保存，便于报告和后续评估使用。
    image_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    image_paths = _image_paths(source)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found under: {source}")

    names = getattr(model, "names", None)
    if names is None:
        names = getattr(getattr(model, "model", None), "names", {})
    config = {
        "weights": str(Path(args.weights).expanduser().resolve()),
        "source": str(source),
        "imgsz": args.imgsz,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "conf": args.conf,
        "predict_iou": args.predict_iou,
        "merge_iou": args.merge_iou,
        "max_det": args.max_det,
        "device": args.device,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    timing_rows = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        started = time.perf_counter()
        detections, timing = predict_image(
            model,
            image,
            imgsz=args.imgsz,
            tile_size=args.tile_size,
            overlap=args.overlap,
            conf=args.conf,
            predict_iou=args.predict_iou,
            merge_iou=args.merge_iou,
            max_det=args.max_det,
            device=args.device,
        )
        timing["total_ms"] = (time.perf_counter() - started) * 1000
        stem = image_path.stem
        # 同时保存 YOLO 文本和 JSON，分别服务于评估脚本和结果复核。
        lines = xyxy_to_yolo(detections, image.shape[1], image.shape[0])
        (prediction_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        json_boxes = [
            {"x1": float(row[0]), "y1": float(row[1]), "x2": float(row[2]), "y2": float(row[3]),
             "confidence": float(row[4]), "class_id": int(row[5])}
            for row in detections.tolist()
        ]
        (prediction_dir / f"{stem}.json").write_text(
            json.dumps({"image": image_path.name, "tiles": timing["tiles"], "boxes": json_boxes}, indent=2),
            encoding="utf-8",
        )
        cv2.imwrite(str(image_dir / image_path.name), _draw_detections(image, detections, names))
        timing_rows.append({"image": image_path.name, **timing})

    # 汇总每张图耗时、切片数和 NMS 前后框数量。
    with (output_dir / "timing.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(timing_rows[0]))
        writer.writeheader()
        writer.writerows(timing_rows)
    summary = {
        "images": len(timing_rows),
        "total_detections": sum(row["detections_after_nms"] for row in timing_rows),
        "mean_total_ms": sum(row["total_ms"] for row in timing_rows) / len(timing_rows),
        "mean_tiles": sum(row["tiles"] for row in timing_rows) / len(timing_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Processed {len(image_paths)} image(s); results saved to {output_dir}")


# 定义脚本命令行参数，方便替换权重和调整切片实验参数。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to a YOLO .pt model")
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--predict-iou", type=float, default=0.70)
    parser.add_argument("--merge-iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


# 程序入口：加载 YOLO 权重，再调用批量推理流程。
def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    run_directory(model, args)


if __name__ == "__main__":
    main()
