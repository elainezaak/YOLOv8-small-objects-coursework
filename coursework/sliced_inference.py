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
    *,
    return_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
    valid = (mapped[:, 2] > mapped[:, 0]) & (mapped[:, 3] > mapped[:, 1])
    filtered = mapped[valid]
    return (filtered, valid) if return_mask else filtered


def tile_edge_distances(boxes, tile_width: int, tile_height: int) -> torch.Tensor:
    """Return each local box's distance to the nearest tile edge.

    A small value means that the detector may have seen only a truncated part
    of the object.  The metadata is kept separate from the six-column YOLO box
    tensor so existing prediction and evaluation formats remain unchanged.
    """
    import torch

    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError("boxes must have shape (N, 6): xyxy, confidence, class")
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile dimensions must be positive")
    if boxes.shape[0] == 0:
        return boxes.new_zeros((0,), dtype=torch.float32)
    left = boxes[:, 0].clamp(0, tile_width)
    top = boxes[:, 1].clamp(0, tile_height)
    right = boxes[:, 2].clamp(0, tile_width)
    bottom = boxes[:, 3].clamp(0, tile_height)
    return torch.minimum(
        torch.minimum(left, top),
        torch.minimum(tile_width - right, tile_height - bottom),
    )


def intersection_over_smaller(boxes1, boxes2) -> torch.Tensor:
    """Compute pairwise IoS for two ``xyxy`` box tensors.

    IoS is intersection area divided by the area of the smaller box. Unlike
    IoU, it identifies a small prediction almost fully contained in a larger
    duplicate box, which is useful for suppressing tiled-inference duplicates.
    The return shape is ``(len(boxes1), len(boxes2))``.
    """
    import torch

    if boxes1.ndim != 2 or boxes1.shape[1] != 4:
        raise ValueError("boxes1 must have shape (N, 4): xyxy")
    if boxes2.ndim != 2 or boxes2.shape[1] != 4:
        raise ValueError("boxes2 must have shape (M, 4): xyxy")
    if boxes1.device != boxes2.device:
        raise ValueError("boxes1 and boxes2 must be on the same device")

    # Empty inputs still return a correctly shaped matrix for vectorized callers.
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    # IoS is defined for floating-point coordinates; normalize integer inputs.
    if not boxes1.is_floating_point():
        boxes1 = boxes1.float()
    if not boxes2.is_floating_point():
        boxes2 = boxes2.float()

    left = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    top = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    right = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    bottom = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    intersection = (right - left).clamp_min(0) * (bottom - top).clamp_min(0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
    smaller_area = torch.minimum(area1[:, None], area2[None, :])
    return intersection / smaller_area.clamp_min(torch.finfo(intersection.dtype).eps)


def _contained_keep_indices(
    detections,
    ios_threshold: float,
    *,
    edge_distances=None,
    boundary_aware: bool = False,
    edge_margin: float = 32.0,
):
    """Return indices that survive same-class IoS suppression."""
    import torch

    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError("detections must have shape (N, 6): xyxy, confidence, class")
    if not 0 < ios_threshold <= 1:
        raise ValueError("ios_threshold must satisfy 0 < ios_threshold <= 1")
    if edge_margin < 0:
        raise ValueError("edge_margin must be non-negative")
    if edge_distances is not None:
        if edge_distances.ndim != 1 or len(edge_distances) != len(detections):
            raise ValueError("edge_distances must have shape (N,)")
        edge_distances = edge_distances.to(device=detections.device)
    if detections.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.long, device=detections.device)

    # When enabled, complete boxes get priority over boxes touching a tile
    # boundary.  The original confidence remains unchanged in output JSON.
    rank = detections[:, 4]
    if boundary_aware and edge_distances is not None:
        rank = rank + (edge_distances >= edge_margin).to(rank.dtype) * 2.0
    order = torch.argsort(rank, descending=True)
    kept = []
    for index in order.tolist():
        candidate = detections[index]
        if kept:
            kept_tensor = detections[torch.tensor(kept, device=detections.device)]
            same_class = kept_tensor[:, 5] == candidate[5]
            if same_class.any():
                candidate_box = candidate[:4].reshape(1, 4)
                ios = intersection_over_smaller(candidate_box, kept_tensor[same_class, :4])
                if bool((ios >= ios_threshold).any()):
                    continue
        kept.append(index)
    kept_tensor = torch.tensor(kept, dtype=torch.long, device=detections.device)
    return kept_tensor[torch.argsort(detections[kept_tensor, 4], descending=True)]


def suppress_contained_detections(
    detections,
    ios_threshold: float,
    *,
    edge_distances=None,
    boundary_aware: bool = False,
    edge_margin: float = 32.0,
) -> torch.Tensor:
    """Suppress same-class boxes whose IoS exceeds ``ios_threshold``.

    IoS is optional.  If boundary-aware mode is enabled, boxes away from tile
    edges are considered before edge-touching boxes when duplicates compete.
    """
    keep = _contained_keep_indices(
        detections,
        ios_threshold,
        edge_distances=edge_distances,
        boundary_aware=boundary_aware,
        edge_margin=edge_margin,
    )
    return detections[keep]


def filter_small_detections(
    detections,
    small_area: float | None = None,
    small_conf: float | None = None,
) -> tuple[torch.Tensor, int]:
    """Optionally remove only low-confidence boxes below an area threshold.

    Both values must be supplied to enable the filter.  This protects real
    VisDrone small objects from a blanket area-based deletion rule.
    """
    if (small_area is None) != (small_conf is None):
        raise ValueError("small_area and small_conf must be provided together")
    if small_area is None:
        return detections, 0
    if small_area <= 0:
        raise ValueError("small_area must be positive")
    if not 0 <= small_conf <= 1:
        raise ValueError("small_conf must satisfy 0 <= small_conf <= 1")
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError("detections must have shape (N, 6): xyxy, confidence, class")
    widths = (detections[:, 2] - detections[:, 0]).clamp_min(0)
    heights = (detections[:, 3] - detections[:, 1]).clamp_min(0)
    remove = (widths * heights < small_area) & (detections[:, 4] < small_conf)
    return detections[~remove], int(remove.sum().item())


# 对所有切片映射回来的框执行按类别 NMS，去掉重叠重复框。
def merge_detections(
    detections,
    iou_threshold: float = 0.50,
    max_det: int = 300,
    ios_threshold: float | None = None,
    boundary_aware: bool = False,
    edge_distances=None,
    edge_margin: float = 32.0,
    small_area: float | None = None,
    small_conf: float | None = None,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, int]]:
    """Apply configurable cross-tile NMS and optional post-processing filters."""
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError("detections must have shape (N, 6): xyxy, confidence, class")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must satisfy 0 <= iou_threshold <= 1")
    if ios_threshold is not None and not 0 < ios_threshold <= 1:
        raise ValueError("ios_threshold must satisfy 0 < ios_threshold <= 1 when enabled")
    if max_det <= 0:
        raise ValueError("max_det must be positive")
    if edge_margin < 0:
        raise ValueError("edge_margin must be non-negative")
    if edge_distances is not None:
        if edge_distances.ndim != 1 or len(edge_distances) != len(detections):
            raise ValueError("edge_distances must have shape (N,)")
        edge_distances = edge_distances.to(device=detections.device)
    if detections.shape[0] == 0:
        empty = detections.reshape(0, 6)
        empty_stats = {"nms_removed": 0, "ios_removed": 0, "small_removed": 0}
        return (empty, empty_stats) if return_stats else empty

    import torch
    from torchvision.ops import nms

    kept_indices = []
    # A large positive offset makes complete boxes win ties against boxes that
    # touch a tile boundary, while leaving their reported confidence intact.
    ranking_scores = detections[:, 4]
    if boundary_aware and edge_distances is not None:
        ranking_scores = ranking_scores + (edge_distances >= edge_margin).to(ranking_scores.dtype) * 2.0
    # 不同类别的框不能互相抑制，因此逐类别执行 NMS。
    for class_id in detections[:, 5].unique(sorted=True):
        class_indices = torch.where(detections[:, 5] == class_id)[0]
        keep = nms(detections[class_indices, :4], ranking_scores[class_indices], iou_threshold)
        kept_indices.append(class_indices[keep])
    # 合并各类别保留的索引，再按原始置信度排序，保证 AP 输入不被改写。
    indices = torch.cat(kept_indices)
    indices = indices[torch.argsort(detections[indices, 4], descending=True)]
    nms_detections = detections[indices]
    nms_edges = edge_distances[indices] if edge_distances is not None else None
    nms_removed = int(len(detections) - len(nms_detections))
    ios_removed = 0
    if ios_threshold is not None:
        keep = _contained_keep_indices(
            nms_detections,
            ios_threshold,
            edge_distances=nms_edges,
            boundary_aware=boundary_aware,
            edge_margin=edge_margin,
        )
        ios_removed = int(len(nms_detections) - len(keep))
        nms_detections = nms_detections[keep]
        if nms_edges is not None:
            nms_edges = nms_edges[keep]
    nms_detections, small_removed = filter_small_detections(nms_detections, small_area, small_conf)
    result = nms_detections[:max_det]
    if return_stats:
        return result, {
            "nms_removed": nms_removed,
            "ios_removed": ios_removed,
            "small_removed": small_removed,
        }
    return result


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
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one tile and return boxes plus Ultralytics stage timings in milliseconds."""
    import torch

    # Ultralytics records preprocessing, neural-network inference, and its own
    # postprocessing separately in ``Results.speed``.
    started = time.perf_counter()
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
    elapsed_ms = (time.perf_counter() - started) * 1000
    speed = getattr(results[0], "speed", {}) if results else {}
    speed = speed or {}
    # Older/custom predictors may not expose ``speed``; use wall time as a
    # conservative fallback so timing.csv is still usable.
    inference_ms = float(speed.get("inference", elapsed_ms))
    preprocess_ms = float(speed.get("preprocess", 0.0))
    model_postprocess_ms = float(speed.get("postprocess", 0.0))
    timing = {
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "model_postprocess_ms": model_postprocess_ms,
    }
    # 没有检测结果时返回空的 (0, 6) 张量，便于后续拼接。
    if not results or results[0].boxes is None:
        return torch.zeros((0, 6), dtype=torch.float32), timing
    return results[0].boxes.data.detach().cpu().float(), timing


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
    ios_threshold: float | None = None,
    boundary_aware: bool = False,
    edge_margin: float = 32.0,
    small_area: float | None = None,
    small_conf: float | None = None,
    max_det: int = 300,
    device: str = "cpu",
    full_image_pass: bool = False,
    full_imgsz: int | None = None,
    full_conf: float | None = None,
    full_max_det: int | None = None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Predict one image with optional full-image fusion and return merged boxes."""
    import numpy as np
    import torch

    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR array with shape (H, W, 3)")
    if full_image_pass:
        full_imgsz = imgsz if full_imgsz is None else full_imgsz
        full_conf = conf if full_conf is None else full_conf
        full_max_det = max_det if full_max_det is None else full_max_det
        if full_imgsz <= 0:
            raise ValueError("full_imgsz must be positive")
        if not 0 <= full_conf <= 1:
            raise ValueError("full_conf must satisfy 0 <= full_conf <= 1")
        if full_max_det <= 0:
            raise ValueError("full_max_det must be positive")
    height, width = image.shape[:2]
    # 端到端计时从图片已在内存中开始，到跨切片 NMS 完成为止。
    total_start = time.perf_counter()
    # 先确定切片位置，再逐块调用模型，避免一次性缩放整张大图。
    tiles = generate_tiles(width, height, tile_size, overlap)
    detections = []
    edge_metadata = []
    preprocess_ms = 0.0
    inference_ms = 0.0
    model_postprocess_ms = 0.0
    tile_preprocess_ms = 0.0
    tile_inference_ms = 0.0
    tile_model_postprocess_ms = 0.0
    for x1, y1, x2, y2 in tiles:
        local_boxes, tile_timing = _predict_tile(
            model,
            image[y1:y2, x1:x2],
            imgsz,
            conf,
            predict_iou,
            max_det,
            device,
        )
        preprocess_ms += tile_timing["preprocess_ms"]
        inference_ms += tile_timing["inference_ms"]
        model_postprocess_ms += tile_timing["model_postprocess_ms"]
        tile_preprocess_ms += tile_timing["preprocess_ms"]
        tile_inference_ms += tile_timing["inference_ms"]
        tile_model_postprocess_ms += tile_timing["model_postprocess_ms"]
        if len(local_boxes):
            # 模型输出仍是切片局部坐标，需要加回切片左上角偏移。
            mapped, valid = map_boxes_to_image(
                local_boxes,
                (x1, y1),
                width,
                height,
                return_mask=True,
            )
            if len(mapped):
                detections.append(mapped)
                if boundary_aware:
                    # Keep edge metadata aligned with boxes that survive clipping.
                    edge_metadata.append(tile_edge_distances(local_boxes, x2 - x1, y2 - y1)[valid])

    tile_boxes_before_fusion = sum(len(item) for item in detections)
    full_image_preprocess_ms = 0.0
    full_image_inference_ms = 0.0
    full_image_model_postprocess_ms = 0.0
    full_image_boxes_before_fusion = 0
    if full_image_pass:
        # Ultralytics returns boxes in the source-image coordinate system, so
        # the full-image branch only needs clipping, not a tile offset.
        full_boxes, full_timing = _predict_tile(
            model,
            image,
            full_imgsz,
            full_conf,
            predict_iou,
            full_max_det,
            device,
        )
        full_image_preprocess_ms = full_timing["preprocess_ms"]
        full_image_inference_ms = full_timing["inference_ms"]
        full_image_model_postprocess_ms = full_timing["model_postprocess_ms"]
        if len(full_boxes):
            mapped_full, _ = map_boxes_to_image(
                full_boxes,
                (0, 0),
                width,
                height,
                return_mask=True,
            )
            if len(mapped_full):
                detections.append(mapped_full)
                full_image_boxes_before_fusion = len(mapped_full)
                if boundary_aware:
                    # A full-image prediction is not truncated by a tile edge.
                    edge_metadata.append(
                        torch.full(
                            (len(mapped_full),),
                            float(max(width, height, edge_margin)),
                            dtype=torch.float32,
                        )
                    )
        preprocess_ms += full_image_preprocess_ms
        inference_ms += full_image_inference_ms
        model_postprocess_ms += full_image_model_postprocess_ms
    # 汇总所有切片结果；同一目标可能在重叠区域出现多次。
    all_boxes = (
        torch.cat(detections, dim=0)
        if detections
        else torch.zeros((0, 6), dtype=torch.float32)
    )
    all_edges = (
        torch.cat(edge_metadata, dim=0)
        if edge_metadata
        else None
    )
    # 在原图坐标系中执行最终 NMS，并记录推理统计信息。
    merged, merge_stats = merge_detections(
        all_boxes,
        merge_iou,
        max_det,
        ios_threshold=ios_threshold,
        boundary_aware=boundary_aware,
        edge_distances=all_edges,
        edge_margin=edge_margin,
        small_area=small_area,
        small_conf=small_conf,
        return_stats=True,
    )
    total_ms = (time.perf_counter() - total_start) * 1000
    # 论文表格只有 Inference 和 Post-process 两列，因此后处理包含：
    # 切片生成、模型预处理/后处理、坐标映射、拼接和跨切片 NMS。
    postprocess_ms = max(0.0, total_ms - inference_ms)
    timing = {
        "tiles": len(tiles),
        "detections_before_nms": int(len(all_boxes)),
        "detections_after_nms": int(len(merged)),
        "tile_detections_before_nms": int(tile_boxes_before_fusion),
        "full_image_detections_before_nms": int(full_image_boxes_before_fusion),
        "full_image_pass": full_image_pass,
        "full_image_tiles": 1 if full_image_pass else 0,
        "ios_enabled": ios_threshold is not None,
        "ios_threshold": ios_threshold if ios_threshold is not None else 0.0,
        "ios_removed": merge_stats["ios_removed"],
        "nms_removed": merge_stats["nms_removed"],
        "small_filter_enabled": small_area is not None,
        "small_area": small_area if small_area is not None else 0.0,
        "small_conf": small_conf if small_conf is not None else 0.0,
        "small_removed": merge_stats["small_removed"],
        "boundary_aware": boundary_aware,
        "edge_margin": edge_margin if boundary_aware else 0.0,
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "model_postprocess_ms": model_postprocess_ms,
        "postprocess_ms": postprocess_ms,
        "total_ms": total_ms,
        "mean_tile_ms": tile_inference_ms / len(tiles),
        "tile_preprocess_ms": tile_preprocess_ms,
        "tile_inference_ms": tile_inference_ms,
        "tile_model_postprocess_ms": tile_model_postprocess_ms,
        "full_image_preprocess_ms": full_image_preprocess_ms,
        "full_image_inference_ms": full_image_inference_ms,
        "full_image_model_postprocess_ms": full_image_model_postprocess_ms,
        "fusion_nms_removed": merge_stats["nms_removed"] if full_image_pass else 0,
        "fusion_ios_removed": merge_stats["ios_removed"] if full_image_pass else 0,
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
def _draw_detections(image, detections, names, draw_conf: float | None = None):
    import cv2

    if draw_conf is not None and not 0 <= draw_conf <= 1:
        raise ValueError("draw_conf must satisfy 0 <= draw_conf <= 1")
    output = image.copy()
    for x1, y1, x2, y2, confidence, class_id in detections.cpu().tolist():
        if draw_conf is not None and confidence < draw_conf:
            continue
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
    full_imgsz = args.full_imgsz if args.full_imgsz is not None else args.imgsz
    full_conf = args.full_conf if args.full_conf is not None else args.conf
    full_max_det = args.full_max_det if args.full_max_det is not None else args.max_det
    # 切片推理不改变网络结构，因此参数量就是加载权重对应模型的参数总数。
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    config = {
        "weights": str(Path(args.weights).expanduser().resolve()),
        "parameters": parameters,
        "parameters_m": parameters / 1e6,
        "source": str(source),
        "imgsz": args.imgsz,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "conf": args.conf,
        "predict_iou": args.predict_iou,
        "merge_iou": args.merge_iou,
        "ios_threshold": args.ios_threshold,
        "boundary_aware": args.boundary_aware,
        "edge_margin": args.edge_margin if args.boundary_aware else None,
        "small_filter_enabled": args.small_filter,
        "small_area": args.small_area if args.small_filter else None,
        "small_conf": args.small_conf if args.small_filter else None,
        "draw_conf": args.draw_conf,
        "max_det": args.max_det,
        "full_image_pass": args.full_image_pass,
        "full_imgsz": full_imgsz if args.full_image_pass else None,
        "full_conf": full_conf if args.full_image_pass else None,
        "full_max_det": full_max_det if args.full_image_pass else None,
        "device": args.device,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        f"[sliced] start: {len(image_paths)} image(s), device={args.device}, "
        f"tile={args.tile_size}, overlap={args.overlap}, conf={args.conf}, "
        f"merge_iou={args.merge_iou}, ios="
        f"{args.ios_threshold if args.ios_threshold is not None else 'off'}, "
        f"boundary={'on' if args.boundary_aware else 'off'}, "
        f"small_filter={'on' if args.small_filter else 'off'}, "
        f"full_image={'on' if args.full_image_pass else 'off'}, "
        f"draw_conf={args.draw_conf if args.draw_conf is not None else 'all'}, "
        f"parameters={parameters:,} ({parameters / 1e6:.4f}M)",
        flush=True,
    )
    timing_rows = []
    for image_index, image_path in enumerate(image_paths, 1):
        print(
            f"[sliced] [{image_index}/{len(image_paths)}] processing {image_path.name}",
            flush=True,
        )
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        detections, timing = predict_image(
            model,
            image,
            imgsz=args.imgsz,
            tile_size=args.tile_size,
            overlap=args.overlap,
            conf=args.conf,
            predict_iou=args.predict_iou,
            merge_iou=args.merge_iou,
            ios_threshold=args.ios_threshold,
            boundary_aware=args.boundary_aware,
            edge_margin=args.edge_margin,
            small_area=args.small_area if args.small_filter else None,
            small_conf=args.small_conf if args.small_filter else None,
            max_det=args.max_det,
            device=args.device,
            full_image_pass=args.full_image_pass,
            full_imgsz=full_imgsz,
            full_conf=full_conf,
            full_max_det=full_max_det,
        )
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
            json.dumps(
                {
                    "image": image_path.name,
                    "tiles": timing["tiles"],
                    "full_image_pass": timing["full_image_pass"],
                    "tile_detections_before_nms": timing["tile_detections_before_nms"],
                    "full_image_detections_before_nms": timing["full_image_detections_before_nms"],
                    "boxes": json_boxes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        cv2.imwrite(
            str(image_dir / image_path.name),
            _draw_detections(image, detections, names, args.draw_conf),
        )
        timing_rows.append({"image": image_path.name, **timing})
        print(
            f"[sliced] [{image_index}/{len(image_paths)}] done: "
            f"tiles={timing['tiles']}, boxes={timing['detections_after_nms']}, "
            f"nms_removed={timing['nms_removed']}, "
            f"ios_removed={timing['ios_removed']}, "
            f"small_removed={timing['small_removed']}, "
            f"inference={timing['inference_ms']:.2f} ms, "
            f"postprocess={timing['postprocess_ms']:.2f} ms, "
            f"total={timing['total_ms']:.2f} ms",
            flush=True,
        )

    # 汇总每张图耗时、切片数和 NMS 前后框数量。
    with (output_dir / "timing.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(timing_rows[0]))
        writer.writeheader()
        writer.writerows(timing_rows)
    summary = {
        "images": len(timing_rows),
        "parameters": parameters,
        "parameters_m": parameters / 1e6,
        "total_detections": sum(row["detections_after_nms"] for row in timing_rows),
        "full_image_pass": args.full_image_pass,
        "full_imgsz": full_imgsz if args.full_image_pass else None,
        "full_conf": full_conf if args.full_image_pass else None,
        "full_max_det": full_max_det if args.full_image_pass else None,
        "total_tile_detections_before_nms": sum(row["tile_detections_before_nms"] for row in timing_rows),
        "total_full_image_detections_before_nms": sum(
            row["full_image_detections_before_nms"] for row in timing_rows
        ),
        "ios_enabled": args.ios_threshold is not None,
        "ios_threshold": args.ios_threshold,
        "total_ios_removed": sum(row["ios_removed"] for row in timing_rows),
        "boundary_aware": args.boundary_aware,
        "edge_margin": args.edge_margin if args.boundary_aware else None,
        "small_filter_enabled": args.small_filter,
        "small_area": args.small_area if args.small_filter else None,
        "small_conf": args.small_conf if args.small_filter else None,
        "total_nms_removed": sum(row["nms_removed"] for row in timing_rows),
        "total_fusion_nms_removed": sum(row["fusion_nms_removed"] for row in timing_rows),
        "total_small_removed": sum(row["small_removed"] for row in timing_rows),
        "total_fusion_ios_removed": sum(row["fusion_ios_removed"] for row in timing_rows),
        "draw_conf": args.draw_conf,
        "mean_preprocess_ms": sum(row["preprocess_ms"] for row in timing_rows) / len(timing_rows),
        "mean_inference_ms": sum(row["inference_ms"] for row in timing_rows) / len(timing_rows),
        "mean_tile_inference_ms": sum(row["tile_inference_ms"] for row in timing_rows) / len(timing_rows),
        "mean_full_image_inference_ms": sum(row["full_image_inference_ms"] for row in timing_rows) / len(timing_rows),
        "mean_postprocess_ms": sum(row["postprocess_ms"] for row in timing_rows) / len(timing_rows),
        "mean_total_ms": sum(row["total_ms"] for row in timing_rows) / len(timing_rows),
        "mean_tiles": sum(row["tiles"] for row in timing_rows) / len(timing_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Processed {len(image_paths)} image(s); parameters={parameters:,} "
        f"({parameters / 1e6:.4f}M); mean total={summary['mean_total_ms']:.2f} ms; "
        f"results saved to {output_dir}",
        flush=True,
    )


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
    parser.add_argument(
        "--ios-threshold",
        type=float,
        default=None,
        help="Enable same-class containment suppression at this IoS threshold; omitted means disabled.",
    )
    parser.add_argument(
        "--boundary-aware",
        action="store_true",
        help="Prefer detections away from tile edges when overlapping boxes compete.",
    )
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=32.0,
        help="Pixels from a tile edge considered boundary-adjacent when boundary-aware mode is enabled.",
    )
    parser.add_argument(
        "--small-filter",
        action="store_true",
        help="Enable low-confidence small-box filtering after NMS/IoS.",
    )
    parser.add_argument(
        "--small-area",
        type=float,
        default=32.0,
        help="Area threshold in original-image pixels for --small-filter.",
    )
    parser.add_argument(
        "--small-conf",
        type=float,
        default=0.10,
        help="Confidence threshold for --small-filter.",
    )
    parser.add_argument(
        "--draw-conf",
        type=float,
        default=None,
        help="Optional visualization-only confidence threshold; JSON predictions are unchanged.",
    )
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--full-image-pass",
        action="store_true",
        help="Run one additional inference pass on the original image and fuse it with tiled detections.",
    )
    parser.add_argument(
        "--full-imgsz",
        type=int,
        default=None,
        help="Input size for the optional full-image pass; defaults to --imgsz.",
    )
    parser.add_argument(
        "--full-conf",
        type=float,
        default=None,
        help="Candidate confidence threshold for the optional full-image pass; defaults to --conf.",
    )
    parser.add_argument(
        "--full-max-det",
        type=int,
        default=None,
        help="Maximum detections for the optional full-image pass; defaults to --max-det.",
    )
    return parser.parse_args()


# 程序入口：加载 YOLO 权重，再调用批量推理流程。
def main() -> None:
    args = parse_args()
    if args.edge_margin < 0:
        raise SystemExit("--edge-margin must be non-negative")
    if args.small_area <= 0:
        raise SystemExit("--small-area must be positive")
    if not 0 <= args.small_conf <= 1:
        raise SystemExit("--small-conf must satisfy 0 <= value <= 1")
    if args.draw_conf is not None and not 0 <= args.draw_conf <= 1:
        raise SystemExit("--draw-conf must satisfy 0 <= value <= 1")
    if args.full_imgsz is not None and args.full_imgsz <= 0:
        raise SystemExit("--full-imgsz must be positive")
    if args.full_conf is not None and not 0 <= args.full_conf <= 1:
        raise SystemExit("--full-conf must satisfy 0 <= value <= 1")
    if args.full_max_det is not None and args.full_max_det <= 0:
        raise SystemExit("--full-max-det must be positive")
    from ultralytics import YOLO

    model = YOLO(args.weights)
    run_directory(model, args)


if __name__ == "__main__":
    main()
