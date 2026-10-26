"""不需要模型权重的切片推理单元测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# 将仓库根目录加入导入路径，确保测试使用当前仓库中的代码。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coursework.sliced_inference import generate_tiles  # noqa: E402

try:
    # 张量相关测试需要 PyTorch；缺少依赖时只跳过这部分测试。
    import torch
    from coursework.evaluate_sliced import _match_image, build_iou_thresholds  # noqa: E402
    from coursework.sliced_inference import (  # noqa: E402
        intersection_over_smaller,
        map_boxes_to_image,
        merge_detections,
        predict_image,
        tile_edge_distances,
        xyxy_to_yolo,
    )
    DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    DEPENDENCIES_AVAILABLE = False


# 测试切片坐标生成，不需要加载模型或深度学习依赖。
class TileGeometryTests(unittest.TestCase):
    def test_tiles_cover_corners_and_have_overlap(self):
        # 1920x1080 图像使用 640 切片、25% 重叠时应生成 8 个切片。
        tiles = generate_tiles(1920, 1080, tile_size=640, overlap=0.25)
        self.assertEqual(len(tiles), 8)
        # 检查左上角和右下角都被覆盖。
        self.assertIn((0, 0, 640, 640), tiles)
        self.assertIn((1280, 440, 1920, 1080), tiles)
        # 切片坐标不应重复，并且至少存在重叠区域。
        self.assertEqual(len(tiles), len(set(tiles)))
        self.assertTrue(any(a[2] > b[0] and a[0] < b[2] for a in tiles for b in tiles if a != b))

    def test_small_image_uses_one_tile(self):
        # 小于切片尺寸的图片不需要补边，直接作为一个完整切片。
        self.assertEqual(generate_tiles(320, 240, 640, 0.25), [(0, 0, 320, 240)])

    def test_invalid_tile_arguments_are_rejected(self):
        # 切片尺寸必须为正数。
        with self.assertRaises(ValueError):
            generate_tiles(100, 100, tile_size=0)
        # overlap 必须在 [0, 1) 范围内，不能完全重叠。
        with self.assertRaises(ValueError):
            generate_tiles(100, 100, overlap=1.0)


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "torch, numpy, and opencv are required")
class EvaluationMatchingTests(unittest.TestCase):
    def test_matching_matches_ultralytics_validator_assignment(self):
        # 复刻 DetectionValidator.match_predictions 的排序和两次 unique 行为。
        predictions = [
            {"x1": 0, "y1": 0, "x2": 100, "y2": 70, "confidence": 0.99, "class_id": 0},
            {"x1": 0, "y1": 0, "x2": 90, "y2": 90, "confidence": 0.90, "class_id": 0},
        ]
        targets = [(0, 0, 0, 100, 100)]
        tp, _, _, _ = _match_image(predictions, targets, build_iou_thresholds(0.5, 0.5, 0.05))
        self.assertTrue(tp[0, 0])
        self.assertFalse(tp[1, 0])


# 测试依赖 PyTorch 的框坐标转换、NMS 和标签格式转换。
@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "torch, numpy, and opencv are required")
class TensorOperationTests(unittest.TestCase):

    def test_full_image_pass_fuses_an_additional_source(self):
        # The fake model makes the full-image call distinguishable by source
        # shape while exercising the same coordinate/fusion path as P7/P8.
        class FakeBoxes:
            def __init__(self, data):
                self.data = data

        class FakeResult:
            def __init__(self, data):
                self.boxes = FakeBoxes(data)
                self.speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}

        class FakeModel:
            def __init__(self):
                self.calls = []

            def predict(self, source, **kwargs):
                self.calls.append((source.shape[:2], kwargs))
                height, width = source.shape[:2]
                data = torch.tensor([[0, 0, width / 2, height / 2, 0.9, 0]], dtype=torch.float32)
                return [FakeResult(data)]

        import numpy as np

        model = FakeModel()
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        merged, timing = predict_image(
            model,
            image,
            imgsz=640,
            tile_size=64,
            overlap=0.25,
            conf=0.001,
            predict_iou=0.70,
            merge_iou=0.35,
            max_det=300,
            full_image_pass=True,
            full_imgsz=960,
            full_conf=0.001,
            full_max_det=300,
        )
        self.assertEqual(len(model.calls), len(generate_tiles(100, 80, 64, 0.25)) + 1)
        self.assertEqual(model.calls[-1][0], (80, 100))
        self.assertEqual(model.calls[-1][1]["imgsz"], 960)
        self.assertTrue(timing["full_image_pass"])
        self.assertEqual(timing["full_image_tiles"], 1)
        self.assertGreater(timing["tile_detections_before_nms"], 0)
        self.assertEqual(timing["full_image_detections_before_nms"], 1)
        self.assertEqual(merged.shape[1], 6)
        self.assertGreaterEqual(len(merged), 1)

    def test_intersection_over_smaller_detects_containment(self):
        boxes1 = torch.tensor([[0, 0, 100, 100], [200, 200, 220, 220]], dtype=torch.float32)
        boxes2 = torch.tensor([[25, 25, 75, 75], [300, 300, 320, 320]], dtype=torch.float32)
        ios = intersection_over_smaller(boxes1, boxes2)
        self.assertTrue(torch.allclose(ios[0, 0], torch.tensor(1.0)))
        self.assertEqual(float(ios[0, 1]), 0.0)
        self.assertEqual(float(ios[1, 0]), 0.0)

    def test_intersection_over_smaller_supports_empty_inputs(self):
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        other = torch.zeros((2, 4), dtype=torch.float32)
        self.assertEqual(tuple(intersection_over_smaller(boxes, other).shape), (0, 2))

    def test_boxes_are_offset_and_clipped(self):
        # 构造一个超出切片边界的局部框，检查偏移和原图边界裁剪。
        boxes = torch.tensor([[-4, 8, 40, 80, 0.9, 2]], dtype=torch.float32)
        mapped = map_boxes_to_image(boxes, offset=(100, 50), image_width=120, image_height=100)
        expected = torch.tensor([[96.0, 58.0, 120.0, 100.0, 0.9, 2.0]])
        # 使用 allclose 允许 float32 的微小舍入误差。
        self.assertTrue(torch.allclose(mapped, expected, atol=1e-6))

    def test_merge_is_class_aware(self):
        # 前两个框同类且高度重叠，应只保留置信度更高的框；
        # 第三个框类别不同，即使位置相同也不能被错误删除。
        detections = torch.tensor(
            [
                [0, 0, 100, 100, 0.90, 0],
                [5, 5, 95, 95, 0.80, 0],
                [5, 5, 95, 95, 0.70, 1],
            ],
            dtype=torch.float32,
        )
        merged = merge_detections(detections, iou_threshold=0.5, max_det=300)
        self.assertEqual(len(merged), 2)
        self.assertEqual(set(merged[:, 5].tolist()), {0.0, 1.0})

    def test_ios_is_optional_and_suppresses_contained_same_class_box(self):
        detections = torch.tensor(
            [
                [0, 0, 100, 100, 0.90, 0],
                [25, 25, 75, 75, 0.80, 0],
            ],
            dtype=torch.float32,
        )
        without_ios = merge_detections(detections, iou_threshold=0.5, max_det=300)
        with_ios = merge_detections(detections, iou_threshold=0.5, max_det=300, ios_threshold=0.8)
        self.assertEqual(len(without_ios), 2)
        self.assertEqual(len(with_ios), 1)
        self.assertAlmostEqual(float(with_ios[0, 4]), 0.90, places=5)

    def test_boundary_aware_nms_prefers_non_edge_box(self):
        detections = torch.tensor(
            [
                [0, 0, 100, 100, 0.95, 0],
                [0, 0, 100, 100, 0.70, 0],
            ],
            dtype=torch.float32,
        )
        # The first box touches a tile edge; the second is an interior box.
        edge_distances = torch.tensor([0.0, 100.0])
        merged = merge_detections(
            detections,
            iou_threshold=0.5,
            max_det=300,
            boundary_aware=True,
            edge_distances=edge_distances,
            edge_margin=16,
        )
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(float(merged[0, 4]), 0.70, places=5)

    def test_small_filter_requires_low_confidence(self):
        detections = torch.tensor(
            [
                [0, 0, 4, 4, 0.05, 0],
                [20, 20, 24, 24, 0.20, 0],
            ],
            dtype=torch.float32,
        )
        merged = merge_detections(
            detections,
            iou_threshold=0.5,
            max_det=300,
            small_area=32,
            small_conf=0.10,
        )
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(float(merged[0, 4]), 0.20, places=5)

    def test_tile_edge_distances(self):
        boxes = torch.tensor(
            [
                [0, 10, 20, 30, 0.9, 0],
                [30, 30, 70, 70, 0.8, 0],
            ],
            dtype=torch.float32,
        )
        distances = tile_edge_distances(boxes, tile_width=100, tile_height=100)
        self.assertEqual(distances.tolist(), [0.0, 30.0])

    def test_yolo_lines_are_normalized(self):
        # 检查 xyxy 像素坐标能正确转换为 YOLO 的归一化 xywh 格式。
        detections = torch.tensor([[10, 20, 30, 60, 0.875, 3]], dtype=torch.float32)
        self.assertEqual(xyxy_to_yolo(detections, 100, 100), ["3 0.200000 0.400000 0.200000 0.400000 0.875000"])


if __name__ == "__main__":
    unittest.main()
