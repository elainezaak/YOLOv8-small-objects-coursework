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
    from coursework.sliced_inference import (  # noqa: E402
        map_boxes_to_image,
        merge_detections,
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


# 测试依赖 PyTorch 的框坐标转换、NMS 和标签格式转换。
@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "torch, numpy, and opencv are required")
class TensorOperationTests(unittest.TestCase):

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

    def test_yolo_lines_are_normalized(self):
        # 检查 xyxy 像素坐标能正确转换为 YOLO 的归一化 xywh 格式。
        detections = torch.tensor([[10, 20, 30, 60, 0.875, 3]], dtype=torch.float32)
        self.assertEqual(xyxy_to_yolo(detections, 100, 100), ["3 0.200000 0.400000 0.200000 0.400000 0.875000"])


if __name__ == "__main__":
    unittest.main()
