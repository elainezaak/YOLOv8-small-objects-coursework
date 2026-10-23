# 4 号任务书：VisDrone 切片推理二次改进

版本：2026-10-20  
适用仓库：`YOLOv8-small-objects-coursework`，Ultralytics 8.1.20

## 1. 任务结论

本方案可行，建议作为第 4 号的主线改进。仓库已有统一的 `YOLO.predict()` 接口、VisDrone 数据转换脚本和最终模型配置，不需要修改训练损失或网络结构。实现一个独立的切片推理器即可完成第一轮实验：把原图分成有重叠的小块，逐块检测，再将局部坐标映射回原图并进行全图去重。

必须在报告中准确表述：这是“最终模型 + 切片推理（inference-time tiling）”，不是重新训练出的新网络。因此至少要保留以下三组结果：

| 编号 | 模型 | 推理方式 | 用途 |
|---|---|---|---|
| E0 | 原始 YOLOv8n | 整图 640 | 小组 Baseline |
| E1 | 仓库最终模型 CBAM | 整图 640 | 验证仓库结构改进 |
| E2 | 仓库最终模型 CBAM | 切片推理 | 验证第 4 号改进 |

建议增加 E3（原始 YOLOv8n + 相同切片推理）作为控制实验。若 E3 也提升，说明收益主要来自切片；若 E2 相对 E1 的提升明显大于 E3，才能更有把握地说明切片与最终模型组合有效。

## 2. 固定条件与禁止事项

E0、E1、E2、E3 必须使用同一份 VisDrone 验证集（548 张）、同一套类别名称和同一置信度口径。模型训练部分沿用小组约定：`imgsz=640`、相同 epochs、batch、seed=42、相同预训练方式。第 4 号不重新训练最终模型，直接使用第 3 号交付的 `best.pt`，并记录其绝对路径和 git commit。

不要在 E2 中同时改变以下内容，否则无法归因：

- 不要把 `conf`、模型权重、类别过滤和输入尺寸一起改掉；
- 不要在一组实验中混用整图和切片的不同 NMS 阈值而不记录；
- 不要只挑选提升最大的图片计算指标；
- 不要用 test-dev 反复调参。参数只在 val 上确定，test-dev 最后只运行一次。

## 3. 第一阶段：确认仓库和权重

### 3.1 环境检查

在仓库根目录执行：

```powershell
python coursework/check_environment.py
python coursework/check_visdrone.py --dataset-dir ..\datasets\VisDrone
```

GPU 正式实验再执行：

```powershell
python coursework/check_environment.py --require-cuda
```

如果环境检查报告的 Ultralytics 路径不是当前仓库的 `ultralytics\`，先修复环境，不要继续实验。

### 3.2 获取最终模型

从第 3 号获得：

```text
runs/coursework/final/weights/best.pt
```

若第 3 号尚未完成 100 epochs，可先用 20--30 epochs 的权重做代码冒烟测试，但该结果不能作为最终表格。记录：

- 权重文件 SHA256；
- 模型参数量；
- 训练 epochs、batch、imgsz、seed、device；
- 训练日志目录和 `results.png`。

## 4. 第二阶段：实现切片推理器

建议新增 `coursework/sliced_inference.py`，保持现有训练代码不变。脚本至少提供以下参数：

```text
--weights PATH          模型权重
--source PATH           单张图片或图片目录
--output-dir PATH       预测图片、txt 和 json 输出目录
--device 0              cpu 或 GPU 编号
--imgsz 640             每个切片送入模型的尺寸
--tile-size 640        原图切片边长
--overlap 0.25          相邻切片重叠比例
--conf 0.25             单切片置信度阈值
--merge-iou 0.50        全图合并时的 class-aware NMS IoU
--max-det 300           每张原图最大输出框数
```

### 4.1 切片坐标规则

设原图宽高为 `W,H`，切片边长为 `T`，重叠率为 `o`，步长为 `S=max(1, floor(T*(1-o)))`。在 x、y 两个方向生成起点，并强制加入 `W-T`、`H-T` 的最后一个起点，确保右边界和下边界覆盖。边缘不足 `T` 的区域可以使用实际尺寸较小的 crop，不能丢弃。

第一轮锁定：`T=640`、`overlap=0.25`。VisDrone 原图通常明显大于 640，切片后远处目标在模型输入中占据更多像素，正是该方案的预期收益来源。

### 4.2 单切片预测与坐标回映射

对每个 crop 调用仓库的 `model.predict()`，固定 `imgsz=640`、`conf=0.25`、`iou=0.70`、`max_det=300`，关闭自动保存，取得 `Results[0].boxes`。局部框 `[x1,y1,x2,y2]` 加上切片左上角偏移 `(x0,y0)`，得到原图坐标；随后裁剪到 `[0,W] x [0,H]`，去除宽或高小于 1 像素的异常框。

不要把不同切片的结果直接拼接后返回。重叠区中的同一目标会产生多个框，必须执行全图合并。

### 4.3 全图合并

使用 `torchvision.ops.nms` 或仓库已有的 NMS 工具，按类别分别执行 NMS：

1. 汇总所有切片的 `[x1,y1,x2,y2,confidence,class_id]`；
2. 对每个 `class_id` 执行 NMS，阈值为 `merge_iou=0.50`；
3. 合并各类别结果，按置信度降序排序，截断到 `max_det`；
4. 输出原图坐标，不再进行第二次缩放。

`merge_iou` 是推理策略参数，不应与单切片预测的 `iou` 混为一个参数。若 0.50 造成重叠目标被错误合并，只在 val 上比较 0.40、0.50、0.60 三个值，并把选择过程写入日志。

### 4.4 输出格式

每张图片同时保存：

- 带框可视化图片；
- YOLO 格式预测文本：`class x_center y_center width height confidence`，坐标归一化到原图；
- 机器可读的 JSON：原图名、切片数、每张框的坐标、置信度和类别；
- 每张图的总耗时、切片数和平均单切片耗时。

建议目录结构：

```text
runs/coursework/sliced_final_t640_o025/
  predictions/
  images/
  timing.csv
  config.json
  summary.json
```

## 5. 第三阶段：先做小规模冒烟测试

正式跑全验证集前，选固定的 6 张 val 图片，使用 `seed=42`，检查：

- 原图尺寸和切片数量符合预期；
- 右侧、下侧边缘确实生成切片；
- 框坐标没有负数、没有超过原图边界；
- 合并前后的框数量变化合理；
- `--tile-size 640 --overlap 0.25` 能在 CPU 和 GPU 下运行；
- 输出图片能打开，标签文本能被解析。

重点人工查看重叠区域：同一个车辆/行人不能留下两个高度重叠的框；跨切片边界的目标不能系统性消失。冒烟测试只证明代码正确，不用于报告指标。

## 6. 第四阶段：指标评估

### 6.1 整图基线

对 E0 和 E1 使用仓库原生验证接口，保持相同参数：

```python
from ultralytics import YOLO

model = YOLO("PATH_TO_BEST_PT")
metrics = model.val(
    data="..\\datasets\\VisDrone\\VisDrone-local.yaml",
    split="val",
    imgsz=640,
    batch=4,             # 按显存调整，但 E0/E1/E2 保持记录一致
    device="0",
    conf=0.001,          # 验证 mAP 使用低阈值，预测展示可另设 0.25
    iou=0.7,
    plots=True,
    project="runs/coursework/evaluation",
    name="e0_baseline_full"   # E1 改名为 e1_final_full
)
```

验证 mAP 时使用 `conf=0.001`，不要直接拿可视化预测的 `conf=0.25` 当 mAP 输入。记录 `metrics.box.map50`、`metrics.box.map`、`metrics.box.mp`、`metrics.box.mr`，分别对应 mAP50、mAP50-95、Precision、Recall。

### 6.2 切片结果的评估

仓库原生 `model.val()` 不知道如何对一张原图拆成多个 crop，因此不能把 E2 简单地写成一次 `model.val()`。建议新增 `coursework/evaluate_sliced.py`，读取切片推理器生成的原图坐标预测和原始 val 标签，复用：

- `ultralytics.utils.metrics.box_iou` 计算 IoU；
- `DetectionValidator._process_batch` 或等价的 class-aware greedy matching；
- `ultralytics.utils.metrics.ap_per_class` 汇总 IoU=0.50:0.95 的 AP。

评估器必须按“每张原图”匹配 GT，不能把切片当成独立图片重复计算 GT。输出字段至少包括：

```text
images, ground_truth_boxes, predicted_boxes,
precision, recall, map50, map50_95,
per_class_ap50, per_class_ap50_95,
mean_sliced_latency_ms, p95_sliced_latency_ms
```

如果直接复用 `DetectionValidator` 的匹配代码，注明复用的仓库版本；不要引入会改变环境的第三方评估框架。所有 E0--E3 的指标放在一个 `metrics.csv` 中，保留原始 JSON，便于第 5 号复核。

## 7. 实验矩阵

### 7.1 必做实验

| 实验 | 权重 | tile-size | overlap | merge-iou | 目的 |
|---|---|---:|---:|---:|---|
| E0 | YOLOv8n best.pt | 整图 | - | 原生 | Baseline |
| E1 | final best.pt | 整图 | - | 原生 | 结构改进基准 |
| E2 | final best.pt | 640 | 0.25 | 0.50 | 主改进 |
| E3 | YOLOv8n best.pt | 640 | 0.25 | 0.50 | 控制切片本身的收益 |

### 7.2 可选参数消融

只有 E2 能稳定复现后才做参数消融，每次只改一个参数：

| 实验 | tile-size | overlap | merge-iou |
|---|---:|---:|---:|
| A1 | 512 | 0.25 | 0.50 |
| A2 | 640 | 0.25 | 0.50 |
| A3 | 960 | 0.25 | 0.50 |
| A4 | 640 | 0.15 | 0.50 |
| A5 | 640 | 0.35 | 0.50 |
| A6 | 640 | 0.25 | 0.40/0.60 |

A1--A6 只在 val 上筛选，不得把多个最优参数重新组合后声称是无偏 test 结果。建议优先选择 mAP50-95 较高且延迟可接受的配置，而不是只追求 mAP50。

## 8. 速度和资源测量

切片会增加推理次数，速度下降是预期现象，不能隐瞒。对每组模型先 warm-up 10 次，再在固定的 100 张 val 图片上测量，GPU 计时前后使用 `torch.cuda.synchronize()`。分别报告：

- 单张原图平均总延迟和 P95 延迟；
- 平均切片数；
- 单切片平均推理时间；
- 峰值显存（`torch.cuda.max_memory_allocated()`）；
- 每张图输出框数量。

E0/E1 的整图延迟和 E2/E3 的“整张原图端到端延迟”必须放在同一张表里。不要只报告单个 tile 的速度，因为用户实际等待的是整张图完成的时间。

## 9. 结果判定标准

本任务不预设一定提升。推荐按以下规则解释：

- 若 E2 相对 E1 的 mAP50-95 或 Recall 有稳定提升，且结果在全 548 张 val 上成立，方案可作为最终二次改进；
- 若 mAP 提升小于 0.005，但小目标相关类别（pedestrian、people、bicycle）的 Recall 明显提升，应同时报告准确率和速度，不能只写“效果更好”；
- 若 E2 指标下降，优先检查 `overlap`、全图 NMS、坐标映射和 `max_det`，再决定是否做 A1--A6；
- 若 E2 提升但延迟超过 E1 的 2 倍，报告为“精度--速度权衡”，不要宣称全面优于最终模型；
- 即使没有提升，只要实验控制正确、失败原因有证据、代码可复现，也属于有效的消融结论。

## 10. 典型案例与错误分析

从全量 val 结果中自动选取而不是凭印象挑图：

1. E1 漏检、E2 检出的目标；
2. E1 和 E2 都检出但 E2 IoU 更高的目标；
3. E2 新增的误检；
4. 位于切片边界附近、被重复检测或漏检的目标。

至少保存 6 张带框图片，每类案例尽量包含 1--2 张。图片标题或旁边的 CSV 记录原图名、类别、置信度、切片参数和是否属于切片边界，避免只展示主观成功案例。

## 11. 交付物清单

- `coursework/sliced_inference.py`：可从命令行运行的切片推理器；
- `coursework/evaluate_sliced.py`：与仓库指标口径一致的切片评估器；
- `coursework/README_sliced_inference.md`：安装前提、命令、参数和输出目录说明；
- `runs/coursework/sliced_final_t640_o025/`：配置、预测、可视化和计时结果；
- `metrics.csv` 和原始 `summary.json`；
- E0--E3 的权重路径、git commit、运行命令和日志；
- 至少 6 张成功/失败案例图；
- 一页实验结论：精度变化、延迟变化、显存变化、适用场景和限制。

## 12. 建议执行顺序

1. 先完成 6 张图片的切片和合并冒烟测试。
2. 用 20 张 val 图片验证评估器的坐标和指标无异常。
3. 全量运行 E1 和 E2，确认端到端计时和输出文件完整。
4. 补跑 E0、E3，完成控制实验。
5. 只有主实验稳定后，才做 A1--A6 的单参数消融。
6. 把 `metrics.csv`、速度表和案例图交给第 5 号统一整合。

