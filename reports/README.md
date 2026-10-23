# YOLOv8 小目标检测复现实验说明书

> 课程：深度学习  
> 参考论文：*Improved YOLOv8 Network for Small Objects Detection*  
> 实验数据集：VisDrone2019-DET  

## 一、实验概况

本次课程作业基于 YOLOv8n，复现论文提出的四项结构改进，并通过消融实验分析各模块的作用。

论文的改进顺序为：

1. **P2**：增加高分辨率 P2 检测头，以增强小目标检测能力。
2. **P2-NoP5**：移除 P5 检测头，并调整网络通道，实现轻量化。
3. **Fusion**：引入双向特征融合，融合不同分辨率的特征。
4. **Final**：在融合结构基础上加入 CBAM 注意力模块。

本次实际完成：

- 五组模型各训练 30 epochs，作为快速消融实验。
- Baseline 和 Final 额外训练 100 epochs，作为正式对比。
- 七组训练均保存模型权重、训练曲线和实验记录。
- 七组模型均使用各自的 `best.pt` 进行了统一验证。

目前核心训练和精度评估已完成，预测图整理、小目标专项分析和独立推理测速尚待补充。

## 二、实验环境与配置

| 项目 | 实验配置 |
|---|---|
| 数据集 | VisDrone2019-DET |
| 训练集 | 6471 张图片 |
| 验证集 | 548 张图片 |
| 输入尺寸 | 640 × 640 |
| Batch size | 4 |
| 随机种子 | 42 |
| 预训练初始化 | COCO YOLOv8n 权重 |
| 优化器 | SGD |
| 训练设备 | NVIDIA RTX 4090 |
| Python | 3.10.21 |
| PyTorch | 2.2.2+cu121 |
| Ultralytics | 8.1.20 |
| 消融实验 | 30 epochs |
| 正式对比 | 100 epochs |

为保证对比合理性，同轮数实验统一使用数据划分、输入尺寸、batch size、随机种子及预训练策略。

需要注意，自定义网络与 YOLOv8n 的结构不同，能从预训练权重迁移的参数数量可能不同。

## 三、实验结果

### 3.1 五组消融实验（30 epochs）

以下结果来自相同验证条件下，各模型 `best.pt` 的重新评估。

| 模型 | Precision | Recall | mAP50 | mAP50-95 | 参数量 |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.41238 | 0.29955 | 0.29947 | 0.17297 | 3.013M |
| P2 | 0.42449 | 0.31653 | 0.31003 | 0.17732 | 2.928M |
| P2-NoP5 | 0.39314 | 0.31628 | 0.30371 | 0.17396 | 1.383M |
| Fusion | 0.38658 | 0.31219 | 0.29955 | 0.16974 | 1.465M |
| Final | 0.38912 | 0.30241 | 0.29625 | 0.17005 | 1.546M |

完整结果：[ablation_comparison.csv](./ablation_comparison.csv)

**初步发现：**

- P2 的检测精度最高，mAP50-95 相较 Baseline 提升约 0.00435。
- P2-NoP5 的参数量相比 P2 减少约 52.8%，检测精度略有下降。
- Fusion 和 Final 在 30 epochs 条件下暂未表现出明显的精度优势。
- 当前指标是所有目标的整体检测结果，还需要进一步按目标尺寸评价小目标检测性能。

### 3.2 Baseline 与 Final（100 epochs）

| 指标 | Baseline | Final |
|---|---:|---:|
| Precision | 0.43586 | 0.43480 |
| Recall | 0.33722 | 0.35421 |
| mAP50 | 0.33478 | 0.34456 |
| mAP50-95 | 0.19576 | 0.20032 |
| 参数量 | 3.013M | 1.546M |
| 训练时间 | 171.90 分钟 | 241.39 分钟 |

完整结果：[formal_100ep.csv](./formal_100ep.csv)

在本次 100 epochs 实验中，Final 的 mAP50-95 相比 Baseline 提高约 0.00457，参数量减少约 48.7%，但训练耗时增加。

结果目前仅对应一个随机种子，尚不能说明提升具有统计稳定性。

## 四、GitHub 文件目录说明

### 4.1 训练与评估代码

`coursework/train_fair_coursework.py`

本次实际使用的训练脚本，支持五种模型和两类训练阶段，自动按模型、训练阶段和时间戳保存结果。

`coursework/evaluate_coursework.py`

用于后续统一验证、预测图片生成和推理耗时记录的辅助脚本。该脚本已上传，但独立的标准化测速流程尚未完成。

`model_configs/`

保存四种改进网络的 YAML 配置。

`ultralytics/`

仓库包含自定义 Ultralytics 代码和模型模块。运行自定义模型时应使用项目内的代码版本。

### 4.2 实验结果

`reports/ablation_comparison.csv`

五组 30 epochs 消融实验的统一评估结果。

`reports/formal_100ep.csv`

Baseline 和 Final 的 100 epochs 正式实验结果。

`reports/experiment_records/`

保存七组训练记录，每个实验目录主要包含：

| 文件 | 作用 |
|---|---|
| `results.csv` | 每个 epoch 的损失、Precision、Recall、mAP 等 |
| `results.png` | 训练损失与验证指标曲线 |
| `args.yaml` | 实际训练参数 |
| `run_summary.json` | 环境、训练耗时、模型参数量和文件路径等 |

`reports/evaluation/`

保存各模型 `best.pt` 统一复评得到的 `metrics_summary.json`。

`reports/weights_manifest.csv`

保存七组模型的权重名称和原始实验目录信息。

注意：部分 JSON 中保留了原服务器的绝对路径，其他成员在自己的机器上运行时需要改成本机路径。

## 五、模型权重下载与使用

七组最佳模型权重已经上传到 [GitHub Releases](https://github.com/elainezaak/YOLOv8-small-objects-coursework/releases)。

| 权重名称 | 对应模型 |
|---|---|
| `baseline_30ep_best.pt` | 30ep Baseline |
| `p2_30ep_best.pt` | 30ep P2 |
| `p2_nop5_30ep_best.pt` | 30ep P2-NoP5 |
| `fusion_30ep_best.pt` | 30ep Fusion |
| `final_30ep_best.pt` | 30ep Final |
| `baseline_100ep_best.pt` | 100ep Baseline |
| `final_100ep_best.pt` | 100ep Final |

**模型使用建议：**

- 制作正式 Baseline 与 Final 检测效果对比图，使用两份 100ep 权重。
- 分析五组结构的消融效果，统一使用五份 30ep 权重。
- 模型权重均为 `best.pt`，可以直接加载进行预测，无需重新训练。

### 5.1 环境准备

首先克隆仓库、安装依赖，按 [课程环境与数据说明](../coursework/README_zh.md) 配置 VisDrone 数据集。

建议将下载的权重放入项目根目录下新建的 `checkpoints/` 文件夹。

确保 Python 导入的是仓库自带的 `ultralytics/`，尤其是 Final 模型需要自定义模块。

### 5.2 生成预测图片

在仓库根目录激活正确环境后，运行以下 Python 示例：

```python
from ultralytics import YOLO

model = YOLO("checkpoints/final_100ep_best.pt")

model.predict(
    source="path/to/example.jpg",
    imgsz=640,
    conf=0.25,
    device=0,
    save=True,
    project="runs/04_evaluation",
    name="final_demo"
)
```

其中，图片路径和 GPU 编号需要替换为本机实际配置。

使用 `baseline_100ep_best.pt` 对同一张图片执行相同预测设置，即可生成对比图。

建议选择 VisDrone 中目标密集、目标尺寸较小的图像进行展示，同时记录可能出现的漏检和误检情况。

### 5.3 重新训练

本次使用的实际训练脚本：

`coursework/train_fair_coursework.py`

例如重新训练 P2 30 epochs：

```bash
python coursework/train_fair_coursework.py \
  --stage ablation \
  --model p2 \
  --init pretrained \
  --epochs 30 \
  --batch 4 \
  --imgsz 640 \
  --seed 42 \
  --device 0 \
  --workers 4
```

模型名称可选：

`baseline`、`p2`、`p2-nop5`、`fusion`、`final`。

其他组员需先准备 VisDrone 数据 YAML。脚本默认寻找仓库父目录下的 `datasets/VisDrone/VisDrone-local.yaml`，也可通过 `--data` 指定其他位置。

实际训练结果自动保存在 `runs/02_ablation_30ep/` 或 `runs/03_formal_100ep/`，并采用独立的时间戳目录。

## 六、复现过程中遇到的问题与分析（汇报重点）

### 问题 1：复现精度没有达到论文报告的水平

**实验现象**

论文在 VisDrone 验证集上报告 Baseline mAP50-95 为 0.203，Final 为 0.241。

本次 100 epochs 实验中，Baseline 为 0.19576，Final 为 0.20032，特别是 Final 与论文结果仍有明显差距。

**可能原因**

原论文训练 1000 epochs，本次正式实验训练 100 epochs。训练轮数差异可能影响模型收敛。

此外，还需核对模型配置、预训练权重迁移程度、Ultralytics 版本以及实际训练超参数。

一个值得进一步核查的细节是：论文仓库给出的 P2 参数量约为 1.737M，而本次 P2 为 2.928M。这说明相应结构配置或版本可能存在差异，不能直接认为网络完全一致。

**当前结论**

我们复现了主要结构改动和完整的训练评价流程，但尚未复现论文报告的最高精度，差异原因还需要进一步排查。

### 问题 2：为什么增加改进模块后，精度没有持续提高？

**实验现象**

30 epochs 消融中，P2 表现最好。之后加入轻量化结构、双向融合和 CBAM，mAP50-95 没有继续提高。

尤其是 Fusion 相比 P2-NoP5，mAP50-95 从 0.17396 降至 0.16974。

**可能原因**

不同结构改变了特征融合方式与模型复杂度；新增模块可能需要不同的训练时间才能充分学习。

此外，自定义网络与预训练 YOLOv8n 的结构匹配程度不同，部分参数可能需要重新学习。

**当前结论**

实验说明，在本次短轮次训练条件下，增加模块没有必然带来精度提升。要判断结构的长期效果，还需要更充分的对照实验。

### 问题 3：为什么 Final 在 30 epochs 和 100 epochs 下表现不同？

**实验现象**

30 epochs 时：

- Baseline mAP50-95：0.17297
- Final mAP50-95：0.17005

100 epochs 时：

- Baseline mAP50-95：0.19576
- Final mAP50-95：0.20032

Final 从短轮次时略低于 Baseline，变成较长训练后略高于 Baseline。

**初步分析**

这提示模型的收敛过程可能影响实验结论。仅用短轮次结果评价所有结构，可能无法充分反映模型的最终性能。

目前只有 Baseline 与 Final 进行了 100 epochs 训练，所以还无法确定具体是哪一个中间模块在后期产生了收益。

### 问题 4：为什么 Final 参数量减少，训练时间却增加？

**实验现象**

Final 的参数量比 Baseline 减少约 48.7%，但 100 epochs 训练时间从约 172 分钟增加到 241 分钟。

**初步分析**

参数量较少并不必然意味着运行速度更快。

P2 高分辨率特征、额外的融合路径及注意力模块都可能增加计算和内存访问开销。

验证日志中的模型计算量也显示，Baseline 约为 8.1 GFLOPs，Final 约为 11.5 GFLOPs。

另外，各次训练使用的共享服务器负载可能不同，也会影响实际耗时。

后续仍需在统一设备、精度和 batch=1 条件下进行独立测速，才能更准确地比较推理效率。


## 七、目前尚未完成的分析

后续可以在现有权重基础上继续：

1. 整理 Baseline、P2、Final 的同图预测结果，对比漏检和误检。
2. 根据目标尺寸统计小目标检测性能，检验 P2 的具体效果。
3. 统一进行 batch=1 推理测速，记录延迟和 FPS。
4. 结合 `results.csv` 和 `results.png` 分析收敛曲线。
5. 如有额外算力，进一步验证 Fusion、CBAM 在更长训练轮数下的表现，并考虑多随机种子重复实验。

以上工作主要用于解释现有结果，不要求重新执行已经完成的七组训练。

## 八、实验结论

本次完成了 YOLOv8n 及四种改进结构的复现和消融验证。

在 30 epochs 条件下，P2 表现出一定的精度收益，轻量化结构显著降低了参数量，但 Fusion 与 CBAM 没有表现出明显的进一步提升。

在 100 epochs 条件下，Final 的 mAP50-95 略高于 Baseline，同时参数量明显减少，但训练耗时增加。

目前结果说明模型结构调整存在精度、参数量与计算成本之间的权衡。后续仍需结合训练曲线、预测可视化和更充分的对照实验，解释各个模块的真实作用。

**所有结论均限定在本次 VisDrone 验证集、训练配置和单个随机种子下。**
