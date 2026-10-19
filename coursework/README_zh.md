# 小组作业使用说明

这个目录是对原仓库的补充，不修改论文模型的核心实现。建议先把代码跑通，再安排正式训练和改进实验。

## 目录说明

- `check_environment.py`：检查依赖、导入路径和 4 个模型配置。
- `check_visdrone.py`：检查图片、标签、类别分布，并生成抽查图。
- `prepare_visdrone.py`：转换并检查 VisDrone 标注，生成本机数据配置。
- `verify_coursework.py`：一条命令运行环境检查和数据检查。
- `train_coursework.py`：用统一命令训练 baseline 或改进模型。
- `VISDRONE_DATA_REPORT.md`：本次数据检查的统计结果。
- `VERIFIED_ENVIRONMENT.md`：已经通过检查的 Linux/CUDA 参考环境。
- `ONE_EPOCH_RECORD_TEMPLATE.md`：GPU 上完成 1 epoch 后填写的记录模板。
- `../model_configs/`：论文方法对应的模型 YAML 配置。
- `../ultralytics/`：仓库自带的、已经接入自定义模块的 Ultralytics 源码。

## 第一步：获取代码

```bash
git clone https://github.com/elainezaak/YOLOv8-small-objects-coursework.git
cd YOLOv8-small-objects-coursework
```

不要直接修改 `main` 分支。开始任务前先创建自己的分支：

```bash
git switch -c 你的名字-任务名称
```

## 第二步：检查环境

固定版本为 Python 3.10、PyTorch 2.2.2、Torchvision 0.17.2 和 Ultralytics 8.1.20。Linux/NVIDIA 环境统一使用已经验证的 PyTorch `cu121` 构建，不要求安装相同版本的系统 CUDA Toolkit。

在 Linux/NVIDIA 主机上创建独立环境并安装依赖：

```bash
conda create -n yolo_small python=3.10 pip -y
conda activate yolo_small

python -m pip install --upgrade pip
python -m pip install -r requirements-cuda121.txt
python -m pip install -r requirements-coursework.txt
```

如果机器已经正确安装 PyTorch 2.2.2，可以只安装 `requirements-coursework.txt`。进入仓库根目录后运行：

```bash
python coursework/check_environment.py
```

看到 4 行 `OK` 和 `Environment check passed.`，说明仓库的自定义模型可以正常加载。

这个检查不会下载 VisDrone，也不会开始训练。

如果输出的 `Ultralytics path` 没有指向本仓库里的 `ultralytics/` 目录，请不要继续训练，先检查当前目录和 Python 环境。

在 GPU 机器上可以增加 `--require-cuda`，确保 PyTorch 确实识别显卡：

```bash
python coursework/check_environment.py --require-cuda
```

## 准备 VisDrone 数据集

从 [数据发布页](https://github.com/ultralytics/yolov5/releases/tag/v1.0) 下载并解压下面三个文件，不需要下载 `test-challenge`：

- `VisDrone2019-DET-train.zip`
- `VisDrone2019-DET-val.zip`
- `VisDrone2019-DET-test-dev.zip`

然后用下面的命令把原始标注转换成 YOLO 格式：

```bash
python coursework/prepare_visdrone.py --dataset-dir ../datasets/VisDrone
```

脚本会核对图片数量、转换标签并生成 `VisDrone-local.yaml`。数据集和这个本机配置文件都保留在仓库外，不要上传到 GitHub。训练时指定它：

```text
仓库父目录/
├── YOLOv8-small-objects-coursework/
│   ├── coursework/
│   ├── model_configs/
│   └── ultralytics/
└── datasets/
    └── VisDrone/
        ├── VisDrone-local.yaml
        ├── VisDrone2019-DET-train/
        │   ├── images/
        │   ├── annotations/
        │   └── labels/
        ├── VisDrone2019-DET-val/
        ├── VisDrone2019-DET-test-dev/
        ├── downloads/
        └── reports/
```

转换完成后，一条命令检查环境、全部标签和类别分布：

```bash
python coursework/verify_coursework.py
```

成功时最后会显示 `All coursework preflight checks passed.`。本机报告写入 `../datasets/VisDrone/reports/`：

- `dataset_report.md`：数据量和类别分布表。
- `class_distribution.csv`：可用于报告或进一步分析的 CSV。
- `class_distribution.png`：类别分布柱状图。
- `label_samples.jpg`：随机标注抽查图。

GPU 机器上的一键检查命令为：

```bash
python coursework/verify_coursework.py --require-cuda
```

训练脚本默认使用这个本机 YAML，也可以显式指定它：

```bash
python coursework/train_coursework.py \
  --data ../datasets/VisDrone/VisDrone-local.yaml \
  --model final \
  --epochs 5 \
  --device 0
```

## 第三步：训练前先做小规模测试

第一次拿到可用 GPU 时，先运行 1 epoch，完成环境与数据负责人的交付记录：

```bash
python coursework/train_coursework.py \
  --model final \
  --epochs 1 \
  --batch 8 \
  --device 0 \
  --pretrained
```

其中 `--device 0` 表示使用第 1 张可用 GPU。如果显存不足，先把 `--batch 8` 改成 `--batch 4`，不要修改输入尺寸和数据划分。

完成后填写 `ONE_EPOCH_RECORD_TEMPLATE.md`，至少记录 GPU、驱动、PyTorch CUDA build、完整命令、batch、耗时和验证指标。

确认 1 epoch 成功后，再使用 5 个 epochs 做训练流程测试；正式实验的 epochs 由小组统一决定。当前设备没有 GPU 时可以使用 `--device cpu` 做代码检查，但不建议在 CPU 上跑完整 VisDrone 训练。

确认测试成功后，再根据实际算力调整 epochs、batch 和 workers。所有对比实验应尽量使用相同的数据划分、输入尺寸、epochs、batch 和随机种子。

## 模型名称

- `baseline`：官方 YOLOv8n。
- `p2`：增加 P2 检测头。
- `p2-nop5`：增加 P2 并移除 P5 检测头。
- `fusion`：再加入双向特征融合。
- `final`：再加入 CBAM，即论文最终模型。

例如训练 baseline：

```bash
python coursework/train_coursework.py \
  --model baseline \
  --epochs 100 \
  --batch 16 \
  --device 0
```

训练结果默认保存在 `runs/coursework/`，该目录已被 Git 忽略。

## 协作规则

完成修改后提交自己的分支：

```bash
git add .
git commit -m "说明你的修改"
git push -u origin 你的分支名
```

然后在 GitHub 创建 Pull Request，不要把多个成员的实验结果直接混在同一个分支里。

## 不要上传的文件

以下内容不要提交到 GitHub：

```text
VisDrone 数据集
runs/
*.pt
*.onnx
*.engine
训练缓存和临时日志
```

