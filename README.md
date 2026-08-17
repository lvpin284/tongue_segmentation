# tongue_segmentation

这是一个用于舌头（Tongue）图像/视频分割与对比实验的 PyTorch 项目，包含基于 SAM（Segment Anything）改进的 TongueSegSAM 后端、多种对比模型（U-Net、CNN、MedSAM 等）、评估/可视化脚本与数据工具。适合研究人员或工程师用于训练/评估舌头分割模型、复现实验与生成可视化结果。

## 主要特点

- 支持基于 SAM 的自定义分割后端（TongueSegSAM），以及多种对比模型（U-Net、CNN、YOLO 等）。
- 支持静态图像训练/验证以及视频帧的实时/批量预测（仓库集成 sam2 的视频预测器）。
- 提供评估（Dice、IoU 等）、损失函数、提示点生成与可视化工具。
- 包含实验绘图脚本，用于生成对比图（loss / iou / dice）。

## 技术栈
- 语言：Python 3.8+
- 框架 / 运行时：PyTorch（具体版本视 CUDA 环境而定）
- 主要依赖：torch, torchvision, numpy, pandas, opencv-python, matplotlib, scikit-image

## 仓库结构（重要项）

```
ComparativeExperiment/      # 对比实验脚本、数据转换与绘图
  ├── train_unet.py
  ├── train_cnn.py
  ├── train_medsam.py
  ├── convert_coco_to_yolo.py
  ├── dataset.py
  ├── plot_experiment.py
  └── cnn_checkpoint.pth.tar

TongueSeg_Main/            # TongueSeg 专用训练脚本（temporal transformer 等）
  ├── train_temporal_transformer.py
  └── snaps/

models/                     # 模型工厂与 SAM 相关实现
  ├── model_dict.py
  ├── segment_anything/
  └── segment_anything_tongueseg/

plots/                      # 实验结果图与定性示例
  ├── loss_comparison.png
  ├── iou_comparison.png
  └── qualitative/

sam2/                       # 集成的 SAM-v2 相关脚本（图像/视频预测、自动掩码生成）
  ├── sam2_image_predictor.py
  ├── sam2_video_predictor.py
  └── automatic_mask_generator.py

utils/                      # 工具模块：配置、数据集、评估、损失、可视化等
  ├── config.py
  ├── dataset_tongue.py
  ├── data_us.py
  ├── evaluation.py
  ├── generate_prompts.py
  ├── visualization.py
  ├── metrics.py
  └── loss_functions/

train.py                    # 主训练脚本（支持 --task Tongue）
realtime_seg.py             # 实时 / 视频推理
test.py / test_video.py     # 推理测试脚本
run_vis.py                  # 可视化运行脚本
SHAP.py                     # SHAP 分析脚本
```

## 核心设计（简要说明）
- 模型构建由 models/model_dict.py 管理；通过传入 `--modelname` 来选择不同后端（如 TongueSegSAM、UNet、CNN 等）。
- 数据读取：utils/dataset_tongue.py 针对 Tongue 数据集，utils/data_us.py 提供通用的 Image-to-Image 数据加载器。
- 损失函数和提示：utils/loss_functions/ 包含自定义损失（SAM 相关），utils/generate_prompts.py 负责生成点击/提示输入给 SAM。 
- 训练流程与评估：train.py 实现训练循环、学习率调整、日志记录与验证；评估逻辑在 utils/evaluation.py。

## 快速开始（最短路径）

1. 克隆仓库并进入目录：

```bash
git clone https://github.com/lvpin284/tongue_segmentation.git
cd tongue_segmentation
```

2. 推荐使用虚拟环境并安装依赖（示例）：

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.\.venv\Scripts\activate  # Windows (PowerShell/CMD)

# 安装常见依赖（根据 CUDA/torch 版本调整 torch 安装指令）
pip install numpy pandas matplotlib opencv-python scikit-image tqdm tensorboard
# 安装 torch / torchvision：请根据你的 CUDA 版本到 https://pytorch.org/ 获取正确的命令
```

3. 准备 SAM checkpoint（可选，但推荐）：

将 SAM 的预训练权重放到 `checkpoints/sam_vit_b_01ec64.pth`，或通过 train.py 的 `--sam_ckpt` 参数指定路径。

4. 准备数据：

- 使用 utils/dataset_tongue.py 的格式准备你的数据目录（images, masks 等）。
- 若使用 YOLO 格式，参考 ComparativeExperiment/convert_coco_to_yolo.py 与 yolo_data.yaml 转换。

5. 运行训练示例（Tongue 任务）：

```bash
python train.py --task Tongue --modelname TongueSegSAM --encoder_input_size 256 --batch_size 4 --n_gpu 1 --base_lr 5e-4
```

6. 推理 / 可视化：

```bash
# 视频或实时分割
python realtime_seg.py
python test_video.py --video path/to/video.mp4

# 绘制对比图
python ComparativeExperiment/plot_experiment.py
python run_vis.py
```

## 评估指标
- 评估在 utils/evaluation.py 中实现，常见度量包含 Dice、IoU 以及验证损失。
- ComparativeExperiment 下保存有示例指标（yolo_metrics.csv, cnn_metrics.csv），并在 plots/ 下提供可视化图。

## 复现与调试提示
- train.py 中固定了随机种子（1234）以提高可复现性。
- 如果使用多 GPU，train.py 支持 nn.DataParallel（通过 `--n_gpu` 设置）。
- 若出现模型权重加载问题，检查 checkpoint 路径与模型名称是否匹配；脚本对带有 `module.` 前缀的 state_dict 做了兼容处理。
- SPM loss 警告：train.py 提供 `--spm_warn_window`、`--spm_warn_rise_ratio`、`--spm_warn_osc_ratio` 参数用于检测异常损失波动。

## 建议的改进（TODO）
- 添加明确的 requirements.txt / environment.yml，固定各依赖版本以便复现。
- 在 README 中补充数据格式示例和小型样本数据以便快速验证。
- 添加 LICENSE 文件以明确开源许可。
- 将大型二进制文件（如模型 checkpoint）移入 Releases 或使用 Git LFS 管理。

## 贡献
欢迎提交 issue 或 PR：
- 修复 bug、补充示例数据、添加依赖文件（requirements.txt）或完善文档均非常欢迎。

## 许可与致谢
请在仓库中补充 LICENSE 文件以明确许可信息。若使用第三方预训练模型（如 SAM），请保留并遵守对应模型的许可与引用信息。

---

如果你希望我把 README 进一步定制为英文版、添加具体的 requirements.txt，或直接把 README 提交到仓库（我已为你准备好提交），告诉我你要的操作。