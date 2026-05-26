# GigaPath × FFPE Fusion

将 Diffusion-FFPE 增强步骤嵌入 GigaPath 在线推理管线，实现 **病理切片 embedding 的 FFPE 归一化**。

本项目是**自包含**的 — 无需额外安装 GigaPath 或 Diffusion-FFPE 仓库，所有必要代码已 vendored。运行时仅需下载预训练权重。

## 核心思路

FFPE（福尔马林固定石蜡包埋）切片存在染色变异和组织形变，影响 GigaPath 特征一致性。本项目在瓦片编码前插入 **Diffusion-FFPE 风格迁移**，将 FFPE 瓦片向 FF（新鲜冷冻）风格转换后再送入 Tile Encoder，从而：

- 减少 FFPE 批次间变异
- 提升下游分类/聚类任务的一致性
- 无需重新训练 GigaPath 模型（即插即用）

```
原始 WSI → 组织检测 → 瓦片提取 → [FFPE 增强] → Tile Encoder → Slide Encoder → embedding
```

## 项目结构

```
gigapath-ffpe-fusion/
├── ffpe_enhancer.py          # FFPE 增强器（Diffusion-FFPE 模型包装）
├── fusion_dataset.py          # 融合数据集（在线瓦片读取 + 可选增强）
├── fusion_pipeline.py         # 主推理管线（双 GPU 并行 + Slide Encoder）
├── coords.py                  # 组织坐标扫描（GPU 加速 / CPU 回退）
├── vendor/                    # vendored 外部依赖（自包含）
│   ├── gigapath/              #   GigaPath 编码器 (prov-gigapath)
│   └── diffusion_ffpe/        #   Diffusion-FFPE 模型
├── weights/                   # 预训练权重（需下载，.gitignore 已排除）
│   ├── pytorch_model.bin      #   Tile Encoder (ViT-giant/14 DINOV2)
│   ├── slide_encoder.pth      #   Slide Encoder (LongNetViT 12L/768d)
│   ├── model.pkl              #   Diffusion-FFPE (SD-Turbo based LoRA)
│   └── sd-turbo/              #   SD-Turbo 本地缓存（可选）
├── submit_fusion.sh           # SLURM 批量提交脚本
├── run_smoke_test.sh          # 冒烟测试
├── setup.sh                   # 一键环境配置
├── requirements.txt           # Pip 依赖
└── README.md
```

## 依赖

### Python 环境

```bash
conda create -n gigapath python=3.10
conda activate gigapath
pip install -r requirements.txt
```

### 模型权重下载

```bash
# 创建 weights 目录
mkdir -p weights

# 1. Tile Encoder: ViT-giant/14 DINOV2 (~4.4 GB)
#    来源: prov-gigapath
wget -O weights/pytorch_model.bin \
  "https://huggingface.co/prov-gigapath/prov-gigapath/resolve/main/pytorch_model.bin"

# 2. Slide Encoder: LongNetViT 12L/768d (~326 MB)
wget -O weights/slide_encoder.pth \
  "https://huggingface.co/prov-gigapath/prov-gigapath/resolve/main/slide_encoder.pth"

# 3. Diffusion-FFPE 模型权重 (model.pkl)
#    来源: Diffusion-FFPE 项目
#    请联系作者或从训练输出获取 model.pkl，放到 weights/model.pkl

# 4. SD-Turbo 缓存（可选，离线环境推荐）
git clone https://huggingface.co/stabilityai/sd-turbo weights/sd-turbo
```

## 快速开始

### 1. 环境配置

```bash
conda activate gigapath
pip install -r requirements.txt
# 下载模型权重到 weights/ 目录（见上方）
```

### 2. 冒烟测试（单张切片，双 GPU）

```bash
# 直接运行（需在 GPU 节点上）
SLIDE=/path/to/wsi.tif MODE=both MAX_TILES_TEST=100 bash run_smoke_test.sh

# 通过 SLURM 提交
SLIDE=/path/to/wsi.tif MODE=both MAX_TILES_TEST=2000 \
  sbatch --partition=gpu2-l40s --gres=gpu:tesla:2 --cpus-per-task=16 --mem=96G \
    --time=02:00:00 --job-name=ffpe_smoke \
    --wrap 'source ~/miniconda3/bin/activate gigapath; bash run_smoke_test.sh'
```

### 3. 命令行推理

```bash
# 单张切片，baseline + fusion 对比
python fusion_pipeline.py \
    --slide /path/to/wsi.tif \
    --mode both \
    --output_dir ./runs/test_run \
    --device_tile0 cuda:0 \
    --device_tile1 cuda:1 \
    --device_slide cuda:1
```

### 4. 批量推理

```bash
SLIDE_DIR=/path/to/slides MODE=fusion sbatch submit_fusion.sh
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--slide` | — | 单张 WSI 路径 |
| `--slide_dir` | — | WSI 目录（批量） |
| `--mode` | `fusion` | `baseline` / `fusion` / `both` |
| `--output_dir` | `./runs/output` | 输出目录 |
| `--tile_weight` | `weights/pytorch_model.bin` | Tile Encoder 权重 |
| `--slide_weight` | `weights/slide_encoder.pth` | Slide Encoder 权重 |
| `--ffpe_weight` | `weights/model.pkl` | Diffusion-FFPE 权重 |
| `--sd_turbo` | `stabilityai/sd-turbo` | SD-Turbo 路径 |
| `--max_tokens` | `0`（全部） | 最大 token 数 |
| `--tile_size` | `256` | 瓦片尺寸 |
| `--batch_size` | `32` | 每 GPU 批次大小 |
| `--device_tile0` | `cuda:0` | GPU0 设备 |
| `--device_tile1` | `cuda:1` | GPU1 设备 |
| `--device_slide` | `cuda:1` | Slide Encoder 设备 |

## 输出格式

```
output_dir/
├── baseline/
│   └── {slide_name}_slide768.npy    # GigaPath embedding [768]
├── fusion/
│   └── {slide_name}_slide768.npy    # FFPE 增强后 embedding [768]
└── fusion_summary.json              # 汇总（含错误日志）
```

## GPU 需求

| 配置 | GPU | 用途 |
|------|-----|------|
| 双 GPU | 2× L40S (48GB) | 每 GPU 编码一半瓦片，Slide Encoder 共享 GPU1 |

## 引用

本项目基于以下开源工作：

- **Prov-GigaPath**: Xu et al., "A whole-slide foundation model for digital pathology from real-world data", *Nature* (2024). [GitHub](https://github.com/prov-gigapath/prov-gigapath)
- **Diffusion-FFPE**: 基于 Stable Diffusion Turbo 的 FFPE→FF 风格迁移

```bibtex
@article{xu2024gigapath,
  title={A whole-slide foundation model for digital pathology from real-world data},
  author={Xu, Hanwen and Usuyama, Naoto and Bagga, Jaspreet and others},
  journal={Nature},
  year={2024}
}

@misc{gigapath-ffpe-fusion,
  author = {Junxi Liu},
  title = {GigaPath × FFPE Fusion: FFPE-Normalized Pathology Slide Embeddings},
  year = {2026},
}
```

## 许可

本项目代码采用 MIT License。

Vendored 代码保留原始许可：
- `vendor/gigapath/`: Prov-GigaPath (见原始仓库)
- `vendor/diffusion_ffpe/`: Diffusion-FFPE (见原始仓库)
