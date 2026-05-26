# GigaPath × FFPE Fusion

将 Diffusion-FFPE 增强步骤嵌入 GigaPath 在线推理管线，实现 **病理切片 embedding 的 FFPE 归一化**。

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
├── ffpe_enhancer.py       # FFPE 增强器（加载 Diffusion-FFPE 模型）
├── fusion_dataset.py       # 融合数据集（在线瓦片读取 + 可选 FFPE 增强）
├── fusion_pipeline.py      # 主推理管线（双 GPU 并行编码 + Slide Encoder）
├── submit_fusion.sh        # SLURM 批量提交脚本（L40S × 2）
├── run_smoke_test.sh       # 冒烟测试（单张切片验证）
├── setup.sh                # 一键环境配置
├── requirements_extra.txt  # 额外 Python 依赖
├── requirements.txt        # 完整依赖列表
└── README.md
```

## 依赖项

### 前置项目（需在同一文件系统可访问）

| 项目 | 用途 |
|------|------|
| [prov-gigapath-improveV4](https://github.com/prov-gigapath/prov-gigapath) | GigaPath 瓦片/切片编码器 + 组织坐标扫描 |
| [Diffusion-FFPE](https://github.com/****/Diffusion-FFPE) | FFPE→FF 风格迁移模型 |
| [prov-gigapath-main](https://github.com/prov-gigapath/prov-gigapath) | WSI 处理工具 |

### 模型权重

- **Tile Encoder**: `pytorch_model.bin`（ViT-giant/14 DINOV2）
- **Slide Encoder**: `slide_encoder.pth`（LongNetViT 12L/768d）
- **Diffusion-FFPE**: `model.pkl`（SD-Turbo based LoRA）
- **SD-Turbo**: 本地缓存 `sd-turbo/` 或自动下载 `stabilityai/sd-turbo`

### Python 环境

```bash
conda create -n gigapath python=3.10
conda activate gigapath
pip install -r requirements.txt
```

## 快速开始

### 1. 环境配置

```bash
# 编辑 setup.sh 中的路径变量
bash setup.sh
```

### 2. 冒烟测试（单张切片，双 GPU）

```bash
# 直接运行（需在 GPU 节点上）
SLIDE=/path/to/wsi.tif MODE=both MAX_TILES_TEST=100 bash run_smoke_test.sh

# 或通过 SLURM 提交
cd gigapath-ffpe-fusion
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

## 输入类型

支持两种输入格式：

| 输入 | 说明 | 处理方式 |
|------|------|----------|
| 单张 WSI | `.tif`（支持金字塔层级） | 自动检测组织区域，MODE=both 同时跑 baseline/fusion |
| 幻灯片目录 | 包含多张 `.tif` 的文件夹 | `--slide_dir` 批量处理 |

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

如果本项目对你的研究有帮助，请引用：

```bibtex
@misc{gigapath-ffpe-fusion,
  author = {Junxi Liu},
  title = {GigaPath × FFPE Fusion: FFPE-Normalized Pathology Slide Embeddings},
  year = {2026},
}
```

## 许可

MIT License
