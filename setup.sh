#!/bin/bash
# =============================================================================
# GigaPath × FFPE 融合项目 — 一键环境配置
#
# 前置条件: gigapath conda 环境已存在
# 用法: bash setup.sh
# =============================================================================
set -euo pipefail

PROJECT_DIR="/public/home/wang/liujx/gigapath-ffpe-fusion"
cd "${PROJECT_DIR}"

echo "============================================"
echo " GigaPath × FFPE 融合项目 环境配置"
echo "============================================"

# 1. 激活 conda 环境
source /public/home/wang/liujx/miniconda3/bin/activate gigapath
echo "✅ conda 环境: gigapath"

# 2. 安装额外依赖
echo ""
echo "📦 安装额外 Python 依赖..."
pip install -r requirements_extra.txt
echo "✅ 依赖安装完成"

# 3. 验证关键模块
echo ""
echo "🔍 验证关键模块..."

python -c "
import sys
sys.path.insert(0, '/public/home/wang/liujx/Diffusion-FFPE-main')
sys.path.insert(0, '/public/home/wang/liujx/prov-gigapath-improveV4')
sys.path.insert(0, '/public/home/wang/liujx/prov-gigapath-main')

# Diffusion-FFPE
from diffusion_ffpe.model import Diffusion_FFPE, initialize_text_encoder
print('  ✅ Diffusion_FFPE 可导入')

# GigaPath
from gigapath.slide_encoder import create_model
print('  ✅ GigaPath Slide Encoder 可导入')

# V4 coords
from parallel_improve2.wsi_embed.coords import compute_tissue_coords_parallel_strips_gpu
print('  ✅ V4 GPU 坐标扫描可导入')

# 本地模块
from ffpe_enhancer import FFPEEnhancer
from fusion_dataset import FusionTileDataset
print('  ✅ 融合模块可导入')
"

# 4. 检查关键文件
echo ""
echo "📁 检查关键文件..."

check_file() {
    if [ -f "$1" ]; then
        echo "  ✅ $1"
    else
        echo "  ❌ 缺失: $1"
    fi
}

check_file "/public/home/wang/liujx/Diffusion-FFPE-main/checkpoints/model.pkl"
check_file "/public/home/wang/liujx/pytorch_model.bin"
check_file "/public/home/wang/liujx/slide_encoder.pth"
check_file "/public/home/wang/share_group_folder_wang/sd-turbo/scheduler/scheduler_config.json"

echo ""
echo "============================================"
echo " 环境配置完成 ✅"
echo "============================================"
echo ""
echo "下一步:"
echo "  冒烟测试: bash run_smoke_test.sh"
echo "  单张推理: python fusion_pipeline.py --slide <path> --mode fusion"
echo "  批量推理: bash submit_fusion.sh"
