#!/bin/bash
# =============================================================================
# 冒烟测试 — 单张 Slide (10 tiles) 验证整个融合流程
#
# 用法:
#   bash run_smoke_test.sh
#   SLIDE=/path/to/wsi.tif bash run_smoke_test.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── 配置 ──
SLIDE="${SLIDE:-/public/home/wang/liujx/prov-gigapath-main/11111ovarian/finaltif/1925.tif}"
MODE="${MODE:-both}"          # baseline | fusion | both
DEVICE_TILE0="${DEVICE_TILE0:-cuda:0}"
DEVICE_TILE1="${DEVICE_TILE1:-cuda:1}"
DEVICE_SLIDE="${DEVICE_SLIDE:-cuda:1}"
MAX_TILES_TEST="${MAX_TILES_TEST:-100}"  # 只用前 100 个瓦片测试
OUTPUT_DIR="${OUTPUT_DIR:-./runs/smoke_test}"

# ── 环境 ──
module load gcc-toolset/12 2>/dev/null || true
source /public/home/wang/liujx/miniconda3/bin/activate gigapath

export PYTHONPATH="/public/home/wang/liujx/prov-gigapath-main:/public/home/wang/liujx/prov-gigapath-improveV4:/public/home/wang/liujx/Diffusion-FFPE-main:${SCRIPT_DIR}:${PYTHONPATH:-}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# 离线 SD-Turbo
if [ -f "/public/home/wang/share_group_folder_wang/sd-turbo/scheduler/scheduler_config.json" ]; then
    export HF_HUB_OFFLINE=1
fi

echo "============================================"
echo " GigaPath × FFPE 融合 冒烟测试"
echo "============================================"
echo "Slide:    ${SLIDE}"
echo "Mode:     ${MODE}"
echo "GPU tile: ${DEVICE_TILE0} / ${DEVICE_TILE1}"
echo "GPU slide: ${DEVICE_SLIDE}"
echo "最大瓦片: ${MAX_TILES_TEST}"
echo "输出:     ${OUTPUT_DIR}"
echo ""

nvidia-smi -L 2>/dev/null || echo "(非 GPU 节点，请用 sbatch 提交)"

# ── 快速自检：Python 导入 ──
echo "[1/3] Python 导入检查..."
python -c "
from ffpe_enhancer import FFPEEnhancer
from fusion_dataset import FusionTileDataset
from fusion_pipeline import get_tile_encoder, get_slide_encoder, scan_tissue_coords
print('✅ 导入成功')
"

# ── 运行 ──
echo "[2/3] 运行融合 Pipeline..."
rm -rf "${OUTPUT_DIR}"

python "${SCRIPT_DIR}/fusion_pipeline.py" \
    --slide "${SLIDE}" \
    --mode "${MODE}" \
    --output_dir "${OUTPUT_DIR}" \
    --device_tile0 "${DEVICE_TILE0}" \
    --device_tile1 "${DEVICE_TILE1}" \
    --device_slide "${DEVICE_SLIDE}" \
    --max_tokens "${MAX_TILES_TEST}"

# ── 验证输出 ──
echo ""
echo "[3/3] 验证输出..."
echo ""

for mode_name in baseline fusion; do
    MODE_DIR="${OUTPUT_DIR}/${mode_name}"
    if [ -d "${MODE_DIR}" ]; then
        SLIDE_NAME="$(basename "${SLIDE}" .tif)"
        EMB_FILE="${MODE_DIR}/${SLIDE_NAME}_slide768.npy"
        if [ -f "${EMB_FILE}" ]; then
            python -c "
import numpy as np
emb = np.load('${EMB_FILE}')
print(f'  ✅ ${mode_name}: {emb.shape} (mean={emb.mean():.4f}, std={emb.std():.4f})')
"
        else
            echo "  ⚠️  ${mode_name}: embedding 文件未生成"
        fi
    fi
done

echo ""
cat "${OUTPUT_DIR}/fusion_summary.json" 2>/dev/null | python -m json.tool 2>/dev/null || true

echo ""
echo "============================================"
echo " 冒烟测试完成 ✅"
echo "============================================"
