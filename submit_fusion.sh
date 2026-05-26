#!/bin/bash
#SBATCH --job-name=ffpe_fusion
#SBATCH --output=/public/home/wang/liujx/gigapath-ffpe-fusion/logs/slurm_ffpe_fusion_output_%j.log
#SBATCH --error=/public/home/wang/liujx/gigapath-ffpe-fusion/logs/slurm_ffpe_fusion_error_%j.log
#SBATCH --partition=gpu2-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00

# =============================================================================
# GigaPath × FFPE 融合 — 批量 L40S 双 GPU 提交
#
# 提交:
#   cd /public/home/wang/liujx/gigapath-ffpe-fusion
#   sbatch submit_fusion.sh
#
# 自定义参数:
#   SLIDE_DIR=/path/to/slides MODE=both sbatch submit_fusion.sh
# =============================================================================

set -euo pipefail

# ── 配置 ──
BASE="/public/home/wang/liujx/gigapath-ffpe-fusion"
cd "${BASE}"

# 数据参数
SLIDE_DIR="${SLIDE_DIR:-/public/home/wang/liujx/prov-gigapath-main/11111ovarian/finaltif}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE}/runs/$(date +%Y%m%d_%H%M%S)}"
MODE="${MODE:-fusion}"
MAX_SLIDES="${MAX_SLIDES:-0}"

# 设备参数
DEVICE_TILE0="${DEVICE_TILE0:-cuda:0}"
DEVICE_TILE1="${DEVICE_TILE1:-cuda:1}"
DEVICE_SLIDE="${DEVICE_SLIDE:-cuda:1}"

# ── 环境 ──
module load gcc-toolset/12
source /public/home/wang/liujx/miniconda3/bin/activate gigapath

export PYTHONPATH="${BASE}:${PYTHONPATH:-}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

if [ -f "/public/home/wang/share_group_folder_wang/sd-turbo/scheduler/scheduler_config.json" ]; then
    export HF_HUB_OFFLINE=1
fi

# ── 日志 ──
echo "============================================"
echo " GigaPath × FFPE 融合 Pipeline"
echo "============================================"
echo "Slide Dir:  ${SLIDE_DIR}"
echo "Mode:       ${MODE}"
echo "Output:     ${OUTPUT_DIR}"
echo "GPU tile:   ${DEVICE_TILE0} / ${DEVICE_TILE1}"
echo "GPU slide:  ${DEVICE_SLIDE}"
echo "Max slides: ${MAX_SLIDES} (0=全部)"
echo "Node:       $(hostname)"
echo ""
nvidia-smi -L
echo ""
nvidia-smi --query-gpu=memory.total --format=csv,noheader
echo ""

# ── 运行 ──
python "${BASE}/fusion_pipeline.py" \
    --slide_dir "${SLIDE_DIR}" \
    --mode "${MODE}" \
    --output_dir "${OUTPUT_DIR}" \
    --device_tile0 "${DEVICE_TILE0}" \
    --device_tile1 "${DEVICE_TILE1}" \
    --device_slide "${DEVICE_SLIDE}" \
    --max_slides "${MAX_SLIDES}"

echo ""
echo "============================================"
echo " 完成 ✅"
echo " 结果: ${OUTPUT_DIR}/"
echo " 汇总: ${OUTPUT_DIR}/fusion_summary.json"
echo "============================================"
