#!/bin/bash
#SBATCH --job-name=ffpe_resume
#SBATCH --output=/public/home/wang/liujx/gigapath-ffpe-fusion/logs/slurm_ffpe_resume_%j.out
#SBATCH --error=/public/home/wang/liujx/gigapath-ffpe-fusion/logs/slurm_ffpe_resume_%j.err
#SBATCH --partition=gpu2-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00

# =============================================================================
# GigaPath × FFPE 融合 — 断点续跑
#
# 从已有输出目录断点续跑，跳过已完成的 slide。
#
# 用法:
#   export RESUME_OUTPUT_DIR=/path/to/existing/output
#   sbatch submit_resume_fusion.sh
#
# 环境变量:
#   RESUME_OUTPUT_DIR — 已有输出目录（默认: runs/batch_537_20260526）
#   SLIDE_DIR         — 切片目录（默认: 11111ovarian/finaltif）
# =============================================================================

set -euo pipefail

BASE="/public/home/wang/liujx/gigapath-ffpe-fusion"
cd "${BASE}"

# ── 配置 ──
SLIDE_DIR="${SLIDE_DIR:-/public/home/wang/liujx/prov-gigapath-main/11111ovarian/finaltif}"
OUTPUT_DIR="${RESUME_OUTPUT_DIR:-${BASE}/runs/batch_537_20260526}"
FUSION_DIR="${OUTPUT_DIR}/fusion"

echo "============================================"
echo " FFPE 断点续跑"
echo "============================================"
echo "Slide Dir:    ${SLIDE_DIR}"
echo "Output Dir:   ${OUTPUT_DIR}"
echo "Fusion Dir:   ${FUSION_DIR}"
echo "Node:         $(hostname)"
echo ""

# ── 扫描已完成 slide ──
declare -A DONE
if [ -d "${FUSION_DIR}" ]; then
    while IFS= read -r npy; do
        name=$(basename "${npy}" _slide768.npy)
        DONE["${name}"]=1
    done < <(find "${FUSION_DIR}" -maxdepth 1 -name '*_slide768.npy')
fi

echo "已完成: ${#DONE[@]} 张 slide"
echo ""

# ── 收集待处理 slide ──
TEMP_SLIDE_DIR="${OUTPUT_DIR}/.resume_slides_$$"
rm -rf "${TEMP_SLIDE_DIR}"
mkdir -p "${TEMP_SLIDE_DIR}"

TODO_COUNT=0
while IFS= read -r tif; do
    name=$(basename "${tif}" .tif)
    if [ -z "${DONE[${name}]:-}" ]; then
        ln -s "${tif}" "${TEMP_SLIDE_DIR}/$(basename "${tif}")"
        TODO_COUNT=$((TODO_COUNT + 1))
    fi
done < <(find "${SLIDE_DIR}" -maxdepth 1 -name '*.tif' | sort)

echo "待处理: ${TODO_COUNT} 张 slide"
echo ""

if [ "${TODO_COUNT}" -eq 0 ]; then
    echo "✅ 全部已完成，无需续跑。"
    rm -rf "${TEMP_SLIDE_DIR}"
    exit 0
fi

# ── 环境 ──
module load gcc-toolset/12
source /public/home/wang/liujx/miniconda3/bin/activate gigapath

export PYTHONPATH="/public/home/wang/liujx/prov-gigapath-main:/public/home/wang/liujx/prov-gigapath-improveV4:/public/home/wang/liujx/Diffusion-FFPE-main:${BASE}:${PYTHONPATH:-}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export PYTHONUNBUFFERED=1

if [ -f "/public/home/wang/share_group_folder_wang/sd-turbo/scheduler/scheduler_config.json" ]; then
    export HF_HUB_OFFLINE=1
fi

nvidia-smi -L
echo ""
nvidia-smi --query-gpu=memory.total --format=csv,noheader
echo ""

# ── 运行（只跑 fusion 模式，baseline 已完成）──
python "${BASE}/fusion_pipeline.py" \
    --slide_dir "${TEMP_SLIDE_DIR}" \
    --mode fusion \
    --output_dir "${OUTPUT_DIR}" \
    --device_tile0 cuda:0 \
    --device_tile1 cuda:1 \
    --device_slide cuda:1 \
    --max_slides 0

RET=$?

# ── 清理 ──
rm -rf "${TEMP_SLIDE_DIR}"

echo ""
echo "============================================"
if [ ${RET} -eq 0 ]; then
    echo " 续跑完成 ✅"
else
    echo " 续跑异常 (exit=${RET})"
fi
echo " 结果: ${FUSION_DIR}/"
echo "============================================"

exit ${RET}
