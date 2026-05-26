#!/usr/bin/env python3
"""
GigaPath × FFPE 融合 Pipeline

将 Diffusion-FFPE 增强步骤嵌入 prov-gigapath-improveV4 的在线推理管线:
  WSI .tif
    → 组织坐标扫描 (GPU 加速)
    → 坐标二分 → 双 GPU 并行
    → [每个 GPU] 在线取瓦片 → FFPE 增强 → DINOV2 Tile Encoder
    → 合并 → Token 采样
    → LongNet Slide Encoder → 768d embedding

对比运行模式:
  --mode baseline : 无 FFPE 增强 (等同 V4 原始流程)
  --mode fusion   : 加入 FFPE 增强 (融合流程)

用法:
  # 单张 Slide
  python fusion_pipeline.py --slide /path/to/wsi.tif --mode fusion

  # 批量处理目录
  python fusion_pipeline.py --slide_dir /path/to/finaltif --mode both
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fusion_pipeline")

# ── 路径常量 ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
IMPROVEV4_ROOT = Path("/public/home/wang/liujx/prov-gigapath-improveV4")
GIGAPATH_ROOT = Path("/public/home/wang/liujx/prov-gigapath-main")

# 模型权重默认路径
DEFAULT_TILE_WEIGHT = "/public/home/wang/liujx/pytorch_model.bin"
DEFAULT_SLIDE_WEIGHT = "/public/home/wang/liujx/slide_encoder.pth"

# 确保依赖在 Python path
for _p in [str(IMPROVEV4_ROOT), str(GIGAPATH_ROOT), str(PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def setup_paths():
    """确保所有必要模块可导入。"""
    pass  # sys.path 已在上面设置


def get_tile_encoder(device: torch.device, tile_weight: str):
    """加载 DINOV2 ViT-giant/14 Tile Encoder。"""
    import timm

    logger.info("加载 Tile Encoder (ViT-giant/14 DINOV2) → %s", device)
    model = timm.create_model(
        "vit_giant_patch14_dinov2",
        pretrained=True,
        img_size=224,
        in_chans=3,
        pretrained_cfg_overlay=dict(file=tile_weight),
    )
    model = model.to(device)
    model.eval()
    return model


def get_slide_encoder(device: torch.device, slide_weight: str):
    """加载 LongNetViT 12L/768d Slide Encoder。"""
    from gigapath.slide_encoder import create_model

    logger.info("加载 Slide Encoder (LongNetViT 12L/768d) → %s", device)
    model = create_model(
        pretrained=slide_weight,
        model_arch="gigapath_slide_enc12l768d",
        in_chans=1536,
    )
    model = model.to(device)
    model.eval()
    return model


def scan_tissue_coords(
    slide_path: str,
    tile_size: int = 256,
    target_level: int = 0,
    bg_threshold: int = 210,
    scan_step: int = 1,
    gpu_device: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """GPU 加速组织坐标扫描（复用 V4 的 coords.py）。

    Returns:
        List[(x0, y0)] level-0 对齐的组织瓦片坐标。
    """
    from parallel_improve2.wsi_embed.coords import compute_tissue_coords_parallel_strips_gpu

    logger.info("扫描组织坐标: %s", os.path.basename(slide_path))

    # 尝试 GPU 加速扫描
    try:
        coords, _ = compute_tissue_coords_parallel_strips_gpu(
            slide_path,
            tile_size=tile_size,
            target_level=target_level,
            bg_threshold=bg_threshold,
            scan_step=scan_step,
            num_workers=min(48, os.cpu_count() or 16),
        )
    except Exception:
        logger.warning("GPU 扫描失败，回退 CPU 扫描")
        from parallel_improve2.wsi_embed.coords import compute_tissue_coords_vectorized
        coords, _ = compute_tissue_coords_vectorized(slide_path, tile_size=tile_size, target_level=target_level, bg_threshold=bg_threshold, scan_step=scan_step)

    logger.info("检测到 %d 个组织瓦片", len(coords))
    return coords


def encode_tiles_on_gpu(
    slide_path: str,
    coords: List[Tuple[int, int]],
    tile_encoder: torch.nn.Module,
    device: torch.device,
    enhance: bool = False,
    enhancer=None,
    tile_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """在单个 GPU 上编码瓦片（带可选 FFPE 增强）。

    Args:
        coords: 该 GPU 负责的坐标子集。
        enhance: True 时插入 FFPE 增强步骤。

    Returns:
        features: [N, 1536] float32
        coords_scaled: [N, 2] float32
    """
    from fusion_dataset import FusionTileDataset

    dataset = FusionTileDataset(
        slide_path=slide_path,
        coords=coords,
        tile_size=tile_size,
        target_level=0,
        enhance=enhance,
        enhancer=enhancer,
        enhance_batch_size=8,
    )
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if enhance else min(num_workers, 4),
        pin_memory=True,
        drop_last=False,
    )

    all_feats = []
    all_coords = []

    tile_encoder = tile_encoder.to(device)
    with torch.inference_mode():
        for tiles, coords_batch in loader:
            tiles = tiles.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                emb = tile_encoder(tiles)  # [B, 1536]
            all_feats.append(emb.float().cpu())
            all_coords.append(coords_batch.cpu())

    dataset.close()
    return torch.cat(all_feats, dim=0), torch.cat(all_coords, dim=0)


def encode_slide(
    feats: torch.Tensor,
    coords: torch.Tensor,
    slide_encoder: torch.nn.Module,
    device: torch.device,
    max_tokens: int = 12000,
    seed: int = 42,
) -> np.ndarray:
    """Slide Encoder: Token 采样 → LongNet → 768d embedding。

    Args:
        feats: [N, 1536] tile features
        coords: [N, 2] tile coordinates
        max_tokens: 最大 token 数（随机采样）

    Returns:
        embedding: numpy [768] float32
    """
    n = feats.shape[0]
    if n > max_tokens:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)[:max_tokens]
        feats = feats[indices]
        coords = coords[indices]
        logger.info("Token 采样: %d → %d", n, max_tokens)

    feats = feats.unsqueeze(0).to(device, dtype=torch.float16, non_blocking=True)
    coords = coords.unsqueeze(0).to(device, non_blocking=True)

    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            rep = slide_encoder(feats, coords)  # [1, 768]

    return rep[0].squeeze().float().cpu().numpy()


def process_single_slide(
    slide_path: str,
    tile_encoder: torch.nn.Module,
    slide_encoder: torch.nn.Module,
    device_tile0: torch.device,
    device_tile1: torch.device,
    device_slide: torch.device,
    enhance: bool = False,
    enhancer=None,
    output_dir: Optional[str] = None,
    max_tokens: int = 12000,
) -> Tuple[np.ndarray, Dict]:
    """处理单张 WSI：坐标扫描 → 双 GPU tile 编码 → slide 编码。

    Returns:
        embedding: [768] float32
        stats: 统计信息字典
    """
    t_start = time.time()
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]

    # 1. 坐标扫描
    coords = scan_tissue_coords(slide_path)
    if len(coords) == 0:
        raise ValueError(f"未检测到组织区域: {slide_path}")
    n_coords = len(coords)

    # 2. 坐标二分
    mid = n_coords // 2
    coords_a = coords[:mid]
    coords_b = coords[mid:]
    logger.info("坐标分配: GPU0=%d, GPU1=%d", len(coords_a), len(coords_b))

    # 3. 双 GPU 并行 tile 编码
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(
            encode_tiles_on_gpu,
            slide_path, coords_a, tile_encoder, device_tile0,
            enhance, enhancer,
        )
        future_b = executor.submit(
            encode_tiles_on_gpu,
            slide_path, coords_b, tile_encoder, device_tile1,
            enhance, enhancer,
        )
        feats_a, coords_a = future_a.result()
        feats_b, coords_b = future_b.result()

    # 4. 合并特征
    feats = torch.cat([feats_a, feats_b], dim=0)
    coords_all = torch.cat([coords_a, coords_b], dim=0)
    logger.info("合并 tile features: [%d, 1536]", feats.shape[0])

    # 5. Slide encoding
    embedding = encode_slide(
        feats, coords_all, slide_encoder, device_slide,
        max_tokens=max_tokens,
    )

    # 6. 保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"{slide_name}_slide768.npy"), embedding)

    elapsed = time.time() - t_start
    stats = {
        "slide": slide_name,
        "n_tiles": n_coords,
        "enhance": enhance,
        "elapsed_s": round(elapsed, 1),
        "embedding_shape": list(embedding.shape),
    }
    logger.info("完成 %s: %.1fs (%d tiles)", slide_name, elapsed, n_coords)

    return embedding, stats


def main():
    parser = argparse.ArgumentParser(
        description="GigaPath × FFPE 融合 Pipeline"
    )
    parser.add_argument(
        "--slide",
        type=str,
        default=None,
        help="单张 WSI .tif 路径",
    )
    parser.add_argument(
        "--slide_dir",
        type=str,
        default=None,
        help="批量处理目录（递归搜索 .tif）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./runs",
        help="Embedding 输出目录",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fusion",
        choices=["baseline", "fusion", "both"],
        help="baseline=无增强, fusion=FFPE增强, both=两者都跑",
    )
    parser.add_argument(
        "--device_tile0",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--device_tile1",
        type=str,
        default="cuda:1",
    )
    parser.add_argument(
        "--device_slide",
        type=str,
        default="cuda:1",
    )
    parser.add_argument(
        "--tile_weight",
        type=str,
        default=DEFAULT_TILE_WEIGHT,
    )
    parser.add_argument(
        "--slide_weight",
        type=str,
        default=DEFAULT_SLIDE_WEIGHT,
    )
    parser.add_argument(
        "--ffpe_pretrained",
        type=str,
        default="/public/home/wang/liujx/Diffusion-FFPE-main/checkpoints/model.pkl",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=12000,
    )
    parser.add_argument(
        "--max_slides",
        type=int,
        default=0,
        help="批量模式最多处理张数（0=全部）",
    )
    args = parser.parse_args()

    if not args.slide and not args.slide_dir:
        parser.error("需要 --slide 或 --slide_dir")

    # ── 设置设备 ──
    device_tile0 = torch.device(args.device_tile0)
    device_tile1 = torch.device(args.device_tile1)
    device_slide = torch.device(args.device_slide)

    # ── 加载模型 ──
    logger.info("=" * 60)
    logger.info("加载模型...")

    # Tile Encoder (每个 GPU 一份独立副本)
    tile_enc0 = get_tile_encoder(device_tile0, args.tile_weight)
    tile_enc1 = get_tile_encoder(device_tile1, args.tile_weight)

    # Slide Encoder
    slide_enc = get_slide_encoder(device_slide, args.slide_weight)

    # FFPE Enhancer (按需加载，仅 fusion 模式用)
    enhancer = None
    if args.mode in ("fusion", "both"):
        from ffpe_enhancer import FFPEEnhancer
        logger.info("加载 FFPE 增强器...")
        enhancer = FFPEEnhancer(
            pretrained_path=args.ffpe_pretrained,
            device="cuda:0",  # 与 tile encoder 共用 GPU0
        )

    # ── 收集 slide 列表 ──
    slide_paths: List[str] = []
    if args.slide:
        slide_paths = [args.slide]
    elif args.slide_dir:
        for root, _, files in os.walk(args.slide_dir):
            for f in sorted(files):
                if f.lower().endswith((".tif", ".svs", ".ndpi")):
                    slide_paths.append(os.path.join(root, f))
        if args.max_slides > 0:
            slide_paths = slide_paths[:args.max_slides]

    logger.info("待处理: %d 张 slide", len(slide_paths))

    # ── 处理 ──
    modes_to_run = []
    if args.mode in ("baseline", "both"):
        modes_to_run.append(("baseline", False))
    if args.mode in ("fusion", "both"):
        modes_to_run.append(("fusion", True))

    all_stats = []

    for mode_name, do_enhance in modes_to_run:
        mode_output_dir = os.path.join(args.output_dir, mode_name)
        logger.info("=" * 60)
        logger.info("模式: %s (enhance=%s)", mode_name, do_enhance)

        for i, slide_path in enumerate(slide_paths):
            logger.info("[%d/%d] %s", i + 1, len(slide_paths),
                        os.path.basename(slide_path))
            try:
                _, stats = process_single_slide(
                    slide_path=slide_path,
                    tile_encoder=tile_enc0,  # GPU0
                    slide_encoder=slide_enc,
                    device_tile0=device_tile0,
                    device_tile1=device_tile1,
                    device_slide=device_slide,
                    enhance=do_enhance,
                    enhancer=enhancer,
                    output_dir=mode_output_dir,
                    max_tokens=args.max_tokens,
                )
                # 覆盖 GPU1 的 tile encoder
                stats["mode"] = mode_name
                all_stats.append(stats)

            except Exception as e:
                logger.error("失败 %s: %s", slide_path, e)
                all_stats.append({
                    "slide": os.path.basename(slide_path),
                    "mode": mode_name,
                    "error": str(e),
                })

    # ── 汇总 ──
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "fusion_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    logger.info("汇总保存: %s", summary_path)

    # 简单统计
    success = [s for s in all_stats if "error" not in s]
    logger.info(
        "完成: %d/%d 成功",
        len(success), len(all_stats),
    )
    if success:
        for mode_name in [m[0] for m in modes_to_run]:
            mode_stats = [s for s in success if s.get("mode") == mode_name]
            if mode_stats:
                avg_time = np.mean([s["elapsed_s"] for s in mode_stats])
                logger.info(
                    "  %s: %d slides, 平均 %.1fs/slide",
                    mode_name, len(mode_stats), avg_time,
                )


if __name__ == "__main__":
    main()
