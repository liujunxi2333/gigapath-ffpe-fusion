#!/usr/bin/env python3
"""
融合数据集 — 从 WSI 在线提取瓦片 → FFPE 增强 → 输出给 Tile Encoder。

这是融合项目的核心模块，替换原有 prov-gigapath-improveV4 的
BaselineWSITileDataset，在瓦片预处理中插入 FFPE 增强步骤。

数据流:
  1. 根据坐标 (x0, y0) 从 WSI 读取 256×256 region
  2. [可选] FFPE 增强（256×256 → 256×256 风格迁移）
  3. Tile Encoder 预处理: Resize(256) → CenterCrop(224) → ImageNet Normalize

两种模式:
  - enhance=False: 等同 BaselineWSITileDataset（baseline 对照组）
  - enhance=True:  插入 FFPE 增强（融合实验组）
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# ── Tile Encoder 预处理（与 prov-gigapath-improveV4 保持一致）─────
TILE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class FusionTileDataset(Dataset):
    """融合瓦片数据集。

    对比 BaselineWSITileDataset 多了 enhance 开关和 FFPE 增强步骤。

    Args:
        slide_path: WSI .tif 文件路径
        coords: 组织瓦片坐标列表 [(x0, y0), ...]（level-0 坐标）
        tile_size: 瓦片尺寸（默认 256）
        target_level: OpenSlide 读取层级（0 = 最高分辨率）
        enhance: 是否启用 FFPE 增强
        enhancer: FFPEEnhancer 实例（enhance=True 时必需）
        enhance_batch_size: FFPE 增强批大小
        open_slide: 可复用的 OpenSlide 句柄（多进程下不要传入）
    """

    def __init__(
        self,
        slide_path: str,
        coords: List[Tuple[int, int]],
        tile_size: int = 256,
        target_level: int = 0,
        enhance: bool = False,
        enhancer=None,
        enhance_batch_size: int = 8,
        open_slide=None,
    ):
        self.slide_path = slide_path
        self.coords = coords
        self.tile_size = tile_size
        self.target_level = target_level
        self.enhance = enhance
        self.enhancer = enhancer
        self.enhance_batch_size = enhance_batch_size

        # OpenSlide 句柄
        if open_slide is not None:
            self._slide = open_slide
        else:
            import openslide
            self._slide = openslide.OpenSlide(slide_path)

        # 计算 level-0 tile_size（考虑了 downsample factor）
        self.level0_tile_size = int(
            self.tile_size * self._slide.level_downsamples[self.target_level]
        )

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tile: torch.Tensor [3, 224, 224], float32, ImageNet 标准化后
            coord: torch.Tensor [2], float32，(x_scaled, y_scaled)
        """
        x0, y0 = self.coords[idx]

        # ── 1. 从 WSI 读取 256×256 瓦片 ──
        tile_pil = self._slide.read_region(
            (x0, y0),
            self.target_level,
            (self.tile_size, self.tile_size),
        ).convert("RGB")

        # ── 2. [可选] FFPE 增强 ──
        if self.enhance and self.enhancer is not None:
            tile_pil = self.enhancer.enhance(tile_pil)

        # ── 3. Tile Encoder 预处理 ──
        tile_tensor = TILE_TRANSFORM(tile_pil)

        # ── 4. 坐标缩放（与 prov-gigapath-improveV4 一致） ──
        coord_scaled = torch.tensor(
            [x0 / self.tile_size, y0 / self.tile_size],
            dtype=torch.float32,
        )

        return tile_tensor, coord_scaled

    def close(self):
        """关闭 OpenSlide 句柄。"""
        if hasattr(self, "_slide") and self._slide is not None:
            self._slide.close()
            self._slide = None

    def __del__(self):
        self.close()


class FusionBatchProcessor:
    """批量处理瓦片：读取 + FFPE增强 + 预处理。

    相比 Dataset 逐个 __getitem__ 的方式，本处理器可以将 FFPE 增强
    批量化，大幅提升 GPU 利用率。
    """

    def __init__(
        self,
        slide_path: str,
        coords: List[Tuple[int, int]],
        enhancer,
        tile_size: int = 256,
        target_level: int = 0,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.slide_path = slide_path
        self.coords = coords
        self.enhancer = enhancer
        self.tile_size = tile_size
        self.target_level = target_level
        self.batch_size = batch_size
        self.num_workers = num_workers

        import openslide
        self.slide = openslide.OpenSlide(slide_path)
        self.level0_tile_size = int(
            tile_size * self.slide.level_downsamples[target_level]
        )

    def process_all(
        self,
        enhance: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理全部坐标，返回 (tiles, coords_scaled)。

        Returns:
            tiles:  [N, 3, 224, 224] float32, ImageNet 标准化后
            coords: [N, 2] float32, (x_scaled, y_scaled)
        """
        if enhance:
            return self._process_with_enhance()
        else:
            return self._process_baseline()

    def _process_baseline(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Baseline: 从 WSI 读取瓦片 → 直接预处理（无 FFPE 增强）。"""
        all_tiles = []
        all_coords = []

        for x0, y0 in self.coords:
            tile_pil = self.slide.read_region(
                (x0, y0), self.target_level, (self.tile_size, self.tile_size)
            ).convert("RGB")
            tile = TILE_TRANSFORM(tile_pil)
            all_tiles.append(tile)
            all_coords.append(torch.tensor(
                [x0 / self.tile_size, y0 / self.tile_size],
                dtype=torch.float32,
            ))

        return torch.stack(all_tiles), torch.stack(all_coords)

    def _process_with_enhance(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """融合流程: 从 WSI 读取瓦片 → FFPE 批量增强 → 预处理。"""
        all_tiles = []
        all_coords = []

        for i in range(0, len(self.coords), self.enhancer.enhance_batch_size * 4):
            batch_end = min(i + self.enhancer.enhance_batch_size * 4, len(self.coords))
            batch_coords = self.coords[i:batch_end]

            # 1. 从 WSI 读取原始瓦片（PIL）
            raw_pils = []
            for x0, y0 in batch_coords:
                tile_pil = self.slide.read_region(
                    (x0, y0), self.target_level, (self.tile_size, self.tile_size)
                ).convert("RGB")
                raw_pils.append(tile_pil)

            # 2. FFPE 批量增强 → tensor [N, 3, H, W] [0,1]
            enhanced_tensor = self.enhancer.enhance_batch(
                raw_pils,
                batch_size=self.enhancer.enhance_batch_size,
                return_tensors=True,
            )

            # 3. 逐张应用 Tile Encoder 预处理
            for j in range(enhanced_tensor.shape[0]):
                # enhanced_tensor[j]: [3, H, W] float32 [0,1]
                single = enhanced_tensor[j]
                # 先转 PIL 再用 transforms（保证与 BaselineWSITileDataset 一致）
                single_pil = transforms.ToPILImage()(single)
                tile = TILE_TRANSFORM(single_pil)
                all_tiles.append(tile)

                x0, y0 = batch_coords[j]
                all_coords.append(torch.tensor(
                    [x0 / self.tile_size, y0 / self.tile_size],
                    dtype=torch.float32,
                ))

        return torch.stack(all_tiles), torch.stack(all_coords)

    def close(self):
        if hasattr(self, "slide") and self.slide is not None:
            self.slide.close()
            self.slide = None

    def __del__(self):
        self.close()


# ── 命令行自检 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="融合数据集自检")
    parser.add_argument("--slide", required=True, help="WSI .tif 路径")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_tiles", type=int, default=10, help="测试瓦片数")
    parser.add_argument("--enhance", action="store_true", default=True)
    parser.add_argument("--no-enhance", dest="enhance", action="store_false")
    args = parser.parse_args()

    # 简易坐标获取 (仅自检用 — 完整扫描用 coords.py)
    import openslide
    slide = openslide.OpenSlide(args.slide)
    thumb_level = slide.level_count - 1
    thumb = np.array(slide.read_region(
        (0, 0), thumb_level, slide.level_dimensions[thumb_level]
    ).convert("L"))
    tissue_mask = thumb < 210
    ys, xs = np.where(tissue_mask)
    if len(ys) == 0:
        logger.error("未检测到组织区域")
        exit(1)

    # 取前 max_tiles 个坐标
    downsample = slide.level_downsamples[thumb_level]
    tile_size_l0 = int(256 * slide.level_downsamples[0])
    coords = []
    for idx in range(min(args.max_tiles, len(xs))):
        y, x = ys[idx], xs[idx]
        x0 = int(x * downsample) // tile_size_l0 * tile_size_l0
        y0 = int(y * downsample) // tile_size_l0 * tile_size_l0
        if (x0 + tile_size_l0 <= slide.dimensions[0] and
                y0 + tile_size_l0 <= slide.dimensions[1]):
            coords.append((x0, y0))
    slide.close()

    if args.enhance:
        from ffpe_enhancer import FFPEEnhancer
        enhancer = FFPEEnhancer(device=args.device)
    else:
        enhancer = None

    ds = FusionTileDataset(
        slide_path=args.slide,
        coords=coords,
        enhance=args.enhance,
        enhancer=enhancer,
    )
    logger.info("数据集长度: %d", len(ds))
    tile, coord = ds[0]
    logger.info("瓦片形状: %s, 坐标: %s", tile.shape, coord)
    logger.info("数据集自检完成 ✅")
