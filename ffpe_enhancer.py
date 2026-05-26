#!/usr/bin/env python3
"""
FFPE 增强包装器 — 加载 Diffusion-FFPE 模型，提供逐瓦片增强接口。

输入: PIL Image (RGB, 256×256) 或任意尺寸
输出: PIL Image (RGB, 与输入同尺寸)，FFPE 风格迁移后

核心流程:
  1. PIL → ToTensor + Normalize(0.5) → [-1, 1]
  2. Diffusion_FFPE.forward(x, direction='a2b', text_emb=...)
  3. Denormalize (*0.5 + 0.5) → [0, 1]
  4. ToPILImage

用法:
  from ffpe_enhancer import FFPEEnhancer
  enhancer = FFPEEnhancer(pretrained_path=".../model.pkl", device="cuda:0")
  enhanced_tile = enhancer.enhance(pil_image)
  # 批量增强
  enhanced_tiles = enhancer.enhance_batch(list_of_pil_images, batch_size=8)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ffpe_enhancer")

# ── 路径常量 ────────────────────────────────────────────────────────
DEFAULT_FFPE_REPO = Path("/public/home/wang/liujx/Diffusion-FFPE-main")
DEFAULT_PRETRAINED = DEFAULT_FFPE_REPO / "checkpoints" / "model.pkl"
DEFAULT_SD_TURBO = "stabilityai/sd-turbo"
SD_TURBO_LOCAL = Path("/public/home/wang/share_group_folder_wang/sd-turbo")


class FFPEEnhancer:
    """FFPE 增强器：加载模型，提供单张/批量增强。"""

    def __init__(
        self,
        pretrained_path: str | Path = str(DEFAULT_PRETRAINED),
        model_path: str = DEFAULT_SD_TURBO,
        device: str = "cuda",
        image_prep: str = "no_resize",
        direction: str = "a2b",
        prompt: str = "paraffin section",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            pretrained_path: Diffusion-FFPE 权重 (model.pkl)
            model_path: SD-Turbo 基座。本地离线路径优先：
                        若 SD_TURBO_LOCAL 存在则自动切换。
            device: "cuda" / "cuda:0"
            image_prep: "no_resize" | "resize_256" | "resize_512"
            direction: "a2b" (FF→FFPE)
            prompt: 文本条件
            dtype: 推理精度 (推荐 float16)
        """
        self.device = torch.device(device)
        self.image_prep = image_prep
        self.direction = direction
        self.dtype = dtype

        # 自动切换本地 SD-Turbo（离线可用）
        self.model_path = self._resolve_model_path(model_path)

        # 确保 Diffusion-FFPE 在 Python path
        if str(DEFAULT_FFPE_REPO) not in sys.path:
            sys.path.insert(0, str(DEFAULT_FFPE_REPO))

        # 加载模型
        logger.info("加载 Diffusion-FFPE 模型: %s", pretrained_path)
        self.model = self._load_model(pretrained_path)
        self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

        # 预计算 text embedding
        logger.info("初始化文本编码器 (prompt=%r) ...", prompt)
        from diffusion_ffpe.model import initialize_text_encoder

        self.tokenizer, self.text_encoder = initialize_text_encoder(self.model_path)
        self.text_encoder = self.text_encoder.to(self.device, dtype=self.dtype)
        self.text_encoder.eval()
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            self.text_emb = self.text_encoder(
                text_inputs.input_ids.to(self.device)
            )[0]
        logger.info("FFPE 增强器就绪 (device=%s, dtype=%s)", self.device, self.dtype)

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        """若本地 sd-turbo 存在则离线加载。"""
        if model_path == "stabilityai/sd-turbo" and (SD_TURBO_LOCAL / "scheduler" / "scheduler_config.json").is_file():
            import os
            os.environ["HF_HUB_OFFLINE"] = "1"
            logger.info("使用本地 SD-Turbo: %s", SD_TURBO_LOCAL)
            return str(SD_TURBO_LOCAL)
        return model_path

    def _load_model(self, pretrained_path: str | Path) -> nn.Module:
        from diffusion_ffpe.model import Diffusion_FFPE

        return Diffusion_FFPE(
            pretrained_path=str(pretrained_path),
            model_path=self.model_path,
            enable_xformers_memory_efficient_attention=True,
            multi_view=True,
        )

    @torch.no_grad()
    def enhance(self, image: Image.Image) -> Image.Image:
        """增强单张 PIL Image。

        Args:
            image: PIL RGB 图像，任意尺寸。

        Returns:
            PIL RGB 图像，与输入同尺寸。
        """
        orig_size = image.size  # (W, H)
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        x = to_tensor(image).unsqueeze(0).to(self.device, dtype=self.dtype)
        batch_size = x.shape[0]

        text_emb = self.text_emb.expand(batch_size, -1, -1)
        out = self.model(x, direction=self.direction, text_emb=text_emb)

        # Denormalize: [-1,1] → [0,1]
        out = out * 0.5 + 0.5
        out = out.clamp(0, 1)

        # → PIL
        out_img = transforms.ToPILImage()(out[0].float().cpu())
        if out_img.size != orig_size:
            out_img = out_img.resize(orig_size, Image.LANCZOS)
        return out_img

    @torch.no_grad()
    def enhance_batch(
        self,
        images: List[Image.Image],
        batch_size: int = 8,
        return_tensors: bool = False,
    ) -> List[Image.Image] | torch.Tensor:
        """批量增强多张 PIL Image。

        Args:
            images: PIL RGB 图像列表。
            batch_size: GPU 批大小。
            return_tensors: True 返回 torch.Tensor [N, 3, H, W] (float32, [0,1])；
                            False 返回 List[PIL.Image]。

        Returns:
            增强后的图像列表或 tensor。
        """
        if not images:
            return [] if not return_tensors else torch.empty(0)

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        all_out: List[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([
                to_tensor(img) for img in batch
            ]).to(self.device, dtype=self.dtype)
            n = tensors.shape[0]

            text_emb = self.text_emb.expand(n, -1, -1)
            out = self.model(tensors, direction=self.direction, text_emb=text_emb)
            out = out * 0.5 + 0.5  # [-1,1] → [0,1]
            out = out.clamp(0, 1)
            all_out.append(out.float().cpu())

        big_tensor = torch.cat(all_out, dim=0)  # [N, 3, H, W]

        if return_tensors:
            return big_tensor

        return [transforms.ToPILImage()(big_tensor[j]) for j in range(big_tensor.shape[0])]

    def __repr__(self) -> str:
        return (
            f"FFPEEnhancer(device={self.device}, dtype={self.dtype}, "
            f"model_path={self.model_path})"
        )


# ── 命令行自检 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FFPE 增强器自检")
    parser.add_argument(
        "--pretrained",
        default=str(DEFAULT_PRETRAINED),
        help="model.pkl 路径",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="测试图片路径（可选）",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="设备",
    )
    args = parser.parse_args()

    enhancer = FFPEEnhancer(
        pretrained_path=args.pretrained,
        device=args.device,
    )

    if args.image:
        test_img = Image.open(args.image).convert("RGB")
        logger.info("原始尺寸: %s", test_img.size)
        result = enhancer.enhance(test_img)
        out_path = Path(args.image).stem + "_ffpe.png"
        result.save(out_path)
        logger.info("增强结果保存: %s", out_path)
    else:
        # 随机张量冒烟
        dummy = Image.fromarray(
            (torch.rand(3, 256, 256) * 255).byte().permute(1, 2, 0).numpy()
        )
        result = enhancer.enhance(dummy)
        logger.info("随机张量测试通过，输出尺寸: %s", result.size)

    logger.info("FFPE 增强器自检完成 ✅")
