#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 SDXL inpainting + ControlNet 的批量推理脚本（YZAtool）。

功能概述：
- 输入一个测试集根目录 --input_dir，递归扫描所有原始图片；
- 对每张原图 xxx.ext，自动在同目录查找 xxx_sketch.* 与 xxx_mask.* 作为条件输入；
- 使用训练时相同风格的 SDXL inpainting + ControlNet 模型进行批量推理；
- 在 --output_dir 中镜像 input_dir 的目录结构，保存 xxx_inpaint.ext 结果图片；
- batch size 在文件顶部通过 DEFAULT_BATCH_SIZE 定义，也可通过命令行 --batch_size 覆盖。

注意：
- 假定使用的是基于 inpainting SDXL base（如 diffusers/stable-diffusion-xl-1.0-inpainting-0.1）
  训练出来的 ControlNet，mask 语义为：白色=需要修补的区域，黑色=保留原图。
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
    UniPCMultistepScheduler,
)


logger = logging.getLogger(__name__)


# ============================================================
# 批大小默认值：可以在这里直接修改，或用 --batch_size 覆盖
# ============================================================
DEFAULT_BATCH_SIZE: int = 4


# ============================================================
# 与 YZApatch.config 对齐的图片扩展名 / 递归策略
# 尽量从 YZApatch.config 读取；若导入失败，则退回到内置默认值。
# ============================================================

def _load_yzapatch_image_config():
    """
    尝试从 YZApatch.config 读取 IMAGE_EXTENSIONS 和 RECURSIVE_SCAN。
    如果导入失败，则返回一组安全的默认值。
    """
    current_dir = Path(__file__).resolve().parent
    yzapatch_dir = current_dir.parent / "YZApatch"

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    recursive_scan = True

    if yzapatch_dir.exists():
        if str(yzapatch_dir) not in sys.path:
            sys.path.insert(0, str(yzapatch_dir))
        try:
            from config import IMAGE_EXTENSIONS, RECURSIVE_SCAN  # type: ignore

            image_extensions = [ext.lower() for ext in IMAGE_EXTENSIONS]
            recursive_scan = bool(RECURSIVE_SCAN)
            logger.info(
                "Loaded IMAGE_EXTENSIONS and RECURSIVE_SCAN from YZApatch.config: %s, recursive=%s",
                image_extensions,
                recursive_scan,
            )
        except Exception as e:  # pragma: no cover - 容错路径
            logger.warning(
                "Failed to import IMAGE_EXTENSIONS / RECURSIVE_SCAN from YZApatch.config: %s. "
                "Falling back to built-in defaults.",
                e,
            )
    else:
        logger.info("YZApatch directory not found, using built-in IMAGE_EXTENSIONS defaults")

    return image_extensions, recursive_scan


@dataclass
class Sample:
    """单个推理样本的信息。"""

    orig_path: Path
    sketch_path: Path
    mask_path: Path
    rel_path: Path  # 相对于 input_dir 的相对路径（用于镜像输出结构）


def _iter_candidate_images(
    root_dir: Path,
    image_extensions: Sequence[str],
    recursive_scan: bool,
) -> Iterable[Path]:
    """
    参考 YZAtool.generate_test_edges.find_candidate_images：
    - 仅返回扩展名在 image_extensions 中的文件；
    - 跳过 *_sketch / *_mask 文件；
    """
    if not root_dir.exists():
        raise ValueError(f"输入目录不存在: {root_dir}")

    image_extensions = [ext.lower() for ext in image_extensions]

    if recursive_scan:
        iterator = root_dir.rglob("*")
    else:
        iterator = root_dir.glob("*")

    for path in iterator:
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in image_extensions:
            continue

        stem = path.stem
        if stem.endswith("_sketch") or stem.endswith("_mask"):
            # 已经是派生文件，跳过
            continue

        yield path


def _find_matching_derivatives(
    img_path: Path,
    image_extensions: Sequence[str],
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    给定原图路径 xxx.ext，在同目录寻找：
    - xxx_sketch.*（优先使用原扩展名；如果不存在，则在 image_extensions 中依次尝试）
    - xxx_mask.*   （同上）
    若找不到则返回 (None, None) 或者对应位置为 None。
    """
    stem = img_path.stem
    suffix = img_path.suffix.lower()
    parent = img_path.parent

    # 优先尝试与原图相同扩展名
    sketch_candidates: List[Path] = [parent / f"{stem}_sketch{suffix}"]
    mask_candidates: List[Path] = [parent / f"{stem}_mask{suffix}"]

    # 其他扩展名补充尝试
    for ext in image_extensions:
        ext = ext.lower()
        if ext == suffix:
            continue
        sketch_candidates.append(parent / f"{stem}_sketch{ext}")
        mask_candidates.append(parent / f"{stem}_mask{ext}")

    sketch_path: Optional[Path] = None
    mask_path: Optional[Path] = None

    for p in sketch_candidates:
        if p.is_file():
            sketch_path = p
            break

    for p in mask_candidates:
        if p.is_file():
            mask_path = p
            break

    return sketch_path, mask_path


def build_samples(
    input_dir: Path,
    strict_mode: bool = True,
) -> List[Sample]:
    """
    扫描 input_dir，构建所有样本列表。

    规则：
    - 递归策略与图片扩展名来自 YZApatch.config（或内置默认）；
    - 对于每个原图 xxx.ext，必须能在同目录找到 xxx_sketch.* 和 xxx_mask.*；
      - 若 strict_mode=True：缺失即报错退出；
      - 若 strict_mode=False：打印 warning 并跳过该样本。
    """
    image_extensions, recursive_scan = _load_yzapatch_image_config()

    samples: List[Sample] = []
    missing_count = 0

    for img_path in _iter_candidate_images(input_dir, image_extensions, recursive_scan):
        sketch_path, mask_path = _find_matching_derivatives(img_path, image_extensions)

        if sketch_path is None or mask_path is None:
            missing_parts = []
            if sketch_path is None:
                missing_parts.append("sketch")
            if mask_path is None:
                missing_parts.append("mask")
            missing_str = "/".join(missing_parts)
            msg = f"[MISSING] {img_path} 缺少派生文件: {missing_str}"

            if strict_mode:
                raise FileNotFoundError(msg)
            else:
                logger.warning(msg)
                missing_count += 1
                continue

        rel_path = img_path.relative_to(input_dir)
        samples.append(
            Sample(
                orig_path=img_path,
                sketch_path=sketch_path,
                mask_path=mask_path,
                rel_path=rel_path,
            )
        )

    if not samples:
        logger.warning("在 %s 中未找到任何有效样本（strict_mode=%s）", input_dir, strict_mode)

    if missing_count > 0 and not strict_mode:
        logger.warning("共有 %d 个样本因缺少 sketch/mask 被跳过。", missing_count)

    return samples


# ============================================================
# 模型 / Pipeline 构建
# ============================================================

def _resolve_device(device: str) -> torch.device:
    device = device.lower()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 device=cuda，但当前环境没有可用的 CUDA 设备。")
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"不支持的 device: {device}，仅支持 auto/cuda/cpu。")


def build_pipeline(args: argparse.Namespace):
    """
    构建 SDXL inpainting + ControlNet 推理 pipeline。

    - base model 通常是 diffusers/stable-diffusion-xl-1.0-inpainting-0.1；
    - controlnet_model_name_or_path 为你训练得到的 ControlNet 权重；
    - 若指定 pretrained_vae_model_name_or_path，则与训练保持一致。
    """
    device = _resolve_device(args.device)

    # dtype 策略：CUDA 上使用 fp16，CPU 上使用 fp32
    if device.type == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    logger.info("Loading ControlNet from %s", args.controlnet_model_name_or_path)
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model_name_or_path,
        torch_dtype=torch_dtype,
    )

    vae = None
    if args.pretrained_vae_model_name_or_path is not None:
        logger.info("Loading VAE from %s", args.pretrained_vae_model_name_or_path)
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_vae_model_name_or_path,
            torch_dtype=torch_dtype,
        )

    logger.info("Loading base SDXL inpainting model from %s", args.pretrained_model_name_or_path)
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch_dtype,
    )

    # 调整 scheduler，与 README_sdxl 中一致
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if device.type == "cuda":
        pipe.to(device)
        # 为了节省显存，也可以启用以下优化（可选）：
        if args.enable_xformers_memory_efficient_attention:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:  # pragma: no cover - 环境相关
                logger.warning("启用 xformers 失败: %s", e)
    else:
        # CPU 情况下，尽量使用 cpu_offload 减少内存压力
        pipe.enable_model_cpu_offload()

    return pipe, device, torch_dtype


# ============================================================
# 单样本推理逻辑
# ============================================================

def _resize_for_inference(
    image: Image.Image,
    resolution: int,
    is_mask: bool = False,
) -> Image.Image:
    """
    按训练时的风格，将图像 resize 到 resolution x resolution。
    - 对于 mask 使用 NEAREST 避免插值导致的灰度混合；
    - 对于普通图像使用 LANCZOS。
    """
    if is_mask:
        return image.resize((resolution, resolution), Image.NEAREST)
    return image.resize((resolution, resolution), Image.LANCZOS)


def _load_pil_image(path: Path, mode: str = "RGB") -> Image.Image:
    try:
        img = Image.open(path)
        if mode is not None:
            img = img.convert(mode)
        return img
    except Exception as e:
        raise RuntimeError(f"加载图片失败: {path}，错误: {e}")


def run_inference_for_sample(
    sample: Sample,
    pipe: StableDiffusionXLControlNetInpaintPipeline,
    args: argparse.Namespace,
    device: torch.device,
    generator: Optional[torch.Generator],
    output_root: Path,
) -> Optional[Path]:
    """
    对单个样本执行推理：
    - 加载原图 / sketch / mask；
    - resize 到指定分辨率；
    - 调用 inpaint + ControlNet pipeline；
    - 将结果保存到 output_root 镜像目录下的 xxx_inpaint.ext。
    """
    # 1. 加载与预处理图像
    orig = _load_pil_image(sample.orig_path, mode="RGB")
    cond = _load_pil_image(sample.sketch_path, mode="RGB")
    mask = _load_pil_image(sample.mask_path, mode="L")  # 单通道，白=修补区域

    orig = _resize_for_inference(orig, args.resolution, is_mask=False)
    cond = _resize_for_inference(cond, args.resolution, is_mask=False)
    mask = _resize_for_inference(mask, args.resolution, is_mask=True)

    # 2. 构造输出路径（镜像 input_dir 结构）
    out_rel_dir = sample.rel_path.parent  # 目录部分
    out_dir = output_root / out_rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = sample.orig_path.stem
    suffix = sample.orig_path.suffix
    out_path = out_dir / f"{stem}_inpaint{suffix}"

    # 3. 调用 pipeline
    prompt = args.prompt or ""
    negative_prompt = args.negative_prompt

    # 注意：StableDiffusionXLControlNetInpaintPipeline 接口
    # image: 原图；mask_image: 白=要修补；controlnet_conditioning_image: 条件图（这里用 sketch）
    with torch.autocast(device.type) if device.type == "cuda" else torch.no_grad():  # type: ignore[arg-type]
        result = pipe(
            prompt=prompt,
            image=orig,
            mask_image=mask,
            controlnet_conditioning_image=cond,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
        )

    image = result.images[0]
    image.save(out_path)
    return out_path


# ============================================================
# CLI / 主逻辑
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 SDXL inpainting + ControlNet 的批量推理脚本（YZAtool）。",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="测试集根目录，将在该目录下递归扫描原始图片，并匹配同目录下 *_sketch / *_mask。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="推理结果输出根目录，将镜像 input_dir 的目录结构。",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help=(
            "SDXL inpainting 基模型名称或路径，建议与训练时一致，如 "
            '"diffusers/stable-diffusion-xl-1.0-inpainting-0.1"。'
        ),
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        required=True,
        help="ControlNet 模型权重路径或 Hub 仓库名（与你训练时保存的位置一致）。",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="可选：若训练时使用了自定义 VAE（如 sdxl-vae-fp16-fix），此处需保持一致。",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="推理时的分辨率，默认 512，需与训练时 --resolution 保持一致（并且被 8 整除）。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"一次处理的样本数量（逻辑批大小），默认 {DEFAULT_BATCH_SIZE}。",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="推理步数（采样步数），默认 20。",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale，默认 5.0。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子；若不指定则使用随机生成器。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="推理使用的设备：auto / cuda / cpu，默认 auto 优先使用 CUDA。",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="统一使用的文本提示词；若为空字符串，则相当于无文本条件（与部分训练设定一致）。",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="统一使用的负向提示词（可选）。",
    )
    parser.add_argument(
        "--strict_mode",
        action="store_true",
        help=(
            "严格模式：若某张原图缺少对应的 *_sketch 或 *_mask，则立刻报错退出。"
            " 若未指定，则默认严格模式开启。"
        ),
    )
    parser.add_argument(
        "--non_strict",
        action="store_true",
        help="关闭严格模式：缺少 *_sketch / *_mask 的样本会被跳过，仅打印 warning。",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="在 CUDA 环境下尝试启用 xFormers 省显存注意力（若已正确安装）。",
    )

    args = parser.parse_args()

    # 解析 strict 模式：默认严格，除非显式传入 --non_strict
    if args.non_strict:
        args.strict_mode = False
    elif not getattr(args, "strict_mode", False):
        # 未显式指定 --strict_mode 时，默认 True
        args.strict_mode = True

    if args.resolution % 8 != 0:
        raise ValueError("resolution 必须能被 8 整除，以与 SDXL 的 latent 分辨率对齐。")

    return args


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("输入目录: %s", input_dir)
    logger.info("输出目录: %s", output_root)

    # 1. 构建样本列表（递归扫描 + *_sketch / *_mask 匹配）
    samples = build_samples(input_dir, strict_mode=args.strict_mode)
    if not samples:
        logger.warning("没有可用样本，程序结束。")
        return

    logger.info("共发现 %d 个有效样本。", len(samples))

    # 2. 构建 pipeline
    pipe, device, _ = build_pipeline(args)

    # 3. 随机种子 / generator
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None

    # 4. 批量推理
    total = len(samples)
    batch_size = max(1, int(args.batch_size))

    logger.info("开始批量推理：batch_size=%d, num_inference_steps=%d", batch_size, args.num_inference_steps)

    idx = 0
    with tqdm(total=total, desc="Inference", unit="img") as pbar:
        while idx < total:
            batch = samples[idx : idx + batch_size]
            for sample in batch:
                try:
                    _ = run_inference_for_sample(
                        sample=sample,
                        pipe=pipe,
                        args=args,
                        device=device,
                        generator=generator,
                        output_root=output_root,
                    )
                except Exception as e:
                    logger.error("样本 %s 推理失败：%s", sample.orig_path, e)
                    if args.strict_mode:
                        raise
                finally:
                    pbar.update(1)
            idx += batch_size

    logger.info("全部推理完成。")


if __name__ == "__main__":
    main()

