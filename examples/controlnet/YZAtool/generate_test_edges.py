#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量为测试集图片生成 edge / sketch / mask 的工具脚本。

设计目标：
- 完全复用 YZApatch 中已有的边缘提取、sketch 生成、mask 生成逻辑
- 不重新实现算法，避免行为和训练数据不一致
- 在指定的 input_dir 下递归扫描候选图片，并在原目录生成
  `*_edge`, `*_sketch`, `*_mask` 三种派生图像
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


# ============================================================
# 可选：在这里覆写 sketch / mask 的部分参数
# 默认留空，行为与 YZApatch.config 中的 SKETCH_PARAMS / MASK_PARAMS 完全一致
# 后续如果需要微调，可以直接在下面两个字典里填入键值对
# 例如：SKETCH_OVERRIDE_PARAMS = {"sigma_mean": 15.0}
# ============================================================
SKETCH_OVERRIDE_PARAMS: Dict = {}
MASK_OVERRIDE_PARAMS: Dict = {}


def _import_yzapatch_modules():
    """
    动态导入 YZApatch 相关模块和 sketch_util。

    为了避免路径问题，这里显式把 YZApatch 和 SKETCH_UTIL_DIR
    插入到 sys.path，再按和 custom_dataset.py 一致的方式导入。
    """
    current_dir = Path(__file__).resolve().parent
    yzapatch_dir = current_dir.parent / "YZApatch"

    # 确保 YZApatch 在 sys.path 中
    if str(yzapatch_dir) not in sys.path:
        sys.path.insert(0, str(yzapatch_dir))

    try:
        from config import (  # type: ignore
            SKETCH_UTIL_DIR,
            DEXINED_CHECKPOINT,
            DEXINED_THRESHOLD,
            DEXINED_DEVICE,
            EDGE_CACHE_DIR,
            SKETCH_PARAMS,
            MASK_PARAMS,
            IMAGE_EXTENSIONS,
            RECURSIVE_SCAN,
            EDGE_CACHE_VERSION,
        )
        from edge_cache import get_edge_cache_manager  # type: ignore
        from mask_generator import ComplexMaskGenerator  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"无法从 YZApatch 导入配置/模块，请检查目录结构和 PYTHONPATH：{e}"
        )

    # 导入 sketch_util
    if SKETCH_UTIL_DIR not in sys.path:
        sys.path.insert(0, SKETCH_UTIL_DIR)
    try:
        from dataset.sketch_util import (  # type: ignore
            make_sketch_from_image_or_edge,
            extract_edge,
        )
    except ImportError as e:
        raise ImportError(
            f"无法从 SKETCH_UTIL_DIR 导入 dataset.sketch_util，请检查 config.SKETCH_UTIL_DIR={SKETCH_UTIL_DIR}：{e}"
        )

    return {
        "SKETCH_UTIL_DIR": SKETCH_UTIL_DIR,
        "DEXINED_CHECKPOINT": DEXINED_CHECKPOINT,
        "DEXINED_THRESHOLD": DEXINED_THRESHOLD,
        "DEXINED_DEVICE": DEXINED_DEVICE,
        "EDGE_CACHE_DIR": EDGE_CACHE_DIR,
        "SKETCH_PARAMS": SKETCH_PARAMS,
        "MASK_PARAMS": MASK_PARAMS,
        "IMAGE_EXTENSIONS": IMAGE_EXTENSIONS,
        "RECURSIVE_SCAN": RECURSIVE_SCAN,
        "EDGE_CACHE_VERSION": EDGE_CACHE_VERSION,
        "get_edge_cache_manager": get_edge_cache_manager,
        "ComplexMaskGenerator": ComplexMaskGenerator,
        "make_sketch_from_image_or_edge": make_sketch_from_image_or_edge,
        "extract_edge": extract_edge,
    }


def find_candidate_images(
    root_dir: Path,
    image_extensions: List[str],
    recursive_scan: bool = True,
) -> List[Path]:
    """
    扫描候选图片：
    - 扩展名在 IMAGE_EXTENSIONS 中
    - 文件名（不含扩展名）不以 _edge / _sketch / _mask 结尾
    """
    if not root_dir.exists():
        raise ValueError(f"输入目录不存在: {root_dir}")

    image_extensions = [ext.lower() for ext in image_extensions]
    candidates: List[Path] = []

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
        if stem.endswith("_edge") or stem.endswith("_sketch") or stem.endswith("_mask"):
            # 已经是派生文件，跳过
            continue

        candidates.append(path)

    candidates = sorted(candidates)
    return candidates


def build_extract_fn(
    extract_edge_func,
    dexined_checkpoint: str,
    dexined_threshold: int,
    dexined_device: str,
):
    """
    按照 InpaintingSketchDataset._extract_edge_with_cache 的方式
    构造边缘提取函数。
    """

    def _extract_fn(img_np):
        return extract_edge_func(
            image=img_np,
            method="dexined",
            dexined_checkpoint=dexined_checkpoint,
            dexined_threshold=dexined_threshold,
            dexined_device=dexined_device,
        )

    return _extract_fn


def process_single_image(
    image_path: Path,
    resolution: int,
    enable_edge_cache: bool,
    edge_cache_manager,
    extract_fn,
    sketch_params: Dict,
    mask_generator,
):
    """
    对单张图片执行：加载 -> resize -> edge -> sketch -> mask -> 保存结果。
    逻辑对齐 InpaintingSketchDataset.__getitem__。
    """
    # 1. 加载原图
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] 加载图片失败 {image_path}: {e}")
        return

    # 2. Resize 到目标分辨率
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.LANCZOS)

    # 3. 转为 numpy
    image_np = np.array(image)  # [H, W, 3], RGB, uint8

    # 4. 提取边缘图（使用缓存逻辑）
    if enable_edge_cache and edge_cache_manager is not None:
        edge_image = edge_cache_manager.get_or_compute_edge(
            image_path=str(image_path),
            image_np=image_np,
            enable_cache=True,
            extract_fn=extract_fn,
        )
    else:
        edge_image = extract_fn(image_np)

    # 5. 生成 sketch（与 InpaintingSketchDataset 相同参数）
    seed = np.random.randint(0, 2**31 - 1)
    sp = sketch_params
    sketch_np = mask_generator  # 占位，防止未使用 warning（稍后覆盖）
    del sketch_np  # 实际不会使用该占位

    from dataset.sketch_util import make_sketch_from_image_or_edge  # type: ignore

    sketch_np = make_sketch_from_image_or_edge(
        input_image=edge_image,
        seed=seed,
        is_edge=True,
        enable_edge_extraction=False,
        sigma_mean=sp.get("sigma_mean", 13.0),
        sigma_std=sp.get("sigma_std", 2.6),
        spatial_smooth_sigma=sp.get("spatial_smooth_sigma", 2.0),
        cp_sigma_mean=sp.get("cp_sigma_mean", 2.1),
        cp_sigma_std=sp.get("cp_sigma_std", 0.4),
        cp_spatial_smooth=sp.get("cp_spatial_smooth", 1.5),
    )

    # 6. 生成复杂 mask（基于边缘密度）
    mask_seed = np.random.randint(0, 2**31 - 1)
    mask_np = mask_generator.generate(edge_image, seed=mask_seed)

    # 7. 保存结果到原目录
    out_dir = image_path.parent
    stem = image_path.stem
    suffix = image_path.suffix  # 保持原扩展名

    edge_path = out_dir / f"{stem}_edge{suffix}"
    sketch_path = out_dir / f"{stem}_sketch{suffix}"
    mask_path = out_dir / f"{stem}_mask{suffix}"

    Image.fromarray(edge_image).save(edge_path)
    Image.fromarray(sketch_np).save(sketch_path)
    Image.fromarray(mask_np).save(mask_path)

    print(f"[OK] {image_path} -> {edge_path.name}, {sketch_path.name}, {mask_path.name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="为测试集图片批量生成 edge / sketch / mask（复用 YZApatch 现有数据处理流程）"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="待处理的测试集根目录，将在该目录下递归扫描图片",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="输出图像分辨率（宽高），默认 512，与训练设置一致",
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        help="禁用边缘缓存（默认使用 YZApatch EDGE_CACHE_DIR 进行缓存）",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="最多处理多少张图片（可选，默认处理全部候选图片）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    modules = _import_yzapatch_modules()
    DEXINED_CHECKPOINT = modules["DEXINED_CHECKPOINT"]
    DEXINED_THRESHOLD = modules["DEXINED_THRESHOLD"]
    DEXINED_DEVICE = modules["DEXINED_DEVICE"]
    EDGE_CACHE_DIR = modules["EDGE_CACHE_DIR"]
    SKETCH_PARAMS = modules["SKETCH_PARAMS"]
    MASK_PARAMS = modules["MASK_PARAMS"]
    IMAGE_EXTENSIONS = modules["IMAGE_EXTENSIONS"]
    RECURSIVE_SCAN = modules["RECURSIVE_SCAN"]
    EDGE_CACHE_VERSION = modules["EDGE_CACHE_VERSION"]
    get_edge_cache_manager = modules["get_edge_cache_manager"]
    ComplexMaskGenerator = modules["ComplexMaskGenerator"]
    extract_edge = modules["extract_edge"]

    input_dir = Path(args.input_dir).resolve()
    print(f"[INFO] 输入目录: {input_dir}")

    # 1. 扫描候选图片
    candidates = find_candidate_images(
        input_dir,
        image_extensions=IMAGE_EXTENSIONS,
        recursive_scan=RECURSIVE_SCAN,
    )
    if not candidates:
        print(f"[WARN] 在 {input_dir} 中未找到任何候选图片")
        return

    if args.max_images is not None and args.max_images > 0:
        candidates = candidates[: args.max_images]

    print(f"[INFO] 共发现 {len(candidates)} 张候选图片（已自动跳过 *_edge/_sketch/_mask）")

    # 2. 构建 edge 提取函数 & 缓存管理器
    extract_fn = build_extract_fn(
        extract_edge_func=extract_edge,
        dexined_checkpoint=DEXINED_CHECKPOINT,
        dexined_threshold=DEXINED_THRESHOLD,
        dexined_device=DEXINED_DEVICE,
    )

    enable_edge_cache = not args.disable_cache
    edge_cache_manager = None
    if enable_edge_cache:
        dexined_params = {
            "threshold": DEXINED_THRESHOLD,
            "version": EDGE_CACHE_VERSION,
        }
        edge_cache_manager = get_edge_cache_manager(EDGE_CACHE_DIR, dexined_params)
        print(f"[INFO] 边缘缓存已启用: {EDGE_CACHE_DIR}")
    else:
        print("[INFO] 边缘缓存已禁用，将实时提取边缘")

    # 3. 合并 sketch / mask 参数（允许顶部字典覆盖）
    sketch_params = dict(SKETCH_PARAMS)
    sketch_params.update(SKETCH_OVERRIDE_PARAMS)

    mask_params = dict(MASK_PARAMS)
    mask_params.update(MASK_OVERRIDE_PARAMS)

    mask_generator = ComplexMaskGenerator(mask_params)

    # 4. 逐张处理
    for idx, img_path in enumerate(candidates):
        print(f"\n[PROC] ({idx + 1}/{len(candidates)}) 处理: {img_path}")
        process_single_image(
            image_path=img_path,
            resolution=args.resolution,
            enable_edge_cache=enable_edge_cache,
            edge_cache_manager=edge_cache_manager,
            extract_fn=extract_fn,
            sketch_params=sketch_params,
            mask_generator=mask_generator,
        )

    print("\n[INFO] 全部处理完成。")


if __name__ == "__main__":
    main()

