#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集加载器模块。

提供不同数据集类型的加载逻辑，用于批量推理脚本。
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """单个推理样本的信息。"""

    orig_path: Path
    sketch_path: Path
    mask_path: Path
    rel_path: Path  # 相对于 input_dir 的相对路径（用于镜像输出结构）


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


class DatasetLoader:
    """数据集加载器基类。"""

    def iter_candidate_images(
        self,
        root_dir: Path,
        image_extensions: Sequence[str],
        recursive_scan: bool,
    ) -> Iterable[Path]:
        """
        扫描候选图片文件。

        参数:
            root_dir: 根目录
            image_extensions: 图片扩展名列表
            recursive_scan: 是否递归扫描

        返回:
            候选图片路径的迭代器
        """
        raise NotImplementedError

    def find_matching_derivatives(
        self,
        img_path: Path,
        image_extensions: Sequence[str],
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        查找匹配的派生文件（如 sketch 和 mask）。

        参数:
            img_path: 原图路径
            image_extensions: 图片扩展名列表

        返回:
            (sketch_path, mask_path) 元组
        """
        raise NotImplementedError

    def build_samples(
        self,
        input_dir: Path,
        strict_mode: bool = True,
    ) -> List[Sample]:
        """
        构建样本列表。

        参数:
            input_dir: 输入目录
            strict_mode: 严格模式

        返回:
            样本列表
        """
        raise NotImplementedError


class ArtbenchDatasetLoader(DatasetLoader):
    """Artbench 数据集加载器。"""

    def iter_candidate_images(
        self,
        root_dir: Path,
        image_extensions: Sequence[str],
        recursive_scan: bool,
    ) -> Iterable[Path]:
        """
        参考 YZAtool.generate_test_edges.find_candidate_images：
        - 仅返回扩展名在 image_extensions 中的文件；
        - 跳过 *_sketch / *_mask / *_edge 文件。
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
            if stem.endswith("_sketch") or stem.endswith("_mask") or stem.endswith("_edge"):
                # 已经是派生文件，跳过
                continue

            yield path

    def find_matching_derivatives(
        self,
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
        self,
        input_dir: Path,
        strict_mode: bool = True,
    ) -> List[Sample]:
        """
        扫描 input_dir，构建所有样本列表。

        规则:
        - 递归策略与图片扩展名来自 YZApatch.config（或内置默认）；
        - 对于每个原图 xxx.ext，必须能在同目录找到 xxx_sketch.* 和 xxx_mask.*；
          - 若 strict_mode=True：缺失即报错退出；
          - 若 strict_mode=False：打印 warning 并跳过该样本。
        """
        image_extensions, recursive_scan = _load_yzapatch_image_config()

        samples: List[Sample] = []
        missing_count = 0

        for img_path in self.iter_candidate_images(input_dir, image_extensions, recursive_scan):
            sketch_path, mask_path = self.find_matching_derivatives(img_path, image_extensions)

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


class Mural1DatasetLoader(DatasetLoader):
    """Mural1 数据集加载器。
    
    特点：
    - 原图和 mask 文件名相同，但 mask 在单独的 mask_dir 目录中
    - sketch 仍然使用 _sketch 后缀，与原图在同一目录
    """

    def __init__(self, mask_dir: Optional[Path] = None):
        """
        初始化 Mural1 数据集加载器。

        参数:
            mask_dir: mask 文件夹路径。如果为 None，需要在 build_samples 时提供。
        """
        self.mask_dir: Optional[Path] = mask_dir

    def iter_candidate_images(
        self,
        root_dir: Path,
        image_extensions: Sequence[str],
        recursive_scan: bool,
    ) -> Iterable[Path]:
        """
        扫描候选图片文件。
        - 仅返回扩展名在 image_extensions 中的文件；
        - 跳过 *_sketch / *_edge 文件（但不跳过 _mask，因为 mask 在单独目录）。
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
            if stem.endswith("_sketch") or stem.endswith("_edge"):
                # 跳过派生文件
                continue

            yield path

    def find_matching_derivatives(
        self,
        img_path: Path,
        image_extensions: Sequence[str],
        mask_dir: Optional[Path] = None,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        查找匹配的派生文件（sketch 和 mask）。

        参数:
            img_path: 原图路径
            image_extensions: 图片扩展名列表
            mask_dir: mask 文件夹路径（如果未在 __init__ 中提供）

        返回:
            (sketch_path, mask_path) 元组
        """
        if mask_dir is None:
            mask_dir = self.mask_dir
        if mask_dir is None:
            raise ValueError("Mural1DatasetLoader 需要 mask_dir 参数")

        stem = img_path.stem
        suffix = img_path.suffix.lower()
        parent = img_path.parent

        # 1. 查找 sketch（与原图同目录，使用 _sketch 后缀）
        sketch_candidates: List[Path] = [parent / f"{stem}_sketch{suffix}"]
        for ext in image_extensions:
            ext = ext.lower()
            if ext == suffix:
                continue
            sketch_candidates.append(parent / f"{stem}_sketch{ext}")

        sketch_path: Optional[Path] = None
        for p in sketch_candidates:
            if p.is_file():
                sketch_path = p
                break

        # 2. 查找 mask（在 mask_dir 中，文件名与原图相同）
        mask_dir_path = Path(mask_dir)
        mask_candidates: List[Path] = [mask_dir_path / f"{stem}{suffix}"]
        for ext in image_extensions:
            ext = ext.lower()
            if ext == suffix:
                continue
            mask_candidates.append(mask_dir_path / f"{stem}{ext}")

        mask_path: Optional[Path] = None
        for p in mask_candidates:
            if p.is_file():
                mask_path = p
                break

        return sketch_path, mask_path

    def build_samples(
        self,
        input_dir: Path,
        strict_mode: bool = True,
        mask_dir: Optional[Path] = None,
    ) -> List[Sample]:
        """
        扫描 input_dir，构建所有样本列表。

        规则:
        - 递归策略与图片扩展名来自 YZApatch.config（或内置默认）；
        - 对于每个原图 xxx.ext：
          - sketch: 在同目录查找 xxx_sketch.*
          - mask: 在 mask_dir 目录中查找 xxx.*（文件名相同）
        - 若 strict_mode=True：缺失即报错退出；
        - 若 strict_mode=False：打印 warning 并跳过该样本。

        参数:
            input_dir: 输入目录（原图所在目录）
            strict_mode: 严格模式
            mask_dir: mask 文件夹路径（如果未在 __init__ 中提供）
        """
        if mask_dir is None:
            mask_dir = self.mask_dir
        if mask_dir is None:
            raise ValueError("Mural1DatasetLoader.build_samples 需要 mask_dir 参数")

        mask_dir_path = Path(mask_dir).resolve()
        if not mask_dir_path.exists():
            raise ValueError(f"Mask 目录不存在: {mask_dir_path}")

        image_extensions, recursive_scan = _load_yzapatch_image_config()

        samples: List[Sample] = []
        missing_count = 0

        for img_path in self.iter_candidate_images(input_dir, image_extensions, recursive_scan):
            sketch_path, mask_path = self.find_matching_derivatives(
                img_path, image_extensions, mask_dir=mask_dir_path
            )

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


def get_dataset_loader(dataset_name: str, mask_dir: Optional[Path] = None) -> DatasetLoader:
    """
    根据数据集名称返回对应的加载器。

    参数:
        dataset_name: 数据集名称（'artbench' 或 'mural1'）
        mask_dir: mask 文件夹路径（仅 mural1 需要）

    返回:
        对应的数据集加载器实例
    """
    if dataset_name == "artbench":
        return ArtbenchDatasetLoader()
    elif dataset_name == "mural1":
        return Mural1DatasetLoader(mask_dir=mask_dir)
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_name}，仅支持 'artbench' 或 'mural1'")
