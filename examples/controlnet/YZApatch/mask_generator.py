# -*- coding: utf-8 -*-
"""
复杂Mask生成器
==============

生成5-10个不规则矩形块 + 随机笔刷描边的复杂mask
基于边缘密度的概率采样，优先在边缘密集区域放置mask
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional


class ComplexMaskGenerator:
    """
    复杂mask生成器
    
    特性：
    - 5-10个随机大小的矩形块（支持旋转）
    - 基于边缘密度的概率采样
    - 随机笔刷描边增加不规则性
    - 形态学操作增加复杂度
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化mask生成器
        
        Args:
            params: mask生成参数字典，如果为None则使用默认值
        """
        # 默认参数
        self.default_params = {
            "num_blocks_range": (5, 10),
            "area_ratio_range": (0.2, 0.5),
            "min_block_size": 32,
            "max_block_ratio": 0.3,
            "rotation_prob": 0.7,
            "rotation_range": (-30, 30),
            "brush_stroke_prob": 0.8,
            "brush_width_range": (3, 10),
            "brush_segments_range": (3, 8),
            "edge_density_weight": 2.0,
            "edge_density_threshold": 0.3,
            "morphology_prob": 0.5,
            "morphology_kernel_size": (3, 7),
        }
        
        # 合并用户参数
        self.params = self.default_params.copy()
        if params is not None:
            self.params.update(params)
        # debug模式下输出详细统计（由debug_edge触发）
        self.debug_output_dir = self.params.pop("_debug_output_dir", None)
        self._debug_counter = 0
    
    def compute_edge_density(self, edge_image: np.ndarray, kernel_size: int = 31) -> np.ndarray:
        """
        计算局部边缘密度图
        
        Args:
            edge_image: 边缘图 [H,W,3]，白底黑线
            kernel_size: 密度计算窗口大小
        
        Returns:
            edge_density: 边缘密度图 [H,W]，范围0-1
        """
        # 转为灰度图
        if edge_image.ndim == 3:
            edge_gray = cv2.cvtColor(edge_image, cv2.COLOR_RGB2GRAY)
        else:
            edge_gray = edge_image
        
        # 反转：黑线变白线
        edge_gray = 255 - edge_gray
        
        # 局部窗口求平均（卷积）
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        edge_density = cv2.filter2D(edge_gray.astype(np.float32) / 255.0, -1, kernel)
        # 裁剪到[0,1]，避免浮点误差产生负值导致sqrt NaN
        edge_density = np.clip(edge_density, 0.0, 1.0)
        return edge_density
    
    def sample_center_from_density(self, edge_density: np.ndarray, 
                                   weight: float, threshold: float,
                                   debug_log_prefix: Optional[str] = None) -> Tuple[int, int]:
        """
        根据边缘密度采样mask中心点
        
        Args:
            edge_density: 边缘密度图 [H,W]
            weight: 密度权重（>1则优先选择密集区域）
            threshold: 密度阈值
            debug_log_prefix: debug模式下日志前缀（用于输出详细统计）
        
        Returns:
            (cy, cx): 中心点坐标
        """
        h, w = edge_density.shape
        do_debug = self.debug_output_dir and debug_log_prefix is not None
        
        # 裁剪到[0,1]，避免负值导致sqrt NaN
        edge_density = np.clip(edge_density, 0.0, 1.0)
        
        # weight必须为正，否则1/weight会异常
        weight = max(weight, 1e-6) if weight <= 0 else weight
        
        # 构建概率图
        prob_map = np.where(
            edge_density > threshold,
            edge_density ** weight,
            edge_density ** (1.0 / weight)
        )
        prob_map = np.nan_to_num(prob_map, nan=0.0, posinf=0.0, neginf=0.0)
        
        prob_sum = float(prob_map.sum())
        if prob_sum < 1e-10 or not np.isfinite(prob_sum):
            # 全零时退化为均匀分布
            flat_prob = np.ones(h * w, dtype=np.float64) / (h * w)
            if do_debug:
                self._write_debug_log(debug_log_prefix, edge_density, prob_map, prob_sum, True)
        else:
            flat_prob = (prob_map.flatten()).astype(np.float64) / prob_sum
            if do_debug:
                self._write_debug_log(debug_log_prefix, edge_density, prob_map, prob_sum, False)
        
        # 再次归一化确保sum=1，避免浮点误差导致np.random.choice报错
        flat_prob = flat_prob / flat_prob.sum()
        idx = np.random.choice(len(flat_prob), p=flat_prob)
        cy, cx = divmod(idx, w)
        return int(cy), int(cx)
    
    def _write_debug_log(self, prefix: str, edge_density: np.ndarray, 
                         prob_map: np.ndarray, prob_sum: float, used_uniform: bool) -> None:
        """输出详细debug统计到文件"""
        out_dir = Path(self.debug_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            f"[MaskDebug] {prefix}",
            f"  edge_density: min={edge_density.min():.6f} max={edge_density.max():.6f} "
            f"mean={edge_density.mean():.6f} neg_count={(edge_density < 0).sum()} nan_count={np.isnan(edge_density).sum()}",
            f"  prob_map: min={prob_map.min():.6f} max={prob_map.max():.6f} sum={prob_sum:.6f} "
            f"nan_count={np.isnan(prob_map).sum()}",
            f"  used_uniform_fallback={used_uniform}",
        ]
        msg = "\n".join(lines)
        (out_dir / f"{prefix}.txt").write_text(msg, encoding="utf-8")
        print(msg)
    
    def draw_rotated_rectangle(self, mask: np.ndarray, center: Tuple[int, int], 
                               width: int, height: int, angle: float) -> None:
        """
        在mask上绘制旋转矩形
        
        Args:
            mask: mask数组 [H,W]，会就地修改
            center: 中心点 (cx, cy)
            width: 矩形宽度
            height: 矩形高度
            angle: 旋转角度（度）
        """
        # 使用cv2.ellipse的矩形模式（通过设置角度参数）
        # 或者使用cv2.boxPoints + cv2.fillPoly
        box = cv2.boxPoints(((center[0], center[1]), (width, height), angle))
        box = box.astype(np.int32)
        cv2.fillPoly(mask, [box], 255)
    
    def draw_ellipse(self, mask: np.ndarray, center: Tuple[int, int], 
                     width: int, height: int, angle: float) -> None:
        """
        在mask上绘制旋转椭圆
        
        Args:
            mask: mask数组 [H,W]，会就地修改
            center: 中心点 (cx, cy)
            width: 椭圆宽度（长轴）
            height: 椭圆高度（短轴）
            angle: 旋转角度（度）
        """
        cv2.ellipse(mask, center, (width // 2, height // 2), angle, 0, 360, 255, -1)
    
    def add_brush_strokes(self, mask: np.ndarray, edge_density: np.ndarray,
                          debug_log_prefix: Optional[str] = None) -> None:
        """
        添加随机笔刷描边，模拟真实涂抹效果
        
        Args:
            mask: mask数组 [H,W]，会就地修改
            edge_density: 边缘密度图 [H,W]
            debug_log_prefix: debug模式下日志前缀
        """
        h, w = mask.shape
        params = self.params
        
        # 随机笔刷参数
        brush_width = np.random.randint(*params["brush_width_range"])
        num_segments = np.random.randint(*params["brush_segments_range"])
        
        # 在边缘密集处采样起点
        cy, cx = self.sample_center_from_density(
            edge_density,
            params["edge_density_weight"],
            params["edge_density_threshold"],
            debug_log_prefix=debug_log_prefix
        )
        
        # 生成随机笔画路径
        points = [(cx, cy)]
        for _ in range(num_segments):
            # 随机步长和方向
            angle = np.random.uniform(0, 2 * np.pi)
            step = np.random.randint(20, 60)
            
            last_x, last_y = points[-1]
            new_x = int(last_x + step * np.cos(angle))
            new_y = int(last_y + step * np.sin(angle))
            
            # 边界裁剪
            new_x = np.clip(new_x, 0, w - 1)
            new_y = np.clip(new_y, 0, h - 1)
            
            points.append((new_x, new_y))
        
        # 绘制笔刷线条
        for i in range(len(points) - 1):
            cv2.line(mask, points[i], points[i + 1], 255, brush_width)
    
    def apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """
        应用形态学操作增加不规则性
        
        Args:
            mask: mask数组 [H,W]
        
        Returns:
            processed_mask: 处理后的mask
        """
        kernel_size = np.random.randint(*self.params["morphology_kernel_size"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 随机选择操作类型
        operation = np.random.choice([cv2.MORPH_CLOSE, cv2.MORPH_OPEN, cv2.MORPH_DILATE])
        processed_mask = cv2.morphologyEx(mask, operation, kernel)
        
        return processed_mask
    
    def generate(self, edge_image: np.ndarray, 
                seed: Optional[int] = None) -> np.ndarray:
        """
        生成复杂mask
        
        Args:
            edge_image: 边缘图 [H,W,3] 或 [H,W]，白底黑线
            seed: 随机种子（可选）
        
        Returns:
            mask: [H,W,3] RGB mask，255为mask区域，0为非mask区域
        """
        if seed is not None:
            np.random.seed(seed)
        
        h, w = edge_image.shape[:2]
        params = self.params
        self._debug_counter += 1
        debug_prefix = f"mask_debug_{self._debug_counter:04d}" if self.debug_output_dir else None
        
        # 1. 计算边缘密度图
        edge_density = self.compute_edge_density(edge_image)
        if self.debug_output_dir:
            out_dir = Path(self.debug_output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / f"{debug_prefix}_edge_density.npy", edge_density)
            # 保存可视化（0-1映射到0-255）
            vis = (edge_density * 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{debug_prefix}_edge_density_vis.png"), vis)
            print(f"[MaskDebug] {debug_prefix} edge_density saved, "
                  f"min={edge_density.min():.4f} max={edge_density.max():.4f}")
        
        # 2. 初始化mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 3. 计算目标面积
        total_pixels = h * w
        target_ratio = np.random.uniform(*params["area_ratio_range"])
        target_area = int(total_pixels * target_ratio)
        
        # 4. 生成矩形块
        num_blocks = np.random.randint(*params["num_blocks_range"])
        area_per_block = target_area // num_blocks
        
        for i in range(num_blocks):
            # 采样中心点（基于边缘密度），仅第一次调用输出debug
            cy, cx = self.sample_center_from_density(
                edge_density,
                params["edge_density_weight"],
                params["edge_density_threshold"],
                debug_log_prefix=f"{debug_prefix}_block{i}" if (debug_prefix and i == 0) else None
            )
            
            # 计算块大小（确保不会太小）
            block_area = area_per_block * np.random.uniform(0.5, params["max_block_ratio"] * num_blocks)
            block_area = max(block_area, params["min_block_size"] ** 2)
            
            # 随机长宽比
            aspect_ratio = np.random.uniform(0.5, 2.0)
            block_w = int(np.sqrt(block_area * aspect_ratio))
            block_h = int(block_area / block_w)
            
            # 确保最小尺寸
            block_w = max(block_w, params["min_block_size"])
            block_h = max(block_h, params["min_block_size"])
            
            # 随机旋转角度
            angle = 0
            if np.random.random() < params["rotation_prob"]:
                angle = np.random.uniform(*params["rotation_range"])
            
            # 随机选择形状：矩形或椭圆
            if np.random.random() > 0.5:
                self.draw_rotated_rectangle(mask, (cx, cy), block_w, block_h, angle)
            else:
                self.draw_ellipse(mask, (cx, cy), block_w, block_h, angle)
        
        # 5. 添加笔刷描边
        if np.random.random() < params["brush_stroke_prob"]:
            num_strokes = np.random.randint(1, 4)  # 1-3条笔刷描边
            for stroke_i in range(num_strokes):
                dbg = f"{debug_prefix}_brush{stroke_i}" if (debug_prefix and stroke_i == 0) else None
                self.add_brush_strokes(mask, edge_density, debug_log_prefix=dbg)
        
        # 6. 应用形态学操作
        if np.random.random() < params["morphology_prob"]:
            mask = self.apply_morphology(mask)
        
        # 7. 转为RGB三通道
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        return mask_rgb


def generate_complex_mask(edge_image: np.ndarray, 
                         params: Optional[Dict] = None,
                         seed: Optional[int] = None) -> np.ndarray:
    """
    便捷函数：生成复杂mask
    
    Args:
        edge_image: 边缘图 [H,W,3] 或 [H,W]
        params: mask生成参数（可选）
        seed: 随机种子（可选）
    
    Returns:
        mask: [H,W,3] RGB mask
    """
    generator = ComplexMaskGenerator(params)
    return generator.generate(edge_image, seed)


if __name__ == "__main__":
    # 测试代码
    import sys
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Test complex mask generator")
    parser.add_argument("--edge_image", type=str, required=True, help="Path to edge image")
    parser.add_argument("--output", type=str, default="test_mask.png", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    args = parser.parse_args()
    
    # 加载边缘图
    edge_img = cv2.imread(args.edge_image)
    if edge_img is None:
        print(f"Failed to load image: {args.edge_image}")
        sys.exit(1)
    
    # 生成mask
    for i in range(args.num_samples):
        mask = generate_complex_mask(edge_img, seed=args.seed + i)
        
        # 保存
        output_path = args.output
        if args.num_samples > 1:
            output_dir = Path(args.output).parent
            output_name = Path(args.output).stem
            output_ext = Path(args.output).suffix
            output_path = output_dir / f"{output_name}_{i}{output_ext}"
        
        cv2.imwrite(str(output_path), mask)
        print(f"Saved mask to: {output_path}")
        
        # 可视化：叠加mask到边缘图
        overlay = edge_img.copy()
        overlay[mask[:, :, 0] > 0] = [0, 255, 0]  # 绿色mask
        overlay_path = str(output_path).replace(".png", "_overlay.png")
        cv2.imwrite(overlay_path, overlay)
        print(f"Saved overlay to: {overlay_path}")
