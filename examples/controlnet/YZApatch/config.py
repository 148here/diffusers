# -*- coding: utf-8 -*-
"""
YZApatch配置文件
=================

请根据实际环境修改顶部的路径配置区域。
"""

# ============================================================
# 用户配置区域 - 请根据实际路径修改以下配置
# ============================================================

# DexiNed代码目录路径（包含model.py的目录）
# 示例: "D:/Coding/lab/TSA-inpainting/codes/TSA-inpaint/DexiNed"
DEXINED_CODE_DIR = "D:/Coding/lab/TSA-inpainting/codes/TSA-inpaint/DexiNed"

# DexiNed模型checkpoint路径
# 示例: "D:/Coding/lab/TSA-inpainting/codes/TSA-inpaint/DexiNed/checkpoints/BIPED/10/10_model.pth"
DEXINED_CHECKPOINT = "D:/Coding/lab/TSA-inpainting/codes/TSA-inpaint/DexiNed/checkpoints/BIPED/10/10_model.pth"

# 训练图片数据目录（支持单目录或包含子目录）
# 示例: "D:/Coding/lab/TSA-inpainting/codes/data/train_images"
TRAIN_DATA_DIR = "D:/Coding/lab/TSA-inpainting/codes/data/train_images"

# 边缘缓存目录（用于加速训练，首次运行会自动创建）
# 示例: "D:/Coding/lab/TSA-inpainting/codes/cache/edges"
EDGE_CACHE_DIR = "D:/Coding/lab/TSA-inpainting/codes/cache/edges"

# sketch_util.py所在目录（用于导入）
# 示例: "D:/Coding/lab/TSA-inpainting/codes/TSA-inpaint"
SKETCH_UTIL_DIR = "D:/Coding/lab/TSA-inpainting/codes/TSA-inpaint"

# ============================================================
# 默认参数配置（一般不需要修改）
# ============================================================

# DexiNed参数
DEXINED_THRESHOLD = 45           # 边缘提取阈值（0-255），越高保留的边缘越少
DEXINED_DEVICE = "cuda"          # 推理设备: "cuda" or "cpu"

# Sketch生成参数（来自sketch_util.py的推荐值）
SKETCH_PARAMS = {
    "sigma_mean": 13.0,          # FFD位移强度基准值（像素）
    "sigma_std": 2.6,            # FFD位移倍率标准差（20%变化）
    "spatial_smooth_sigma": 2.0, # FFD空间平滑半径（像素）
    "cp_sigma_mean": 2.1,        # 线段控制点偏移基准值（像素）
    "cp_sigma_std": 0.4,         # 线段控制点倍率标准差（19%变化）
    "cp_spatial_smooth": 1.5,    # 线段控制点平滑半径（像素）
}

# Mask生成参数（复杂度：5-10个矩形块 + 笔刷描边）
MASK_PARAMS = {
    # 矩形块数量范围
    "num_blocks_range": (5, 10),
    
    # mask总面积占比范围（例如0.2表示占图像20%面积）
    "area_ratio_range": (0.2, 0.5),
    
    # 单个块最小尺寸（像素，确保mask不会太小）
    "min_block_size": 32,
    
    # 单个块最大占比（相对于总mask面积）
    "max_block_ratio": 0.3,
    
    # 旋转相关
    "rotation_prob": 0.7,        # 块旋转概率
    "rotation_range": (-30, 30), # 旋转角度范围（度）
    
    # 笔刷描边参数（增加不规则性）
    "brush_stroke_prob": 0.8,    # 添加笔刷描边概率
    "brush_width_range": (3, 10), # 笔刷宽度范围（像素）
    "brush_segments_range": (3, 8), # 每次描边的笔画段数
    
    # 边缘密度权重（>1则优先在边缘密集处放置mask）
    "edge_density_weight": 2.0,
    "edge_density_threshold": 0.3, # 边缘密度阈值（0-1）
    
    # 形态学操作（增加不规则性）
    "morphology_prob": 0.5,      # 应用形态学操作概率
    "morphology_kernel_size": (3, 7), # 形态学操作核大小范围
}

# 数据加载参数
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']  # 支持的图片格式
RECURSIVE_SCAN = True            # 是否递归扫描子目录

# 性能参数
EDGE_CACHE_VERSION = "v1"        # 缓存版本号（修改此值会使旧缓存失效）
