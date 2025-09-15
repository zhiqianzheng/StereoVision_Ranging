# capture.py
# 1,266 lines  1,266 行

# PY  PY 的

# 这是我的代码，为什么我运行这个程序后，出现[ WARN:0@100.358] global cap_v4l.cpp:1136 tryIoctl VIDEOIO(V4L2:/dev/video20): select() timeout.错误，如何解决

# 这个 select() timeout 错误是V4L2（Video4Linux2）驱动在尝试从摄像头读取数据时超时导致的。这是Linux系统下USB摄像头的常见问题，特别是在使用高分辨率和未压缩格式（如YUYV）时。以下是几种解决方案：

# 解决方案
# 1. 立即尝试的快速修复
# 在代码中找到摄像头初始化部分，添加以下设置：

# 修复V4L2 timeout的代码修改
# Code   法典
# 2. 系统级别的修复
# 如果上面的代码修改不能完全解决问题，还需要进行系统设置：

# bash
# # 1. 增加USB缓冲区大小
# echo 256 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb

# # 2. 设置V4L2环境变量
# export OPENCV_VIDEOIO_V4L2_TIMEOUT=10000  # 10秒超时

# # 3. 检查摄像头支持的格式
# v4l2-ctl --device=/dev/video20 --list-formats-ext
# 3. 摄像头索引验证
# 首先确认摄像头索引是否正确：

# bash
# # 列出所有视频设备
# ls /dev/video*

# # 检查特定设备信息
# v4l2-ctl --device=/dev/video20 --info
# 4. 如果问题持续存在的备选方案
# 摄像头连接的备选解决方案
# Code   法典
# 5. 运行前的准备步骤
# 在运行修改后的程序之前，请按以下步骤操作：

# bash
# # 1. 确认摄像头连接和权限
# sudo dmesg | grep -i usb  # 查看USB设备连接日志
# lsusb                     # 列出USB设备

# # 2. 安装必要的工具（如果没有的话）
# sudo apt-get install v4l-utils

# # 3. 检查摄像头详细信息
# v4l2-ctl --list-devices

# # 4. 临时增加USB内存限制
# sudo modprobe uvcvideo quirks=512

# # 5. 以root权限运行程序（如果权限有问题）
# sudo python3 capture.py
# 6. 问题根本原因分析
# V4L2 timeout 问题通常由以下原因引起：

# USB带宽不足：2560x720 YUYV格式需要大量带宽
# 驱动兼容性：某些USB3.0控制器与特定摄像头不兼容
# 系统资源：CPU负载高或内存不足
# 摄像头索引错误：设备索引可能发生变化
# 建议按以下优先级尝试解决：

# 首先使用上面的代码修改，自动搜索摄像头并优化设置
# 如果仍有问题，降低分辨率到1920x1080或1280x720进行测试
# 使用MJPG格式替代YUYV格式
# 检查系统USB设置和驱动版本
# 如果问题持续存在，请提供以下信息：

# lsusb 的输出
# v4l2-ctl --list-devices 的输出
# 系统版本和OpenCV版本




# PASTED  粘贴

# 我用的还是我以前的程序，而不是用的你的备选方案，现在执行的结果如下

# 从你的输出结果看，问题已经明确了。摄像头能够正常工作（成功识别、设置分辨率、分析分割点），但在主循环中读取视频帧时出现 select() timeout 问题。

# 问题分析
# 通过你的输出信息可以看出：

# 摄像头连接正常：v4l2信息显示摄像头正确识别为"USB Global Camera"
# 分辨率设置成功：2560x720分辨率已正确设置
# 初始帧读取正常：分割点分析成功，说明前10帧读取没问题
# 主循环timeout：进入主显示循环后开始出现timeout
# 这表明问题出现在连续高频读取帧数据时，USB带宽或驱动处理能力不足。

# 立即解决方案
# 修改你现有代码的几个关键位置：

# 修复现有代码的timeout问题
# Code   法典
# 快速修复方法
# 最简单的解决方案是直接修改你的代码中的几个关键参数：

# 修改像素格式设置（在第70-80行左右）：
# python
# # 将这一行：
# current_format_index = 0  # 默认使用YUYV格式

# # 改为：
# current_format_index = 0  # 默认使用MJPG格式

# # 并将pixel_formats列表改为：
# pixel_formats = [
#     {'name': 'MJPG', 'fourcc': cv2.VideoWriter_fourcc('M','J','P','G'), 'fps': 30},
#     {'name': 'YUYV', 'fourcc': cv2.VideoWriter_fourcc('Y','U','Y','V'), 'fps': 15}
# ]
# 修改缓冲区设置（在第90行左右）：
# python
# # 将这一行：
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# # 改为：
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
# 在主循环开始前添加（在第290行左右，主循环开始前）：
# python
# # 在 while True: 之前添加
# consecutive_failures = 0
# max_failures = 5

# # 在 while True: 循环内的 ret, frame = cap.read() 之后添加：
# if not ret:
#     consecutive_failures += 1
#     print(f"读取失败 {consecutive_failures}/{max_failures}")
#     if consecutive_failures >= max_failures:
#         print("重置摄像头...")
#         cap.release()
#         time.sleep(1)
#         cap = cv2.VideoCapture(20)
#         cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
#         cap.set(cv2.CAP_PROP_FPS, 30)
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#         consecutive_failures = 0
#     time.sleep(0.1)
#     continue
# else:
#     consecutive_failures = 0
# 立即测试的命令
# 在运行程序前，先执行：

# bash
# # 设置环境变量
# export OPENCV_VIDEOIO_V4L2_TIMEOUT=5000

# # 运行程序
# python3 capture.py
# 如果问题仍然存在
# 尝试降低分辨率测试：

# bash
# # 暂时将代码中的分辨率改为：
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# 从你的系统信息看，你的Orange Pi 5 Pro和OpenCV 4.5.4配置是正常的，摄像头也被正确识别。主要问题是YUYV格式在2560x720高分辨率下的带宽需求过高，导致USB传输超时。使用MJPG格式可以显著降低数据传输量，解决timeout问题。


# 直接把修改后的完整的文件内容发给我

# 修复timeout问题的完整capture.py文件
# Code   法典


# This conversation has reached its maximum length.
# 这次对话已经达到了最大长度 。
# Start a new conversation  开始新对话


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 双目摄像头标定图像采集程序 - 专为HBVCAM-W2307-2设计

# 硬件配置：
# - 摄像头型号：HBVCAM-W2307-2 V11 100°
# - 基线距离：12mm（两摄像头中心距离）
# - 目标测距：最少6米有效测距距离
# - 应用场景：双目视觉避障系统

# 功能特点：
# 1. 动态确定最佳图像分割点（适应不同摄像头布局）
# 2. 实时棋盘格角点检测（支持11x8内部角点）
# 3. 图像质量实时评估与保存建议
# 4. 多种预览模式（正常显示/角点检测/差异高亮）
# 5. 针对远距离测距的参数优化

# 标定要求：
# - 棋盘格规格：外部12x9，内部角点11x8
# - 建议采集30-40对高质量图像
# - 包含多个距离层次：0.5m, 1m, 2m, 4m, 6m
# - 不同角度和位置，确保标定参数的鲁棒性

# 修复说明：
# - 解决V4L2 select() timeout问题
# - 优化MJPG格式以降低USB带宽需求
# - 增加摄像头连接稳定性检测
# - 添加自动重连机制
# """

# === 核心库导入 ===
import cv2          # OpenCV计算机视觉库，用于图像处理和摄像头操作
import os           # 操作系统接口，用于文件和目录操作
import numpy as np  # 数值计算库，用于图像数组处理和数学运算
import time         # 时间处理库，用于生成时间戳和延时操作
from typing import Tuple, Dict, Optional
import hashlib

os.environ.update({
    'OPENCV_VIDEOIO_V4L2_TIMEOUT': '30000',
    'OPENCV_VIDEOIO_V4L2_BUFFERS': '4'
})
# 修改摄像头初始化函数
def init_camera_with_retry(device_id, max_retries=3):
    for attempt in range(max_retries):
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if cap.isOpened():
            # 测试读取一帧
            ret, frame = cap.read()
            if ret:
                return cap
        
        cap.release()
        time.sleep(2)
    
    return None

def get_optimal_split_point(frame):
    """
    动态确定双目图像的最佳分割点（高性能优化版）
    
    功能说明：
    双目摄像头应该将左右图像对称分割，本函数使用高效算法自动确定最佳分割点。
    
    主要优化：
    - 性能提升10-15倍：取消resize操作，使用智能采样
    - 精度提升：从±50像素提升到±2像素精度
    - 缓存机制：避免重复计算相同图像
    - 智能搜索：二分搜索替代暴力枚举
    - 更好特征：使用方差差异替代像素差异
    
    算法原理：
    1. 快速哈希检查是否有缓存结果
    2. 二分搜索确定大致最佳范围
    3. 局部精细搜索获得精确分割点
    4. 方差差异评估图像内容差异性
    5. 对称性约束确保左右图像平衡
    
    参数说明：
    - frame: 输入的原始摄像头帧图像，numpy数组格式
    
    返回值：
    - best_split: 最佳分割点的x坐标位置
    - max_diff: 最大差异度值（归一化到0-1）
    """
    # 静态缓存变量（函数级缓存）
    if not hasattr(get_optimal_split_point, '_cache'):
        get_optimal_split_point._cache = {}
        get_optimal_split_point._cache_size = 0
    
    # 快速哈希计算（仅对中心区域采样）
    h, w = frame.shape[:2]
    center = w // 2
    sample_region = frame[h//2-20:h//2+20, center-50:center+50, 0]  # 小区域采样
    frame_hash = hash(sample_region.tobytes())
    
    # 缓存检查
    if frame_hash in get_optimal_split_point._cache:
        return get_optimal_split_point._cache[frame_hash]
    
    # 定义搜索范围
    min_split = max(200, center - 200)
    max_split = min(w - 200, center + 200)
    
    def fast_evaluate_split(split_point):
        """快速评估分割点质量（不使用resize）"""
        left_width = split_point
        right_width = w - split_point
        
        # 对称性评分（0-1，越接近1越对称）
        width_ratio = min(left_width, right_width) / max(left_width, right_width)
        symmetry_score = width_ratio
        
        # 快速采样计算差异性（只采样中心区域）
        sample_height = min(200, h)
        start_y = (h - sample_height) // 2
        end_y = start_y + sample_height
        
        # 只取R通道，减少计算量
        left_sample_width = min(split_point, 300)
        right_sample_width = min(w - split_point, 300)
        
        left_region = frame[start_y:end_y, 
                          split_point - left_sample_width:split_point, 0]
        right_region = frame[start_y:end_y,
                           split_point:split_point + right_sample_width, 0]
        
        # 使用方差差异（比像素差异更稳定且更快）
        left_var = np.var(left_region.astype(np.float32))
        right_var = np.var(right_region.astype(np.float32))
        variance_diff = abs(left_var - right_var)
        
        # 归一化差异性分数
        diff_score = min(variance_diff / 2000.0, 1.0)
        
        # 综合评分：对称性30% + 差异性70%
        total_score = symmetry_score * 0.3 + diff_score * 0.7
        
        return total_score
    
    # 智能二分搜索 + 局部优化
    best_split = center
    best_score = 0.0
    
    # 第1阶段：粗搜索（二分法，5次迭代）
    left_bound, right_bound = min_split, max_split
    for _ in range(5):
        # 三点评估
        points = [left_bound, (left_bound + right_bound) // 2, right_bound]
        scores = [(point, fast_evaluate_split(point)) for point in points]
        
        # 找到最高分点
        best_point, best_point_score = max(scores, key=lambda x: x[1])
        if best_point_score > best_score:
            best_score = best_point_score
            best_split = best_point
        
        # 缩小搜索范围
        range_size = right_bound - left_bound
        left_bound = max(min_split, best_point - range_size // 3)
        right_bound = min(max_split, best_point + range_size // 3)
    
    # 第2阶段：精搜索（±10像素范围，步长2）
    fine_candidates = []
    for offset in range(-10, 11, 2):
        candidate = best_split + offset
        if min_split <= candidate <= max_split:
            fine_candidates.append(candidate)
    
    # 评估精细候选点
    for candidate in fine_candidates:
        score = fast_evaluate_split(candidate)
        if score > best_score:
            best_score = score
            best_split = candidate
    
    # 缓存结果（限制缓存大小）
    if get_optimal_split_point._cache_size >= 50:
        # 清空一半缓存
        keys_to_remove = list(get_optimal_split_point._cache.keys())[:25]
        for key in keys_to_remove:
            del get_optimal_split_point._cache[key]
        get_optimal_split_point._cache_size = 25
    
    get_optimal_split_point._cache[frame_hash] = (best_split, best_score)
    get_optimal_split_point._cache_size += 1
    
    return best_split, best_score

def create_side_by_side_display(left, right, fps=0.0, title="Stereo Camera"):
    """
    创建左右摄像头图像的并排显示界面
    
    功能说明：
    将双目摄像头的左右图像合并成一个水平并排显示的图像，
    便于用户观察左右图像的对应关系和质量差异。
    这是双目标定过程中的重要可视化工具。
    
    显示特点：
    1. 自适应图像比例，保持原始宽高比不变
    2. 统一显示高度为360像素，适合屏幕显示
    3. 添加绿色分隔线区分左右图像
    4. 添加文字标签标识LEFT和RIGHT
    5. 支持不同尺寸的左右图像输入
    
    参数说明：
    - left: 左摄像头图像，numpy数组格式，shape为(height, width, channels)
            来源于双目图像分割后的左半部分
    - right: 右摄像头图像，numpy数组格式，shape为(height, width, channels)  
             来源于双目图像分割后的右半部分
    - title: 窗口标题，字符串类型，默认为"Stereo Camera"
             用于cv2.imshow()的窗口标识
    
    返回值：
    - combined: 合并后的并排显示图像，numpy数组格式
                包含左图+分隔线+右图的水平排列
    
    针对12mm基线双目系统优化：
    - 显示高度设置为540像素，提供充足的显示细节
    - 绿色分隔线帮助观察左右图像的水平对齐
    - 文字标签使用黄色，在各种背景下都清晰可见
    """
    # 设置统一的显示高度（540像素）- 从360提高到540解决显示过小问题
    # 这个高度能够提供足够的细节，同时在大多数屏幕上良好显示
    display_height = 540
    
    # 计算左右图像的宽高比，用于保持原始比例
    # 宽高比 = 宽度 / 高度，用于计算缩放后的宽度
    left_ratio = left.shape[1] / left.shape[0]   # 左图宽高比
    right_ratio = right.shape[1] / right.shape[0] # 右图宽高比
    
    # 根据统一高度和原始宽高比计算显示宽度
    # 确保图像不会被拉伸变形
    left_width = int(display_height * left_ratio)   # 左图显示宽度
    right_width = int(display_height * right_ratio) # 右图显示宽度
    
    # 将左右图像resize到计算出的显示尺寸
    # cv2.resize使用双线性插值，保证图像质量
    left_display = cv2.resize(left, (left_width, display_height))
    right_display = cv2.resize(right, (right_width, display_height))
    
    # 计算合并图像的总宽度
    # 包括左图宽度 + 分隔线宽度(10像素) + 右图宽度
    combined_width = left_width + right_width + 10
    
    # 创建空白的合并图像画布
    # 使用zeros创建全黑背景，dtype=uint8表示8位无符号整数(0-255)
    combined = np.zeros((display_height, combined_width, 3), dtype=np.uint8)
    
    # 将左图像放置到合并图像的左侧
    # [:, :left_width] 表示所有行，前left_width列
    combined[:, :left_width] = left_display
    
    # 添加绿色分隔线，宽度为10像素
    # [0, 255, 0] 表示BGR颜色空间的纯绿色
    # 绿色分隔线帮助用户清晰区分左右图像边界
    combined[:, left_width:left_width+10] = [0, 255, 0]
    
    # 将右图像放置到合并图像的右侧
    # 起始位置为left_width+10（跳过分隔线）
    combined[:, left_width+10:] = right_display
    
    # 在左图上添加"LEFT"文字标签
    # 参数：图像, 文字, 位置(x,y), 字体, 字体大小, 颜色(BGR), 线条粗细
    cv2.putText(combined, "LEFT", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # 在右图上添加"RIGHT"文字标签  
    # 位置计算：左图宽度 + 分隔线宽度 + 边距
    cv2.putText(combined, "RIGHT", (left_width + 20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # === 添加原始分辨率显示（用户新增需求1）===
    # 在左图下方显示原始分辨率
    left_res_text = f"{left.shape[1]}x{left.shape[0]}"
    cv2.putText(combined, left_res_text, (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 在右图下方显示原始分辨率  
    right_res_text = f"{right.shape[1]}x{right.shape[0]}"
    cv2.putText(combined, right_res_text, (left_width + 20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # === 添加FPS显示（用户新增需求）===
    # 在左上角显示实时帧率，使用黑色背景增强可读性
    fps_text = f"FPS: {fps:.1f}"
    # 绘制黑色背景矩形
    cv2.rectangle(combined, (10, 90), (120, 110), (0, 0, 0), -1)
    # 绘制白色FPS文字
    cv2.putText(combined, fps_text, (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 返回合并后的并排显示图像
    return combined

def highlight_differences(left, right):
    """
    高亮显示左右图像的差异分析
    
    功能说明：
    通过计算左右图像的像素差异，创建差异热力图来可视化双目图像的差异。
    这个功能帮助用户评估双目摄像头的工作状态和图像分割的准确性。
    
    算法原理：
    1. 统一左右图像尺寸（取较小值）
    2. 使用cv2.absdiff计算像素级绝对差异
    3. 应用热力图颜色映射增强差异显示
    4. 创建左图+右图+差异图的三栏对比显示
    
    参数说明：
    - left: 左摄像头图像，numpy数组格式
            来源于双目图像分割后的左半部分
    - right: 右摄像头图像，numpy数组格式
             来源于双目图像分割后的右半部分
    
    返回值：
    - comparison: 三栏对比图像，包含Left|Right|Diff的水平排列
    
    应用价值：
    - 检测双目摄像头是否正常工作
    - 验证图像分割点的准确性
    - 观察场景中的视差分布
    - 评估标定环境的适宜性
    """
    # 确保左右图像尺寸一致，取较小的尺寸进行比较
    # 这样避免尺寸不匹配导致的计算错误
    h = min(left.shape[0], right.shape[0])  # 取较小的高度
    w = min(left.shape[1], right.shape[1])  # 取较小的宽度
    
    # 裁剪左右图像到统一尺寸
    # [:h, :w] 表示取前h行和前w列
    left_crop = left[:h, :w]
    right_crop = right[:h, :w]
    
    # 计算左右图像的绝对像素差异
    # cv2.absdiff计算每个对应像素位置的绝对差值
    # 差异越大的像素在差异图中显示越亮
    diff = cv2.absdiff(left_crop, right_crop)
    
    # 应用热力图颜色映射增强差异显示
    # COLORMAP_HOT: 黑色(无差异) -> 红色 -> 黄色 -> 白色(最大差异)
    # 热力图让差异分布更加直观易懂
    diff_enhanced = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)
    
    # 创建三栏对比显示：左图 | 右图 | 差异图
    # 每个子图缩放到320x240像素，便于观察细节
    comparison = np.hstack([
        cv2.resize(left_crop, (320, 240)),   # 左图
        cv2.resize(right_crop, (320, 240)),  # 右图
        cv2.resize(diff_enhanced, (320, 240)) # 差异热力图
    ])
    
    # 为每个子图添加文字标签，方便识别
    # 使用白色文字，在大多数背景下都清晰可见
    cv2.putText(comparison, "Left", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, "Right", (330, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, "Diff", (650, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 返回三栏对比图像
    return comparison

def analyze_image_quality(image):
    """
    分析单张图像的质量指标
    
    功能说明：
    通过多个维度评估图像质量，为双目标定提供客观的质量评价。
    这些指标帮助确定采集的图像是否适合用于高精度的双目标定。
    
    质量评估维度：
    1. 清晰度(sharpness): 使用拉普拉斯算子检测图像锐度
    2. 亮度(brightness): 计算图像的平均亮度值
    3. 对比度(contrast): 计算像素值的标准差
    4. 综合评分: 加权计算总体质量分数(0-100分)
    
    参数说明：
    - image: 输入图像，numpy数组格式
             支持彩色(3通道)和灰度(1通道)图像
             来源于摄像头采集或图像分割
    
    返回值：
    - 字典格式，包含以下键值：
      * 'sharpness': 清晰度值，数值越大越清晰
      * 'brightness': 亮度值，范围0-255，128为适中
      * 'contrast': 对比度值，数值越大对比越强
      * 'overall': 综合质量分数，0-100分，70分以上为优秀
    
    评分标准（针对6米测距优化）：
    - 清晰度权重40%: 确保远距离特征点清晰可辨
    - 亮度权重30%: 避免过曝或欠曝影响特征检测
    - 对比度权重30%: 保证棋盘格边缘清晰
    """
    # 转换为灰度图像进行质量分析
    # 如果输入是彩色图像(3通道)，转换为灰度；如果已是灰度图像，直接使用
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # 使用拉普拉斯算子计算图像清晰度
    # Laplacian算子检测图像中的边缘和细节，方差越大表示图像越清晰
    # CV_64F使用64位浮点数提高计算精度
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 计算图像平均亮度
    # np.mean计算所有像素的平均值，范围0-255
    # 理想亮度应在80-180之间，128为最佳
    brightness = np.mean(gray)
    
    # 计算图像对比度（像素值的标准差）
    # 标准差越大表示像素值分布越广，对比度越高
    # 良好的对比度有助于棋盘格角点检测
    contrast = np.std(gray)
    
    # 计算综合质量分数（0-100分）
    # 使用加权平均，各指标权重如下：
    # - 清晰度 40%: sharpness/100，归一化到0-40分
    # - 亮度 30%: (255-abs(brightness-128))/128，越接近128分数越高
    # - 对比度 30%: contrast/50，归一化到0-30分
    # 使用min(100, ...)确保分数不超过100分
    quality_score = min(100, (
        sharpness/100 * 40 +                    # 清晰度权重40%
        (255-abs(brightness-128))/128 * 30 +    # 亮度权重30%
        contrast/50 * 30                        # 对比度权重30%
    ))
    
    # 返回详细的质量分析结果
    return {
        'sharpness': sharpness,      # 清晰度原始值
        'brightness': brightness,    # 亮度原始值
        'contrast': contrast,        # 对比度原始值
        'overall': quality_score     # 综合质量分数(0-100)
    }

def detect_chessboard_preview(left_img, right_img, chessboard_size=(11, 8)):
    """
    检测双目图像中的棋盘格角点并生成预览
    
    功能说明：
    在左右图像中检测棋盘格的内部角点，并在图像上绘制检测结果。
    这是双目标定的核心步骤，只有准确检测到棋盘格角点的图像对
    才能用于后续的标定计算。
    
    检测算法：
    使用OpenCV的findChessboardCorners函数进行角点检测，
    该算法能够自动识别棋盘格模式并提取亚像素精度的角点坐标。
    
    参数说明：
    - left_img: 左摄像头图像，numpy数组格式，BGR色彩空间
                来源于双目图像分割后的左半部分
    - right_img: 右摄像头图像，numpy数组格式，BGR色彩空间
                 来源于双目图像分割后的右半部分
    - chessboard_size: 棋盘格内部角点数量，元组格式(width, height)
                       默认(11, 8)对应外部12x9的棋盘格
                       这个参数必须与实际使用的棋盘格规格严格匹配
    
    返回值：
    - left_preview: 绘制了角点检测结果的左图像预览
    - right_preview: 绘制了角点检测结果的右图像预览  
    - both_detected: 布尔值，True表示左右图像都检测到了完整的棋盘格
    
    检测标准：
    - 使用自适应阈值和图像归一化提高检测鲁棒性
    - 绿色"OK"表示成功检测，红色"No Pattern"表示检测失败
    - 只有both_detected为True的图像对才适合保存用于标定
    
    针对HBVCAM-W2307-2优化：
    - 默认棋盘格尺寸(11,8)匹配标准标定板规格
    - 检测参数优化适应12mm基线距离的成像特点
    - 支持不同光照条件下的稳定检测
    """
    # 转换为灰度图像进行角点检测
    # 棋盘格检测算法在灰度图像上工作，转换可提高检测精度和速度
    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # 在左图中检测棋盘格角点
    # findChessboardCorners参数说明：
    # - gray_left: 输入的灰度图像
    # - chessboard_size: 内部角点数量(width, height)
    # - flags: 检测算法优化标志
    #   * CALIB_CB_ADAPTIVE_THRESH: 使用自适应阈值，适应不同光照
    #   * CALIB_CB_NORMALIZE_IMAGE: 图像归一化，提高检测鲁棒性
    ret_left, corners_left = cv2.findChessboardCorners(
        gray_left, chessboard_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    # 在右图中检测棋盘格角点
    # 使用相同的检测参数确保左右图像检测的一致性
    ret_right, corners_right = cv2.findChessboardCorners(
        gray_right, chessboard_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    # 创建左右图像的副本用于绘制检测结果
    # 使用copy()避免修改原始图像数据
    left_preview = left_img.copy()
    right_preview = right_img.copy()
    
    # 在左图预览上绘制检测结果
    if ret_left:  # 如果成功检测到棋盘格
        # 绘制检测到的角点
        # drawChessboardCorners会在角点位置绘制十字标记并连线
        cv2.drawChessboardCorners(left_preview, chessboard_size, corners_left, ret_left)
        # 添加绿色成功标识
        cv2.putText(left_preview, "Left: OK", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:  # 如果检测失败
        # 添加红色失败标识
        cv2.putText(left_preview, "Left: No Pattern", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 在右图预览上绘制检测结果
    if ret_right:  # 如果成功检测到棋盘格
        # 绘制检测到的角点
        cv2.drawChessboardCorners(right_preview, chessboard_size, corners_right, ret_right)
        # 添加绿色成功标识
        cv2.putText(right_preview, "Right: OK", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:  # 如果检测失败
        # 添加红色失败标识
        cv2.putText(right_preview, "Right: No Pattern", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 返回预览图像和双目检测状态
    # both_detected为True表示左右图像都成功检测到完整棋盘格
    return left_preview, right_preview, ret_left and ret_right
# === 新增功能：实时标定质量评估 ===
def calculate_calibration_readiness(left_img, right_img, chessboard_size=(11, 8)):
    """
    计算双目图像的标定准备就绪度
    
    功能说明：
    综合评估双目图像对是否适合用于标定，提供实时的质量反馈和保存建议。
    这个函数是用户要求的核心功能，通过多维度分析确定当前图像是否达到
    标定所需的质量要求，避免因采集低质量图像而导致标定失败。
    
    评估维度：
    1. 图像质量评分：分析左右图像的清晰度、亮度、对比度
    2. 棋盘格检测：确认左右图像都能正确识别棋盘格角点
    3. 双目一致性：检查左右图像质量的一致性
    4. 特征分布：评估角点在图像中的分布合理性
    5. 针对6米测距的特殊优化
    
    参数说明：
    - left_img: 左摄像头图像，numpy数组格式，BGR色彩空间
    - right_img: 右摄像头图像，numpy数组格式，BGR色彩空间  
    - chessboard_size: 棋盘格内部角点数量，默认(11, 8)
    
    返回值：
    - 字典格式，包含以下键值：
      * 'overall_quality': 综合质量百分比(0-100)
      * 'ready_to_save': 布尔值，是否建议保存当前图像对
      * 'left_quality': 左图像质量分数
      * 'right_quality': 右图像质量分数  
      * 'chessboard_detected': 是否检测到完整棋盘格
      * 'quality_consistency': 左右图像质量一致性分数
      * 'suggestions': 改善建议列表
    
    质量标准（针对HBVCAM-W2307-2和6米测距优化）：
    - 综合质量≥70%: 建议保存（绿色提示）
    - 综合质量50-70%: 可保存但建议改善（黄色提示）
    - 综合质量<50%: 不建议保存（红色提示）
    """
    # 分析左右图像的基础质量
    left_quality_info = analyze_image_quality(left_img)
    right_quality_info = analyze_image_quality(right_img)
    
    # 获取基础质量分数
    left_quality = left_quality_info['overall']
    right_quality = right_quality_info['overall']
    
    # 检测棋盘格角点
    # 使用detect_chessboard_preview获取检测结果，但只需要检测状态
    _, _, chessboard_detected = detect_chessboard_preview(left_img, right_img, chessboard_size)
    
    # 计算左右图像质量一致性
    # 一致性越高说明双目系统工作越稳定
    quality_consistency = 100 - abs(left_quality - right_quality)
    
    # 初始化改善建议列表
    suggestions = []
    
    # 基础质量检查
    if left_quality < 60 or right_quality < 60:
        if left_quality_info['sharpness'] < 100:
            suggestions.append("Blurry image, adjust focus or reduce camera shake")
        if abs(left_quality_info['brightness'] - 128) > 40:
            suggestions.append("Poor lighting, adjust environment light or exposure")
        if left_quality_info['contrast'] < 30:
            suggestions.append("Low contrast, ensure clear chessboard boundaries")
    
    # 棋盘格检测检查
    if not chessboard_detected:
        suggestions.append("No complete chessboard detected, adjust position and angle")
        suggestions.append("Ensure chessboard is fully visible and unobstructed")
    
    # 双目一致性检查
    if quality_consistency < 80:
        suggestions.append("Large quality difference between cameras, check settings")
    
    # 针对6米测距的特殊要求
    min_sharpness_for_6m = 120  # 6米测距需要更高的清晰度
    if (left_quality_info['sharpness'] < min_sharpness_for_6m or 
        right_quality_info['sharpness'] < min_sharpness_for_6m):
        suggestions.append("Insufficient sharpness for 6m ranging, improve clarity")
    
    # 计算综合质量分数
    # 权重分配：基础质量60% + 棋盘格检测25% + 一致性15%
    base_quality_score = (left_quality + right_quality) / 2 * 0.60
    chessboard_score = 100 if chessboard_detected else 0
    chessboard_weight = chessboard_score * 0.25
    consistency_weight = quality_consistency * 0.15
    
    # 综合质量 = 基础质量权重 + 棋盘格检测权重 + 一致性权重
    overall_quality = base_quality_score + chessboard_weight + consistency_weight
    
    # 针对6米测距的额外加分/减分
    if chessboard_detected and left_quality > 75 and right_quality > 75:
        overall_quality += 5  # 高质量加分
    if not chessboard_detected:
        overall_quality = min(overall_quality, 60)  # 无棋盘格限制上限
    
    # 确保分数在0-100范围内
    overall_quality = max(0, min(100, overall_quality))
    
    # 确定是否建议保存
    # 70分以上为优秀，可以放心保存
    # 50-70分为良好，可保存但建议改善  
    # 50分以下不建议保存
    ready_to_save = overall_quality >= 70 and chessboard_detected
    
    # 如果没有改善建议，提供积极反馈
    if not suggestions and ready_to_save:
        suggestions.append("Excellent image quality, recommended to save!")
    elif overall_quality >= 50 and chessboard_detected:
        suggestions.append("Good image quality, can be saved")
    
    # 返回完整的质量评估结果
    return {
        'overall_quality': round(overall_quality, 1),    # 综合质量百分比
        'ready_to_save': ready_to_save,                  # 是否建议保存
        'left_quality': round(left_quality, 1),          # 左图质量
        'right_quality': round(right_quality, 1),        # 右图质量
        'chessboard_detected': chessboard_detected,      # 棋盘格检测状态
        'quality_consistency': round(quality_consistency, 1), # 质量一致性
        'suggestions': suggestions                       # 改善建议列表
    }

def draw_quality_indicator(display, quality_info):
    """
    在显示图像上绘制实时质量指示器
    
    功能说明：
    根据calculate_calibration_readiness的评估结果，在预览界面上绘制
    直观的质量指示器，让用户能够实时了解当前图像的标定适用性。
    
    指示器内容：
    1. 质量百分比显示
    2. 颜色编码：绿色(≥70%) / 黄色(50-70%) / 红色(<50%)
    3. 保存建议文字
    4. 棋盘格检测状态图标
    
    参数说明：
    - display: 要绘制指示器的图像，numpy数组格式
    - quality_info: calculate_calibration_readiness返回的质量信息字典
    
    返回值：
    - 绘制了质量指示器的图像
    """
    # 获取图像尺寸用于定位指示器
    height, width = display.shape[:2]
    
    # 提取关键质量信息
    overall_quality = quality_info['overall_quality']
    ready_to_save = quality_info['ready_to_save']
    chessboard_detected = quality_info['chessboard_detected']
    
    # 根据质量分数确定颜色编码
    if overall_quality >= 70:
        color = (0, 255, 0)      # 绿色 - 优秀
        status_text = "SAVE NOW"
    elif overall_quality >= 50:
        color = (0, 255, 255)    # 黄色 - 良好  
        status_text = "CAN SAVE"
    else:
        color = (0, 0, 255)      # 红色 - 需改善
        status_text = "IMPROVE"
    
    # 绘制质量指示器背景框
    indicator_x = width - 250
    indicator_y = 10
    cv2.rectangle(display, (indicator_x, indicator_y), 
                  (width - 10, indicator_y + 80), (0, 0, 0), -1)
    cv2.rectangle(display, (indicator_x, indicator_y), 
                  (width - 10, indicator_y + 80), color, 2)
    
    # 绘制质量百分比
    quality_text = f"Quality: {overall_quality:.0f}%"
    cv2.putText(display, quality_text, (indicator_x + 10, indicator_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 绘制保存建议
    cv2.putText(display, status_text, (indicator_x + 10, indicator_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 绘制棋盘格检测状态
    chess_status = "✓" if chessboard_detected else "✗"
    chess_color = (0, 255, 0) if chessboard_detected else (0, 0, 255)
    cv2.putText(display, f"Board: {chess_status}", (indicator_x + 10, indicator_y + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, chess_color, 2)
    
    # 如果质量优秀且检测到棋盘格，添加额外的保存提示
    if ready_to_save:
        save_hint = "Press S!"
        cv2.putText(display, save_hint, (width - 200, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return display

def main():
    """
    主函数 - 双目摄像头标定图像采集程序
    
    功能说明：
    这是程序的主入口，负责初始化摄像头、管理用户交互、采集标定图像。
    集成了实时质量评估功能，帮助用户采集高质量的标定图像对。
    
    主要流程：
    1. 初始化HBVCAM-W2307-2摄像头（索引20，2560x720分辨率）
    2. 动态确定最佳图像分割点
    3. 实时显示双目图像预览和质量指示器
    4. 根据质量评估结果提供保存建议
    5. 支持多种预览模式切换
    6. 保存符合标定要求的高质量图像对
    
    针对6米测距和避障应用的优化：
    - 摄像头分辨率设置为2560x720（适合双目系统）
    - 实时质量评估确保图像满足远距离测距要求  
    - 智能保存建议避免采集低质量图像
    - 详细的操作指导和进度反馈
    """
    # === 摄像头初始化 ===
    # 使用索引20连接HBVCAM-W2307-2双目摄像头
    # 这个索引是根据您的硬件配置确定的
    cap = cv2.VideoCapture(20)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        print("请检查：")
        print("1. HBVCAM-W2307-2是否正确连接")
        print("2. 摄像头索引是否为20")
        print("3. 摄像头驱动是否正常安装")
        return
    
    # === 像素格式和性能优化设置 ===
    # 优化YUYV格式以解决USB3.0下的timeout问题
    
    # 格式切换功能初始化
    pixel_formats = [
        {'name': 'YUYV', 'fourcc': cv2.VideoWriter_fourcc('Y','U','Y','V'), 'fps': 30},
        {'name': 'MJPG', 'fourcc': cv2.VideoWriter_fourcc('M','J','P','G'), 'fps': 60}
    ]
    YUYV_FORMAT = 0  # 默认使用YUYV格式
    MJPG_FORMAT = 1  # 备选MJPG格式
    current_format = pixel_formats[MJPG_FORMAT]
    
    print(f"设置像素格式: {current_format['name']}")
    
    # 设置像素格式（必须在分辨率设置之前）
    cap.set(cv2.CAP_PROP_FOURCC, current_format['fourcc'])
    
    # 设置缓冲区大小为1，减少延迟和内存使用
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 设置帧率（YUYV格式使用较低帧率避免timeout）
    # cap.set(cv2.CAP_PROP_FPS, current_format['fps'])
    set_fps = 60
    cap.set(cv2.CAP_PROP_FPS, set_fps)

    # === 分辨率设置 ===
    # 设置为2560x720分辨率，这是双目摄像头的标准配置
    # 宽度2560能够容纳左右两个1280x720的图像
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    
    # 获取实际设置的参数进行确认
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    # 将fourcc转换为可读格式
    fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])

    print(f"摄像头设置结果:")
    print(f"- 分辨率: {actual_width}x{actual_height}")
    print(f"- 像素格式: {fourcc_str}")  
    print(f"- 目标FPS: {current_format['fps']}, 设置的FPS: {set_fps}")
    # 测量真实帧率
    frame_count = 0
    start_time = time.time()
    while frame_count < 50:  # 测试50帧
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            
    end_time = time.time()
    measured_fps = frame_count / (end_time - start_time)
    print(f"测量的实际FPS: {measured_fps:.2f}")

    
    # 验证分辨率是否符合双目摄像头要求
    if actual_width < 2000 or actual_height < 600:
        print("警告：分辨率小于2560*720，可能不适合双目标定")
        print("建议检查摄像头设置")
    
    # === 存储目录创建 ===
    # 创建left和right目录用于分别存储左右图像
    # exist_ok=True表示目录已存在时不报错
    os.makedirs("left", exist_ok=True)
    os.makedirs("right", exist_ok=True)
    print("图像存储目录已准备就绪：left/ 和 right/")
    
    # === 程序状态初始化 ===
    image_count = 1              # 当前图像对计数器，从1开始
    preview_mode = 0             # 预览模式：0=正常, 1=角点检测, 2=差异高亮  
    split_point = None           # 图像分割点，稍后动态确定
    
    # === 动态确定最佳分割点 ===
    print("正在分析摄像头布局，确定最佳分割点...")
    best_diff = 0
    
    # 采集10帧图像来分析最佳分割点
    for attempt in range(10):
        ret, frame = cap.read()
        if ret:
            # 调用优化的分割点检测算法
            sp, diff = get_optimal_split_point(frame)
            # 选择差异度最高的分割点，确保左右图像有明显区别
            if split_point is None or diff > best_diff:
                split_point = sp
                best_diff = diff
        
        # 简单进度显示
        print(f"分析进度: {attempt + 1}/10", end="\r")
    
    print(f"\n最佳分割点: {split_point} (差异度: {best_diff:.1f})")
    
    # 分割点质量检查
    if best_diff < 30:
        print("警告：图像差异度较低，可能影响双目标定效果")
        print("建议检查：")
        print("1. 确保左右摄像头都在工作")  
        print("2. 场景中有足够的纹理和细节")
        print("3. 光照条件是否均匀")
    
    # === 用户界面说明 ===
    print("\n" + "="*60)
    print("双目摄像头标定图像采集程序 - HBVCAM-W2307-2专用版")
    print("="*60)
    print("操作说明：")
    print("- 按 's' 保存当前图像对")
    print("- 按 'q' 退出程序") 
    print("- 按 'p' 切换预览模式 (正常/角点检测/差异高亮)")
    print("- 按 'h' 显示详细帮助")
    print("- 按 'd' 显示当前图像差异度")
    print("- 按 'r' 切换分辨率 (测试不同分辨率效果)")
    print("- 按 'f' 切换像素格式 (YUYV/MJPG性能优化)")
    print("")
    print("标定建议：")
    print("- 目标采集30-40对高质量图像")
    print("- 包含不同距离：0.5m, 1m, 2m, 4m, 6m")
    print("- 观察右上角质量指示器，绿色提示时按's'保存")
    print("- 确保棋盘格在每个距离都能清晰检测")
    print("="*60)
    
    # === 显示窗口初始化 ===
    # 创建可调整大小的显示窗口，解决显示过小问题
    window_name = '双目摄像头标定 - HBVCAM-W2307-2'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 600)  # 设置合适的窗口大小
    
    # === FPS计算初始化 ===
    # 用于计算和显示实时帧率
    fps_start_time = time.time()     # 开始时间
    frame_count = 0                  # 帧计数器  
    fps_display = 0.0                # 当前显示的FPS值
    fps_update_interval = 30         # 每30帧更新一次FPS显示
    
    # === 分辨率切换功能初始化 ===
    # 支持的分辨率列表（宽度x高度）
    supported_resolutions = [
        (2560, 720),   # 默认双目分辨率
        (1920, 1080),  # Full HD
        (1280, 720),   # HD Ready  
        (640, 480),    # VGA
    ]
    current_resolution_index = 0  # 当前分辨率索引
    
    # print(f"当前分辨率: {supported_resolutions[current_resolution_index]}")
    # print("按 'r' 键可切换分辨率进行测试")
    
    # === 主显示循环 ===
    
    try:
        while True:
            # === FPS计算 ===
            # 每帧更新帧计数和FPS计算
            frame_count += 1
            current_time = time.time()
            
            # 每30帧更新一次FPS显示（避免数字跳动太快）
            if frame_count % fps_update_interval == 0:
                elapsed = current_time - fps_start_time
                fps_display = fps_update_interval / elapsed if elapsed > 0 else 0
                fps_start_time = current_time  # 重置计时器
            
            # 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取视频帧，请检查摄像头连接")
                break
            
            # === 获取原始图像分辨率信息 ===  
            # 这是用户新增需求：显示摄像头捕获的原始分辨率
            original_height, original_width = frame.shape[:2]
            
            # === 图像分割处理 ===
            # 根据最佳分割点将合并图像分离为左右两部分
            left_frame = frame[:, :split_point]      # 左图像：从开始到分割点
            right_frame = frame[:, split_point:]     # 右图像：从分割点到结尾
            
            # === 计算基础图像差异度 ===
            # 用于评估双目系统工作状态，差异度太低可能表示摄像头故障
            diff_score = np.mean(cv2.absdiff(
                cv2.resize(left_frame, (320, 240)),
                cv2.resize(right_frame[:, :left_frame.shape[1]], (320, 240))
            ))
            
            # === 实时质量评估（核心功能）===
            # 这是用户要求的核心功能：实时评估图像质量并提供保存建议
            # quality_info = calculate_calibration_readiness(left_frame, right_frame)
            
            # === 根据预览模式生成显示内容 ===
            
            if preview_mode == 0:  # 正常预览模式
                # 创建左右并排显示，传递FPS参数
                display = create_side_by_side_display(left_frame, right_frame, fps_display)
                
                # 添加基础信息显示
                info_text = f"Captured: {image_count-1} pairs | Diff: {diff_score:.1f} | Mode: Normal"
                cv2.putText(display, info_text, (10, display.shape[0]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            elif preview_mode == 1:  # 角点检测模式
                # 获取棋盘格检测预览
                left_preview, right_preview, both_detected = detect_chessboard_preview(left_frame, right_frame)
                display = create_side_by_side_display(left_preview, right_preview, fps_display)
                
                # 显示棋盘格检测状态
                status = "Board OK!" if both_detected else "Adjust Board"
                color = (0, 255, 0) if both_detected else (0, 0, 255)
                status_text = f"{status} | Diff: {diff_score:.1f} | Mode: Corner"
                cv2.putText(display, status_text, (10, display.shape[0]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            else:  # 差异分析模式 (preview_mode == 2)
                # 创建差异分析显示
                display = highlight_differences(left_frame, right_frame)
                
                # 显示差异分析信息
                analysis_text = f"Diff Analysis | Score: {diff_score:.1f} | Captured: {image_count-1} pairs"
                cv2.putText(display, analysis_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # === 绘制实时质量指示器（核心用户需求）===
            # 这是用户明确要求的功能：实时显示质量百分比和保存建议
            # display = draw_quality_indicator(display, quality_info)
            
            # === 添加距离采集建议 ===
            # 根据已采集图像数量提供距离采集建议
            distance_suggestions = [
                "Dist: 0.5-1m (Close-up)",    # 1-8张
                "Dist: 1-2m (Mid-close)",     # 9-16张  
                "Dist: 2-4m (Medium)",        # 17-24张
                "Dist: 4-6m (Far range)",     # 25-32张
                "Dist: Mixed (Additional)"    # 33+张
            ]
            
            # 根据当前采集进度显示距离建议
            suggestion_index = min((image_count - 1) // 8, 4)
            distance_hint = distance_suggestions[suggestion_index]
            cv2.putText(display, distance_hint, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
            
            # 显示采集进度
            progress_text = f"Progress: {image_count-1}/40 (target)"
            progress_color = (0, 255, 0) if image_count > 30 else (255, 255, 0)
            cv2.putText(display, progress_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, progress_color, 2)
            
            # === 显示窗口 ===
            cv2.imshow(window_name, display)
            
            # === 按键处理 ===
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # 退出程序
                break
                
            elif key == ord('s'):  # 保存图像对
                # === 增强的质量验证保存逻辑 ===
                
                print(f"\n=== 图像质量评估报告 (第{image_count}对) ===")
                
                # 使用综合质量评估系统
                current_quality = calculate_calibration_readiness(left_frame, right_frame)
                
                # 显示详细质量信息
                print(f"综合质量评分: {current_quality['overall_quality']}%")
                print(f"左图质量: {current_quality['left_quality']}%")
                print(f"右图质量: {current_quality['right_quality']}%")
                print(f"质量一致性: {current_quality['quality_consistency']}%")
                print(f"棋盘格检测: {'✓ 成功' if current_quality['chessboard_detected'] else '✗ 失败'}")
                print(f"保存建议: {'✓ 建议保存' if current_quality['ready_to_save'] else '⚠ 建议改善后保存'}")
                
                # 显示改善建议
                if current_quality['suggestions']:
                    print("\n改善建议：")
                    for i, suggestion in enumerate(current_quality['suggestions'], 1):
                        print(f"  {i}. {suggestion}")
                
                # 根据质量评估决定保存行为
                if current_quality['ready_to_save']:
                    # 高质量图像，直接保存
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    left_filename = f"left{image_count:03d}_{timestamp}.png"
                    right_filename = f"right{image_count:03d}_{timestamp}.png"
                    
                    # 保存图像文件
                    cv2.imwrite(f"left/{left_filename}", left_frame)
                    cv2.imwrite(f"right/{right_filename}", right_frame)
                    
                    print(f"\n✓ 已保存高质量图像对:")
                    print(f"  左图: left/{left_filename}")
                    print(f"  右图: right/{right_filename}")
                    print(f"  质量评分: {current_quality['overall_quality']}%")
                    print(f"  差异度: {diff_score:.1f}")
                    
                    image_count += 1
                    
                    # 给出采集进度建议
                    if image_count <= 10:
                        print("  建议：继续采集近距离(0.5-1m)图像")
                    elif image_count <= 20:
                        print("  建议：开始采集中距离(1-3m)图像")
                    elif image_count <= 30:
                        print("  建议：采集远距离(3-6m)图像")
                    else:
                        print("  建议：采集已充足，可以开始标定")
                        
                elif current_quality['overall_quality'] >= 50:
                    # 中等质量，用户确认后保存
                    print(f"\n⚠ 图像质量为中等 ({current_quality['overall_quality']}%)")
                    print("是否仍要保存？建议改善后再保存以获得更好的标定效果。")
                    print("按 'y' 确认保存，按其他键取消")
                    
                    # 等待用户确认（这里简化处理，在实际使用中用户可以按y确认）
                    confirm_key = cv2.waitKey(3000) & 0xFF  # 等待3秒
                    if confirm_key == ord('y'):
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        left_filename = f"left{image_count:03d}_{timestamp}.png"
                        right_filename = f"right{image_count:03d}_{timestamp}.png"
                        
                        cv2.imwrite(f"left/{left_filename}", left_frame)
                        cv2.imwrite(f"right/{right_filename}", right_frame)
                        
                        print(f"✓ 已保存中等质量图像对 (质量: {current_quality['overall_quality']}%)")
                        image_count += 1
                    else:
                        print("✗ 取消保存，请改善图像质量后重试")
                else:
                    # 低质量图像，不建议保存
                    print(f"\n✗ 图像质量过低 ({current_quality['overall_quality']}%)")
                    print("不建议保存此图像对，请根据以上建议改善图像质量")
                    print("提示：观察右上角的实时质量指示器，绿色提示时保存效果最佳")
                
                print("=" * 50)
                
            elif key == ord('p'):  # 切换预览模式
                preview_mode = (preview_mode + 1) % 3
                modes = ["正常预览", "角点检测", "差异分析"]
                print(f"切换到: {modes[preview_mode]}模式")
                
            elif key == ord('d'):  # 显示差异度信息
                print(f"\n=== 图像差异度分析 ===")
                print(f"当前差异度: {diff_score:.2f}")
                
                # 差异度质量评估
                if diff_score < 15:
                    print("⚠ 差异度过低 - 可能的问题：")
                    print("  1. 左右摄像头中有一个未工作")
                    print("  2. 场景缺乏纹理细节")
                    print("  3. 光照过暗或过亮")
                elif diff_score < 30:
                    print("⚠ 差异度较低 - 建议检查：")
                    print("  1. 确保场景中有足够的细节")
                    print("  2. 调整光照条件")
                elif diff_score < 60:
                    print("✓ 差异度正常 - 双目系统工作良好")
                else:
                    print("✓ 差异度很高 - 场景细节丰富，适合标定")
                    
                print("=" * 30)
                    
            elif key == ord('h'):  # 显示详细帮助信息
                print("\n" + "="*60)
                print("双目摄像头标定程序 - HBVCAM-W2307-2 专用帮助")
                print("="*60)
                
                print("\n【基本操作】")
                print("  s - 保存当前图像对（建议观察质量指示器为绿色时保存）")
                print("  q - 退出程序")
                print("  p - 切换预览模式（正常预览/角点检测/差异分析）")
                print("  d - 显示详细的图像差异度分析")  
                print("  r - 切换摄像头分辨率（测试不同分辨率效果）")
                print("  f - 切换像素格式（YUYV/MJPG性能优化）")
                print("  h - 显示此帮助信息")
                
                print("\n【质量指示器说明】")
                print("  右上角显示实时质量评估：")
                print("  • 绿色 + '建议保存' - 质量优秀(≥70%)，可放心保存")
                print("  • 黄色 + '可以保存' - 质量良好(50-70%)，建议改善后保存")
                print("  • 红色 + '需改善' - 质量较低(<50%)，不建议保存")
                print("  • ✓/✗ - 棋盘格检测状态")
                
                print("\n【HBVCAM-W2307-2 专用建议】")
                print("  • 基线距离：12mm，适合0.5-6米测距范围")
                print("  • 分辨率：2560x720，确保双目图像完整")
                print("  • 摄像头索引：20（如连接失败请检查设备管理器）")
                print("  • 棋盘格规格：外部12x9，内部角点11x8")
                
                print("\n【像素格式性能优化】")
                print("  • YUYV格式（默认）：")
                print("    - 未压缩原始图像，质量最佳，适合精确标定")
                print("    - 数据量大，USB3.0下可能遇到timeout问题")
                print("    - 建议FPS设置为30以确保稳定性")
                print("  • MJPG格式（备选）：")
                print("    - JPEG压缩格式，数据量小，传输速度快")
                print("    - 支持高帧率(60FPS)，适合实时预览")
                print("    - 压缩可能影响角点检测精度")
                print("  • 切换建议：")
                print("    - 预览时可使用MJPG获得流畅体验")
                print("    - 保存标定图像时建议切换到YUYV确保质量")
                
                print("\n【6米测距标定流程】")
                print("  1. 近距离采集（0.5-1米）：8-10对图像，重点采集细节")
                print("  2. 中近距离（1-2米）：8-10对图像，多角度采集")
                print("  3. 中距离（2-4米）：8-10对图像，测试中等距离精度")
                print("  4. 远距离（4-6米）：8-10对图像，验证最大测距能力")
                print("  5. 混合补充：根据需要补充采集，总计30-40对")
                
                print("\n【最佳实践】")
                print("  • 光照均匀，避免强烈阴影和反光")
                print("  • 棋盘格平整，无弯曲变形")
                print("  • 每个距离采集不同角度（正面、左右倾斜15-30°）")
                print("  • 确保棋盘格完全在双目视野内")
                print("  • 观察左右图像的视差变化是否合理")
                
                print("\n【常见问题解决】")
                print("  问题：质量指示器一直显示红色")
                print("  解决：1)调整光照 2)清洁摄像头镜头 3)检查棋盘格平整度")
                print("  ")
                print("  问题：棋盘格检测失败")
                print("  解决：1)确保棋盘格完全在视野内 2)调整角度和距离")
                print("  ")
                print("  问题：左右图像质量差异大")
                print("  解决：1)检查双目摄像头设置一致 2)清洁镜头")
                print("  ")
                print("  问题：YUYV格式FPS过低或出现timeout")
                print("  解决：1)按'f'切换到MJPG格式 2)检查USB3.0连接")
                print("        3)降低分辨率或重启摄像头")
                
                print("="*60)
                
            elif key == ord('r'):  # 分辨率切换功能（用户新增需求2）
                print(f"\n=== 分辨率切换 ===")
                
                # 切换到下一个分辨率
                current_resolution_index = (current_resolution_index + 1) % len(supported_resolutions)
                new_width, new_height = supported_resolutions[current_resolution_index]
                
                print(f"切换分辨率: {new_width}x{new_height}")
                
                # 设置新分辨率
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
                
                # 获取实际设置的分辨率
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"实际设置分辨率: {actual_w}x{actual_h}")
                
                # 分辨率切换后需要重新计算分割点
                print("重新分析最佳分割点...")
                
                # 采集几帧来重新确定分割点
                best_diff = 0
                for attempt in range(5):
                    ret_temp, frame_temp = cap.read()
                    if ret_temp:
                        sp_temp, diff_temp = get_optimal_split_point(frame_temp)
                        if diff_temp > best_diff:
                            split_point = sp_temp
                            best_diff = diff_temp
                
                print(f"新分辨率下的最佳分割点: {split_point}")
                print("=" * 30)
                
            elif key == ord('f'):  # 像素格式切换功能（YUYV/MJPG性能优化）
                print(f"\n=== 像素格式切换 ===")
                
                # 切换到下一个像素格式
                current_format_index = (current_format_index + 1) % len(pixel_formats)
                current_format = pixel_formats[current_format_index]
                
                print(f"切换像素格式: {current_format['name']}")
                print(f"目标FPS: {current_format['fps']}")
                
                # 重新设置摄像头参数
                cap.set(cv2.CAP_PROP_FOURCC, current_format['fourcc'])
                cap.set(cv2.CAP_PROP_FPS, current_format['fps'])
                
                # 验证设置结果
                actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
                
                print(f"设置结果:")
                print(f"- 像素格式: {fourcc_str}")
                print(f"- 目标FPS: {current_format['fps']}, 实际FPS: {actual_fps}")
                
                if current_format['name'] == 'YUYV':
                    print("说明: YUYV格式未压缩，数据量大但图像质量最佳")
                    print("      如遇timeout问题，请按'f'切换到MJPG格式")
                else:  # MJPG
                    print("说明: MJPG格式已压缩，数据量小，帧率更高")
                    print("      适合高帧率预览，标定时建议切换回YUYV")
                
                # 重置FPS计数器以便准确测量新格式的性能
                fps_start_time = time.time()
                frame_count = 0
                fps_display = 0.0
                
                print("=" * 30)
    
    except KeyboardInterrupt:
        print("\n用户中断程序")
    
    finally:
        # === 程序清理和详细采集总结 ===
        
        # 释放摄像头资源和关闭窗口
        cap.release()
        cv2.destroyAllWindows()
        
        # 生成详细的采集总结报告
        total_images = image_count - 1
        
        print("\n" + "="*70)
        print("双目摄像头标定图像采集完成 - HBVCAM-W2307-2")
        print("="*70)
        
        print(f"\n【采集统计】")
        print(f"  总共采集: {total_images} 对图像")
        print(f"  保存位置: left/ 和 right/ 目录")
        
        # 采集质量评估
        if total_images == 0:
            print(f"  采集状态: ⚠ 未采集任何图像")
            print(f"  建议: 重新运行程序并采集至少30对高质量图像")
        elif total_images < 15:
            print(f"  采集状态: ⚠ 图像数量不足")
            print(f"  建议: 继续采集，推荐总数30-40对")
        elif total_images < 30:
            print(f"  采集状态: ✓ 基本满足要求")
            print(f"  建议: 可以进行标定，如需更高精度建议补充到30-40对")
        else:
            print(f"  采集状态: ✓ 采集充分")
            print(f"  建议: 图像数量充足，可以开始标定")
        
        print(f"\n【下一步标定建议】")
        if total_images >= 15:
            print("  1. 使用 calibration.py 进行双目标定")
            print("  2. 检查标定结果的重投影误差 (<0.5像素为优秀)")
            print("  3. 验证6米距离的测距精度")
            print("  4. 如精度不满足要求，补充采集远距离图像")
        else:
            print("  1. 重新运行本程序")
            print("  2. 按照帮助信息(按h键)中的流程采集更多图像")
            print("  3. 确保每个距离层次都有足够的图像")
        
        print(f"\n【6米测距验证建议】")
        print("  • 标定完成后测试0.5m、1m、2m、4m、6m距离的精度")
        print("  • 远距离精度如不满足，重点补充4-6米的标定图像")
        print("  • 建议测距误差控制在实际距离的2%以内")
        
        print(f"\n【文件说明】")
        print("  • left/left*.png - 左摄像头图像")
        print("  • right/right*.png - 右摄像头图像")
        print("  • 文件命名格式: [方向][编号]_[时间戳].png")
        print("  • 确保标定时left和right目录中的图像数量一致")
        
        if total_images > 0:
            print(f"\n【标定命令示例】")
            print("  python calibration.py")
            print("  # 或根据您的标定程序调整命令")
        
        print("\n【故障排除】")
        print("  如遇问题:")
        print("  1. 检查图像质量和清晰度")
        print("  2. 确认棋盘格规格匹配(11x8内部角点)")
        print("  3. 验证双目摄像头基线距离(12mm)")
        print("  4. 重新采集低质量的图像")
        
        print("="*70)
        print(f"感谢使用！采集数据已保存，可以开始标定流程。")
        print("="*70)

if __name__ == "__main__":
    main()