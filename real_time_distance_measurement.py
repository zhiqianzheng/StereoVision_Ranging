#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双目相机实时测距程序
功能：基于标定结果，实时测量物体距离
依赖：需要先运行stereo_calibration.py生成camera_config.py
使用：鼠标点击画面中的物体，显示距离
"""

import cv2
import numpy as np
import sys
import os
import time
from typing import Optional, Tuple

# 导入相机配置
sys.path.append(r'two_vision_calibration/calibration_code/calibration_results')
try:
    import camera_config as config
    print("✅ 成功导入相机配置")
    print(f"   基线距离: {config.BASELINE_MM:.2f}mm")
    print(f"   焦距: {config.FOCAL_LENGTH:.2f}pixels")
    print(f"   推荐最大测距: {config.get_recommended_max_distance()/1000:.1f}m")
except ImportError as e:
    print("❌ 错误：无法导入相机配置文件")
    print("   请确保已运行 stereo_calibration.py 生成配置文件")
    print(f"   错误详情: {e}")
    sys.exit(1)


class StereoDistanceDetector:
    """双目距离检测器"""
    
    def __init__(self):
        """初始化距离检测器"""
        # 鼠标点击位置
        self.mouse_x = 320
        self.mouse_y = 240
        
        # 创建立体匹配器
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=160,  # 必须是16的倍数
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # 创建WLS滤波器
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereo_matcher)
        self.wls_filter.setLambda(80000)
        self.wls_filter.setSigmaColor(1.2)
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo_matcher)
        
        # 创建校正映射表
        self._create_rectify_maps()
        
        print("🎯 双目距离检测器初始化完成")
        
    def _create_rectify_maps(self):
        """创建校正映射表"""
        # 图像尺寸（双目摄像头）
        image_size = (config.CAMERA_WIDTH//2, config.CAMERA_HEIGHT)
        
        # 左相机校正映射
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            config.LEFT_CAMERA_MATRIX,
            config.LEFT_DIST_COEFFS,
            config.RECTIFY_R1,
            config.RECTIFY_P1,
            image_size,
            cv2.CV_16SC2
        )
        
        # 右相机校正映射
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            config.RIGHT_CAMERA_MATRIX,
            config.RIGHT_DIST_COEFFS,
            config.RECTIFY_R2,
            config.RECTIFY_P2,
            image_size,
            cv2.CV_16SC2
        )
        
        print("📐 校正映射表创建完成")
    
    def rectify_images(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """校正立体图像对"""
        rectified_left = cv2.remap(img_left, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        return rectified_left, rectified_right
    
    def compute_disparity(self, rectified_left: np.ndarray, rectified_right: np.ndarray) -> np.ndarray:
        """计算视差图"""
        # 转换为灰度图
        if len(rectified_left.shape) == 3:
            gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = rectified_left
            gray_right = rectified_right
        
        # 计算视差
        disparity_left = self.stereo_matcher.compute(gray_left, gray_right)
        disparity_right = self.right_matcher.compute(gray_right, gray_left)
        
        # 转换数据类型
        disparity_left = disparity_left.astype(np.float32) / 16.0
        disparity_right = disparity_right.astype(np.float32) / 16.0
        
        # 使用WLS滤波器优化
        filtered_disparity = self.wls_filter.filter(
            disparity_left, gray_left, None, disparity_right
        )
        
        return filtered_disparity
    
    def disparity_to_distance(self, disparity: np.ndarray) -> np.ndarray:
        """将视差转换为距离（毫米）"""
        # 避免除零
        disparity_safe = np.where(disparity > 0, disparity, 0.1)
        
        # 使用配置文件中的函数计算距离
        distance = np.zeros_like(disparity)
        for i in range(disparity.shape[0]):
            for j in range(disparity.shape[1]):
                if disparity[i,j] > 0:
                    distance[i,j] = config.get_distance_from_disparity(disparity[i,j])
        
        # 限制测距范围
        distance = np.where(
            (distance >= config.MIN_VALID_DISTANCE_MM) & (distance <= config.MAX_VALID_DISTANCE_MM),
            distance, 0
        )
        
        return distance
    
    def get_distance_at_point(self, distance_map: np.ndarray, x: int, y: int, window_size: int = 5) -> Optional[float]:
        """获取指定点的距离"""
        h, w = distance_map.shape
        
        # 边界检查
        if x < 0 or x >= w or y < 0 or y >= h:
            return None
        
        # 在指定点周围取平均值
        half_window = window_size // 2
        y1 = max(0, y - half_window)
        y2 = min(h, y + half_window + 1)
        x1 = max(0, x - half_window)
        x2 = min(w, x + half_window + 1)
        
        window_distances = distance_map[y1:y2, x1:x2]
        valid_distances = window_distances[window_distances > 0]
        
        if len(valid_distances) > 0:
            return np.median(valid_distances)
        return None
    
    def create_distance_visualization(self, distance_map: np.ndarray) -> np.ndarray:
        """创建距离图的彩色可视化"""
        # 归一化距离图
        distance_viz = distance_map.copy()
        
        # 设置可视化范围
        min_dist = config.MIN_VALID_DISTANCE_MM
        max_dist = min(config.MAX_VALID_DISTANCE_MM, 3000)  # 限制最大显示范围
        
        # 归一化到0-255
        distance_viz = np.clip(distance_viz, min_dist, max_dist)
        distance_viz = ((distance_viz - min_dist) / (max_dist - min_dist) * 255).astype(np.uint8)
        
        # 无效区域设为0
        distance_viz[distance_map == 0] = 0
        
        # 应用颜色映射
        colored_distance = cv2.applyColorMap(distance_viz, cv2.COLORMAP_JET)
        
        # 无效区域设为黑色
        colored_distance[distance_map == 0] = [0, 0, 0]
        
        return colored_distance
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        self.mouse_x = x
        self.mouse_y = y
        
    def run_real_time_detection(self, camera_id: int = 20):
        """运行实时距离检测"""
        print("🎥 启动实时距离检测...")
        
        # 打开摄像头（HBVCAM-W2307-2双目摄像头）
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ 无法打开摄像头 {camera_id}")
            print("   请检查摄像头连接或更改camera_id")
            return False
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        print("✅ 摄像头启动成功")
        print("📖 操作说明:")
        print("   - 鼠标移动：选择测距点")
        print("   - 按 's'：保存当前距离图")
        print("   - 按 'q'：退出程序")
        
        # 创建窗口并设置鼠标回调
        cv2.namedWindow('Stereo Distance Detection', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Stereo Distance Detection', self.mouse_callback)
        
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        
        while True:
            # 读取图像
            ret, frame = cap.read()
            
            if not ret:
                print("❌ 读取摄像头失败")
                continue
            
            try:
                # 分离左右图像
                height, width = frame.shape[:2]
                left_img = frame[:, :width//2]
                right_img = frame[:, width//2:]
                
                # 图像校正
                rectified_left, rectified_right = self.rectify_images(left_img, right_img)
                
                # 计算视差
                disparity = self.compute_disparity(rectified_left, rectified_right)
                
                # 转换为距离
                distance_map = self.disparity_to_distance(disparity)
                
                # 获取鼠标点的距离
                point_distance = self.get_distance_at_point(distance_map, self.mouse_x, self.mouse_y)
                
                # 创建显示图像
                display_left = rectified_left.copy()
                distance_colored = self.create_distance_visualization(distance_map)
                
                # 绘制十字线
                cv2.line(display_left, (self.mouse_x - 20, self.mouse_y), 
                        (self.mouse_x + 20, self.mouse_y), (0, 255, 0), 2)
                cv2.line(display_left, (self.mouse_x, self.mouse_y - 20), 
                        (self.mouse_x, self.mouse_y + 20), (0, 255, 0), 2)
                cv2.circle(display_left, (self.mouse_x, self.mouse_y), 5, (0, 255, 0), -1)
                
                # 显示距离信息
                if point_distance is not None:
                    distance_text = f"Distance: {point_distance:.0f}mm ({point_distance/10:.1f}cm)"
                    cv2.putText(display_left, distance_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # 根据距离显示颜色提示
                    if point_distance < 1000:
                        color = (0, 0, 255)  # 红色 - 很近
                        warning = "VERY CLOSE"
                    elif point_distance < 2000:
                        color = (0, 165, 255)  # 橙色 - 较近
                        warning = "CLOSE"
                    elif point_distance < 4000:
                        color = (0, 255, 255)  # 黄色 - 中等
                        warning = "MEDIUM"
                    else:
                        color = (0, 255, 0)  # 绿色 - 远
                        warning = "FAR"
                    
                    cv2.putText(display_left, warning, (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.putText(display_left, "No valid distance", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # 计算FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                
                # 显示FPS
                cv2.putText(display_left, f"FPS: {fps:.1f}", (10, display_left.shape[0] - 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # 组合显示
                combined_display = np.hstack([display_left, distance_colored])
                cv2.imshow('Stereo Distance Detection', combined_display)
                
                # 键盘输入处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存距离图
                    timestamp = int(time.time())
                    save_path = f"distance_map_{timestamp}.png"
                    cv2.imwrite(save_path, distance_colored)
                    print(f"💾 距离图已保存: {save_path}")
                
            except Exception as e:
                print(f"❌ 处理错误: {e}")
                continue
        
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 程序已退出")
        
        return True


def main():
    """主函数"""
    print("=" * 60)
    print("🎯 双目相机实时测距系统")
    print("基于立体视觉的距离测量")
    print("=" * 60)
    
    try:
        # 创建距离检测器
        detector = StereoDistanceDetector()
        
        # 运行实时检测
        print("\n🚀 启动实时测距...")
        detector.run_real_time_detection(config.CAMERA_INDEX)
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()