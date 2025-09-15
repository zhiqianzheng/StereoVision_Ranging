#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双目相机通用测距程序
功能：基于标定结果进行实时距离测量
使用方式：
1. 先运行 stereo_calibration.py 生成 camera_config.py
2. 运行此程序进行实时测距
3. 鼠标点击画面中的物体查看距离

特点：完全通用，通过导入camera_config配置文件获取参数
"""

import cv2
import numpy as np
import sys
import os
import time
from typing import Optional, Tuple

# 动态导入相机配置
try:
    # 添加配置文件路径
    config_path = r'two_vision_calibration/calibration_code/calibration_results'
    if config_path not in sys.path:
        sys.path.append(config_path)
    
    import camera_config
    print("✅ 相机配置加载成功")
    
except ImportError as e:
    print("❌ 无法导入相机配置文件")
    print("   请确保已运行 stereo_calibration.py 生成 camera_config.py")
    print(f"   错误详情: {e}")
    sys.exit(1)


class UniversalStereoRangeFinder:
    """通用双目测距器"""
    
    def __init__(self):
        """初始化测距器，使用导入的配置"""
        self.config = camera_config
        
        # 鼠标位置
        self.mouse_x = 320
        self.mouse_y = 240
        
        # 初始化立体匹配器
        self._init_stereo_matcher()
        
        # 创建图像校正映射
        self._create_rectification_maps()
        
        print("🎯 通用双目测距器初始化完成")
        
    def _init_stereo_matcher(self):
        """初始化立体匹配器"""
        # 使用SGBM算法
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
        
        # WLS滤波器
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo_matcher)
        self.wls_filter.setLambda(80000)
        self.wls_filter.setSigmaColor(1.2)
        
        # 右视图匹配器
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo_matcher)
        
    def _create_rectification_maps(self):
        """创建图像校正映射"""
        # 计算图像尺寸
        image_width = self.config.CAMERA_WIDTH // 2  # 双目摄像头分左右
        image_height = self.config.CAMERA_HEIGHT
        image_size = (image_width, image_height)
        
        # 左相机校正映射
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            self.config.LEFT_CAMERA_MATRIX,
            self.config.LEFT_DIST_COEFFS,
            self.config.RECTIFY_R1,
            self.config.RECTIFY_P1,
            image_size,
            cv2.CV_16SC2
        )
        
        # 右相机校正映射
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            self.config.RIGHT_CAMERA_MATRIX,
            self.config.RIGHT_DIST_COEFFS,
            self.config.RECTIFY_R2,
            self.config.RECTIFY_P2,
            image_size,
            cv2.CV_16SC2
        )
        
        print("📐 图像校正映射创建完成")
        
    def rectify_images(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """校正图像对"""
        left_rectified = cv2.remap(left_img, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_img, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
        return left_rectified, right_rectified
        
    def compute_disparity(self, left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
        """计算视差图"""
        # 计算左右视差
        disparity_left = self.stereo_matcher.compute(left_gray, right_gray)
        disparity_right = self.right_matcher.compute(right_gray, left_gray)
        
        # 转换数据类型
        disparity_left = disparity_left.astype(np.float32) / 16.0
        disparity_right = disparity_right.astype(np.float32) / 16.0
        
        # 使用WLS滤波器优化
        filtered_disparity = self.wls_filter.filter(
            disparity_left, left_gray, None, disparity_right
        )
        
        return filtered_disparity
        
    def disparity_to_distance(self, disparity: np.ndarray) -> np.ndarray:
        """将视差转换为距离（使用配置文件中的函数）"""
        distance_map = np.zeros_like(disparity)
        
        # 使用配置文件中的距离计算函数
        valid_mask = disparity > 0
        distance_map[valid_mask] = self.config.get_distance_from_disparity(disparity[valid_mask])
        
        # 应用距离范围限制
        distance_map = np.where(
            (distance_map >= self.config.MIN_VALID_DISTANCE_MM) & 
            (distance_map <= self.config.MAX_VALID_DISTANCE_MM),
            distance_map, 0
        )
        
        return distance_map
        
    def get_point_distance(self, distance_map: np.ndarray, x: int, y: int) -> Optional[float]:
        """获取指定点的距离"""
        h, w = distance_map.shape
        
        if x < 0 or x >= w or y < 0 or y >= h:
            return None
            
        # 使用配置文件中的窗口大小
        window_size = self.config.DISTANCE_FILTER_WINDOW
        half_window = window_size // 2
        
        y1 = max(0, y - half_window)
        y2 = min(h, y + half_window + 1)
        x1 = max(0, x - half_window)
        x2 = min(w, x + half_window + 1)
        
        window_region = distance_map[y1:y2, x1:x2]
        valid_distances = window_region[window_region > 0]
        
        if len(valid_distances) > 0:
            return float(np.median(valid_distances))
        return None
        
    def create_depth_visualization(self, distance_map: np.ndarray) -> np.ndarray:
        """创建深度图可视化"""
        # 归一化处理
        viz_map = distance_map.copy()
        
        # 使用配置文件中的范围
        min_dist = self.config.MIN_VALID_DISTANCE_MM
        max_dist = min(self.config.MAX_VALID_DISTANCE_MM, 3000)
        
        # 归一化到0-255
        viz_map = np.clip(viz_map, min_dist, max_dist)
        viz_map = ((viz_map - min_dist) / (max_dist - min_dist) * 255).astype(np.uint8)
        viz_map[distance_map == 0] = 0
        
        # 应用颜色映射
        colored_depth = cv2.applyColorMap(viz_map, cv2.COLORMAP_JET)
        colored_depth[distance_map == 0] = [0, 0, 0]
        
        return colored_depth
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调"""
        self.mouse_x = x
        self.mouse_y = y
        
    def run(self):
        """运行实时测距"""
        print("🚀 启动实时测距系统...")
        
        # 打开摄像头
        cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
        
        if not cap.isOpened():
            print(f"❌ 无法打开摄像头 {self.config.CAMERA_INDEX}")
            return False
            
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
        
        print("✅ 摄像头启动成功")
        print("📋 操作说明:")
        print("   - 移动鼠标选择测距点")
        print("   - 按 's' 保存深度图")
        print("   - 按 'q' 退出程序")
        
        # 创建窗口
        cv2.namedWindow('Universal Stereo Range Finder', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Universal Stereo Range Finder', self.mouse_callback)
        
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            try:
                # 分离左右图像
                height, width = frame.shape[:2]
                left_image = frame[:, :width//2]
                right_image = frame[:, width//2:]
                
                # 图像校正
                left_rect, right_rect = self.rectify_images(left_image, right_image)
                
                # 转换为灰度图
                left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
                
                # 计算视差
                disparity = self.compute_disparity(left_gray, right_gray)
                
                # 转换为距离
                distance_map = self.disparity_to_distance(disparity)
                
                # 获取鼠标点距离
                point_distance = self.get_point_distance(distance_map, self.mouse_x, self.mouse_y)
                
                # 创建显示图像
                display_image = left_rect.copy()
                depth_colored = self.create_depth_visualization(distance_map)
                
                # 绘制测距十字线
                cv2.line(display_image, (self.mouse_x - 15, self.mouse_y), 
                        (self.mouse_x + 15, self.mouse_y), (0, 255, 0), 2)
                cv2.line(display_image, (self.mouse_x, self.mouse_y - 15), 
                        (self.mouse_x, self.mouse_y + 15), (0, 255, 0), 2)
                cv2.circle(display_image, (self.mouse_x, self.mouse_y), 3, (0, 255, 0), -1)
                
                # 显示距离信息
                if point_distance:
                    dist_text = f"Distance: {point_distance:.0f}mm ({point_distance/10:.1f}cm)"
                    cv2.putText(display_image, dist_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 距离状态指示
                    if point_distance < 1000:
                        status_color = (0, 0, 255)  # 红色
                        status_text = "CLOSE"
                    elif point_distance < 3000:
                        status_color = (0, 255, 255)  # 黄色
                        status_text = "MEDIUM"
                    else:
                        status_color = (0, 255, 0)  # 绿色
                        status_text = "FAR"
                        
                    cv2.putText(display_image, status_text, (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                else:
                    cv2.putText(display_image, "No distance data", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 计算并显示FPS
                fps_counter += 1
                if fps_counter >= 30:
                    current_fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    fps_counter = 0
                
                cv2.putText(display_image, f"FPS: {current_fps:.1f}", 
                          (10, display_image.shape[0] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 合并显示
                combined_view = np.hstack([display_image, depth_colored])
                cv2.imshow('Universal Stereo Range Finder', combined_view)
                
                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    save_name = f"depth_map_{timestamp}.png"
                    cv2.imwrite(save_name, depth_colored)
                    print(f"💾 深度图已保存: {save_name}")
                    
            except Exception as e:
                print(f"❌ 处理异常: {e}")
                continue
                
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 程序结束")
        return True


def main():
    """主函数"""
    print("=" * 60)
    print("🎯 通用双目相机测距系统")
    print("支持任意双目摄像头配置")
    print("=" * 60)
    
    try:
        # 创建测距器
        range_finder = UniversalStereoRangeFinder()
        
        # 运行测距
        range_finder.run()
        
    except KeyboardInterrupt:
        print("\n⏹️ 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 运行异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()